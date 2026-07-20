#!/usr/bin/env python3
"""Validate the files and metadata used by the training/validation loader."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torchaudio
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crowd_reaction.data import (  # noqa: E402
    _truthy_strong_label,
    build_split_records,
    normalize_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that all metadata rows resolve to the data used by training."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Open every referenced WAV header and check chunk bounds (slower).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalized_index(paths: list[Path], label: str, errors: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in paths:
        key = normalize_name(path.name)
        if key in index and index[key] != path:
            errors.append(
                f"ambiguous {label} names after normalization: {index[key].name!r} and {path.name!r}"
            )
        else:
            index[key] = path
    return index


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    data = config["data"]

    info_csv = Path(data["audios_info_csv"])
    weak_csv = Path(data["weak_labels_csv"])
    strong_dir = Path(data["strong_labels_dir"])
    original_dir = Path(data["original_audio_dir"])
    negative_value = data.get("negative_data_dir")
    negative_dir = Path(negative_value) if negative_value else None

    errors: list[str] = []
    warnings: list[str] = []
    for label, path, kind in (
        ("audios_info_csv", info_csv, "file"),
        ("weak_labels_csv", weak_csv, "file"),
        ("strong_labels_dir", strong_dir, "directory"),
        ("original_audio_dir", original_dir, "directory"),
    ):
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            errors.append(f"{label} is missing or is not a {kind}: {path}")

    if negative_dir is not None and not negative_dir.is_dir():
        warnings.append(
            f"optional negative_data_dir is absent (the loader silently skips it): {negative_dir}"
        )

    if errors:
        for message in errors:
            print(f"ERROR: {message}")
        for message in warnings:
            print(f"WARNING: {message}")
        return 1

    original_paths = sorted(original_dir.glob("*.wav"))
    strong_paths = sorted(strong_dir.glob("*.txt"))
    original_index = normalized_index(original_paths, "original WAV", errors)
    strong_index = normalized_index(strong_paths, "strong TXT", errors)

    info_df = pd.read_csv(info_csv)
    weak_df = pd.read_csv(weak_csv)
    info_required = {"title", "strong_label"}
    weak_required = {
        "source_file", "start_sec", "end_sec", "no_crowd",
        "clear_disapproval", "unclear_disapproval", "unclear_approval",
        "clear_approval", "crowd_chorus",
    }
    for column in sorted(info_required - set(info_df.columns)):
        errors.append(f"{info_csv}: missing required column {column!r}")
    for column in sorted(weak_required - set(weak_df.columns)):
        errors.append(f"{weak_csv}: missing required column {column!r}")
    if errors:
        for message in errors:
            print(f"ERROR: {message}")
        return 1

    weak_keys: list[str] = []
    missing_weak: Counter[str] = Counter()
    referenced_paths: set[Path] = set()
    for row_number, row in weak_df.iterrows():
        source = str(row["source_file"])
        key = normalize_name(source)
        weak_keys.append(key)
        audio_path = original_index.get(key)
        if audio_path is None:
            missing_weak[source] += 1
            continue
        referenced_paths.add(audio_path)
        try:
            start, end = float(row["start_sec"]), float(row["end_sec"])
            if start < 0 or end <= start:
                errors.append(
                    f"weak CSV row {row_number + 2} has invalid interval [{start}, {end}] for {source!r}"
                )
        except (TypeError, ValueError):
            errors.append(f"weak CSV row {row_number + 2} has non-numeric start_sec/end_sec")

    for source, count in missing_weak.items():
        errors.append(f"{count} weak row(s) for {source!r} have no matching original WAV")

    for key, txt_path in strong_index.items():
        audio_path = original_index.get(key)
        if audio_path is None:
            errors.append(f"strong TXT {txt_path.name!r} has no matching original WAV")
        else:
            referenced_paths.add(audio_path)

    strong_metadata_keys: set[str] = set()
    for _, row in info_df.iterrows():
        if _truthy_strong_label(row["strong_label"]):
            key = normalize_name(row["title"])
            strong_metadata_keys.add(key)
            if key not in original_index:
                errors.append(f"strong-labelled metadata title {row['title']!r} has no matching original WAV")
            elif key not in strong_index:
                errors.append(f"strong-labelled metadata title {row['title']!r} has no matching strong TXT")

    if args.check_audio:
        duration_by_path: dict[Path, float] = {}
        all_used_paths = set(referenced_paths)
        if negative_dir and negative_dir.is_dir():
            all_used_paths.update(negative_dir.glob("*.wav"))
        for path in sorted(all_used_paths):
            try:
                audio_info = torchaudio.info(str(path))
                duration_by_path[path] = audio_info.num_frames / audio_info.sample_rate
                if audio_info.num_frames <= 0 or audio_info.sample_rate <= 0:
                    errors.append(f"empty or invalid audio: {path}")
            except Exception as exc:  # torchaudio backend errors vary by installation
                errors.append(f"cannot read audio {path}: {exc}")
        for row_number, row in weak_df.iterrows():
            path = original_index.get(normalize_name(str(row["source_file"])))
            if path in duration_by_path:
                try:
                    end = float(row["end_sec"])
                    if end > duration_by_path[path] + 0.05:
                        errors.append(
                            f"weak CSV row {row_number + 2} ends at {end:.3f}s, beyond "
                            f"{path.name!r} duration {duration_by_path[path]:.3f}s"
                        )
                except (TypeError, ValueError):
                    pass

    # Finally exercise the production record builder itself, catching any rule the
    # explicit diagnostics above did not cover.
    try:
        splits = build_split_records(
            audios_info_csv=str(info_csv),
            weak_labels_csv=str(weak_csv),
            strong_labels_dir=str(strong_dir),
            original_audio_dir=str(original_dir),
            negative_data_dir=str(negative_dir) if negative_dir else None,
            chunk_sec=float(data["chunk_sec"]),
            unclear_label_weight=float(config.get("loss", {}).get("unclear_label_weight", 0.5)),
        )
    except Exception as exc:
        errors.append(f"training loader rejected the data: {type(exc).__name__}: {exc}")
        splits = None

    print(f"Original WAVs: {len(original_paths)}")
    print(f"Weak-label rows: {len(weak_df)} ({len(set(weak_keys))} unique normalized sources)")
    print(f"Strong-label TXTs: {len(strong_paths)}")
    print(f"Strong-labelled metadata titles: {len(strong_metadata_keys)}")
    if splits is not None:
        print(f"Training records: {len(splits.train_records)}")
        print(f"Validation records: {len(splits.val_records)}")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors:
        print(f"FAIL: found {len(errors)} problem(s)")
        return 1
    print("OK: every metadata-derived training/validation item resolves correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
