#!/usr/bin/env python3
"""Validate the files and metadata used by the training/validation loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import torch
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
    parser.add_argument("--config", default="configs/beats.yaml")
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Open every referenced WAV header and check chunk bounds (slower).",
    )
    parser.add_argument(
        "--decode-audio", action="store_true",
        help="Decode all referenced audio in blocks and check finite samples (implies --check-audio).",
    )
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Write a SHA-256 inventory of checked data after a successful audit.")
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
        errors.append(
            f"configured negative_data_dir is absent (training would silently lose negatives): {negative_dir}"
        )

    if errors:
        for message in errors:
            print(f"ERROR: {message}")
        for message in warnings:
            print(f"WARNING: {message}")
        return 1

    if negative_dir is not None and not list(negative_dir.glob("*.wav")):
        errors.append(f"configured negative_data_dir contains no WAV files: {negative_dir}")

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
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
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
            if key not in original_index:
                # Downloaded filenames may include a YouTube ID or the literal
                # " - video" suffix. Resolve only these explicit conventions;
                # never use a fuzzy title match that could hide a missing file.
                parsed_url = urlparse(str(row.get("url", "")))
                video_id = None
                if parsed_url.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
                    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
                elif parsed_url.hostname == "youtu.be":
                    video_id = parsed_url.path.strip("/")
                candidates = [
                    candidate_key for candidate_key, path in original_index.items()
                    if (video_id and path.name.removeprefix("noise_").startswith(f"Youtube-{video_id}-"))
                    or candidate_key == normalize_name(str(row["title"]) + " - video")
                ]
                if len(candidates) == 1:
                    key = candidates[0]
                    warnings.append(f"metadata title {row['title']!r} resolves by download naming convention to {original_index[key].name!r}")
                elif len(candidates) > 1:
                    errors.append(f"ambiguous downloaded audio for metadata title {row['title']!r}")
            strong_metadata_keys.add(key)
            if key not in original_index:
                errors.append(f"strong-labelled metadata title {row['title']!r} has no matching original WAV")
            elif key not in strong_index:
                errors.append(f"strong-labelled metadata title {row['title']!r} has no matching strong TXT")

    strong_intervals: list[tuple[Path, int, float, float]] = []
    for key, label_path in strong_index.items():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                onset, offset, label = line.strip().split("\t")
                onset, offset = float(onset), float(offset)
                if not math.isfinite(onset) or not math.isfinite(offset) or onset < 0 or offset <= onset:
                    raise ValueError("interval must be finite with 0 <= onset < offset")
                if key in original_index:
                    strong_intervals.append((original_index[key], line_number, onset, offset))
            except ValueError as exc:
                errors.append(f"invalid strong interval in {label_path.name}:{line_number}: {exc}")

    if args.check_audio or args.decode_audio:
        duration_by_path: dict[Path, float] = {}
        all_used_paths = set(referenced_paths)
        if negative_dir and negative_dir.is_dir():
            all_used_paths.update(negative_dir.glob("*.wav"))
        for path in sorted(all_used_paths):
            try:
                audio_info = torchaudio.info(str(path))
                if audio_info.num_frames <= 0 or audio_info.sample_rate <= 0:
                    errors.append(f"empty or invalid audio: {path}")
                    continue
                duration_by_path[path] = audio_info.num_frames / audio_info.sample_rate
                if args.decode_audio:
                    block_size = audio_info.sample_rate * 60
                    for offset in range(0, audio_info.num_frames, block_size):
                        expected = min(block_size, audio_info.num_frames - offset)
                        waveform, rate = torchaudio.load(str(path), frame_offset=offset, num_frames=expected)
                        if rate != audio_info.sample_rate or waveform.shape[-1] != expected:
                            raise ValueError(f"truncated or inconsistent audio at frame {offset}")
                        if not torch.isfinite(waveform).all():
                            raise ValueError(f"non-finite samples at frame {offset}")
            except Exception as exc:  # torchaudio backend errors vary by installation
                errors.append(f"cannot read audio {path}: {exc}")
        for audio_path, line_number, onset, offset in strong_intervals:
            if audio_path in duration_by_path and offset > duration_by_path[audio_path] + 0.05:
                errors.append(f"strong interval for {audio_path.name}:{line_number} ends beyond audio duration")
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
        if not splits.train_records or not splits.val_records:
            errors.append("training and validation splits must both be nonempty")
        train_sources = {normalize_name(record.speech_id) for record in splits.train_records}
        val_sources = {normalize_name(record.speech_id) for record in splits.val_records}
        overlap = train_sources & val_sources
        if overlap:
            errors.append(f"train/validation source overlap: {sorted(overlap)}")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors:
        print(f"FAIL: found {len(errors)} problem(s)")
        return 1
    if args.manifest is not None:
        inventory_paths = set(original_paths + strong_paths + [info_csv, weak_csv])
        if negative_dir is not None:
            inventory_paths.update(negative_dir.glob("*.wav"))
        inventory = []
        for path in sorted(inventory_paths):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            inventory.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                              "sha256": digest.hexdigest()})
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"config": str(config_path.resolve()),
                                            "decode_audio": args.decode_audio,
                                            "files": inventory}, indent=2) + "\n")
        print(f"SHA-256 inventory: {args.manifest} ({len(inventory)} files)")
    print("OK: every metadata-derived training/validation item resolves correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
