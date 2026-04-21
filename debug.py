#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


SEGMENT_SUFFIX_PATTERN = re.compile(r"_seg\d+$", flags=re.IGNORECASE)


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = normalized.replace("\\", "/").split("/")[-1]
    if normalized.lower().startswith("noise_"):
        normalized = normalized[6:]
    normalized = re.sub(r"\.[^.]+$", "", normalized)
    normalized = SEGMENT_SUFFIX_PATTERN.sub("", normalized)

    translation = str.maketrans(
        {
            "’": "'",
            "‘": "'",
            "‚": "'",
            "‛": "'",
            "“": '"',
            "”": '"',
            "„": '"',
            "‟": '"',
            "–": "-",
            "—": "-",
            "―": "-",
            "‐": "-",
            "｜": "|",
            "：": ":",
            "？": "?",
            "！": "!",
            "＂": '"',
            "／": "/",
            "＆": "&",
            "＃": "#",
        }
    )
    normalized = normalized.translate(translation)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = "".join(ch for ch in normalized if ch.isalnum())
    return normalized


def build_original_audio_index(original_audio_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for audio_path in sorted(original_audio_dir.glob("*.wav")):
        key = normalize_name(audio_path.name)
        index.setdefault(key, []).append(audio_path)
    return index


def debug_weak_metadata(weak_labels_csv: Path, original_audio_index: dict[str, list[Path]]) -> None:
    print("=== Weak Metadata Check ===")
    seen = set()
    mismatches = []

    with weak_labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_file = row["source_file"]
            if source_file in seen:
                continue
            seen.add(source_file)

            normalized = normalize_name(source_file)
            matches = original_audio_index.get(normalized, [])
            if matches:
                print(f"[OK]  {source_file}")
                print(f"      normalized: {normalized}")
                print(f"      original:   {matches[0].name}")
            else:
                mismatches.append((source_file, normalized))
                print(f"[MISS] {source_file}")
                print(f"       normalized: {normalized}")

    print()
    print(f"Weak unique source_file count: {len(seen)}")
    print(f"Weak source_file mismatches:   {len(mismatches)}")


def debug_strong_metadata(strong_labels_dir: Path, original_audio_index: dict[str, list[Path]]) -> None:
    print()
    print("=== Strong TXT Check ===")
    txt_paths = sorted(strong_labels_dir.glob("noise_*.txt"))
    mismatches = []

    for txt_path in txt_paths:
        normalized = normalize_name(txt_path.name)
        matches = original_audio_index.get(normalized, [])
        if matches:
            print(f"[OK]  {txt_path.name}")
            print(f"      normalized: {normalized}")
            print(f"      original:   {matches[0].name}")
        else:
            mismatches.append((txt_path.name, normalized))
            print(f"[MISS] {txt_path.name}")
            print(f"       normalized: {normalized}")

    print()
    print(f"Strong TXT count:        {len(txt_paths)}")
    print(f"Strong TXT mismatches:   {len(mismatches)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug weak/strong metadata filename reconciliation")
    parser.add_argument("--weak-labels-csv", default="data/weak_labelling/_weak_labels.csv")
    parser.add_argument("--strong-labels-dir", default="data/strong_labelling")
    parser.add_argument("--original-audio-dir", default="data/original_audio_files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_audio_dir = Path(args.original_audio_dir)
    weak_labels_csv = Path(args.weak_labels_csv)
    strong_labels_dir = Path(args.strong_labels_dir)

    original_audio_index = build_original_audio_index(original_audio_dir)
    print(f"Indexed {sum(len(v) for v in original_audio_index.values())} original audio files from {original_audio_dir}")
    debug_weak_metadata(weak_labels_csv, original_audio_index)
    debug_strong_metadata(strong_labels_dir, original_audio_index)


if __name__ == "__main__":
    main()
