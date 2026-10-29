#!/usr/bin/env python3
"""Download the primary classifier checkpoints from a pinned GitHub release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_weights.json"


def verified(path: Path, model: dict) -> bool:
    if path.stat().st_size != model["size_bytes"]:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == model["sha256"]


def download_weight(url: str, model: dict, output_dir: Path, *, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / model["filename"]
    if target.exists() and not force:
        if verified(target, model):
            print(f"Already verified: {target}", flush=True)
            return target
        raise ValueError(f"{target} differs from the release checkpoint; use --force to replace it")

    print(f"Downloading {model['filename']} ({model['size_bytes'] / 1024**2:.1f} MiB)...", flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{target.name}.", suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            with urlopen(url, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        if not verified(temporary, model):
            raise ValueError(f"Size or SHA-256 mismatch for {model['filename']}; download discarded")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Verified and saved: {target}", flush=True)
    return target


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=tuple(manifest["models"]),
                        help="Download only these models (default: all three)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "weights",
                        help="Destination directory (default: repository weights/)")
    parser.add_argument("--force", action="store_true", help="Redownload and replace existing checkpoints after verification")
    args = parser.parse_args()
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}"
    try:
        for name in dict.fromkeys(args.model or manifest["models"]):
            model = manifest["models"][name]
            download_weight(f"{base}/{model['filename']}", model, args.output_dir, force=args.force)
    except (OSError, URLError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
