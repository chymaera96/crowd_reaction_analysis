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

from importlib.resources import files

MANIFEST = files("crowd_reaction").joinpath("resources/model_weights.json")


def cache_directory() -> Path:
    """Per-user, release-specific cache; override with CROWD_REACTION_CACHE."""
    override = os.environ.get("CROWD_REACTION_CACHE")
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "crowd_reaction"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "crowd_reaction"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "crowd_reaction"
    return root / json.loads(MANIFEST.read_text())["release"]


def model_config(model: str = "cnn_tcn") -> dict:
    import yaml
    manifest = json.loads(MANIFEST.read_text())
    if model not in manifest["models"]:
        raise ValueError(f"Unknown model {model!r}; choose from {tuple(manifest['models'])}")
    filename = Path(manifest["models"][model]["config"]).name
    return yaml.safe_load(files("crowd_reaction").joinpath("resources", filename).read_text())


def get_weights(model: str = "cnn_tcn", *, cache_dir: str | Path | None = None) -> Path:
    """Fetch only the selected model, verifying existing and downloaded weights."""
    manifest = json.loads(MANIFEST.read_text())
    if model not in manifest["models"]:
        raise ValueError(f"Unknown model {model!r}; choose from {tuple(manifest['models'])}")
    entry = manifest["models"][model]
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}"
    return download_weight(f"{base}/{entry['filename']}", entry,
                           Path(cache_dir).expanduser() if cache_dir is not None else cache_directory())


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


def main(*, default_output_dir: Path | None = None, default_models=None) -> None:
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=tuple(manifest["models"]),
                        help="Download only these models (default: cnn_tcn)")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir or cache_directory(),
                        help="Destination directory (default: shared model cache)")
    parser.add_argument("--force", action="store_true", help="Redownload and replace existing checkpoints after verification")
    args = parser.parse_args()
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}"
    try:
        for name in dict.fromkeys(args.model or default_models or ["cnn_tcn"]):
            model = manifest["models"][name]
            download_weight(f"{base}/{model['filename']}", model, args.output_dir, force=args.force)
    except (OSError, URLError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
