from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
from urllib.error import URLError

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_weights", Path(__file__).resolve().parents[1] / "scripts/download_weights.py"
)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def metadata(content):
    return {"filename": "model.pt", "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def test_download_and_skip_verified_checkpoint(tmp_path, monkeypatch):
    content = b"complete checkpoint"
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(downloader, "urlopen", fetch)
    target = downloader.download_weight("https://example.test/model.pt", metadata(content), tmp_path)
    assert target.read_bytes() == content
    downloader.download_weight("https://example.test/model.pt", metadata(content), tmp_path)
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == [target]


def test_existing_different_checkpoint_requires_force(tmp_path, monkeypatch):
    target = tmp_path / "model.pt"
    target.write_bytes(b"custom weights")
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: pytest.fail("Must not download"))
    with pytest.raises(ValueError, match="--force"):
        downloader.download_weight("https://example.test/model.pt", metadata(b"released weights"), tmp_path)
    assert target.read_bytes() == b"custom weights"


@pytest.mark.parametrize("replacement", [b"bad", b"wrong checkpoint"])
def test_corrupt_download_preserves_existing_file(tmp_path, monkeypatch, replacement):
    target = tmp_path / "model.pt"
    target.write_bytes(b"original weights")
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: io.BytesIO(replacement))
    with pytest.raises(ValueError, match="mismatch"):
        downloader.download_weight("https://example.test/model.pt", metadata(b"valid checkpoint"), tmp_path, force=True)
    assert target.read_bytes() == b"original weights"
    assert list(tmp_path.iterdir()) == [target]


def test_network_failure_cleans_temporary_download(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("connection interrupted")

    monkeypatch.setattr(downloader, "urlopen", fail)
    with pytest.raises(URLError):
        downloader.download_weight("https://example.test/model.pt", metadata(b"weights"), tmp_path)
    assert not list(tmp_path.iterdir())


def test_force_replaces_file_after_verification(tmp_path, monkeypatch):
    target = tmp_path / "model.pt"
    target.write_bytes(b"old")
    content = b"released checkpoint"
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: io.BytesIO(content))
    downloader.download_weight("https://example.test/model.pt", metadata(content), tmp_path, force=True)
    assert target.read_bytes() == content
    assert list(tmp_path.iterdir()) == [target]
