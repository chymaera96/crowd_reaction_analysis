from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from crowd_reaction import CrowdDetector
from crowd_reaction import api
from crowd_reaction.model import CrowdReactionModel
from crowd_reaction.weights import model_config


@pytest.fixture
def checkpoint(tmp_path):
    config = model_config()
    model = CrowdReactionModel(encoder_type="cnn_tcn", head_hidden_dim=64,
                               sample_rate=16000, chunk_sec=20, instance_sec=0.04,
                               tasks_config=config["tasks"], cnn_config=config["model"]["cnn_config"])
    path = tmp_path / "classifier.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    return path


def test_default_detector_downloads_only_cnn_and_reuses_model(tmp_path, checkpoint, monkeypatch):
    calls = []
    original = api.infer_utils.load_model

    def get_weights(name, **kwargs):
        assert name == "cnn_tcn"
        calls.append("weights")
        return checkpoint

    def load_model(*args, **kwargs):
        calls.append("load")
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "get_weights", get_weights)
    monkeypatch.setattr(api.infer_utils, "load_model", load_model)
    # Stereo input at another sample rate exercises decoding, mono mixing, and resampling.
    audio = tmp_path / "example.wav"
    sf.write(audio, np.zeros((22050, 2), dtype=np.float32), 22050)
    detector = CrowdDetector()
    result = detector.predict(audio, mode="event", event_threshold=0, median_filter_sec=0, show_progress=False)
    detector.predict(audio, mode="polarity", show_progress=False)
    assert calls == ["weights", "load"]
    assert result.scores.shape == (25, 1)
    assert np.isfinite(result.scores).all()
    assert result.region_mean_score(*result.predicted_regions[0]) >= 0
    output = tmp_path / "annotations"
    result.save(output, label_scores=True)
    with (output / "predicted_segments.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows and rows[0][2].startswith("crowd mean=")
    assert float(rows[0][1]) == pytest.approx(1)
    assert "relevant_event" in json.loads((output / "scores.json").read_text())
    assert (output / "plot.png").stat().st_size > 0


def test_explicit_checkpoint_works_offline_and_validates_arguments(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr(api, "get_weights", lambda *a, **k: pytest.fail("Must not download"))
    detector = CrowdDetector(checkpoint_path=checkpoint)
    with pytest.raises(ValueError, match="batch_size"):
        detector.predict("unused.wav", batch_size=0)
    with pytest.raises(ValueError, match="min_mean_score"):
        detector.predict("unused.wav", min_mean_score=2)


@pytest.mark.parametrize("name,rate", [("cnn_tcn", 0.04), ("beats", 1.0), ("wav2vec2", 0.5)])
def test_packaged_model_configurations(name, rate):
    config = model_config(name)
    assert config["model"]["encoder_type"] == name
    assert config["data"]["instance_sec"] == rate
