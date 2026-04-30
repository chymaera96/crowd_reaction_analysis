from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from crowd_reaction.data import StrongEvent
from crowd_reaction.eval import SpeechChunkPrediction, evaluate_multitask_weak, evaluate_strong
from crowd_reaction.model import CrowdReactionModel, DummyFeatureExtractor, mmm_bag_loss


_INFER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer.py"
_INFER_SPEC = importlib.util.spec_from_file_location("crowd_reaction_infer", _INFER_PATH)
assert _INFER_SPEC is not None and _INFER_SPEC.loader is not None
infer_module = importlib.util.module_from_spec(_INFER_SPEC)
_INFER_SPEC.loader.exec_module(infer_module)


def test_mmm_loss_matches_manual_negative_case() -> None:
    logits = torch.zeros((1, 2, 1), dtype=torch.float32)
    labels = torch.zeros((1, 1), dtype=torch.float32)
    loss = mmm_bag_loss(logits, labels)
    expected = 3.0 * math.log(2.0)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-5)


def test_mmm_loss_returns_zero_when_all_bags_are_masked() -> None:
    logits = torch.zeros((2, 3, 2), dtype=torch.float32)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mask = torch.zeros((2,), dtype=torch.float32)
    loss = mmm_bag_loss(logits, labels, bag_mask=mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_weak_eval_respects_masks() -> None:
    metrics = evaluate_multitask_weak(
        {
            "event": {
                "targets": np.array([[1.0], [0.0], [1.0]], dtype=np.float32),
                "probs": np.array([[0.9], [0.1], [0.2]], dtype=np.float32),
                "mask": np.array([1.0, 1.0, 0.0], dtype=np.float32),
            },
            "approval": {
                "targets": np.array([[1.0], [0.0]], dtype=np.float32),
                "probs": np.array([[0.8], [0.3]], dtype=np.float32),
                "mask": np.array([1.0, 0.0], dtype=np.float32),
            },
        },
        threshold=0.5,
    )
    assert metrics["event"]["num_valid"] == 2
    assert metrics["approval"]["num_valid"] == 1


def test_strong_eval_merges_overlapping_chunks() -> None:
    pytest.importorskip("sed_eval")
    predictions = [
        SpeechChunkPrediction(
            speech_id="speech-1",
            chunk_start_sec=0.0,
            chunk_end_sec=4.0,
            instance_probs=np.array(
                [
                    [0.1],
                    [0.9],
                    [0.8],
                    [0.9],
                ],
                dtype=np.float32,
            ),
        ),
        SpeechChunkPrediction(
            speech_id="speech-1",
            chunk_start_sec=2.0,
            chunk_end_sec=6.0,
            instance_probs=np.array(
                [
                    [0.7],
                    [0.9],
                    [0.1],
                    [0.1],
                ],
                dtype=np.float32,
            ),
        ),
    ]
    strong_events = [
        StrongEvent(speech_id="speech-1", event_class=0, onset_sec=1.0, offset_sec=3.0),
        StrongEvent(speech_id="speech-1", event_class=0, onset_sec=3.0, offset_sec=4.0),
    ]
    metrics = evaluate_strong(
        predictions,
        strong_events,
        num_classes=1,
        instance_sec=1.0,
        speech_durations={"speech-1": 6.0},
        threshold=0.5,
    )
    assert metrics["segment_macro_f1"] > 0.9
    assert metrics["event_f1"] > 0.9


def test_synthetic_one_step_training_smoke() -> None:
    model = CrowdReactionModel(feature_extractor=DummyFeatureExtractor(output_dim=8), chunk_sec=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    instances = torch.randn(3, 20, 32)
    outputs = model(instances=instances)
    loss = (
        mmm_bag_loss(outputs.instance_logits["event"], torch.tensor([[1.0], [0.0], [1.0]]))
        + 0.5 * mmm_bag_loss(outputs.instance_logits["approval"], torch.tensor([[1.0], [0.0], [1.0]]), bag_mask=torch.tensor([1.0, 1.0, 0.5]))
        + 0.5 * mmm_bag_loss(outputs.instance_logits["disapproval"], torch.tensor([[0.0], [1.0], [1.0]]), bag_mask=torch.tensor([1.0, 1.0, 0.5]))
    )
    loss.backward()
    optimizer.step()

    assert outputs.instance_logits["event"].shape == (3, 20, 1)
    assert outputs.bag_probabilities["event"].shape == (3, 1)
    assert outputs.instance_logits["approval"].shape == (3, 20, 1)
    assert outputs.instance_logits["disapproval"].shape == (3, 20, 1)
    assert float(loss.detach().item()) > 0.0


def test_infer_predicted_regions_and_export_format(tmp_path: Path) -> None:
    probs = np.array(
        [
            [0.2, 0.1, 0.9],
            [0.8, 0.6, 0.4],
            [0.9, 0.7, 0.6],
            [0.4, 0.2, 0.6],
            [0.7, 0.1, 0.6],
        ],
        dtype=np.float32,
    )
    regions = infer_module.predicted_regions_from_probs(
        probs,
        label_names=["relevant_event", "approval", "disapproval"],
        event_threshold=0.5,
        attribute_threshold=0.5,
        instance_sec=1.0,
    )

    assert regions == [
        (1.0, 3.0, "approval"),
        (2.0, 3.0, "disapproval"),
        (4.0, 5.0, "disapproval"),
    ]

    output_path = tmp_path / "predicted.csv"
    infer_module.write_sonic_visualiser_regions(output_path, regions)
    assert output_path.read_text(encoding="utf-8") == (
        "1.000000,2.000000,approval\n"
        "2.000000,1.000000,disapproval\n"
        "4.000000,1.000000,disapproval\n"
    )


def test_infer_aggregate_multitask_probs_flattens_task_outputs() -> None:
    aggregated = infer_module.aggregate_multitask_probs(
        {
            "event": [
                SpeechChunkPrediction(
                    speech_id="speech-1",
                    chunk_start_sec=0.0,
                    chunk_end_sec=3.0,
                    instance_probs=np.array([[0.1], [0.9], [0.2]], dtype=np.float32),
                )
            ],
            "approval": [
                SpeechChunkPrediction(
                    speech_id="speech-1",
                    chunk_start_sec=0.0,
                    chunk_end_sec=3.0,
                    instance_probs=np.array([[0.7], [0.2], [0.3]], dtype=np.float32),
                )
            ],
            "disapproval": [
                SpeechChunkPrediction(
                    speech_id="speech-1",
                    chunk_start_sec=0.0,
                    chunk_end_sec=3.0,
                    instance_probs=np.array([[0.1], [0.8], [0.4]], dtype=np.float32),
                )
            ],
        },
        instance_sec=1.0,
        speech_durations={"speech-1": 3.0},
    )
    assert aggregated["speech-1"].shape == (3, 3)
    assert np.allclose(
        aggregated["speech-1"],
        np.array(
            [
                [0.1, 0.7, 0.1],
                [0.9, 0.2, 0.8],
                [0.2, 0.3, 0.4],
            ],
            dtype=np.float32,
        ),
    )


def test_infer_ground_truth_regions_export_only_approval_disapproval() -> None:
    regions = infer_module.regions_from_annotations(
        {
            "clear_approval": [(0.0, 1.0)],
            "unclear_approval": [(2.0, 3.0)],
            "clear_disapproval": [(4.0, 5.0)],
            "unclear_disapproval": [(6.0, 7.0)],
            "crowd_chorus": [(8.0, 9.0)],
        }
    )

    assert regions == [
        (0.0, 1.0, "approval"),
        (2.0, 3.0, "approval"),
        (4.0, 5.0, "disapproval"),
        (6.0, 7.0, "disapproval"),
    ]


def test_infer_formats_time_as_mmss() -> None:
    assert infer_module.format_seconds_mmss(0.0) == "00:00"
    assert infer_module.format_seconds_mmss(65.1) == "01:05"
