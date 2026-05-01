from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from crowd_reaction.data import StrongEvent, parse_strong_label_file_by_task
from crowd_reaction.eval import SpeechChunkPrediction, evaluate_multitask_weak, evaluate_strong
from crowd_reaction.model import CrowdReactionModel, DummyFeatureExtractor, mmm_bag_loss, mmm_bag_loss_from_probs


_INFER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer.py"
_INFER_SPEC = importlib.util.spec_from_file_location("crowd_reaction_infer", _INFER_PATH)
assert _INFER_SPEC is not None and _INFER_SPEC.loader is not None
infer_module = importlib.util.module_from_spec(_INFER_SPEC)
_INFER_SPEC.loader.exec_module(infer_module)

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("crowd_reaction_train", _TRAIN_PATH)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
train_module = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(train_module)


class DummyWav2Vec2StyleFeatureExtractor(torch.nn.Module):
    def __init__(self, output_dim: int = 8) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.proj = torch.nn.Linear(1, self.output_dim)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        return self.proj(instances.mean(dim=-1, keepdim=True))


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


def test_mmm_loss_from_probs_matches_logit_wrapper() -> None:
    logits = torch.tensor([[[0.2], [-0.4], [1.1]], [[-0.7], [0.8], [0.1]]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    mask = torch.tensor([1.0, 0.5], dtype=torch.float32)

    assert torch.isclose(
        mmm_bag_loss_from_probs(torch.sigmoid(logits), labels, bag_mask=mask),
        mmm_bag_loss(logits, labels, bag_mask=mask),
        atol=1e-6,
    )


def test_conditional_attribute_loss_uses_detached_event_gate() -> None:
    event_logits = torch.tensor([[[2.0], [-2.0]]], dtype=torch.float32, requires_grad=True)
    approval_logits = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32, requires_grad=True)
    outputs = SimpleNamespace(
        instance_logits={
            "event": event_logits,
            "approval": approval_logits,
        }
    )
    targets = {
        "event_target": torch.tensor([[1.0]], dtype=torch.float32),
        "event_mask": torch.tensor([0.0], dtype=torch.float32),
        "approval_target": torch.tensor([[1.0]], dtype=torch.float32),
        "approval_mask": torch.tensor([1.0], dtype=torch.float32),
    }

    loss, _ = train_module.compute_multitask_loss(
        outputs,
        targets,
        loss_config={"conditional_attribute_loss": True, "lambda_approval": 1.0},
        task_class_weights={},
    )
    loss.backward()

    expected = mmm_bag_loss_from_probs(
        torch.sigmoid(event_logits.detach()) * torch.sigmoid(approval_logits.detach()),
        targets["approval_target"],
        bag_mask=targets["approval_mask"],
    )
    assert torch.isclose(loss.detach(), expected, atol=1e-6)
    assert event_logits.grad is None or torch.allclose(event_logits.grad, torch.zeros_like(event_logits))
    assert approval_logits.grad is not None
    assert float(approval_logits.grad.abs().sum().item()) > 0.0


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
    assert metrics["event"]["macro_precision"] == 1.0
    assert metrics["approval"]["num_valid"] == 1


def test_train_wandb_payload_logs_only_precision_and_f1_validation_metrics() -> None:
    metrics = {
        "epoch": 3,
        "train_loss": 1.0,
        "train_event_loss": 0.5,
        "train_approval_loss": 0.25,
        "train_disapproval_loss": 0.125,
        "weak": {
            "event": {"macro_precision": 0.8, "macro_f1": 0.7, "macro_average_precision": 0.9, "macro_auroc": 0.95},
            "approval": {"macro_precision": 0.6, "macro_f1": 0.5, "macro_average_precision": 0.7, "macro_auroc": 0.75},
            "disapproval": {"macro_precision": 0.4, "macro_f1": 0.3, "macro_average_precision": 0.5, "macro_auroc": 0.55},
        },
        "strong": {
            "segment_macro_precision": 0.9,
            "segment_macro_recall": 0.2,
            "segment_macro_f1": 0.8,
            "event_precision": 0.7,
            "event_recall": 0.1,
            "event_f1": 0.6,
            "approval": {
                "segment_macro_precision": 0.5,
                "segment_macro_recall": 0.2,
                "segment_macro_f1": 0.4,
                "event_precision": 0.3,
                "event_recall": 0.1,
                "event_f1": 0.2,
            },
            "disapproval": {
                "segment_macro_precision": 0.55,
                "segment_macro_recall": 0.25,
                "segment_macro_f1": 0.45,
                "event_precision": 0.35,
                "event_recall": 0.15,
                "event_f1": 0.25,
            },
        },
    }

    payload = train_module.wandb_validation_payload(metrics)

    assert payload["strong.segment_macro_precision"] == 0.9
    assert payload["strong.segment_macro_f1"] == 0.8
    assert payload["strong.event_precision"] == 0.7
    assert payload["strong.event_f1"] == 0.6
    assert payload["strong.approval.segment_macro_precision"] == 0.5
    assert payload["strong.approval.segment_macro_f1"] == 0.4
    assert payload["strong.approval.event_precision"] == 0.3
    assert payload["strong.approval.event_f1"] == 0.2
    assert payload["strong.disapproval.segment_macro_precision"] == 0.55
    assert payload["strong.disapproval.segment_macro_f1"] == 0.45
    assert payload["strong.disapproval.event_precision"] == 0.35
    assert payload["strong.disapproval.event_f1"] == 0.25
    assert not any(
        key.startswith("weak.") or "average_precision" in key or "auroc" in key or "recall" in key
        for key in payload
    )


def test_train_validation_scores_use_separate_strong_event_and_segment_f1() -> None:
    metrics = {
        "weak": {"event": {"macro_f1": 0.1}},
        "strong": {
            "segment_macro_f1": 0.8,
            "event_f1": 0.6,
        },
    }

    assert train_module.validation_score(metrics, "segment_macro_f1") == 0.8
    assert train_module.validation_score(metrics, "event_f1") == 0.6


def test_parse_strong_label_file_by_task_splits_event_and_attributes(tmp_path: Path) -> None:
    txt_path = tmp_path / "noise_sample.txt"
    txt_path.write_text(
        "0.0\t1.0\tclear_disapproval\n"
        "1.0\t2.0\tcrowd_chorus\n"
        "2.0\t3.0\tunclear_approval\n"
        "3.0\t4.0\thard_annotation\n",
        encoding="utf-8",
    )

    events_by_task = parse_strong_label_file_by_task(str(txt_path), speech_id="speech-1")

    assert len(events_by_task["event"]) == 3
    assert [(event.onset_sec, event.offset_sec) for event in events_by_task["approval"]] == [(2.0, 3.0)]
    assert [(event.onset_sec, event.offset_sec) for event in events_by_task["disapproval"]] == [(0.0, 1.0)]


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
        StrongEvent(speech_id="speech-1", event_class=0, onset_sec=1.0, offset_sec=4.0),
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


def test_wav2vec2_style_feature_extractor_supports_one_hz_bins() -> None:
    extractor = DummyWav2Vec2StyleFeatureExtractor(output_dim=8)
    instances = torch.randn(2, 20, 16000)
    embeddings = extractor(instances)

    assert embeddings.shape == (2, 20, 8)


def test_model_with_wav2vec2_style_features_outputs_one_hz_logits() -> None:
    model = CrowdReactionModel(
        feature_extractor=DummyWav2Vec2StyleFeatureExtractor(output_dim=8),
        chunk_sec=20.0,
        instance_sec=1.0,
    )

    outputs = model(instances=torch.randn(2, 20, 16000))

    assert outputs.instance_logits["event"].shape == (2, 20, 1)
    assert outputs.instance_logits["approval"].shape == (2, 20, 1)
    assert outputs.instance_logits["disapproval"].shape == (2, 20, 1)


def test_encoder_configs_keep_expected_training_recipes() -> None:
    root = Path(__file__).resolve().parents[1]
    default_config = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    wav2vec2_config = yaml.safe_load((root / "configs" / "wav2vec2.yaml").read_text(encoding="utf-8"))

    assert default_config["model"]["encoder_type"] == "beats"
    assert default_config["data"]["instance_sec"] == 1.0
    assert default_config["loss"]["unclear_label_weight"] == 0.75
    assert default_config["loss"]["conditional_attribute_loss"] is True
    assert wav2vec2_config["model"]["encoder_type"] == "wav2vec2"
    assert wav2vec2_config["data"]["instance_sec"] == 1.0
    assert wav2vec2_config["loss"]["unclear_label_weight"] == 0.75
    assert wav2vec2_config["loss"]["conditional_attribute_loss"] is True


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
