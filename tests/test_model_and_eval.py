from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from crowd_reaction.data import StrongEvent, parse_strong_label_file_by_task
from crowd_reaction.eval import SpeechChunkPrediction, evaluate_multitask_weak, evaluate_strong
from crowd_reaction.model import (
    CrowdReactionModel,
    DummyFeatureExtractor,
    FrozenWav2Vec2FeatureExtractor,
    mmm_bag_loss,
    mmm_bag_loss_from_probs,
)


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

_RESULTS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "results.py"
_RESULTS_SPEC = importlib.util.spec_from_file_location("crowd_reaction_results", _RESULTS_PATH)
assert _RESULTS_SPEC is not None and _RESULTS_SPEC.loader is not None
results_module = importlib.util.module_from_spec(_RESULTS_SPEC)
_RESULTS_SPEC.loader.exec_module(results_module)

_API_PATH = Path(__file__).resolve().parents[1] / "scripts" / "api.py"
_API_SPEC = importlib.util.spec_from_file_location("crowd_reaction_api", _API_PATH)
assert _API_SPEC is not None and _API_SPEC.loader is not None
api_module = importlib.util.module_from_spec(_API_SPEC)
sys.modules[_API_SPEC.name] = api_module
_API_SPEC.loader.exec_module(api_module)


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
    assert payload["strong.polarity.segment_macro_f1"] == pytest.approx(0.425)
    assert payload["strong.polarity.event_f1"] == pytest.approx(0.225)
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


def test_train_polarity_validation_score_macro_averages_approval_and_disapproval() -> None:
    metrics = {
        "strong": {
            "approval": {"segment_macro_f1": 0.8, "event_f1": 0.6},
            "disapproval": {"segment_macro_f1": 0.4, "event_f1": 0.2},
        }
    }

    assert train_module.polarity_validation_score(metrics, "segment_macro_f1") == pytest.approx(0.6)
    assert train_module.polarity_validation_score(metrics, "event_f1") == pytest.approx(0.4)


def test_results_thresholds_use_inference_threshold_policy() -> None:
    thresholds = results_module.thresholds_from_config(
        {
            "val": {
                "threshold": 0.1,
                "event_threshold": 0.2,
                "attribute_threshold": 0.3,
            }
        }
    )

    assert thresholds == {
        "relevant_event": 0.2,
        "approval": 0.3,
        "disapproval": 0.3,
    }


def test_results_payload_shape() -> None:
    payload = results_module.build_results_payload(
        config_path="configs/wav2vec2.yaml",
        checkpoint_path="outputs/run/best_segment_f1.pt",
        thresholds={"relevant_event": 0.5, "approval": 0.4, "disapproval": 0.4},
        metrics={
            "relevant_event": {
                "segment_per_class": [{"class_index": 0}],
                "segment_macro_precision": 0.9,
                "segment_macro_recall": 0.8,
                "segment_macro_f1": 0.85,
                "event_per_class": [{"class_index": 0}],
                "event_precision": 0.7,
                "event_recall": 0.6,
                "event_f1": 0.65,
            },
            "approval": {
                "segment_macro_precision": 0.5,
                "segment_macro_recall": 0.4,
                "segment_macro_f1": 0.45,
                "event_precision": 0.3,
                "event_recall": 0.2,
                "event_f1": 0.25,
            },
        },
    )

    assert payload["config"] == "configs/wav2vec2.yaml"
    assert payload["checkpoint"] == "outputs/run/best_segment_f1.pt"
    assert payload["thresholds"]["relevant_event"] == 0.5
    assert payload["metrics"]["relevant_event"]["segment"]["precision"] == 0.9
    assert payload["metrics"]["relevant_event"]["segment"]["f1"] == 0.85
    assert payload["metrics"]["approval"]["event"]["f1"] == 0.25
    assert "segment_per_class" not in payload["metrics"]["relevant_event"]
    assert "segment_macro_f1" not in payload["metrics"]["relevant_event"]


def test_results_parse_args_defaults_output_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "results.py",
            "--config",
            "configs/wav2vec2.yaml",
            "--checkpoint",
            "outputs/run/best_segment_f1.pt",
        ],
    )

    args = results_module.parse_args()

    assert args.config == "configs/wav2vec2.yaml"
    assert args.checkpoint == "outputs/run/best_segment_f1.pt"
    assert args.output is None


def test_train_wav2vec2_layer_cli_override_is_applied_to_effective_config() -> None:
    config = {"model": {"wav2vec2_layer_index": 8}}
    args = SimpleNamespace(wav2vec2_layer=3)

    effective = train_module.apply_cli_overrides(config, args)

    assert effective["model"]["wav2vec2_layer_index"] == 3


@pytest.mark.parametrize("module", [infer_module, results_module])
def test_evaluation_loads_wav2vec2_layer_from_checkpoint_config(
    monkeypatch: pytest.MonkeyPatch, module
) -> None:
    constructed: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            constructed.update(kwargs)

        def to(self, device):
            return self

        def load_state_dict(self, state_dict) -> None:
            constructed["state_dict"] = state_dict

        def eval(self) -> None:
            constructed["eval"] = True

    checkpoint = {
        "model_state_dict": {"heads.event.weight": torch.ones(1)},
        "config": {"model": {"wav2vec2_layer_index": 7}},
    }
    monkeypatch.setattr(module, "CrowdReactionModel", FakeModel)
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: checkpoint)
    config = {
        "model": {"encoder_type": "wav2vec2", "wav2vec2_layer_index": 2},
        "data": {"sample_rate": 16000, "chunk_sec": 20.0, "instance_sec": 0.5},
    }

    module.load_model(config, "checkpoint.pt", torch.device("cpu"))

    assert constructed["wav2vec2_layer_index"] == 7
    assert constructed["eval"] is True


@pytest.mark.parametrize("module", [infer_module, results_module])
def test_evaluation_rejects_scalar_fusion_checkpoint(
    monkeypatch: pytest.MonkeyPatch, module
) -> None:
    checkpoint = {
        "model_state_dict": {"feature_extractor.scalar_mix.gamma": torch.tensor(1.0)},
        "config": {"model": {"wav2vec2_layer_indices": [3, 6, 9, 12]}},
    }
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: checkpoint)

    with pytest.raises(RuntimeError, match="obsolete wav2vec2 scalar layer fusion"):
        module.load_model({"model": {}, "data": {}}, "checkpoint.pt", torch.device("cpu"))


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


def test_wav2vec2_style_feature_extractor_supports_half_second_bins() -> None:
    extractor = DummyWav2Vec2StyleFeatureExtractor(output_dim=8)
    instances = torch.randn(2, 40, 8000)
    embeddings = extractor(instances)

    assert embeddings.shape == (2, 40, 8)


@pytest.mark.parametrize("layer_index", [1, 3, 12])
def test_frozen_wav2vec2_extractor_selects_one_configured_hidden_layer(
    monkeypatch: pytest.MonkeyPatch, layer_index: int
) -> None:
    class FakeWav2Vec2Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=12)
            self.encoder_weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool = False):
            assert output_hidden_states is True
            batch = waveform.shape[0]
            hidden_states = tuple(
                torch.full((batch, 7, 4), float(layer_index), device=waveform.device)
                for layer_index in range(13)
            )
            return SimpleNamespace(hidden_states=hidden_states)

    class FakeWav2Vec2Model:
        @staticmethod
        def from_pretrained(model_name: str) -> FakeWav2Vec2Encoder:
            assert model_name == "fake/wav2vec2"
            return FakeWav2Vec2Encoder()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Wav2Vec2Model = FakeWav2Vec2Model
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    extractor = FrozenWav2Vec2FeatureExtractor("fake/wav2vec2", layer_index=layer_index)
    embeddings = extractor(torch.randn(2, 40, 8000))

    assert embeddings.shape == (2, 40, 4)
    assert torch.allclose(embeddings, torch.full((2, 40, 4), float(layer_index)))
    assert extractor.layer_index == layer_index
    assert all(not parameter.requires_grad for parameter in extractor.encoder.parameters())
    assert not any("scalar_mix" in name for name, _ in extractor.named_parameters())


@pytest.mark.parametrize("layer_index", [0, 13])
def test_frozen_wav2vec2_extractor_rejects_non_transformer_layers(
    monkeypatch: pytest.MonkeyPatch, layer_index: int
) -> None:
    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=12)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Wav2Vec2Model = SimpleNamespace(from_pretrained=lambda _: FakeEncoder())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(ValueError, match="between 1 and 12"):
        FrozenWav2Vec2FeatureExtractor("fake/wav2vec2", layer_index=layer_index)


def test_frozen_wav2vec2_extractor_rejects_unavailable_returned_hidden_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=12)

        def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool = False):
            return SimpleNamespace(hidden_states=tuple(torch.zeros(waveform.shape[0], 2, 4) for _ in range(3)))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Wav2Vec2Model = SimpleNamespace(from_pretrained=lambda _: FakeEncoder())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    extractor = FrozenWav2Vec2FeatureExtractor("fake/wav2vec2", layer_index=3)

    with pytest.raises(ValueError, match="returned 3 hidden-state tensors"):
        extractor(torch.randn(1, 2, 8))


def test_model_with_wav2vec2_style_features_outputs_half_second_logits() -> None:
    model = CrowdReactionModel(
        feature_extractor=DummyWav2Vec2StyleFeatureExtractor(output_dim=8),
        chunk_sec=20.0,
        instance_sec=0.5,
    )

    outputs = model(instances=torch.randn(2, 40, 8000))

    assert outputs.instance_logits["event"].shape == (2, 40, 1)
    assert outputs.instance_logits["approval"].shape == (2, 40, 1)
    assert outputs.instance_logits["disapproval"].shape == (2, 40, 1)


def test_encoder_configs_keep_expected_training_recipes() -> None:
    root = Path(__file__).resolve().parents[1]
    default_config = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    wav2vec2_config = yaml.safe_load((root / "configs" / "wav2vec2.yaml").read_text(encoding="utf-8"))

    assert default_config["model"]["encoder_type"] == "beats"
    assert default_config["data"]["instance_sec"] == 1.0
    assert default_config["loss"]["unclear_label_weight"] == 0.75
    assert default_config["loss"]["conditional_attribute_loss"] is True
    assert default_config["augmentation"]["enabled"] is True
    assert default_config["trainer"]["epochs"] == 15
    assert wav2vec2_config["model"]["encoder_type"] == "wav2vec2"
    assert wav2vec2_config["model"]["wav2vec2_layer_index"] == 3
    assert wav2vec2_config["data"]["instance_sec"] == 0.5
    assert wav2vec2_config["loss"]["unclear_label_weight"] == 0.75
    assert wav2vec2_config["loss"]["conditional_attribute_loss"] is True
    assert wav2vec2_config["augmentation"]["enabled"] is True
    assert wav2vec2_config["trainer"]["epochs"] == 15


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


def test_infer_conditions_attribute_scores_by_event_probability() -> None:
    probs_by_task = {
        "event": np.array([[[0.2], [0.8]]], dtype=np.float32),
        "approval": np.array([[[0.5], [0.5]]], dtype=np.float32),
        "disapproval": np.array([[[0.25], [0.75]]], dtype=np.float32),
    }

    conditioned = infer_module.condition_attribute_probs_by_event(probs_by_task)

    assert np.allclose(conditioned["event"], np.array([[[0.2], [0.8]]], dtype=np.float32))
    assert np.allclose(conditioned["approval"], np.array([[[0.1], [0.4]]], dtype=np.float32))
    assert np.allclose(conditioned["disapproval"], np.array([[[0.05], [0.6]]], dtype=np.float32))
    assert np.allclose(probs_by_task["approval"], np.array([[[0.5], [0.5]]], dtype=np.float32))


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


def test_api_scores_json_uses_class_keyed_lists(tmp_path: Path) -> None:
    result = api_module.InferenceResult(
        audio_path="example.wav",
        instance_sec=1.0,
        event_threshold=0.5,
        attribute_threshold=0.5,
        label_names=("relevant_event", "approval", "disapproval"),
        times_sec=np.array([0.5, 1.5], dtype=np.float32),
        scores=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        predicted_regions=[],
    )

    output_path = tmp_path / "scores.json"
    api_module.write_scores_json(result, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert list(payload) == ["relevant_event", "approval", "disapproval"]
    assert payload["relevant_event"] == [0.10000000149011612, 0.4000000059604645]
    assert payload["approval"] == [0.20000000298023224, 0.5]
    assert payload["disapproval"] == [0.30000001192092896, 0.6000000238418579]


def test_api_predicted_segments_csv_uses_duration(tmp_path: Path) -> None:
    result = api_module.InferenceResult(
        audio_path="example.wav",
        instance_sec=1.0,
        event_threshold=0.5,
        attribute_threshold=0.5,
        label_names=("relevant_event", "approval", "disapproval"),
        times_sec=np.array([0.5, 1.5, 2.5], dtype=np.float32),
        scores=np.zeros((3, 3), dtype=np.float32),
        predicted_regions=[(2.0, 5.0, "approval"), (10.0, 11.5, "disapproval")],
    )

    output_path = tmp_path / "predicted_segments.csv"
    api_module.write_predicted_segments_csv(result, output_path)

    assert output_path.read_text(encoding="utf-8") == (
        "2.000000,3.000000,approval\n"
        "10.000000,1.500000,disapproval\n"
    )


def test_api_region_thresholding_matches_infer_event_gating() -> None:
    scores = np.array(
        [
            [0.2, 0.9, 0.9],
            [0.8, 0.6, 0.4],
            [0.8, 0.4, 0.6],
        ],
        dtype=np.float32,
    )

    regions = infer_module.predicted_regions_from_probs(
        scores,
        label_names=["relevant_event", "approval", "disapproval"],
        event_threshold=0.5,
        attribute_threshold=0.5,
        instance_sec=1.0,
    )
    result = api_module.InferenceResult(
        audio_path="example.wav",
        instance_sec=1.0,
        event_threshold=0.5,
        attribute_threshold=0.5,
        label_names=("relevant_event", "approval", "disapproval"),
        times_sec=np.array([0.5, 1.5, 2.5], dtype=np.float32),
        scores=scores,
        predicted_regions=regions,
    )

    assert result.predicted_regions == [
        (1.0, 2.0, "approval"),
        (2.0, 3.0, "disapproval"),
    ]


def test_api_event_mode_selects_only_relevant_event_outputs() -> None:
    scores = np.array(
        [
            [0.2, 0.9, 0.1],
            [0.8, 0.1, 0.9],
        ],
        dtype=np.float32,
    )

    selected_scores, selected_labels, export_labels = api_module._select_mode_outputs(
        scores,
        label_names=("relevant_event", "approval", "disapproval"),
        active_label_names=["relevant_event", "approval", "disapproval"],
        mode="event",
    )

    assert selected_labels == ("relevant_event",)
    assert export_labels == ("relevant_event",)
    assert np.allclose(selected_scores, np.array([[0.2], [0.8]], dtype=np.float32))


def test_api_event_mode_exports_relevant_event_regions() -> None:
    scores = np.array(
        [
            [0.2],
            [0.8],
            [0.9],
            [0.1],
        ],
        dtype=np.float32,
    )

    regions = api_module.predicted_regions_with_median_filter(
        scores,
        label_names=["relevant_event"],
        event_threshold=0.5,
        attribute_threshold=0.5,
        instance_sec=1.0,
        median_filter_sec=0.0,
        export_labels=("relevant_event",),
    )

    assert regions == [(1.0, 3.0, "relevant_event")]


def test_api_median_filter_removes_short_predicted_regions() -> None:
    scores = np.array(
        [
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    unfiltered_regions = api_module.predicted_regions_with_median_filter(
        scores,
        label_names=["relevant_event", "approval", "disapproval"],
        event_threshold=0.5,
        attribute_threshold=0.5,
        instance_sec=0.5,
        median_filter_sec=0.0,
    )
    filtered_regions = api_module.predicted_regions_with_median_filter(
        scores,
        label_names=["relevant_event", "approval", "disapproval"],
        event_threshold=0.5,
        attribute_threshold=0.5,
        instance_sec=0.5,
        median_filter_sec=3.0,
    )

    assert unfiltered_regions == [(0.5, 2.0, "approval"), (4.5, 7.0, "approval")]
    assert filtered_regions == [(4.5, 7.0, "approval")]


def test_api_plot_scores_use_median_filtered_attribute_masks() -> None:
    scores = np.array(
        [
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = api_module.InferenceResult(
        audio_path="example.wav",
        instance_sec=0.5,
        event_threshold=0.5,
        attribute_threshold=0.5,
        label_names=("relevant_event", "approval", "disapproval"),
        times_sec=np.arange(scores.shape[0], dtype=np.float32),
        scores=scores,
        predicted_regions=[],
        median_filter_sec=3.0,
    )

    plot_scores = api_module.scores_for_plot(result)

    assert np.allclose(plot_scores[:, 0], scores[:, 0])
    assert np.allclose(plot_scores[:, 1], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32))
    assert np.allclose(plot_scores[:, 2], np.zeros(scores.shape[0], dtype=np.float32))


def test_api_predicted_segments_csv_merges_consecutive_bins_like_infer(tmp_path: Path) -> None:
    scores = np.array(
        [
            [0.9, 0.7, 0.1],
            [0.9, 0.7, 0.1],
            [0.9, 0.7, 0.1],
            [0.1, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    regions = infer_module.predicted_regions_from_probs(
        scores,
        label_names=["relevant_event", "approval", "disapproval"],
        event_threshold=0.5,
        attribute_threshold=0.5,
        instance_sec=1.0,
    )
    result = api_module.InferenceResult(
        audio_path="example.wav",
        instance_sec=1.0,
        event_threshold=0.5,
        attribute_threshold=0.5,
        label_names=("relevant_event", "approval", "disapproval"),
        times_sec=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        scores=scores,
        predicted_regions=regions,
    )

    output_path = tmp_path / "predicted_segments.csv"
    api_module.write_predicted_segments_csv(result, output_path)

    assert regions == [(0.0, 3.0, "approval")]
    assert output_path.read_text(encoding="utf-8") == "0.000000,3.000000,approval\n"


def test_api_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "api.py",
            "--config",
            "configs/wav2vec2.yaml",
            "--checkpoint",
            "outputs/run/best_segment_f1.pt",
            "--audio",
            "input.wav",
            "--output-dir",
            "api_outputs/example",
            "--no-score-functions",
        ],
    )

    args = api_module.parse_args()

    assert args.config == "configs/wav2vec2.yaml"
    assert args.checkpoint == "outputs/run/best_segment_f1.pt"
    assert args.audio == "input.wav"
    assert args.output_dir == "api_outputs/example"
    assert args.mode == "polarity"
    assert args.median_filter_sec == 3.0
    assert args.no_progress is False
    assert args.no_score_functions is True
