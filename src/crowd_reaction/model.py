from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


TASK_OUTPUT_DIMS = {
    "event": 1,
    "approval": 1,
    "disapproval": 1,
}


class FeatureExtractor(Protocol):
    output_dim: int

    def __call__(self, instances: torch.Tensor) -> torch.Tensor:
        ...


class DummyFeatureExtractor(nn.Module):
    """Small deterministic feature extractor used for tests and smoke checks."""

    def __init__(self, output_dim: int = 16) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(4, self.output_dim)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        stats = torch.stack(
            [
                instances.mean(dim=-1),
                instances.std(dim=-1),
                instances.abs().mean(dim=-1),
                instances.square().mean(dim=-1),
            ],
            dim=-1,
        )
        batch, steps, channels = stats.shape
        return self.proj(stats.view(batch * steps, channels)).view(batch, steps, self.output_dim)


class FrozenBEATsFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        from .beats import BEATs, BEATsConfig

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except EOFError as exc:
            raise RuntimeError(f"BEATs checkpoint at {checkpoint_path} appears truncated or corrupt") from exc

        config = BEATsConfig(checkpoint["cfg"])
        self.encoder = BEATs(config)
        self.encoder.load_state_dict(checkpoint["model"])
        self.output_dim = int(self.encoder.cfg.encoder_embed_dim)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        batch, steps, samples = instances.shape
        flat = instances.reshape(batch * steps, samples)
        with torch.no_grad():
            features, _ = self.encoder.extract_features(flat, feature_only=True)
            pooled = features.mean(dim=1)
        return pooled.view(batch, steps, self.output_dim)


class Wav2Vec2LayerScalarMix(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        if int(num_layers) <= 1:
            raise ValueError("Scalar fusion requires at least two layers")
        self.scalar_weights = nn.Parameter(torch.zeros(int(num_layers), dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
        if len(layer_features) != self.scalar_weights.numel():
            raise ValueError(f"Expected {self.scalar_weights.numel()} layer tensors, got {len(layer_features)}")
        stacked = torch.stack(layer_features, dim=0)
        weights = torch.softmax(self.scalar_weights, dim=0)
        view_shape = (weights.shape[0],) + (1,) * (stacked.dim() - 1)
        return self.gamma * (stacked * weights.view(view_shape)).sum(dim=0)


class FrozenWav2Vec2FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        layer_index: int | None = 3,
        layer_indices: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Wav2Vec2 encoder requires the `transformers` package. "
                "Install project dependencies before using model.encoder_type=wav2vec2."
            ) from exc

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = int(self.encoder.config.hidden_size)
        num_hidden_layers = int(self.encoder.config.num_hidden_layers)
        self.layer_indices = None if layer_indices is None else tuple(int(index) for index in layer_indices)
        if self.layer_indices is not None:
            if len(self.layer_indices) < 2:
                raise ValueError("wav2vec2 layer_indices must contain at least two transformer layers")
            invalid = [index for index in self.layer_indices if not 1 <= index <= num_hidden_layers]
            if invalid:
                raise ValueError(
                    f"wav2vec2 layer indices must be between 1 and {num_hidden_layers}, got {invalid}"
                )
            if len(set(self.layer_indices)) != len(self.layer_indices):
                raise ValueError("wav2vec2 layer_indices must not contain duplicates")
            self.layer_index = None
            self.scalar_mix = Wav2Vec2LayerScalarMix(len(self.layer_indices))
        else:
            self.layer_index = int(3 if layer_index is None else layer_index)
            if not 1 <= self.layer_index <= num_hidden_layers:
                raise ValueError(
                    f"wav2vec2 layer_index must be a transformer layer between 1 and "
                    f"{num_hidden_layers}, got {self.layer_index}"
                )

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        batch, steps, samples = instances.shape
        waveform = instances.reshape(batch, steps * samples)
        waveform = waveform - waveform.mean(dim=1, keepdim=True)
        waveform = waveform / waveform.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-7)
        with torch.no_grad():
            outputs = self.encoder(waveform, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Wav2Vec2Model did not return hidden states")
            requested_indices = self.layer_indices or (self.layer_index,)
            missing_indices = [index for index in requested_indices if index is None or index >= len(hidden_states)]
            if missing_indices:
                raise ValueError(
                    f"Requested wav2vec2 hidden-state indices {missing_indices}, "
                    f"but model returned {len(hidden_states)} hidden-state tensors"
                )
            pooled_layers = [
                F.adaptive_avg_pool1d(hidden_states[index].transpose(1, 2), output_size=steps).transpose(1, 2)
                for index in requested_indices
            ]
        if self.layer_indices is not None:
            return self.scalar_mix(pooled_layers)
        return pooled_layers[0]


class TemporalClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = embeddings.shape
        logits = self.network(embeddings.view(batch * steps, dim))
        return logits.view(batch, steps, -1)


@dataclass
class MultiTaskOutputs:
    instance_logits: dict[str, torch.Tensor]
    bag_probabilities: dict[str, torch.Tensor]


class CrowdReactionModel(nn.Module):
    def __init__(
        self,
        *,
        feature_extractor: FeatureExtractor | None = None,
        encoder_type: str = "beats",
        beats_checkpoint_path: str | None = None,
        wav2vec2_model_name: str = "facebook/wav2vec2-base",
        wav2vec2_layer_index: int = 3,
        wav2vec2_layer_indices: tuple[int, ...] | list[int] | None = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        sample_rate: int = 16000,
        chunk_sec: float = 20.0,
        instance_sec: float = 1.0,
        tasks_config: dict[str, dict[str, bool]] | None = None,
    ) -> None:
        super().__init__()
        if feature_extractor is None:
            encoder_type = str(encoder_type).strip().lower()
            if encoder_type == "beats":
                if beats_checkpoint_path is None:
                    raise ValueError("beats_checkpoint_path is required when encoder_type='beats'")
                feature_extractor = FrozenBEATsFeatureExtractor(beats_checkpoint_path)
            elif encoder_type == "wav2vec2":
                feature_extractor = FrozenWav2Vec2FeatureExtractor(
                    wav2vec2_model_name,
                    layer_index=wav2vec2_layer_index,
                    layer_indices=wav2vec2_layer_indices,
                )
            else:
                raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.feature_extractor = feature_extractor
        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.instance_sec = float(instance_sec)
        self.instance_num_samples = int(round(self.sample_rate * self.instance_sec))
        self.instances_per_chunk = int(round(self.chunk_sec / self.instance_sec))
        self.enabled_tasks = self._resolve_enabled_tasks(tasks_config)
        self.heads = nn.ModuleDict(
            {
                task_name: TemporalClassifierHead(
                    input_dim=int(feature_extractor.output_dim),
                    num_classes=TASK_OUTPUT_DIMS[task_name],
                    hidden_dim=int(head_hidden_dim),
                    dropout=float(head_dropout),
                )
                for task_name in self.enabled_tasks
            }
        )

    @staticmethod
    def _resolve_enabled_tasks(tasks_config: dict[str, dict[str, bool]] | None) -> tuple[str, ...]:
        enabled = ["event"]
        for task_name in ("approval", "disapproval"):
            task_cfg = (tasks_config or {}).get(task_name, {})
            if bool(task_cfg.get("enabled", True)):
                enabled.append(task_name)
        return tuple(enabled)

    def forward(
        self,
        waveform: torch.Tensor | None = None,
        instances: torch.Tensor | None = None,
    ) -> MultiTaskOutputs:
        if instances is None:
            if waveform is None:
                raise ValueError("Either waveform or instances must be provided")
            if waveform.dim() != 2:
                raise ValueError(f"Expected waveform shape [B, samples], got {tuple(waveform.shape)}")
            expected_samples = self.instances_per_chunk * self.instance_num_samples
            if waveform.shape[1] != expected_samples:
                raise ValueError(
                    f"Expected waveform with {expected_samples} samples for {self.chunk_sec}s chunks, got {waveform.shape[1]}"
                )
            instances = waveform.view(waveform.shape[0], self.instances_per_chunk, self.instance_num_samples)

        embeddings = self.feature_extractor(instances)
        instance_logits = {task_name: head(embeddings) for task_name, head in self.heads.items()}
        bag_probabilities = {task_name: torch.sigmoid(task_logits).amax(dim=1) for task_name, task_logits in instance_logits.items()}
        return MultiTaskOutputs(instance_logits=instance_logits, bag_probabilities=bag_probabilities)


@dataclass
class MMMTargets:
    max_target: float
    mean_target: float
    min_target: float


def _targets_for_label(label: float, *, positive_mean_target: float = 0.5) -> MMMTargets:
    positive_mean_target = float(positive_mean_target)
    if not 0.0 <= positive_mean_target <= 1.0:
        raise ValueError(f"positive_mean_target must be between 0 and 1, got {positive_mean_target}")
    if float(label) >= 0.5:
        return MMMTargets(max_target=1.0, mean_target=positive_mean_target, min_target=0.0)
    return MMMTargets(max_target=0.0, mean_target=0.0, min_target=0.0)


def mmm_bag_loss(
    instance_logits: torch.Tensor,
    bag_labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    bag_mask: torch.Tensor | None = None,
    positive_mean_target: float = 0.5,
) -> torch.Tensor:
    return mmm_bag_loss_from_probs(
        torch.sigmoid(instance_logits),
        bag_labels,
        class_weights=class_weights,
        bag_mask=bag_mask,
        positive_mean_target=positive_mean_target,
    )


def mmm_bag_loss_from_probs(
    instance_probs: torch.Tensor,
    bag_labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    bag_mask: torch.Tensor | None = None,
    positive_mean_target: float = 0.5,
) -> torch.Tensor:
    if instance_probs.dim() != 3:
        raise ValueError(f"Expected instance_probs [B, T, C], got {tuple(instance_probs.shape)}")
    if bag_labels.shape != (instance_probs.shape[0], instance_probs.shape[2]):
        raise ValueError("bag_labels must have shape [B, C]")

    probs = instance_probs.clamp(min=1e-7, max=1.0 - 1e-7)
    max_probs = probs.amax(dim=1)
    mean_probs = probs.mean(dim=1)
    min_probs = probs.amin(dim=1)

    targets = torch.zeros_like(bag_labels)
    mean_targets = torch.zeros_like(bag_labels)
    for batch_index in range(bag_labels.shape[0]):
        for class_index in range(bag_labels.shape[1]):
            label_targets = _targets_for_label(
                float(bag_labels[batch_index, class_index].item()),
                positive_mean_target=positive_mean_target,
            )
            targets[batch_index, class_index] = label_targets.max_target
            mean_targets[batch_index, class_index] = label_targets.mean_target

    max_loss = F.binary_cross_entropy(max_probs, targets, reduction="none")
    mean_loss = F.binary_cross_entropy(mean_probs, mean_targets, reduction="none")
    min_loss = F.binary_cross_entropy(min_probs, torch.zeros_like(bag_labels), reduction="none")
    losses = max_loss + mean_loss + min_loss

    if class_weights is not None:
        losses = losses * class_weights.view(1, -1)

    if bag_mask is None:
        return losses.mean()

    if bag_mask.dim() == 1:
        bag_mask = bag_mask.view(-1, 1)
    if bag_mask.shape[0] != losses.shape[0]:
        raise ValueError("bag_mask must have the same batch dimension as instance_logits")

    masked_losses = losses * bag_mask
    normalizer = bag_mask.sum() * losses.shape[1]
    if float(normalizer.detach().item()) <= 0.0:
        return losses.sum() * 0.0
    return masked_losses.sum() / normalizer
