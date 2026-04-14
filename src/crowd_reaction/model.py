from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


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


class CrowdReactionModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 2,
        feature_extractor: FeatureExtractor | None = None,
        beats_checkpoint_path: str | None = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        sample_rate: int = 16000,
        chunk_sec: float = 30.0,
        instance_sec: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_extractor is None:
            if beats_checkpoint_path is None:
                raise ValueError("beats_checkpoint_path is required when feature_extractor is not provided")
            feature_extractor = FrozenBEATsFeatureExtractor(beats_checkpoint_path)
        self.feature_extractor = feature_extractor
        self.num_classes = int(num_classes)
        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.instance_sec = float(instance_sec)
        self.instance_num_samples = int(round(self.sample_rate * self.instance_sec))
        self.instances_per_chunk = int(round(self.chunk_sec / self.instance_sec))
        self.head = TemporalClassifierHead(
            input_dim=int(feature_extractor.output_dim),
            num_classes=self.num_classes,
            hidden_dim=int(head_hidden_dim),
            dropout=float(head_dropout),
        )

    def forward(self, waveform: torch.Tensor | None = None, instances: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        instance_logits = self.head(embeddings)
        bag_probabilities = torch.sigmoid(instance_logits).amax(dim=1)
        return instance_logits, bag_probabilities


@dataclass
class MMMTargets:
    max_target: float
    mean_target: float
    min_target: float


def _targets_for_label(label: float) -> MMMTargets:
    if float(label) >= 0.5:
        return MMMTargets(max_target=1.0, mean_target=0.5, min_target=0.0)
    return MMMTargets(max_target=0.0, mean_target=0.0, min_target=0.0)


def mmm_bag_loss(
    instance_logits: torch.Tensor,
    bag_labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if instance_logits.dim() != 3:
        raise ValueError(f"Expected instance_logits [B, T, C], got {tuple(instance_logits.shape)}")
    if bag_labels.shape != (instance_logits.shape[0], instance_logits.shape[2]):
        raise ValueError("bag_labels must have shape [B, C]")

    probs = torch.sigmoid(instance_logits)
    max_probs = probs.amax(dim=1)
    mean_probs = probs.mean(dim=1)
    min_probs = probs.amin(dim=1)

    targets = torch.zeros_like(bag_labels)
    mean_targets = torch.zeros_like(bag_labels)
    for batch_index in range(bag_labels.shape[0]):
        for class_index in range(bag_labels.shape[1]):
            label_targets = _targets_for_label(float(bag_labels[batch_index, class_index].item()))
            targets[batch_index, class_index] = label_targets.max_target
            mean_targets[batch_index, class_index] = label_targets.mean_target

    max_loss = F.binary_cross_entropy(max_probs, targets, reduction="none")
    mean_loss = F.binary_cross_entropy(mean_probs, mean_targets, reduction="none")
    min_loss = F.binary_cross_entropy(min_probs, torch.zeros_like(bag_labels), reduction="none")
    losses = max_loss + mean_loss + min_loss

    if class_weights is not None:
        losses = losses * class_weights.view(1, -1)
    return losses.mean()
