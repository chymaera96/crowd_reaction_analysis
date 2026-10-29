"""Small spectrogram encoders trained on continuous audio chunks."""
from __future__ import annotations

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        layers = []
        for _ in range(2):
            layers.extend([
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
                nn.GroupNorm(8, width), nn.ReLU(), nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features + self.layers(features))


class SpectrogramCNNEncoder(nn.Module):
    """64-bin log-mel CNN, optionally followed by a local TCN or a BiGRU.

    The instance tensor is reassembled before the spectrogram: instance boundaries
    are output bins, not independently encoded audio crops. No pretrained weights.
    """
    output_dim = 64

    def __init__(self, architecture: str = "cnn", sample_rate: int = 16000,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if architecture not in {"cnn", "tcn", "gru"}:
            raise ValueError(f"Unknown CNN architecture: {architecture}")
        self.architecture = architecture
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, win_length=round(sample_rate * 0.025),
            hop_length=round(sample_rate * 0.010), n_mels=64,
            f_min=50, f_max=sample_rate / 2, power=2.0,
        )
        blocks = []
        channels = 1
        for width, time_pool in [(16, 1), (32, 2), (64, 2)]:
            blocks.append(nn.Sequential(
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.GroupNorm(8, width), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.GroupNorm(8, width), nn.ReLU(),
                nn.AvgPool2d((2, time_pool)), nn.Dropout2d(dropout),
            ))
            channels = width
        self.cnn = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(64 * 4, 64, 1)
        if architecture == "tcn":
            self.temporal = nn.Sequential(*[
                TemporalResidualBlock(64, dilation, dropout) for dilation in (1, 2, 4, 8)
            ])
        elif architecture == "gru":
            self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
            self.gru_projection = nn.Linear(128, 64)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = instances.shape
        waveform = instances.reshape(batch, -1)
        features = 10.0 * self.mel(waveform).clamp_min(1e-10).log10()
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-5)
        features = self.cnn(((features - mean) / std).unsqueeze(1))
        features = F.adaptive_avg_pool2d(features, (4, features.shape[-1]))
        features = F.relu(self.projection(features.flatten(1, 2)))
        if self.architecture == "tcn":
            features = self.temporal(features)
        elif self.architecture == "gru":
            sequence, _ = self.gru(features.transpose(1, 2))
            features = self.gru_projection(sequence).transpose(1, 2)
        features = F.adaptive_avg_pool1d(features, steps)
        return features.transpose(1, 2).contiguous()
