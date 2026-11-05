"""SoundFile audio I/O, independent of TorchAudio's removed decoder APIs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    num_frames: int
    num_channels: int


def info(path: str | Path) -> AudioInfo:
    metadata = sf.info(str(path))
    return AudioInfo(metadata.samplerate, metadata.frames, metadata.channels)


def load(path: str | Path, *, frame_offset: int = 0, num_frames: int = -1):
    samples, rate = sf.read(str(path), start=frame_offset, frames=num_frames,
                            dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), rate


def save(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    sf.write(str(path), waveform.detach().cpu().numpy().T, sample_rate)
