from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio


DEFAULT_WEAK_COLUMNS = (
    "audio_path",
    "speech_id",
    "chunk_start_sec",
    "chunk_end_sec",
    "label_0",
    "label_1",
)

DEFAULT_STRONG_COLUMNS = (
    "speech_id",
    "event_class",
    "onset_sec",
    "offset_sec",
)


@dataclass(frozen=True)
class WeakChunkRecord:
    audio_path: str
    speech_id: str
    chunk_start_sec: float
    chunk_end_sec: float
    labels: tuple[float, ...]


@dataclass(frozen=True)
class StrongEvent:
    speech_id: str
    event_class: int
    onset_sec: float
    offset_sec: float


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], csv_path: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")


def read_weak_metadata(csv_path: str, num_classes: int = 2) -> list[WeakChunkRecord]:
    df = pd.read_csv(csv_path)
    required = DEFAULT_WEAK_COLUMNS[:4] + tuple(f"label_{idx}" for idx in range(num_classes))
    _validate_columns(df, required, csv_path)
    base_dir = str(Path(csv_path).resolve().parent)

    records: list[WeakChunkRecord] = []
    for row in df.itertuples(index=False):
        labels = tuple(float(getattr(row, f"label_{idx}")) for idx in range(num_classes))
        records.append(
            WeakChunkRecord(
                audio_path=resolve_path(base_dir, str(getattr(row, "audio_path"))),
                speech_id=str(getattr(row, "speech_id")),
                chunk_start_sec=float(getattr(row, "chunk_start_sec")),
                chunk_end_sec=float(getattr(row, "chunk_end_sec")),
                labels=labels,
            )
        )
    return records


def read_strong_events(csv_path: str) -> list[StrongEvent]:
    df = pd.read_csv(csv_path)
    _validate_columns(df, DEFAULT_STRONG_COLUMNS, csv_path)
    events: list[StrongEvent] = []
    for row in df.itertuples(index=False):
        events.append(
            StrongEvent(
                speech_id=str(getattr(row, "speech_id")),
                event_class=int(getattr(row, "event_class")),
                onset_sec=float(getattr(row, "onset_sec")),
                offset_sec=float(getattr(row, "offset_sec")),
            )
        )
    return events


def _resample_if_needed(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform[0]
    return waveform.mean(dim=0)


def load_audio_mono(audio_path: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    waveform = _resample_if_needed(waveform, sr, sample_rate)
    return _convert_to_mono(waveform).to(torch.float32)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(float(seconds) * float(sample_rate)))


def slice_waveform(
    waveform: torch.Tensor,
    start_sec: float,
    end_sec: float,
    sample_rate: int,
    target_num_samples: int | None = None,
) -> torch.Tensor:
    start_sample = max(0, seconds_to_samples(start_sec, sample_rate))
    end_sample = max(start_sample, seconds_to_samples(end_sec, sample_rate))
    chunk = waveform[start_sample:end_sample]
    if target_num_samples is None:
        return chunk

    if chunk.numel() >= target_num_samples:
        return chunk[:target_num_samples]

    padded = torch.zeros(target_num_samples, dtype=waveform.dtype)
    padded[: chunk.numel()] = chunk
    return padded


def split_into_instances(
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    instance_sec: float,
    chunk_sec: float,
) -> torch.Tensor:
    instance_samples = seconds_to_samples(instance_sec, sample_rate)
    expected_samples = seconds_to_samples(chunk_sec, sample_rate)
    if instance_samples <= 0:
        raise ValueError("instance_sec must be positive")
    if expected_samples <= 0:
        raise ValueError("chunk_sec must be positive")

    if chunk_waveform.numel() != expected_samples:
        chunk_waveform = slice_waveform(
            chunk_waveform,
            start_sec=0.0,
            end_sec=chunk_sec,
            sample_rate=sample_rate,
            target_num_samples=expected_samples,
        )

    num_instances = expected_samples // instance_samples
    if num_instances * instance_samples != expected_samples:
        raise ValueError("chunk_sec must be divisible by instance_sec")
    return chunk_waveform.view(num_instances, instance_samples)


class WeakChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_csv: str,
        *,
        sample_rate: int = 16000,
        chunk_sec: float = 30.0,
        instance_sec: float = 1.0,
        num_classes: int = 2,
    ) -> None:
        self.metadata_csv = metadata_csv
        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.instance_sec = float(instance_sec)
        self.num_classes = int(num_classes)
        self.records = read_weak_metadata(metadata_csv, num_classes=num_classes)
        self.chunk_num_samples = seconds_to_samples(self.chunk_sec, self.sample_rate)
        self.instances_per_chunk = int(round(self.chunk_sec / self.instance_sec))
        if self.instances_per_chunk * seconds_to_samples(self.instance_sec, self.sample_rate) != self.chunk_num_samples:
            raise ValueError("chunk_sec must be divisible by instance_sec at the configured sample_rate")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = load_audio_mono(record.audio_path, sample_rate=self.sample_rate)
        chunk = slice_waveform(
            waveform,
            start_sec=record.chunk_start_sec,
            end_sec=record.chunk_end_sec,
            sample_rate=self.sample_rate,
            target_num_samples=self.chunk_num_samples,
        )
        instances = split_into_instances(
            chunk,
            sample_rate=self.sample_rate,
            instance_sec=self.instance_sec,
            chunk_sec=self.chunk_sec,
        )
        return {
            "waveform": chunk,
            "instances": instances,
            "labels": torch.tensor(record.labels, dtype=torch.float32),
            "speech_id": record.speech_id,
            "audio_path": record.audio_path,
            "chunk_start_sec": torch.tensor(record.chunk_start_sec, dtype=torch.float32),
            "chunk_end_sec": torch.tensor(record.chunk_end_sec, dtype=torch.float32),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.stack([item["waveform"] for item in batch], dim=0),
        "instances": torch.stack([item["instances"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "speech_id": [item["speech_id"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
        "chunk_start_sec": torch.stack([item["chunk_start_sec"] for item in batch], dim=0),
        "chunk_end_sec": torch.stack([item["chunk_end_sec"] for item in batch], dim=0),
    }


def speech_durations_from_records(records: list[WeakChunkRecord]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for record in records:
        durations[record.speech_id] = max(durations.get(record.speech_id, 0.0), float(record.chunk_end_sec))
    return durations


def resolve_path(base_dir: str | None, maybe_relative_path: str) -> str:
    path = Path(maybe_relative_path)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((Path(base_dir) / path).resolve())
