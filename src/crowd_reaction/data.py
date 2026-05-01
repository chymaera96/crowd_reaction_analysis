from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio


APPROVAL_LABELS = {
    "clear_approval",
    "unclear_approval",
}
DISAPPROVAL_LABELS = {
    "clear_disapproval",
    "unclear_disapproval",
}
POSITIVE_EVENT_LABELS = APPROVAL_LABELS | DISAPPROVAL_LABELS | {"crowd_chorus"}
NEGATIVE_CROWD_LABEL = "no_crowd"
STRONG_TEXT_GLOB = "noise_*.txt"
SEGMENT_SUFFIX_PATTERN = re.compile(r"_seg\d+$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class WeakBagTargets:
    event_target: tuple[float, ...]
    event_mask: float
    approval_target: tuple[float, ...]
    approval_mask: float
    disapproval_target: tuple[float, ...]
    disapproval_mask: float


@dataclass(frozen=True)
class WeakChunkRecord:
    audio_path: str
    speech_id: str
    chunk_start_sec: float
    chunk_end_sec: float
    targets: WeakBagTargets
    split: str


@dataclass(frozen=True)
class StrongEvent:
    speech_id: str
    event_class: int
    onset_sec: float
    offset_sec: float


@dataclass(frozen=True)
class SplitDatasets:
    train_records: list[WeakChunkRecord]
    val_records: list[WeakChunkRecord]
    strong_events: list[StrongEvent]


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = normalized.replace("\\", "/").split("/")[-1]
    if normalized.lower().startswith("noise_"):
        normalized = normalized[6:]
    normalized = re.sub(r"\.[^.]+$", "", normalized)
    normalized = SEGMENT_SUFFIX_PATTERN.sub("", normalized)

    translation = str.maketrans(
        {
            "’": "'",
            "‘": "'",
            "‚": "'",
            "‛": "'",
            "“": '"',
            "”": '"',
            "„": '"',
            "‟": '"',
            "–": "-",
            "—": "-",
            "―": "-",
            "‐": "-",
            "｜": "|",
            "：": ":",
            "？": "?",
            "！": "!",
            "＂": '"',
            "／": "/",
            "＆": "&",
            "＃": "#",
        }
    )
    normalized = normalized.translate(translation)
    normalized = re.sub(r"[^\x00-\x7F]+", "", normalized)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = "".join(ch for ch in normalized if ch.isalnum())
    return normalized


def _row_flag(row: pd.Series, column: str) -> int:
    value = row.get(column, 0)
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0
    return int(float(value))


def _label_confidence(row: pd.Series, *, clear_label: str, unclear_label: str, unclear_weight: float) -> float:
    if _row_flag(row, clear_label) > 0:
        return 1.0
    if _row_flag(row, unclear_label) > 0:
        return float(unclear_weight)
    return 0.0


def weak_row_to_targets(row: pd.Series, *, unclear_label_weight: float = 0.5) -> WeakBagTargets:
    approval_confidence = max(
        _label_confidence(
            row,
            clear_label="clear_approval",
            unclear_label="unclear_approval",
            unclear_weight=unclear_label_weight,
        ),
        0.0,
    )
    disapproval_confidence = max(
        _label_confidence(
            row,
            clear_label="clear_disapproval",
            unclear_label="unclear_disapproval",
            unclear_weight=unclear_label_weight,
        ),
        0.0,
    )
    approval = approval_confidence > 0.0
    disapproval = disapproval_confidence > 0.0
    crowd_chorus = _row_flag(row, "crowd_chorus") > 0
    no_crowd = _row_flag(row, NEGATIVE_CROWD_LABEL) > 0

    event_positive = approval or disapproval or crowd_chorus
    contradictory_event = event_positive and no_crowd
    event_mask = 0.0 if contradictory_event or (not event_positive and not no_crowd) else 1.0
    event_target = (1.0,) if event_positive and not contradictory_event else (0.0,)

    approval_target = (1.0 if approval else 0.0,)
    disapproval_target = (1.0 if disapproval else 0.0,)
    attribute_mask = 0.0 if no_crowd or crowd_chorus or contradictory_event or not event_positive else max(approval_confidence, disapproval_confidence)
    approval_mask = approval_confidence if approval and attribute_mask > 0.0 else attribute_mask
    disapproval_mask = disapproval_confidence if disapproval and attribute_mask > 0.0 else attribute_mask

    return WeakBagTargets(
        event_target=event_target,
        event_mask=event_mask,
        approval_target=approval_target,
        approval_mask=approval_mask,
        disapproval_target=disapproval_target,
        disapproval_mask=disapproval_mask,
    )


def weak_row_to_labels(row: pd.Series) -> tuple[float, ...]:
    return weak_row_to_targets(row).event_target


def strong_label_to_class(label: str) -> int | None:
    label = str(label).strip()
    if label in POSITIVE_EVENT_LABELS:
        return 0
    return None


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


def _truthy_strong_label(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "0", "false", "False", "nan", "None"}
    return bool(value)


def _build_original_audio_index(original_audio_dir: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for audio_path in sorted(Path(original_audio_dir).glob("*.wav")):
        key = normalize_name(audio_path.name)
        if key in index and index[key] != audio_path:
            raise ValueError(f"Ambiguous original audio filename normalization for {audio_path.name}")
        index[key] = audio_path.resolve()
    return index


def _build_audio_info_index(audios_info_csv: str) -> dict[str, bool]:
    df = pd.read_csv(audios_info_csv)
    required = {"title", "strong_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {audios_info_csv}: {missing}")

    index: dict[str, bool] = {}
    for row in df.itertuples(index=False):
        key = normalize_name(getattr(row, "title"))
        strong = _truthy_strong_label(getattr(row, "strong_label"))
        index[key] = index.get(key, False) or strong
    return index


def _infer_validation_key(
    source_key: str,
    *,
    audio_info_index: dict[str, bool],
    strong_txt_by_key: dict[str, Path],
) -> bool:
    if source_key in audio_info_index:
        return bool(audio_info_index[source_key])
    if source_key in strong_txt_by_key:
        return True
    return False


def parse_strong_label_file(strong_txt_path: str, speech_id: str) -> list[StrongEvent]:
    events: list[StrongEvent] = []
    with open(strong_txt_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed strong label line {line_number} in {strong_txt_path}: {stripped}")
            onset_sec, offset_sec, label = parts
            class_index = strong_label_to_class(label)
            if class_index is None:
                continue
            events.append(
                StrongEvent(
                    speech_id=speech_id,
                    event_class=class_index,
                    onset_sec=float(onset_sec),
                    offset_sec=float(offset_sec),
                )
            )
    return events


def build_split_records(
    *,
    audios_info_csv: str,
    weak_labels_csv: str,
    strong_labels_dir: str,
    original_audio_dir: str,
    negative_data_dir: str | None = None,
    chunk_sec: float = 20.0,
    unclear_label_weight: float = 0.5,
) -> SplitDatasets:
    audio_info_index = _build_audio_info_index(audios_info_csv)
    original_audio_index = _build_original_audio_index(original_audio_dir)
    weak_df = pd.read_csv(weak_labels_csv)

    required_weak_columns = {
        "source_file",
        "start_sec",
        "end_sec",
        "no_crowd",
        "clear_disapproval",
        "unclear_disapproval",
        "unclear_approval",
        "clear_approval",
        "crowd_chorus",
    }
    missing = sorted(required_weak_columns - set(weak_df.columns))
    if missing:
        raise ValueError(f"Missing columns in {weak_labels_csv}: {missing}")

    strong_txt_by_key = {
        normalize_name(path.name): path.resolve()
        for path in sorted(Path(strong_labels_dir).glob(STRONG_TEXT_GLOB))
    }

    train_records: list[WeakChunkRecord] = []
    val_records: list[WeakChunkRecord] = []
    validation_speech_ids: dict[str, Path] = {}

    for row in weak_df.itertuples(index=False):
        source_file = str(getattr(row, "source_file"))
        source_key = normalize_name(source_file)
        audio_path = original_audio_index.get(source_key)
        if audio_path is None:
            raise ValueError(f"Weak row source file {source_file} does not match any original audio file")

        speech_id = audio_path.stem
        targets = weak_row_to_targets(pd.Series(row._asdict()), unclear_label_weight=unclear_label_weight)
        split = "val" if _infer_validation_key(source_key, audio_info_index=audio_info_index, strong_txt_by_key=strong_txt_by_key) else "train"
        record = WeakChunkRecord(
            audio_path=str(audio_path),
            speech_id=speech_id,
            chunk_start_sec=float(getattr(row, "start_sec")),
            chunk_end_sec=float(getattr(row, "end_sec")),
            targets=targets,
            split=split,
        )
        if split == "val":
            val_records.append(record)
            validation_speech_ids[source_key] = audio_path
        else:
            train_records.append(record)

    if negative_data_dir:
        negative_dir = Path(negative_data_dir)
        if negative_dir.exists():
            negative_targets = weak_row_to_targets(
                pd.Series(
                    {
                        "clear_disapproval": 0,
                        "unclear_disapproval": 0,
                        "unclear_approval": 0,
                        "clear_approval": 0,
                        "no_crowd": 1,
                        "crowd_chorus": 0,
                    }
                ),
                unclear_label_weight=unclear_label_weight,
            )
            for audio_path in sorted(negative_dir.glob("*.wav")):
                train_records.append(
                    WeakChunkRecord(
                        audio_path=str(audio_path.resolve()),
                        speech_id=audio_path.stem,
                        chunk_start_sec=0.0,
                        chunk_end_sec=float(chunk_sec),
                        targets=negative_targets,
                        split="train",
                    )
                )

    strong_events: list[StrongEvent] = []
    for source_key, audio_path in validation_speech_ids.items():
        strong_txt_path = strong_txt_by_key.get(source_key)
        if strong_txt_path is None:
            raise ValueError(f"Validation file {audio_path.name} is marked strong-labeled but has no matching TXT file")
        strong_events.extend(parse_strong_label_file(str(strong_txt_path), speech_id=audio_path.stem))

    for source_key, is_strong in audio_info_index.items():
        if not is_strong:
            continue
        if source_key in original_audio_index and source_key not in strong_txt_by_key:
            raise ValueError(f"Strong-labeled title key {source_key} has no matching strong TXT file")

    return SplitDatasets(train_records=train_records, val_records=val_records, strong_events=strong_events)


class WeakChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[WeakChunkRecord],
        *,
        sample_rate: int = 16000,
        chunk_sec: float = 20.0,
        instance_sec: float = 1.0,
        num_classes: int = 1,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.instance_sec = float(instance_sec)
        self.num_classes = int(num_classes)
        self.records = list(records)
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
        targets = {
            "event_target": torch.tensor(record.targets.event_target, dtype=torch.float32),
            "event_mask": torch.tensor(record.targets.event_mask, dtype=torch.float32),
            "approval_target": torch.tensor(record.targets.approval_target, dtype=torch.float32),
            "approval_mask": torch.tensor(record.targets.approval_mask, dtype=torch.float32),
            "disapproval_target": torch.tensor(record.targets.disapproval_target, dtype=torch.float32),
            "disapproval_mask": torch.tensor(record.targets.disapproval_mask, dtype=torch.float32),
        }
        return {
            "waveform": chunk,
            "instances": instances,
            "targets": targets,
            "labels": targets["event_target"],
            "speech_id": record.speech_id,
            "audio_path": record.audio_path,
            "chunk_start_sec": torch.tensor(record.chunk_start_sec, dtype=torch.float32),
            "chunk_end_sec": torch.tensor(record.chunk_end_sec, dtype=torch.float32),
            "split": record.split,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.stack([item["waveform"] for item in batch], dim=0),
        "instances": torch.stack([item["instances"] for item in batch], dim=0),
        "targets": {
            "event_target": torch.stack([item["targets"]["event_target"] for item in batch], dim=0),
            "event_mask": torch.stack([item["targets"]["event_mask"] for item in batch], dim=0),
            "approval_target": torch.stack([item["targets"]["approval_target"] for item in batch], dim=0),
            "approval_mask": torch.stack([item["targets"]["approval_mask"] for item in batch], dim=0),
            "disapproval_target": torch.stack([item["targets"]["disapproval_target"] for item in batch], dim=0),
            "disapproval_mask": torch.stack([item["targets"]["disapproval_mask"] for item in batch], dim=0),
        },
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "speech_id": [item["speech_id"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
        "chunk_start_sec": torch.stack([item["chunk_start_sec"] for item in batch], dim=0),
        "chunk_end_sec": torch.stack([item["chunk_end_sec"] for item in batch], dim=0),
        "split": [item["split"] for item in batch],
    }


def speech_durations_from_records(records: list[WeakChunkRecord]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for record in records:
        durations[record.speech_id] = max(durations.get(record.speech_id, 0.0), float(record.chunk_end_sec))
    return durations
