from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data import StrongEvent


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def binary_f1_score(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    predictions = probs >= threshold
    tp = float(np.logical_and(predictions, targets == 1).sum())
    fp = float(np.logical_and(predictions, targets == 0).sum())
    fn = float(np.logical_and(~predictions, targets == 1).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return _safe_divide(2.0 * precision * recall, precision + recall)


def binary_auroc(targets: np.ndarray, probs: np.ndarray) -> float:
    positives = targets == 1
    negatives = targets == 0
    num_pos = int(positives.sum())
    num_neg = int(negatives.sum())
    if num_pos == 0 or num_neg == 0:
        return 0.0

    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[positives].sum())
    return (pos_rank_sum - (num_pos * (num_pos + 1) / 2.0)) / float(num_pos * num_neg)


def binary_average_precision(targets: np.ndarray, probs: np.ndarray) -> float:
    num_pos = int((targets == 1).sum())
    if num_pos == 0:
        return 0.0
    order = np.argsort(-probs)
    sorted_targets = targets[order]
    tp = np.cumsum(sorted_targets == 1)
    fp = np.cumsum(sorted_targets == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / num_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    gain = recall - recall_prev
    return float(np.sum(precision * gain))


def evaluate_weak(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    if targets.shape != probs.shape:
        raise ValueError("targets and probs must have the same shape")

    per_class = []
    for class_index in range(targets.shape[1]):
        class_targets = targets[:, class_index].astype(np.int64)
        class_probs = probs[:, class_index].astype(np.float64)
        per_class.append(
            {
                "class_index": class_index,
                "auroc": binary_auroc(class_targets, class_probs),
                "average_precision": binary_average_precision(class_targets, class_probs),
                "f1": binary_f1_score(class_targets, class_probs, threshold=threshold),
            }
        )

    return {
        "per_class": per_class,
        "macro_auroc": float(np.mean([item["auroc"] for item in per_class])) if per_class else 0.0,
        "macro_average_precision": float(np.mean([item["average_precision"] for item in per_class])) if per_class else 0.0,
        "macro_f1": float(np.mean([item["f1"] for item in per_class])) if per_class else 0.0,
    }


@dataclass(frozen=True)
class SpeechChunkPrediction:
    speech_id: str
    chunk_start_sec: float
    chunk_end_sec: float
    instance_probs: np.ndarray


def aggregate_chunk_predictions(
    predictions: list[SpeechChunkPrediction],
    *,
    num_classes: int,
    instance_sec: float,
    speech_durations: dict[str, float],
) -> dict[str, np.ndarray]:
    aggregated: dict[str, np.ndarray] = {}
    for speech_id, duration_sec in speech_durations.items():
        num_bins = int(np.ceil(duration_sec / instance_sec))
        aggregated[speech_id] = np.zeros((num_bins, num_classes), dtype=np.float32)

    for prediction in predictions:
        speech_bins = aggregated.setdefault(
            prediction.speech_id,
            np.zeros((int(np.ceil(prediction.chunk_end_sec / instance_sec)), num_classes), dtype=np.float32),
        )
        start_bin = int(round(prediction.chunk_start_sec / instance_sec))
        for offset, values in enumerate(prediction.instance_probs):
            bin_index = start_bin + offset
            if 0 <= bin_index < speech_bins.shape[0]:
                speech_bins[bin_index] = np.maximum(speech_bins[bin_index], values.astype(np.float32))
    return aggregated


def strong_events_to_bin_targets(
    strong_events: list[StrongEvent],
    *,
    num_classes: int,
    instance_sec: float,
    speech_durations: dict[str, float],
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for speech_id, duration_sec in speech_durations.items():
        num_bins = int(np.ceil(duration_sec / instance_sec))
        targets[speech_id] = np.zeros((num_bins, num_classes), dtype=np.int64)

    for event in strong_events:
        speech_bins = targets.setdefault(
            event.speech_id,
            np.zeros((int(np.ceil(event.offset_sec / instance_sec)), num_classes), dtype=np.int64),
        )
        start_bin = max(0, int(np.floor(event.onset_sec / instance_sec)))
        end_bin = max(start_bin + 1, int(np.ceil(event.offset_sec / instance_sec)))
        speech_bins[start_bin:end_bin, event.event_class] = 1
    return targets


def contiguous_regions(binary_bins: np.ndarray, *, instance_sec: float) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    start = None
    for idx, value in enumerate(binary_bins.tolist()):
        if value and start is None:
            start = idx
        if not value and start is not None:
            regions.append((start * instance_sec, idx * instance_sec))
            start = None
    if start is not None:
        regions.append((start * instance_sec, len(binary_bins) * instance_sec))
    return regions


def _import_sed_eval():
    try:
        import sed_eval  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Strong evaluation now requires `sed_eval` and its dependency `dcase_util`. "
            "Install project dependencies before running validation."
        ) from exc
    return sed_eval


def _event_dict(file_id: str, class_index: int, onset_sec: float, offset_sec: float) -> dict[str, Any]:
    return {
        "file": file_id,
        "event_label": str(class_index),
        "event_onset": float(onset_sec),
        "event_offset": float(offset_sec),
    }


def _build_reference_event_lists(
    strong_events: list[StrongEvent],
    *,
    speech_ids: list[str],
    num_classes: int,
) -> dict[str, list[dict[str, Any]]]:
    reference_by_speech = {speech_id: [] for speech_id in speech_ids}
    for event in strong_events:
        reference_by_speech.setdefault(event.speech_id, [])
        reference_by_speech[event.speech_id].append(
            _event_dict(event.speech_id, event.event_class, event.onset_sec, event.offset_sec)
        )
    return reference_by_speech


def evaluate_strong(
    predictions: list[SpeechChunkPrediction],
    strong_events: list[StrongEvent],
    *,
    num_classes: int,
    instance_sec: float,
    speech_durations: dict[str, float],
    threshold: float = 0.5,
    event_collar_sec: float = 1.0,
    event_offset_ratio: float = 0.2,
) -> dict[str, Any]:
    sed_eval = _import_sed_eval()
    aggregated = aggregate_chunk_predictions(
        predictions,
        num_classes=num_classes,
        instance_sec=instance_sec,
        speech_durations=speech_durations,
    )
    targets = strong_events_to_bin_targets(
        strong_events,
        num_classes=num_classes,
        instance_sec=instance_sec,
        speech_durations=speech_durations,
    )

    event_labels = [str(class_index) for class_index in range(num_classes)]
    segment_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=event_labels,
        time_resolution=float(instance_sec),
    )
    event_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=event_labels,
        t_collar=float(event_collar_sec),
        percentage_of_length=float(event_offset_ratio),
    )

    speech_ids = sorted(set(aggregated) | set(targets) | set(speech_durations))
    reference_by_speech = _build_reference_event_lists(strong_events, speech_ids=speech_ids, num_classes=num_classes)
    estimated_by_speech: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for speech_id in speech_ids:
        pred_bins = aggregated.get(speech_id)
        if pred_bins is None:
            pred_bins = np.zeros((int(np.ceil(speech_durations.get(speech_id, 0.0) / instance_sec)), num_classes), dtype=np.float32)

        for class_index in range(num_classes):
            pred_binary = (pred_bins[:, class_index] >= threshold).astype(np.int64)
            for onset_sec, offset_sec in contiguous_regions(pred_binary, instance_sec=instance_sec):
                estimated_by_speech[speech_id].append(_event_dict(speech_id, class_index, onset_sec, offset_sec))

        segment_metrics.evaluate(
            reference_event_list=reference_by_speech.get(speech_id, []),
            estimated_event_list=estimated_by_speech.get(speech_id, []),
            evaluated_length_seconds=float(speech_durations.get(speech_id, 0.0)),
        )
        event_metrics.evaluate(
            reference_event_list=reference_by_speech.get(speech_id, []),
            estimated_event_list=estimated_by_speech.get(speech_id, []),
        )

    segment_overall = segment_metrics.results_overall_metrics()
    segment_class_wise = segment_metrics.results_class_wise_metrics()
    event_overall = event_metrics.results_overall_metrics()
    event_class_wise = event_metrics.results_class_wise_metrics()

    per_class_segment = []
    per_class_event = []
    for class_index, label in enumerate(event_labels):
        segment_class = segment_class_wise[label]["f_measure"]
        event_class = event_class_wise[label]["f_measure"]
        per_class_segment.append(
            {
                "class_index": class_index,
                "precision": float(segment_class["precision"]),
                "recall": float(segment_class["recall"]),
                "f1": float(segment_class["f_measure"]),
            }
        )
        per_class_event.append(
            {
                "class_index": class_index,
                "precision": float(event_class["precision"]),
                "recall": float(event_class["recall"]),
                "f1": float(event_class["f_measure"]),
            }
        )

    return {
        "segment_per_class": per_class_segment,
        "segment_macro_precision": float(np.mean([item["precision"] for item in per_class_segment])) if per_class_segment else 0.0,
        "segment_macro_recall": float(np.mean([item["recall"] for item in per_class_segment])) if per_class_segment else 0.0,
        "segment_macro_f1": float(np.mean([item["f1"] for item in per_class_segment])) if per_class_segment else 0.0,
        "event_per_class": per_class_event,
        "event_precision": float(event_overall["f_measure"]["precision"]),
        "event_recall": float(event_overall["f_measure"]["recall"]),
        "event_f1": float(event_overall["f_measure"]["f_measure"]),
    }


@torch.no_grad()
def collect_strong_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[SpeechChunkPrediction]]:
    all_targets = []
    all_probs = []
    chunk_predictions: list[SpeechChunkPrediction] = []

    for batch in dataloader:
        instances = batch["instances"].to(device)
        labels = batch["labels"].cpu().numpy()
        logits, bag_probs = model(instances=instances)
        instance_probs = torch.sigmoid(logits).cpu().numpy()
        all_targets.append(labels)
        all_probs.append(bag_probs.cpu().numpy())

        for batch_index in range(instances.shape[0]):
            chunk_predictions.append(
                SpeechChunkPrediction(
                    speech_id=batch["speech_id"][batch_index],
                    chunk_start_sec=float(batch["chunk_start_sec"][batch_index].item()),
                    chunk_end_sec=float(batch["chunk_end_sec"][batch_index].item()),
                    instance_probs=instance_probs[batch_index],
                )
            )

    weak_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 0), dtype=np.float32)
    weak_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0), dtype=np.float32)
    return weak_targets, weak_probs, chunk_predictions
