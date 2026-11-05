#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import inference as infer_utils
from . import audio as audio_io
from .weights import get_weights, model_config
from crowd_reaction.data import WeakChunkDataset, chunk_records_for_strong_audio, collate_batch  # noqa: E402


API_MODES = ("polarity", "event")
# Label written to predicted_segments.csv for each model output; event mode
# exports crowd presence (the relevant_event head) as "crowd".
EXPORT_LABEL_NAMES = {"relevant_event": "crowd"}


@dataclass
class InferenceResult:
    audio_path: str
    instance_sec: float
    event_threshold: float
    attribute_threshold: float
    label_names: tuple[str, ...]
    times_sec: np.ndarray
    scores: np.ndarray
    predicted_regions: list[tuple[float, float, str]]
    median_filter_sec: float | None = 3.0
    mode: str = "polarity"

    def region_mean_score(self, onset: float, offset: float, label: str) -> float:
        return region_mean_score(self, onset, offset, label)

    def save(self, output_dir: str | Path, *, label_scores: bool = False,
             plot: bool = True, annotations=None, plot_score_functions: bool = True) -> None:
        """Write Sonic Visualiser CSV, scores JSON, and optionally a plot."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_scores_json(self, output_dir / "scores.json")
        write_predicted_segments_csv(self, output_dir / "predicted_segments.csv", label_scores=label_scores)
        if plot:
            plot_inference_result(self, output_dir / "plot.png", annotations,
                                  plot_score_functions=plot_score_functions)


class CrowdDetector:
    """Load once and annotate multiple recordings with the same classifier.

    The default joint CNN–TCN downloads to the shared cache on first use.
    Supply checkpoint_path for offline use or a custom trained checkpoint.
    """

    def __init__(self, model: str = "cnn_tcn", *, device: str | torch.device = "cpu",
                 cache_dir: str | Path | None = None,
                 config_path: str | Path | None = None,
                 checkpoint_path: str | Path | None = None) -> None:
        self.config = infer_utils.load_config(str(config_path)) if config_path is not None else model_config(model)
        encoder = self.config["model"].get("encoder_type", "beats")
        if encoder == "wav2vec2":
            try:
                import transformers  # noqa: F401
            except ImportError as exc:
                raise RuntimeError('Install wav2vec2 support with pip install "crowd-reaction-analysis[wav2vec2]"') from exc
        self.device = _resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path is not None else get_weights(encoder, cache_dir=cache_dir)
        self._model = infer_utils.load_model(self.config, str(self.checkpoint_path), self.device)

    def predict(self, audio_path: str | Path, *, mode: str = "polarity",
                event_threshold: float | None = None, attribute_threshold: float | None = None,
                median_filter_sec: float | None = 3.0, batch_size: int = 1,
                min_mean_score: float | None = None, show_progress: bool = True) -> InferenceResult:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if min_mean_score is not None and not 0 <= min_mean_score <= 1:
            raise ValueError("min_mean_score must be between 0 and 1")
        if median_filter_sec is not None and median_filter_sec < 0:
            raise ValueError("median_filter_sec must be nonnegative")
        for threshold in (event_threshold, attribute_threshold):
            if threshold is not None and not 0 <= threshold <= 1:
                raise ValueError("probability thresholds must be between 0 and 1")
        result = _run_audio_inference(
            audio_path, None, self.checkpoint_path, event_threshold, attribute_threshold,
            median_filter_sec, batch_size, self.device, mode, show_progress,
            _model=self._model, _config=self.config,
        )
        if min_mean_score is not None:
            result.predicted_regions = [
                (onset, offset, label) for onset, offset, label in result.predicted_regions
                if region_mean_score(result, onset, offset, label) > min_mean_score
            ]
        return result


def run_audio_inference(
    audio_path: str | Path,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    event_threshold: float | None = None,
    attribute_threshold: float | None = None,
    median_filter_sec: float | None = 3.0,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    mode: str = "polarity",
    show_progress: bool = True,
    *, model: str = "cnn_tcn", cache_dir: str | Path | None = None,
    min_mean_score: float | None = None,
) -> InferenceResult:
    """One-shot inference; reuse CrowdDetector for multiple audio files."""
    detector = CrowdDetector(model, device=device or "cpu", config_path=config_path,
                             checkpoint_path=checkpoint_path, cache_dir=cache_dir)
    return detector.predict(audio_path, mode=mode, event_threshold=event_threshold,
                            attribute_threshold=attribute_threshold, median_filter_sec=median_filter_sec,
                            batch_size=batch_size if batch_size is not None else 1,
                            min_mean_score=min_mean_score, show_progress=show_progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crowd-reaction inference on one audio file")
    parser.add_argument("--model", choices=("cnn_tcn", "beats", "wav2vec2"), default="cnn_tcn", help="Model to use (default: cnn_tcn)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Model cache directory")
    parser.add_argument("--config", default=None, help="Optional custom YAML configuration")
    parser.add_argument("--checkpoint", default=None, help="Optional local trained checkpoint")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--output-dir", required=True, help="Directory to save API outputs")
    parser.add_argument(
        "--mode",
        choices=API_MODES,
        default="polarity",
        help="polarity exports approval/disapproval boxes; event exports only crowd boxes (relevant_event head)",
    )
    parser.add_argument("--event-threshold", type=float, default=None, help="Override relevant-event probability threshold")
    parser.add_argument("--attribute-threshold", type=float, default=None, help="Override approval/disapproval probability threshold")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size")
    parser.add_argument("--device", default=None, help="Torch device string, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--annotations", default=None, help="Optional strong-label TXT annotations to overlay on the plot")
    parser.add_argument(
        "--median-filter-sec",
        type=float,
        default=3.0,
        help="Median-filter window for predicted approval/disapproval regions; set 0 to disable",
    )
    parser.add_argument(
        "--no-score-functions",
        action="store_true",
        help="Hide probability score functions and threshold lines from plot.png",
    )
    parser.add_argument(
        "--label-scores",
        action="store_true",
        help="Append each segment's mean probability to its label, e.g. 'crowd mean=0.83'",
    )
    parser.add_argument(
        "--min-mean-score",
        type=float,
        default=None,
        help="Keep only segments whose mean probability is above this value (e.g. 0.8)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the inference batch progress bar",
    )
    return parser.parse_args()


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _thresholds_from_config(
    config: dict[str, Any],
    *,
    event_threshold: float | None,
    attribute_threshold: float | None,
) -> tuple[float, float]:
    resolved_event_threshold = (
        float(event_threshold)
        if event_threshold is not None
        else float(config["val"].get("event_threshold", infer_utils.sigmoid(infer_utils.DEFAULT_EVENT_LOGIT_THRESHOLD)))
    )
    resolved_attribute_threshold = (
        float(attribute_threshold)
        if attribute_threshold is not None
        else float(config["val"].get("attribute_threshold", config["val"].get("threshold", 0.5)))
    )
    return resolved_event_threshold, resolved_attribute_threshold


def _build_single_audio_loader(
    *,
    audio_path: Path,
    config: dict[str, Any],
    batch_size: int | None,
) -> tuple[DataLoader, float]:
    records = chunk_records_for_strong_audio(
        audio_path=audio_path,
        chunk_sec=float(config["data"]["chunk_sec"]),
        split="infer",
    )
    dataset = WeakChunkDataset(
        records,
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
    )
    audio_info = audio_io.info(str(audio_path))
    audio_duration_sec = float(audio_info.num_frames) / float(audio_info.sample_rate)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config["val"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config["val"].get("num_workers", 0)),
        collate_fn=collate_batch,
    )
    return loader, audio_duration_sec


def _progress_loader(loader: DataLoader, *, enabled: bool):
    if not enabled:
        return loader
    return tqdm(loader, total=len(loader), desc="Inference batches", unit="batch")


def _scores_with_fixed_label_order(
    aggregated_scores: np.ndarray,
    active_label_names: list[str],
    *,
    num_bins: int,
) -> np.ndarray:
    fixed_scores = np.zeros((num_bins, len(infer_utils.PLOTTED_LABEL_ORDER)), dtype=np.float32)
    for output_index, label_name in enumerate(infer_utils.PLOTTED_LABEL_ORDER):
        if label_name not in active_label_names:
            continue
        input_index = active_label_names.index(label_name)
        copy_bins = min(num_bins, aggregated_scores.shape[0])
        fixed_scores[:copy_bins, output_index] = aggregated_scores[:copy_bins, input_index]
    return fixed_scores


def _select_mode_outputs(
    scores: np.ndarray,
    *,
    label_names: tuple[str, ...],
    active_label_names: list[str],
    mode: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if mode == "polarity":
        return scores, label_names, ("approval", "disapproval")
    if mode == "event":
        if "relevant_event" not in active_label_names:
            raise ValueError("event mode requires a checkpoint with a relevant_event/event head")
        event_index = label_names.index("relevant_event")
        return scores[:, event_index : event_index + 1], ("relevant_event",), ("relevant_event",)
    raise ValueError(f"Unknown API mode: {mode}")


def _median_filter_binary(values: np.ndarray, *, kernel_bins: int) -> np.ndarray:
    binary = np.asarray(values, dtype=np.int64).reshape(-1)
    if kernel_bins <= 1 or binary.size == 0:
        return binary.copy()
    if kernel_bins % 2 == 0:
        kernel_bins += 1
    pad_bins = kernel_bins // 2
    padded = np.pad(binary, (pad_bins, pad_bins), mode="constant", constant_values=0)
    filtered = np.zeros_like(binary)
    for index in range(binary.size):
        filtered[index] = int(np.median(padded[index : index + kernel_bins]) >= 0.5)
    return filtered


def _thresholded_prediction_masks(
    predicted_probs: np.ndarray,
    *,
    label_names: list[str],
    event_threshold: float,
    attribute_threshold: float,
    instance_sec: float,
    median_filter_sec: float | None,
    filtered_labels: tuple[str, ...] = ("approval", "disapproval"),
) -> np.ndarray:
    if predicted_probs.shape[1] != len(label_names):
        raise ValueError("predicted_probs second dimension must match label_names")
    predicted_binary = np.zeros_like(predicted_probs, dtype=np.int64)
    for class_index, label in enumerate(label_names):
        threshold = event_threshold if label == "relevant_event" else attribute_threshold
        predicted_binary[:, class_index] = (predicted_probs[:, class_index] >= threshold).astype(np.int64)
    if "relevant_event" in label_names:
        event_index = label_names.index("relevant_event")
        event_active = predicted_binary[:, event_index].astype(bool)
        for gated_label in ("approval", "disapproval"):
            if gated_label in label_names:
                gated_index = label_names.index(gated_label)
                predicted_binary[:, gated_index] = np.logical_and(
                    event_active,
                    predicted_binary[:, gated_index].astype(bool),
                ).astype(np.int64)

    if median_filter_sec is not None and float(median_filter_sec) > 0.0:
        kernel_bins = int(np.ceil(float(median_filter_sec) / float(instance_sec)))
        for label in filtered_labels:
            if label in label_names:
                label_index = label_names.index(label)
                predicted_binary[:, label_index] = _median_filter_binary(
                    predicted_binary[:, label_index],
                    kernel_bins=kernel_bins,
                )
    return predicted_binary


def predicted_regions_with_median_filter(
    predicted_probs: np.ndarray,
    *,
    label_names: list[str],
    event_threshold: float,
    attribute_threshold: float,
    instance_sec: float,
    median_filter_sec: float | None = 3.0,
    export_labels: tuple[str, ...] = ("approval", "disapproval"),
) -> list[tuple[float, float, str]]:
    predicted_binary = _thresholded_prediction_masks(
        predicted_probs,
        label_names=label_names,
        event_threshold=event_threshold,
        attribute_threshold=attribute_threshold,
        instance_sec=instance_sec,
        median_filter_sec=median_filter_sec,
        filtered_labels=export_labels,
    )

    regions: list[tuple[float, float, str]] = []
    export_label_set = set(export_labels)
    for class_index, label in enumerate(label_names):
        if label not in export_label_set:
            continue
        for onset_sec, offset_sec in infer_utils.contiguous_regions(predicted_binary[:, class_index], instance_sec=instance_sec):
            regions.append((float(onset_sec), float(offset_sec), label))
    return sorted(regions, key=lambda item: (item[0], item[1], item[2]))


def scores_for_plot(result: InferenceResult) -> np.ndarray:
    plot_scores = np.asarray(result.scores, dtype=np.float32).copy()
    predicted_binary = _thresholded_prediction_masks(
        plot_scores,
        label_names=list(result.label_names),
        event_threshold=float(result.event_threshold),
        attribute_threshold=float(result.attribute_threshold),
        instance_sec=float(result.instance_sec),
        median_filter_sec=result.median_filter_sec,
        filtered_labels=("approval", "disapproval"),
    )
    for label in ("approval", "disapproval"):
        if label in result.label_names:
            label_index = result.label_names.index(label)
            plot_scores[:, label_index] = predicted_binary[:, label_index].astype(np.float32)
    return plot_scores


def _run_audio_inference(
    audio_path: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    event_threshold: float | None = None,
    attribute_threshold: float | None = None,
    median_filter_sec: float | None = 3.0,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    mode: str = "polarity",
    show_progress: bool = True,
    *,
    _model=None,
    _config=None,
) -> InferenceResult:
    if mode not in API_MODES:
        raise ValueError(f"mode must be one of {API_MODES}, got {mode!r}")
    audio_path = Path(audio_path)
    config = _config if _config is not None else infer_utils.load_config(str(config_path))
    resolved_device = _resolve_device(device)
    resolved_event_threshold, resolved_attribute_threshold = _thresholds_from_config(
        config,
        event_threshold=event_threshold,
        attribute_threshold=attribute_threshold,
    )
    loader, audio_duration_sec = _build_single_audio_loader(
        audio_path=audio_path,
        config=config,
        batch_size=batch_size,
    )
    model = _model if _model is not None else infer_utils.load_model(config, str(checkpoint_path), resolved_device)
    chunk_predictions_by_task = infer_utils.collect_multitask_chunk_predictions(
        model,
        _progress_loader(loader, enabled=show_progress),
        device=resolved_device,
    )
    active_label_names = infer_utils.active_label_order(chunk_predictions_by_task)
    aggregated_by_speech = infer_utils.aggregate_multitask_probs(
        chunk_predictions_by_task,
        instance_sec=float(config["data"]["instance_sec"]),
        speech_durations={audio_path.stem: audio_duration_sec},
    )
    aggregated_scores = aggregated_by_speech.get(audio_path.stem)
    instance_sec = float(config["data"]["instance_sec"])
    num_bins = int(np.ceil(audio_duration_sec / instance_sec))
    if aggregated_scores is None:
        aggregated_scores = np.zeros((num_bins, 0), dtype=np.float32)
    scores = _scores_with_fixed_label_order(
        aggregated_scores,
        active_label_names,
        num_bins=num_bins,
    )
    label_names = tuple(infer_utils.PLOTTED_LABEL_ORDER)
    scores, label_names, export_labels = _select_mode_outputs(
        scores,
        label_names=label_names,
        active_label_names=active_label_names,
        mode=mode,
    )
    predicted_regions = predicted_regions_with_median_filter(
        scores,
        label_names=list(label_names),
        event_threshold=resolved_event_threshold,
        attribute_threshold=resolved_attribute_threshold,
        instance_sec=instance_sec,
        median_filter_sec=median_filter_sec,
        export_labels=export_labels,
    )
    times_sec = (np.arange(num_bins, dtype=np.float32) + 0.5) * instance_sec
    return InferenceResult(
        audio_path=str(audio_path),
        mode=mode,
        instance_sec=instance_sec,
        event_threshold=resolved_event_threshold,
        attribute_threshold=resolved_attribute_threshold,
        label_names=label_names,
        times_sec=times_sec,
        scores=scores,
        predicted_regions=predicted_regions,
        median_filter_sec=median_filter_sec,
    )


def scores_as_dict(result: InferenceResult) -> dict[str, list[float]]:
    return {
        label_name: [float(value) for value in result.scores[:, label_index]]
        for label_index, label_name in enumerate(result.label_names)
    }


def write_scores_json(result: InferenceResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(scores_as_dict(result), handle, indent=2)


def region_mean_score(result: InferenceResult, onset: float, offset: float, label: str) -> float:
    """Mean probability of label over the bins of [onset, offset)."""
    column = result.scores[:, result.label_names.index(label)]
    first = int(round(onset / result.instance_sec))
    last = max(first + 1, int(round(offset / result.instance_sec)))
    values = column[first:last]
    return float(values.mean())


def write_predicted_segments_csv(
    result: InferenceResult, output_path: str | Path, *, label_scores: bool = False
) -> None:
    regions = []
    for onset, offset, label in result.predicted_regions:
        text = EXPORT_LABEL_NAMES.get(label, label)
        if label_scores:
            mean = region_mean_score(result, onset, offset, label)
            text = f"{text} mean={mean:.2f}"
        regions.append((onset, offset, text))
    infer_utils.write_sonic_visualiser_regions(Path(output_path), regions)


def plot_inference_result(
    result: InferenceResult,
    output_path: str | Path,
    annotations: dict[str, list[tuple[float, float]]] | None = None,
    *,
    plot_score_functions: bool = True,
) -> None:
    audio_info = audio_io.info(result.audio_path)
    infer_utils.plot_speech(
        audio_path=result.audio_path,
        speech_id=Path(result.audio_path).stem,
        predicted_probs=scores_for_plot(result),
        predicted_regions=result.predicted_regions,
        label_names=list(result.label_names),
        ground_truth_annotations=annotations or {},
        sample_rate=int(audio_info.sample_rate),
        instance_sec=float(result.instance_sec),
        event_threshold=float(result.event_threshold),
        attribute_threshold=float(result.attribute_threshold),
        output_path=Path(output_path),
        plot_score_functions=plot_score_functions,
        score_axis_max=2.0,
        score_alpha=0.4,
    )


def _plot_inference_result_with_config(
    result: InferenceResult,
    output_path: Path,
    *,
    config: dict[str, Any],
    annotations: dict[str, list[tuple[float, float]]] | None,
    event_threshold: float,
    attribute_threshold: float,
    plot_score_functions: bool,
) -> None:
    infer_utils.plot_speech(
        audio_path=result.audio_path,
        speech_id=Path(result.audio_path).stem,
        predicted_probs=scores_for_plot(result),
        predicted_regions=result.predicted_regions,
        label_names=list(result.label_names),
        ground_truth_annotations=annotations or {},
        sample_rate=int(config["data"]["sample_rate"]),
        instance_sec=float(result.instance_sec),
        event_threshold=float(event_threshold),
        attribute_threshold=float(attribute_threshold),
        output_path=output_path,
        plot_score_functions=plot_score_functions,
        score_axis_max=2.0,
        score_alpha=0.4,
    )


def main() -> None:
    args = parse_args()
    config = infer_utils.load_config(args.config) if args.config else model_config(args.model)
    event_threshold, attribute_threshold = _thresholds_from_config(
        config,
        event_threshold=args.event_threshold,
        attribute_threshold=args.attribute_threshold,
    )
    result = run_audio_inference(
        audio_path=args.audio,
        model=args.model,
        cache_dir=args.cache_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        event_threshold=event_threshold,
        attribute_threshold=attribute_threshold,
        median_filter_sec=args.median_filter_sec,
        batch_size=args.batch_size,
        device=args.device,
        mode=args.mode,
        show_progress=not args.no_progress,
        min_mean_score=args.min_mean_score,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = (
        infer_utils.parse_raw_strong_annotations(str(args.annotations))
        if args.annotations is not None
        else None
    )
    scores_path = output_dir / "scores.json"
    segments_path = output_dir / "predicted_segments.csv"
    plot_path = output_dir / "plot.png"
    write_scores_json(result, scores_path)
    write_predicted_segments_csv(result, segments_path, label_scores=args.label_scores)
    _plot_inference_result_with_config(
        result,
        plot_path,
        config=config,
        annotations=annotations,
        event_threshold=event_threshold,
        attribute_threshold=attribute_threshold,
        plot_score_functions=not args.no_score_functions,
    )
    print(f"Saved {scores_path}")
    print(f"Saved {segments_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
