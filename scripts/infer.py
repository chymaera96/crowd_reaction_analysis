#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crowd_reaction.data import (
    WeakChunkDataset,
    build_strong_validation_split,
    collate_batch,
    normalize_name,
    speech_durations_from_records,
)
from crowd_reaction.eval import SpeechChunkPrediction, aggregate_chunk_predictions, contiguous_regions
from crowd_reaction.model import CrowdReactionModel


PLOTTED_LABEL_ORDER = [
    "relevant_event",
    "approval",
    "disapproval",
]
TASK_EXPORT_SPECS = {
    "event": ["relevant_event"],
    "approval": ["approval"],
    "disapproval": ["disapproval"],
}
PREDICTION_COLORS = {
    "relevant_event": "#000000",
    "approval": "#2ca02c",
    "disapproval": "#d62728",
}
SCORE_LINE_COLORS = {
    "relevant_event": "#000000",
    "approval": "#2ca02c",
    "disapproval": "#d62728",
}
DEFAULT_EVENT_LOGIT_THRESHOLD = -1.5
GROUND_TRUTH_LABEL_COLORS = {
    "clear_disapproval": "#d62728",
    "unclear_disapproval": "#ff7f0e",
    "unclear_approval": "#17becf",
    "clear_approval": "#2ca02c",
    "crowd_chorus": "#9467bd",
}
GROUND_TRUTH_LABEL_ORDER = [
    "clear_disapproval",
    "unclear_disapproval",
    "unclear_approval",
    "clear_approval",
    "crowd_chorus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on strong-labeled validation files and save overlay plots")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (e.g. best_segment_f1.pt)")
    parser.add_argument("--output-dir", required=True, help="Directory to save inference plots")
    parser.add_argument("--threshold", type=float, default=None, help="Override event detection probability threshold")
    parser.add_argument(
        "--attribute-threshold",
        type=float,
        default=None,
        help="Override approval/disapproval probability threshold",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size for inference")
    parser.add_argument("--run-id", default=None, help="Optional W&B run id for inference logging")
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=("online", "offline", "disabled"),
        help="Optional W&B mode for inference logging",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_val_loader(config: dict[str, Any], records) -> DataLoader:
    loader_cfg = dict(config["val"])
    if loader_cfg.get("batch_size") is None:
        loader_cfg["batch_size"] = 4
    dataset = WeakChunkDataset(
        records,
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
    )


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("W&B logging requested for inference, but `wandb` is not installed.") from exc
    return wandb


def init_wandb(config: dict[str, Any], output_dir: Path, *, run_id: str | None, wandb_mode: str | None):
    wandb_config = config.get("wandb", {})
    enabled = bool(wandb_config.get("enabled", False)) or (wandb_mode is not None and wandb_mode != "disabled") or (run_id is not None)
    if not enabled:
        return None

    resolved_mode = wandb_mode if wandb_mode is not None else wandb_config.get("mode")
    if resolved_mode:
        os.environ["WANDB_MODE"] = str(resolved_mode)
    if wandb_config.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_config["project"]))
    if wandb_config.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(wandb_config["entity"]))

    wandb = _import_wandb()
    return wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        id=run_id,
        name=run_id if run_id is not None else None,
        tags=(wandb_config.get("tags") or []) + ["inference"],
        notes=wandb_config.get("notes"),
        dir=str(output_dir),
        config=config,
    )


def get_wandb_module_if_needed(wandb_run):
    if wandb_run is None:
        return None
    return _import_wandb()


def sanitize_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")


def truncate_plot_title(value: str, limit: int = 10) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def format_seconds_mmss(value: float, _pos: float | None = None) -> str:
    total_seconds = max(0, int(round(float(value))))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


def regions_from_annotations(annotations: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float, str]]:
    export_label_map = {
        "clear_approval": "approval",
        "unclear_approval": "approval",
        "clear_disapproval": "disapproval",
        "unclear_disapproval": "disapproval",
    }
    regions: list[tuple[float, float, str]] = []
    for label in GROUND_TRUTH_LABEL_ORDER:
        export_label = export_label_map.get(label)
        if export_label is None:
            continue
        for onset_sec, offset_sec in annotations.get(label, []):
            regions.append((float(onset_sec), float(offset_sec), export_label))

    return sorted(regions, key=lambda item: (item[0], item[1], item[2]))


def predicted_regions_from_probs(
    predicted_probs: np.ndarray,
    *,
    label_names: list[str],
    event_threshold: float,
    attribute_threshold: float,
    instance_sec: float,
    export_labels: tuple[str, ...] = ("approval", "disapproval"),
) -> list[tuple[float, float, str]]:
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
    regions: list[tuple[float, float, str]] = []
    export_label_set = set(export_labels)
    for class_index, label in enumerate(label_names):
        if label not in export_label_set:
            continue
        for onset_sec, offset_sec in contiguous_regions(predicted_binary[:, class_index], instance_sec=instance_sec):
            regions.append((float(onset_sec), float(offset_sec), label))
    return sorted(regions, key=lambda item: (item[0], item[1], item[2]))


def write_sonic_visualiser_regions(output_path: Path, regions: list[tuple[float, float, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for onset_sec, offset_sec, label in sorted(regions, key=lambda item: (item[0], item[1], item[2])):
            duration_sec = max(0.0, float(offset_sec) - float(onset_sec))
            writer.writerow([f"{float(onset_sec):.6f}", f"{duration_sec:.6f}", label])


def prediction_diagnostics_row(
    *,
    speech_id: str,
    predicted_probs: np.ndarray,
    label_names: list[str],
    instance_sec: float,
    event_threshold: float,
) -> dict[str, str]:
    row: dict[str, str] = {
        "speech_id": speech_id,
        "num_bins": str(int(predicted_probs.shape[0])),
        "instance_sec": f"{float(instance_sec):.6f}",
    }
    if "relevant_event" not in label_names:
        return row

    event_probs = predicted_probs[:, label_names.index("relevant_event")]
    event_active = event_probs >= float(event_threshold)
    top_event_count = max(1, int(math.ceil(0.1 * event_probs.shape[0]))) if event_probs.size else 0
    top_event_indices = np.argsort(-event_probs)[:top_event_count] if top_event_count else np.array([], dtype=np.int64)

    for label_name in ("relevant_event", "approval", "disapproval"):
        if label_name not in label_names:
            continue
        probs = predicted_probs[:, label_names.index(label_name)]
        max_index = int(np.argmax(probs)) if probs.size else 0
        row[f"{label_name}_max"] = f"{float(np.max(probs)):.6f}" if probs.size else ""
        row[f"{label_name}_mean"] = f"{float(np.mean(probs)):.6f}" if probs.size else ""
        row[f"{label_name}_argmax_sec"] = f"{(max_index + 0.5) * float(instance_sec):.6f}" if probs.size else ""
        if label_name == "relevant_event":
            row["relevant_event_active_bins"] = str(int(event_active.sum()))
            continue
        row[f"{label_name}_mean_when_event_active"] = (
            f"{float(np.mean(probs[event_active])):.6f}" if event_active.any() else ""
        )
        row[f"{label_name}_max_when_event_active"] = (
            f"{float(np.max(probs[event_active])):.6f}" if event_active.any() else ""
        )
        row[f"{label_name}_mean_top_event_decile"] = (
            f"{float(np.mean(probs[top_event_indices])):.6f}" if top_event_indices.size else ""
        )
        if probs.size > 1 and float(np.std(event_probs)) > 0.0 and float(np.std(probs)) > 0.0:
            row[f"{label_name}_event_corr"] = f"{float(np.corrcoef(event_probs, probs)[0, 1]):.6f}"
        else:
            row[f"{label_name}_event_corr"] = ""

    return row


def write_prediction_diagnostics(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "speech_id",
        "num_bins",
        "instance_sec",
        "relevant_event_max",
        "relevant_event_mean",
        "relevant_event_argmax_sec",
        "relevant_event_active_bins",
    ]
    ordered_fieldnames = [field for field in preferred if field in fieldnames]
    ordered_fieldnames += [field for field in fieldnames if field not in preferred]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_attribute_probs_by_event(instance_probs_by_task: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    conditioned = dict(instance_probs_by_task)
    event_probs = conditioned.get("event")
    if event_probs is None:
        return conditioned
    for task_name in ("approval", "disapproval"):
        if task_name in conditioned:
            conditioned[task_name] = event_probs[:, :, :1] * conditioned[task_name]
    return conditioned


def collect_multitask_chunk_predictions(
    model: CrowdReactionModel,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, list[SpeechChunkPrediction]]:
    predictions_by_task = {task_name: [] for task_name in TASK_EXPORT_SPECS}
    with torch.no_grad():
        for batch in dataloader:
            instances = batch["instances"].to(device)
            outputs = model(instances=instances)
            instance_probs_by_task = {
                task_name: torch.sigmoid(task_logits).cpu().numpy()
                for task_name, task_logits in outputs.instance_logits.items()
            }
            instance_probs_by_task = condition_attribute_probs_by_event(instance_probs_by_task)
            for task_name, label_names in TASK_EXPORT_SPECS.items():
                if task_name not in instance_probs_by_task:
                    continue
                task_instance_probs = instance_probs_by_task[task_name]
                for batch_index in range(instances.shape[0]):
                    predictions_by_task[task_name].append(
                        SpeechChunkPrediction(
                            speech_id=batch["speech_id"][batch_index],
                            chunk_start_sec=float(batch["chunk_start_sec"][batch_index].item()),
                            chunk_end_sec=float(batch["chunk_end_sec"][batch_index].item()),
                            instance_probs=task_instance_probs[batch_index, :, : len(label_names)],
                        )
                    )
    return predictions_by_task


def aggregate_multitask_probs(
    chunk_predictions_by_task: dict[str, list[SpeechChunkPrediction]],
    *,
    instance_sec: float,
    speech_durations: dict[str, float],
) -> dict[str, np.ndarray]:
    aggregated_by_task: dict[str, dict[str, np.ndarray]] = {}
    for task_name, label_names in TASK_EXPORT_SPECS.items():
        predictions = chunk_predictions_by_task.get(task_name, [])
        if not predictions:
            continue
        aggregated_by_task[task_name] = aggregate_chunk_predictions(
            predictions,
            num_classes=len(label_names),
            instance_sec=instance_sec,
            speech_durations=speech_durations,
        )

    flattened: dict[str, np.ndarray] = {}
    speech_ids = sorted({speech_id for task_map in aggregated_by_task.values() for speech_id in task_map})
    for speech_id in speech_ids:
        matrices = []
        for task_name, label_names in TASK_EXPORT_SPECS.items():
            task_map = aggregated_by_task.get(task_name)
            if task_map is None or speech_id not in task_map:
                continue
            matrices.append(task_map[speech_id][:, : len(label_names)])
        if matrices:
            flattened[speech_id] = np.concatenate(matrices, axis=1)
    return flattened


def active_label_order(chunk_predictions_by_task: dict[str, list[SpeechChunkPrediction]]) -> list[str]:
    enabled = set()
    for task_name, names in TASK_EXPORT_SPECS.items():
        if chunk_predictions_by_task.get(task_name):
            enabled.update(names)
    return [label_name for label_name in PLOTTED_LABEL_ORDER if label_name in enabled]


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> CrowdReactionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    checkpoint_model_config = checkpoint.get("config", {}).get("model", {})
    if "wav2vec2_layer_indices" in checkpoint_model_config or any("scalar_mix" in key for key in state_dict):
        raise RuntimeError(
            "This checkpoint uses the obsolete wav2vec2 scalar layer fusion and is "
            "incompatible with single-layer ablation models."
        )
    layer_index = int(checkpoint_model_config.get("wav2vec2_layer_index", config["model"].get("wav2vec2_layer_index", 3)))
    model = CrowdReactionModel(
        encoder_type=config["model"].get("encoder_type", "beats"),
        beats_checkpoint_path=config["model"].get("beats_checkpoint_path"),
        wav2vec2_model_name=config["model"].get("wav2vec2_model_name", "facebook/wav2vec2-base"),
        wav2vec2_layer_index=layer_index,
        head_hidden_dim=int(config["model"].get("head_hidden_dim", 256)),
        head_dropout=float(config["model"].get("head_dropout", 0.1)),
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
        tasks_config=config.get("tasks"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_spectrogram(waveform: torch.Tensor, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Bias the spectrogram toward better time detail so short reactions are easier to inspect.
    n_fft = 512
    hop_length = 128
    win_length = 512
    window = torch.hann_window(win_length)
    spectrogram = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    magnitude = spectrogram.abs().clamp_min(1e-8)
    db = 20.0 * torch.log10(magnitude)
    times = np.arange(db.shape[1]) * hop_length / sample_rate
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return db.numpy(), times, freqs


def build_strong_text_index(strong_labels_dir: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(Path(strong_labels_dir).glob("*.txt")):
        key = normalize_name(path.name)
        resolved_path = path.resolve()
        if key in index and index[key] != resolved_path:
            raise ValueError(
                "Ambiguous strong label filename normalization for "
                f"{index[key].name} and {path.name}"
            )
        index[key] = resolved_path
    return index


def parse_raw_strong_annotations(strong_txt_path: str) -> dict[str, list[tuple[float, float]]]:
    annotations: dict[str, list[tuple[float, float]]] = {}
    with open(strong_txt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            onset_sec, offset_sec, label = stripped.split("\t")
            label = label.strip()
            if label == "hard_annotation":
                continue
            annotations.setdefault(label, []).append((float(onset_sec), float(offset_sec)))
    return annotations


def draw_intervals(
    ax: plt.Axes,
    intervals: list[tuple[float, float]],
    *,
    y_min: float,
    y_max: float,
    color: str,
    label: str | None,
    alpha: float = 0.9,
) -> None:
    label_used = False
    for onset_sec, offset_sec in intervals:
        ax.axvspan(
            onset_sec,
            offset_sec,
            ymin=y_min,
            ymax=y_max,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label if not label_used else None,
        )
        label_used = True


def overlay_events(
    ax: plt.Axes,
    *,
    predicted_regions: list[tuple[float, float, str]],
    ground_truth_annotations: dict[str, list[tuple[float, float]]],
    instance_sec: float,
) -> None:
    band_height = 0.06
    gap = 0.01
    base = 0.99
    num_gt_bands = len([label for label in GROUND_TRUTH_LABEL_ORDER if label in ground_truth_annotations])

    current_band = 0
    for label in GROUND_TRUTH_LABEL_ORDER:
        gt_intervals = ground_truth_annotations.get(label)
        if not gt_intervals:
            continue
        top = base - current_band * (band_height + gap)
        gt_bottom = max(0.0, top - band_height)
        draw_intervals(
            ax,
            gt_intervals,
            y_min=gt_bottom,
            y_max=top,
            color=GROUND_TRUTH_LABEL_COLORS.get(label, "#1f77b4"),
            label=f"GT {label}",
        )
        current_band += 1

    for label in ("relevant_event", "approval", "disapproval"):
        pred_intervals = [(onset_sec, offset_sec) for onset_sec, offset_sec, region_label in predicted_regions if region_label == label]
        if not pred_intervals:
            continue
        top = base - current_band * (band_height + gap)
        pred_bottom = max(0.0, top - band_height)
        draw_intervals(
            ax,
            pred_intervals,
            y_min=pred_bottom,
            y_max=top,
            color=PREDICTION_COLORS[label],
            label=f"Pred {label}",
            alpha=0.35,
        )
        current_band += 1


def plot_speech(
    *,
    audio_path: str,
    speech_id: str,
    predicted_probs: np.ndarray,
    predicted_regions: list[tuple[float, float, str]],
    label_names: list[str],
    ground_truth_annotations: dict[str, list[tuple[float, float]]],
    sample_rate: int,
    instance_sec: float,
    event_threshold: float,
    attribute_threshold: float,
    output_path: Path,
    plot_score_functions: bool = True,
    score_axis_max: float = 1.0,
    score_alpha: float = 0.95,
) -> None:
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]

    spectrogram_db, times, freqs = compute_spectrogram(waveform, sample_rate)
    fig, ax = plt.subplots(figsize=(14, 5))
    extent = [0.0, times[-1] if len(times) else waveform.numel() / sample_rate, 0.0, freqs[-1] if len(freqs) else sample_rate / 2.0]
    image = ax.imshow(
        spectrogram_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        alpha=0.55,
    )
    fig.colorbar(image, ax=ax, format="%+2.0f dB")

    overlay_events(
        ax,
        predicted_regions=predicted_regions,
        ground_truth_annotations=ground_truth_annotations,
        instance_sec=instance_sec,
    )

    title_parts = [f"{truncate_plot_title(speech_id)}"]
    if "relevant_event" in label_names:
        title_parts.append(f"event={event_threshold:.2f}")
    if any(label in label_names for label in ("approval", "disapproval")):
        title_parts.append(f"attr={attribute_threshold:.2f}")
    ax.set_title(" | ".join(title_parts))
    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Frequency (Hz)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(FuncFormatter(format_seconds_mmss))
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b: list[Any] = []
    labels_b: list[str] = []
    if plot_score_functions:
        score_ax = ax.twinx()
        num_steps = predicted_probs.shape[0]
        centers = (np.arange(num_steps, dtype=np.float32) + 0.5) * float(instance_sec)
        for label_name in ("relevant_event", "approval", "disapproval"):
            if label_name not in label_names:
                continue
            class_index = label_names.index(label_name)
            linewidth = 1.75 if label_name == "relevant_event" else 1.25
            score_ax.plot(
                centers,
                predicted_probs[:, class_index],
                color=SCORE_LINE_COLORS[label_name],
                linewidth=linewidth,
                alpha=score_alpha,
                drawstyle="steps-mid",
                label=f"Score {label_name}",
            )
        if "relevant_event" in label_names:
            score_ax.axhline(
                event_threshold,
                color=SCORE_LINE_COLORS["relevant_event"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                label="Event threshold",
            )
        if any(label in label_names for label in ("approval", "disapproval")):
            score_ax.axhline(
                attribute_threshold,
                color="#444444",
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
                label="Approval/disapproval threshold",
            )
        score_ax.set_ylim(0.0, float(score_axis_max))
        score_ax.set_ylabel("Predicted probability")
        score_ax.grid(False)
        score_ax.xaxis.set_major_formatter(FuncFormatter(format_seconds_mmss))
        handles_b, labels_b = score_ax.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size is not None:
        config["val"]["batch_size"] = int(args.batch_size)

    event_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config["val"].get("event_threshold", sigmoid(DEFAULT_EVENT_LOGIT_THRESHOLD)))
    )
    attribute_threshold = (
        float(args.attribute_threshold)
        if args.attribute_threshold is not None
        else float(config["val"].get("attribute_threshold", config["val"].get("threshold", 0.5)))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb(config, output_dir, run_id=args.run_id, wandb_mode=args.wandb_mode)
    wandb_module = get_wandb_module_if_needed(wandb_run)

    split_data = build_strong_validation_split(
        strong_labels_dir=config["data"]["strong_labels_dir"],
        original_audio_dir=config["data"]["original_audio_dir"],
        chunk_sec=float(config["data"]["chunk_sec"]),
    )
    strong_text_index = build_strong_text_index(config["data"]["strong_labels_dir"])
    val_loader = build_val_loader(config, split_data.val_records)
    model = load_model(config, args.checkpoint, device)

    chunk_predictions_by_task = collect_multitask_chunk_predictions(model, val_loader, device=device)
    label_names = active_label_order(chunk_predictions_by_task)

    record_by_speech = {}
    for record in split_data.val_records:
        record_by_speech[record.speech_id] = record

    speech_durations = speech_durations_from_records(split_data.val_records)
    for event in split_data.strong_events:
        speech_durations[event.speech_id] = max(speech_durations.get(event.speech_id, 0.0), float(event.offset_sec))

    aggregated_probs = aggregate_multitask_probs(
        chunk_predictions_by_task,
        instance_sec=float(config["data"]["instance_sec"]),
        speech_durations=speech_durations,
    )

    diagnostics_rows: list[dict[str, str]] = []
    for speech_id in sorted(aggregated_probs):
        if speech_id not in record_by_speech:
            continue
        record = record_by_speech[speech_id]
        audio_info = torchaudio.info(record.audio_path)
        audio_duration = float(audio_info.num_frames) / float(audio_info.sample_rate)
        num_bins = max(
            int(math.ceil(audio_duration / float(config["data"]["instance_sec"]))),
            aggregated_probs[speech_id].shape[0],
        )

        pred_pad = np.zeros((num_bins, len(label_names)), dtype=np.float32)
        pred_pad[: aggregated_probs[speech_id].shape[0]] = aggregated_probs[speech_id]
        strong_txt_path = strong_text_index.get(normalize_name(record.audio_path))
        ground_truth_annotations = parse_raw_strong_annotations(str(strong_txt_path)) if strong_txt_path is not None else {}
        predicted_regions = predicted_regions_from_probs(
            pred_pad,
            label_names=label_names,
            event_threshold=event_threshold,
            attribute_threshold=attribute_threshold,
            instance_sec=float(config["data"]["instance_sec"]),
        )
        ground_truth_regions = regions_from_annotations(ground_truth_annotations)

        stem = sanitize_stem(speech_id)
        output_path = output_dir / f"{stem}.png"
        predicted_annotation_path = output_dir / f"{stem}.predicted.csv"
        ground_truth_annotation_path = output_dir / f"{stem}.ground_truth.csv"
        plot_speech(
            audio_path=record.audio_path,
            speech_id=speech_id,
            predicted_probs=pred_pad,
            predicted_regions=predicted_regions,
            label_names=label_names,
            ground_truth_annotations=ground_truth_annotations,
            sample_rate=int(config["data"]["sample_rate"]),
            instance_sec=float(config["data"]["instance_sec"]),
            event_threshold=event_threshold,
            attribute_threshold=attribute_threshold,
            output_path=output_path,
        )
        write_sonic_visualiser_regions(predicted_annotation_path, predicted_regions)
        write_sonic_visualiser_regions(ground_truth_annotation_path, ground_truth_regions)
        diagnostics_rows.append(
            prediction_diagnostics_row(
                speech_id=speech_id,
                predicted_probs=pred_pad,
                label_names=label_names,
                instance_sec=float(config["data"]["instance_sec"]),
                event_threshold=event_threshold,
            )
        )
        print(f"Saved {output_path}")
        print(f"Saved {predicted_annotation_path}")
        print(f"Saved {ground_truth_annotation_path}")
        if wandb_run is not None and wandb_module is not None:
            wandb_run.log({f"inference/{speech_id}": wandb_module.Image(str(output_path))})

    diagnostics_path = output_dir / "prediction_diagnostics.csv"
    write_prediction_diagnostics(diagnostics_path, diagnostics_rows)
    if diagnostics_rows:
        print(f"Saved {diagnostics_path}")

    num_saved_plots = len(list(output_dir.glob("*.png")))
    print(f"Total plots saved in {output_dir}: {num_saved_plots}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
