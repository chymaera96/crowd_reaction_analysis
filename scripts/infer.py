#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from crowd_reaction.data import WeakChunkDataset, build_split_records, collate_batch, normalize_name, speech_durations_from_records
from crowd_reaction.eval import aggregate_chunk_predictions, collect_strong_predictions, contiguous_regions
from crowd_reaction.model import CrowdReactionModel


CLASS_NAMES = {
    0: "crowd",
}
PREDICTION_COLORS = {
    0: "#ff9896",
}
SCORE_LINE_COLORS = {
    0: "#ff4d4d",
}
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
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (e.g. best_val.pt)")
    parser.add_argument("--output-dir", required=True, help="Directory to save inference plots")
    parser.add_argument("--threshold", type=float, default=None, help="Override detection threshold")
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
        num_classes=int(config["data"]["num_classes"]),
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


def regions_from_annotations(annotations: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float, str]]:
    regions: list[tuple[float, float, str]] = []
    for label in GROUND_TRUTH_LABEL_ORDER:
        for onset_sec, offset_sec in annotations.get(label, []):
            regions.append((float(onset_sec), float(offset_sec), label))

    seen_labels = set(GROUND_TRUTH_LABEL_ORDER)
    for label in sorted(annotations):
        if label in seen_labels:
            continue
        for onset_sec, offset_sec in annotations[label]:
            regions.append((float(onset_sec), float(offset_sec), label))
    return regions


def predicted_regions_from_probs(
    predicted_probs: np.ndarray,
    *,
    threshold: float,
    instance_sec: float,
) -> list[tuple[float, float, str]]:
    predicted_binary = (predicted_probs >= threshold).astype(np.int64)
    regions: list[tuple[float, float, str]] = []
    for class_index in range(predicted_binary.shape[1]):
        label = CLASS_NAMES[class_index]
        for onset_sec, offset_sec in contiguous_regions(predicted_binary[:, class_index], instance_sec=instance_sec):
            regions.append((float(onset_sec), float(offset_sec), label))
    return regions


def write_sonic_visualiser_regions(output_path: Path, regions: list[tuple[float, float, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for onset_sec, offset_sec, label in sorted(regions, key=lambda item: (item[0], item[1], item[2])):
            duration_sec = max(0.0, float(offset_sec) - float(onset_sec))
            handle.write(f"{float(onset_sec):.6f}\t{duration_sec:.6f}\t{label}\n")


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> CrowdReactionModel:
    model = CrowdReactionModel(
        num_classes=int(config["model"]["num_classes"]),
        beats_checkpoint_path=config["model"]["beats_checkpoint_path"],
        head_hidden_dim=int(config["model"].get("head_hidden_dim", 256)),
        head_dropout=float(config["model"].get("head_dropout", 0.1)),
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
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
    return {
        normalize_name(path.name): path.resolve()
        for path in sorted(Path(strong_labels_dir).glob("noise_*.txt"))
    }


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
) -> None:
    label_used = False
    for onset_sec, offset_sec in intervals:
        ax.axvspan(
            onset_sec,
            offset_sec,
            ymin=y_min,
            ymax=y_max,
            color=color,
            alpha=0.9,
            linewidth=0,
            label=label if not label_used else None,
        )
        label_used = True


def overlay_events(
    ax: plt.Axes,
    *,
    predicted_bins: np.ndarray,
    ground_truth_annotations: dict[str, list[tuple[float, float]]],
    instance_sec: float,
) -> None:
    band_height = 0.06
    gap = 0.01
    base = 0.99
    num_gt_bands = len([label for label in GROUND_TRUTH_LABEL_ORDER if label in ground_truth_annotations])
    total_bands = num_gt_bands + predicted_bins.shape[1]

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

    for class_index in range(predicted_bins.shape[1]):
        top = base - current_band * (band_height + gap)
        pred_bottom = max(0.0, top - band_height)
        pred_intervals = contiguous_regions(predicted_bins[:, class_index].astype(np.int64), instance_sec=instance_sec)
        draw_intervals(
            ax,
            pred_intervals,
            y_min=pred_bottom,
            y_max=top,
            color=PREDICTION_COLORS[class_index],
            label=f"Pred {CLASS_NAMES[class_index]}",
        )
        current_band += 1


def plot_speech(
    *,
    audio_path: str,
    speech_id: str,
    predicted_probs: np.ndarray,
    ground_truth_annotations: dict[str, list[tuple[float, float]]],
    sample_rate: int,
    instance_sec: float,
    threshold: float,
    output_path: Path,
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

    predicted_binary = (predicted_probs >= threshold).astype(np.int64)
    overlay_events(
        ax,
        predicted_bins=predicted_binary,
        ground_truth_annotations=ground_truth_annotations,
        instance_sec=instance_sec,
    )

    score_ax = ax.twinx()
    num_steps = predicted_probs.shape[0]
    centers = (np.arange(num_steps, dtype=np.float32) + 0.5) * float(instance_sec)
    for class_index in range(predicted_probs.shape[1]):
        score_ax.plot(
            centers,
            predicted_probs[:, class_index],
            color=SCORE_LINE_COLORS[class_index],
            linewidth=1.75,
            alpha=0.95,
            drawstyle="steps-mid",
            label=f"Score {CLASS_NAMES[class_index]}",
        )
    score_ax.set_ylim(0.0, 1.0)
    score_ax.set_ylabel("Predicted probability")
    score_ax.grid(False)

    ax.set_title(f"{truncate_plot_title(speech_id)} | threshold={threshold:.2f}")
    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Frequency (Hz)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(FuncFormatter(format_seconds_mmss))
    score_ax.xaxis.set_major_formatter(FuncFormatter(format_seconds_mmss))
    handles_a, labels_a = ax.get_legend_handles_labels()
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

    threshold = float(args.threshold) if args.threshold is not None else float(config["val"].get("threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb(config, output_dir, run_id=args.run_id, wandb_mode=args.wandb_mode)
    wandb_module = get_wandb_module_if_needed(wandb_run)

    split_data = build_split_records(
        audios_info_csv=config["data"]["audios_info_csv"],
        weak_labels_csv=config["data"]["weak_labels_csv"],
        strong_labels_dir=config["data"]["strong_labels_dir"],
        original_audio_dir=config["data"]["original_audio_dir"],
    )
    strong_text_index = build_strong_text_index(config["data"]["strong_labels_dir"])
    val_loader = build_val_loader(config, split_data.val_records)
    model = load_model(config, args.checkpoint, device)

    _, _, chunk_predictions = collect_strong_predictions(model, val_loader, device=device)

    record_by_speech = {}
    for record in split_data.val_records:
        record_by_speech[record.speech_id] = record

    speech_durations = speech_durations_from_records(split_data.val_records)
    for event in split_data.strong_events:
        speech_durations[event.speech_id] = max(speech_durations.get(event.speech_id, 0.0), float(event.offset_sec))

    aggregated_probs = aggregate_chunk_predictions(
        chunk_predictions,
        num_classes=int(config["model"]["num_classes"]),
        instance_sec=float(config["data"]["instance_sec"]),
        speech_durations=speech_durations,
    )

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

        pred_pad = np.zeros((num_bins, int(config["model"]["num_classes"])), dtype=np.float32)
        pred_pad[: aggregated_probs[speech_id].shape[0]] = aggregated_probs[speech_id]
        strong_txt_path = strong_text_index.get(normalize_name(record.audio_path))
        ground_truth_annotations = parse_raw_strong_annotations(str(strong_txt_path)) if strong_txt_path is not None else {}
        predicted_regions = predicted_regions_from_probs(
            pred_pad,
            threshold=threshold,
            instance_sec=float(config["data"]["instance_sec"]),
        )
        ground_truth_regions = regions_from_annotations(ground_truth_annotations)

        stem = sanitize_stem(speech_id)
        output_path = output_dir / f"{stem}.png"
        predicted_annotation_path = output_dir / f"{stem}.predicted.tsv"
        ground_truth_annotation_path = output_dir / f"{stem}.ground_truth.tsv"
        plot_speech(
            audio_path=record.audio_path,
            speech_id=speech_id,
            predicted_probs=pred_pad,
            ground_truth_annotations=ground_truth_annotations,
            sample_rate=int(config["data"]["sample_rate"]),
            instance_sec=float(config["data"]["instance_sec"]),
            threshold=threshold,
            output_path=output_path,
        )
        write_sonic_visualiser_regions(predicted_annotation_path, predicted_regions)
        write_sonic_visualiser_regions(ground_truth_annotation_path, ground_truth_regions)
        print(f"Saved {output_path}")
        print(f"Saved {predicted_annotation_path}")
        print(f"Saved {ground_truth_annotation_path}")
        if wandb_run is not None and wandb_module is not None:
            wandb_run.log({f"inference/{speech_id}": wandb_module.Image(str(output_path))})

    num_saved_plots = len(list(output_dir.glob("*.png")))
    print(f"Total plots saved in {output_dir}: {num_saved_plots}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
