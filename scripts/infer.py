#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crowd_reaction.data import WeakChunkDataset, build_split_records, collate_batch, speech_durations_from_records
from crowd_reaction.eval import aggregate_chunk_predictions, collect_strong_predictions, contiguous_regions, strong_events_to_bin_targets
from crowd_reaction.model import CrowdReactionModel


CLASS_NAMES = {
    0: "disapproval",
    1: "approval",
}
GROUND_TRUTH_COLORS = {
    0: "#d62728",
    1: "#2ca02c",
}
PREDICTION_COLORS = {
    0: "#ff9896",
    1: "#98df8a",
}
SCORE_LINE_COLORS = {
    0: "#ff4d4d",
    1: "#32cd32",
}


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


def sanitize_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")


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
    n_fft = 1024
    hop_length = 256
    win_length = 1024
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
            alpha=0.35,
            linewidth=0,
            label=label if not label_used else None,
        )
        label_used = True


def overlay_events(
    ax: plt.Axes,
    *,
    predicted_bins: np.ndarray,
    target_bins: np.ndarray,
    instance_sec: float,
) -> None:
    band_height = 0.06
    gap = 0.01
    base = 0.99
    for class_index in range(target_bins.shape[1]):
        top = base - class_index * 2 * (band_height + gap)
        gt_bottom = max(0.0, top - band_height)
        pred_top = max(0.0, gt_bottom - gap)
        pred_bottom = max(0.0, pred_top - band_height)

        gt_intervals = contiguous_regions(target_bins[:, class_index].astype(np.int64), instance_sec=instance_sec)
        pred_intervals = contiguous_regions(predicted_bins[:, class_index].astype(np.int64), instance_sec=instance_sec)

        draw_intervals(
            ax,
            gt_intervals,
            y_min=gt_bottom,
            y_max=top,
            color=GROUND_TRUTH_COLORS[class_index],
            label=f"GT {CLASS_NAMES[class_index]}",
        )
        draw_intervals(
            ax,
            pred_intervals,
            y_min=pred_bottom,
            y_max=pred_top,
            color=PREDICTION_COLORS[class_index],
            label=f"Pred {CLASS_NAMES[class_index]}",
        )


def plot_speech(
    *,
    audio_path: str,
    speech_id: str,
    predicted_probs: np.ndarray,
    target_bins: np.ndarray,
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
    fig, ax = plt.subplots(figsize=(16, 7))
    extent = [0.0, times[-1] if len(times) else waveform.numel() / sample_rate, 0.0, freqs[-1] if len(freqs) else sample_rate / 2.0]
    image = ax.imshow(
        spectrogram_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    fig.colorbar(image, ax=ax, format="%+2.0f dB")

    predicted_binary = (predicted_probs >= threshold).astype(np.int64)
    overlay_events(
        ax,
        predicted_bins=predicted_binary,
        target_bins=target_bins,
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
            linewidth=2.0,
            alpha=0.95,
            label=f"Score {CLASS_NAMES[class_index]}",
        )
    score_ax.set_ylim(0.0, 1.0)
    score_ax.set_ylabel("Predicted probability")
    score_ax.grid(False)

    ax.set_title(f"{speech_id} | threshold={threshold:.2f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = score_ax.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
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

    split_data = build_split_records(
        audios_info_csv=config["data"]["audios_info_csv"],
        weak_labels_csv=config["data"]["weak_labels_csv"],
        strong_labels_dir=config["data"]["strong_labels_dir"],
        original_audio_dir=config["data"]["original_audio_dir"],
    )
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
    target_bins = strong_events_to_bin_targets(
        split_data.strong_events,
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
            target_bins.get(speech_id, np.zeros((0, int(config["model"]["num_classes"])), dtype=np.int64)).shape[0],
        )

        pred_pad = np.zeros((num_bins, int(config["model"]["num_classes"])), dtype=np.float32)
        pred_pad[: aggregated_probs[speech_id].shape[0]] = aggregated_probs[speech_id]
        tgt_existing = target_bins.get(speech_id)
        tgt_pad = np.zeros((num_bins, int(config["model"]["num_classes"])), dtype=np.int64)
        if tgt_existing is not None:
            tgt_pad[: tgt_existing.shape[0]] = tgt_existing

        output_path = output_dir / f"{sanitize_stem(speech_id)}.png"
        plot_speech(
            audio_path=record.audio_path,
            speech_id=speech_id,
            predicted_probs=pred_pad,
            target_bins=tgt_pad,
            sample_rate=int(config["data"]["sample_rate"]),
            instance_sec=float(config["data"]["instance_sec"]),
            threshold=threshold,
            output_path=output_path,
        )
        print(f"Saved {output_path}")
        if wandb_run is not None:
            wandb_run.log({f"inference/{speech_id}": wandb_run.Image(str(output_path))})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
