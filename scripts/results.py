#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crowd_reaction.data import WeakChunkDataset, build_split_records, collate_batch, speech_durations_from_records
from crowd_reaction.eval import collect_strong_predictions, evaluate_strong
from crowd_reaction.model import CrowdReactionModel
from torch.utils.data import DataLoader


TASK_LABELS = {
    "event": "relevant_event",
    "approval": "approval",
    "disapproval": "disapproval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute strong validation results for a trained checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", default=None, help="Optional output JSON path; defaults to checkpoint directory/results.json")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_val_loader(config: dict[str, Any], records, *, batch_size: int | None) -> DataLoader:
    loader_cfg = dict(config["val"])
    if batch_size is not None:
        loader_cfg["batch_size"] = int(batch_size)
    dataset = WeakChunkDataset(
        records,
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
    )


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> CrowdReactionModel:
    model = CrowdReactionModel(
        encoder_type=config["model"].get("encoder_type", "beats"),
        beats_checkpoint_path=config["model"].get("beats_checkpoint_path"),
        wav2vec2_model_name=config["model"].get("wav2vec2_model_name", "facebook/wav2vec2-base"),
        head_hidden_dim=int(config["model"].get("head_hidden_dim", 256)),
        head_dropout=float(config["model"].get("head_dropout", 0.1)),
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
        tasks_config=config.get("tasks"),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def thresholds_from_config(config: dict[str, Any]) -> dict[str, float]:
    val_config = config.get("val", {})
    event_threshold = float(val_config.get("event_threshold", val_config.get("threshold", 0.5)))
    attribute_threshold = float(val_config.get("attribute_threshold", val_config.get("threshold", 0.5)))
    return {
        "relevant_event": event_threshold,
        "approval": attribute_threshold,
        "disapproval": attribute_threshold,
    }


def build_results_payload(
    *,
    config_path: str,
    checkpoint_path: str,
    thresholds: dict[str, float],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "thresholds": thresholds,
        "metrics": clean_results_metrics(metrics),
    }


def clean_strong_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "segment": {
            "precision": float(metrics["segment_macro_precision"]),
            "recall": float(metrics["segment_macro_recall"]),
            "f1": float(metrics["segment_macro_f1"]),
        },
        "event": {
            "precision": float(metrics["event_precision"]),
            "recall": float(metrics["event_recall"]),
            "f1": float(metrics["event_f1"]),
        },
    }


def clean_results_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        label_name: clean_strong_metrics(label_metrics)
        for label_name, label_metrics in metrics.items()
    }


def compute_strong_results(
    *,
    model: CrowdReactionModel,
    val_loader: DataLoader,
    strong_events_by_task,
    thresholds: dict[str, float],
    instance_sec: float,
    event_collar_sec: float,
    event_offset_ratio: float,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    _, chunk_predictions_by_task = collect_strong_predictions(model, val_loader, device=device)
    speech_durations = speech_durations_from_records(val_loader.dataset.records)
    for task_events in strong_events_by_task.values():
        for event in task_events:
            speech_durations[event.speech_id] = max(speech_durations.get(event.speech_id, 0.0), float(event.offset_sec))

    metrics: dict[str, dict[str, Any]] = {}
    for task_name, label_name in TASK_LABELS.items():
        task_events = strong_events_by_task.get(task_name, [])
        task_predictions = chunk_predictions_by_task.get(task_name, [])
        if not task_events and not task_predictions:
            continue
        metrics[label_name] = evaluate_strong(
            task_predictions,
            task_events,
            num_classes=1,
            instance_sec=instance_sec,
            speech_durations=speech_durations,
            threshold=float(thresholds[label_name]),
            event_collar_sec=event_collar_sec,
            event_offset_ratio=event_offset_ratio,
        )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output is not None else checkpoint_path.parent / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = build_split_records(
        audios_info_csv=config["data"]["audios_info_csv"],
        weak_labels_csv=config["data"]["weak_labels_csv"],
        strong_labels_dir=config["data"]["strong_labels_dir"],
        original_audio_dir=config["data"]["original_audio_dir"],
        negative_data_dir=config["data"].get("negative_data_dir"),
        chunk_sec=float(config["data"]["chunk_sec"]),
        unclear_label_weight=float(config.get("loss", {}).get("unclear_label_weight", 0.5)),
    )
    val_loader = build_val_loader(config, split_data.val_records, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, str(checkpoint_path), device)
    thresholds = thresholds_from_config(config)
    metrics = compute_strong_results(
        model=model,
        val_loader=val_loader,
        strong_events_by_task=split_data.strong_events_by_task,
        thresholds=thresholds,
        instance_sec=float(config["data"]["instance_sec"]),
        event_collar_sec=float(config["val"].get("event_collar_sec", config["data"]["instance_sec"])),
        event_offset_ratio=float(config["val"].get("event_offset_ratio", 0.2)),
        device=device,
    )
    payload = build_results_payload(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        thresholds=thresholds,
        metrics=metrics,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
