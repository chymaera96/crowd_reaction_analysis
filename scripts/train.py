#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crowd_reaction.data import WeakChunkDataset, build_split_records, collate_batch, speech_durations_from_records
from crowd_reaction.eval import collect_strong_predictions, evaluate_multitask_weak, evaluate_strong
from crowd_reaction.model import CrowdReactionModel, mmm_bag_loss, mmm_bag_loss_from_probs


TASK_SPECS = {
    "event": {"target_key": "event_target", "mask_key": "event_mask", "loss_weight": 1.0},
    "approval": {"target_key": "approval_target", "mask_key": "approval_mask", "loss_weight_key": "lambda_approval"},
    "disapproval": {"target_key": "disapproval_target", "mask_key": "disapproval_mask", "loss_weight_key": "lambda_disapproval"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crowd reaction SED with frozen BEATs and hierarchical MMM MIL loss")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics")
    parser.add_argument("--run-id", default=None, help="Run identifier used for W&B and checkpoint subdirectory naming")
    parser.add_argument(
        "--wav2vec2-layer",
        type=int,
        choices=range(1, 13),
        default=None,
        metavar="N",
        help="Override the wav2vec2 transformer layer (1-12)",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=("online", "offline", "disabled"),
        help="Override W&B mode for this run",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable training and validation progress bars",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.wav2vec2_layer is not None:
        config.setdefault("model", {})["wav2vec2_layer_index"] = int(args.wav2vec2_layer)
    return config


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "W&B logging is enabled in the config, but the `wandb` package is not installed."
        ) from exc
    return wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    data_config: dict[str, Any],
    loader_config: dict[str, Any],
    records,
    shuffle: bool,
    *,
    augmentation_config: dict[str, Any] | None = None,
) -> DataLoader:
    dataset = WeakChunkDataset(
        records,
        sample_rate=int(data_config["sample_rate"]),
        chunk_sec=float(data_config["chunk_sec"]),
        instance_sec=float(data_config["instance_sec"]),
        augmentation_config=augmentation_config,
    )
    return DataLoader(
        dataset,
        batch_size=int(loader_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(loader_config.get("num_workers", 0)),
        collate_fn=collate_batch,
    )


def _task_class_weights(loss_config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for task_name in ("event", "approval", "disapproval"):
        key = f"{task_name}_class_weights"
        value = loss_config.get(key)
        if value is not None:
            weights[task_name] = torch.tensor(value, dtype=torch.float32, device=device)
    return weights


def compute_multitask_loss(
    outputs,
    batch_targets: dict[str, torch.Tensor],
    *,
    loss_config: dict[str, Any],
    task_class_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = None
    loss_values: dict[str, float] = {}
    conditional_attribute_loss = bool(loss_config.get("conditional_attribute_loss", False))
    event_probs = None
    if conditional_attribute_loss and "event" in outputs.instance_logits:
        event_probs = torch.sigmoid(outputs.instance_logits["event"]).detach()

    for task_name, spec in TASK_SPECS.items():
        if task_name not in outputs.instance_logits:
            continue
        target_tensor = batch_targets[spec["target_key"]]
        mask_tensor = batch_targets[spec["mask_key"]]
        if task_name != "event" and conditional_attribute_loss and event_probs is not None:
            task_probs = event_probs * torch.sigmoid(outputs.instance_logits[task_name])
            task_loss = mmm_bag_loss_from_probs(
                task_probs,
                target_tensor,
                class_weights=task_class_weights.get(task_name),
                bag_mask=mask_tensor,
            )
        else:
            task_loss = mmm_bag_loss(
                outputs.instance_logits[task_name],
                target_tensor,
                class_weights=task_class_weights.get(task_name),
                bag_mask=mask_tensor,
            )
        if task_name == "event":
            weighted_loss = task_loss
        else:
            weighted_loss = float(loss_config.get(spec["loss_weight_key"], 0.5)) * task_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        loss_values[f"{task_name}_loss"] = float(task_loss.detach().cpu().item())

    if total_loss is None:
        raise RuntimeError("No task losses were computed; check enabled task configuration")
    loss_values["total_loss"] = float(total_loss.detach().cpu().item())
    return total_loss, loss_values


def evaluate_epoch(
    model: CrowdReactionModel,
    val_loader: DataLoader,
    *,
    strong_events_by_task,
    instance_sec: float,
    threshold: float,
    event_collar_sec: float,
    event_offset_ratio: float,
    device: torch.device,
    show_progress: bool = True,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> dict[str, Any]:
    desc = "Validation"
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs} val"
    elif epoch is not None:
        desc = f"Epoch {epoch} val"
    val_batches = (
        tqdm(val_loader, total=len(val_loader), desc=desc, unit="batch", leave=False)
        if show_progress
        else val_loader
    )
    weak_predictions, chunk_predictions_by_task = collect_strong_predictions(model, val_batches, device=device)
    weak_metrics = evaluate_multitask_weak(weak_predictions, threshold=threshold)

    strong_metrics: dict[str, Any] | None = None
    if any(strong_events_by_task.values()):
        speech_durations = speech_durations_from_records(val_loader.dataset.records)
        strong_metrics = {}
        for task_name in ("event", "approval", "disapproval"):
            task_events = strong_events_by_task.get(task_name, [])
            task_predictions = chunk_predictions_by_task.get(task_name, [])
            if not task_events and not task_predictions:
                continue
            strong_metrics[task_name] = evaluate_strong(
                task_predictions,
                task_events,
                num_classes=1,
                instance_sec=instance_sec,
                speech_durations=speech_durations,
                threshold=threshold,
                event_collar_sec=event_collar_sec,
                event_offset_ratio=event_offset_ratio,
            )
        if "event" in strong_metrics:
            strong_metrics.update(strong_metrics["event"])

    return {
        "weak": weak_metrics,
        "strong": strong_metrics,
    }


def save_checkpoint(
    path: Path,
    *,
    model: CrowdReactionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    return score


def validation_score(metrics: dict[str, Any], metric_name: str) -> float:
    strong = metrics.get("strong")
    if strong is not None:
        score = _finite_float(strong.get(metric_name))
        if score is not None:
            return score
    return float(metrics["weak"]["event"]["macro_f1"])


def polarity_validation_score(metrics: dict[str, Any], metric_name: str) -> float:
    strong = metrics.get("strong")
    if strong is None:
        raise ValueError("Strong validation metrics are required for polarity checkpoint selection")
    scores = []
    for task_name in ("approval", "disapproval"):
        task_metrics = strong.get(task_name)
        if task_metrics is None or metric_name not in task_metrics:
            raise ValueError(f"Missing strong {task_name}.{metric_name} for polarity checkpoint selection")
        score = _finite_float(task_metrics[metric_name])
        if score is None:
            score = 0.0
        scores.append(score)
    return float(sum(scores) / len(scores))


def _add_strong_task_payload(
    payload: dict[str, float | int | None],
    metrics: dict[str, Any],
    *,
    prefix: str,
    use_main_names: bool,
) -> None:
    if use_main_names:
        payload["strong.segment_macro_precision"] = _finite_float(metrics.get("segment_macro_precision"))
        payload["strong.segment_macro_f1"] = _finite_float(metrics.get("segment_macro_f1"))
        payload["strong.event_precision"] = _finite_float(metrics.get("event_precision"))
        payload["strong.event_f1"] = _finite_float(metrics.get("event_f1"))
        return
    payload[f"{prefix}.segment_macro_precision"] = _finite_float(metrics.get("segment_macro_precision"))
    payload[f"{prefix}.segment_macro_f1"] = _finite_float(metrics.get("segment_macro_f1"))
    payload[f"{prefix}.event_precision"] = _finite_float(metrics.get("event_precision"))
    payload[f"{prefix}.event_f1"] = _finite_float(metrics.get("event_f1"))


def wandb_validation_payload(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    payload: dict[str, float | int | None] = {
        "epoch": int(metrics["epoch"]),
        "train.loss": float(metrics["train_loss"]),
        "train.event_loss": float(metrics["train_event_loss"]),
        "train.approval_loss": metrics.get("train_approval_loss"),
        "train.disapproval_loss": metrics.get("train_disapproval_loss"),
    }

    strong = metrics.get("strong")
    if strong is not None:
        _add_strong_task_payload(payload, strong, prefix="strong.relevant_event", use_main_names=True)
        for task_name in ("approval", "disapproval"):
            task_metrics = strong.get(task_name)
            if task_metrics is not None:
                _add_strong_task_payload(payload, task_metrics, prefix=f"strong.{task_name}", use_main_names=False)
        if strong.get("approval") is not None and strong.get("disapproval") is not None:
            payload["strong.polarity.segment_macro_f1"] = polarity_validation_score(metrics, "segment_macro_f1")
            payload["strong.polarity.event_f1"] = polarity_validation_score(metrics, "event_f1")
    return {key: value for key, value in payload.items() if value is not None}


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
    run = wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        id=run_id,
        name=run_id if run_id is not None else wandb_config.get("run_name"),
        tags=wandb_config.get("tags"),
        notes=wandb_config.get("notes"),
        dir=str(output_dir),
        config=config,
    )
    return run


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    output_dir = Path(args.output_dir)
    if args.run_id:
        output_dir = output_dir / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb(config, output_dir, run_id=args.run_id, wandb_mode=args.wandb_mode)

    split_datasets = build_split_records(
        audios_info_csv=config["data"]["audios_info_csv"],
        weak_labels_csv=config["data"]["weak_labels_csv"],
        strong_labels_dir=config["data"]["strong_labels_dir"],
        original_audio_dir=config["data"]["original_audio_dir"],
        negative_data_dir=config["data"].get("negative_data_dir"),
        chunk_sec=float(config["data"]["chunk_sec"]),
        unclear_label_weight=float(config.get("loss", {}).get("unclear_label_weight", 0.5)),
    )

    train_loader = build_dataloader(
        config["data"],
        config["train"],
        split_datasets.train_records,
        shuffle=True,
        augmentation_config=config.get("augmentation"),
    )
    val_loader = build_dataloader(config["data"], config["val"], split_datasets.val_records, shuffle=False)

    model = CrowdReactionModel(
        encoder_type=config["model"].get("encoder_type", "beats"),
        beats_checkpoint_path=config["model"].get("beats_checkpoint_path"),
        wav2vec2_model_name=config["model"].get("wav2vec2_model_name", "facebook/wav2vec2-base"),
        wav2vec2_layer_index=int(config["model"].get("wav2vec2_layer_index", 3)),
        head_hidden_dim=int(config["model"].get("head_hidden_dim", 256)),
        head_dropout=float(config["model"].get("head_dropout", 0.1)),
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_sec=float(config["data"]["chunk_sec"]),
        instance_sec=float(config["data"]["instance_sec"]),
        tasks_config=config.get("tasks"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )

    task_class_weights = _task_class_weights(config.get("loss", {}), device)

    best_polarity_segment_f1 = float("-inf")
    best_polarity_event_f1 = float("-inf")
    best_relevant_event_f1 = float("-inf")
    history = []

    show_progress = not bool(args.no_progress)
    total_epochs = int(config["trainer"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_totals = defaultdict(float)
        batches = 0

        train_batches = (
            tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{total_epochs} train", unit="batch", leave=False)
            if show_progress
            else train_loader
        )
        for batch in train_batches:
            optimizer.zero_grad()
            instances = batch["instances"].to(device)
            batch_targets = {key: value.to(device) for key, value in batch["targets"].items()}
            outputs = model(instances=instances)
            loss, loss_values = compute_multitask_loss(
                outputs,
                batch_targets,
                loss_config=config.get("loss", {}),
                task_class_weights=task_class_weights,
            )
            loss.backward()
            optimizer.step()

            for key, value in loss_values.items():
                running_totals[key] += value
            batches += 1
            if show_progress:
                train_batches.set_postfix(loss=f"{running_totals['total_loss'] / max(batches, 1):.4f}")

        model.eval()
        metrics = evaluate_epoch(
            model,
            val_loader,
            strong_events_by_task=split_datasets.strong_events_by_task,
            instance_sec=float(config["data"]["instance_sec"]),
            threshold=float(config["val"].get("threshold", 0.5)),
            event_collar_sec=float(config["val"].get("event_collar_sec", config["data"]["instance_sec"])),
            event_offset_ratio=float(config["val"].get("event_offset_ratio", 0.2)),
            device=device,
            show_progress=show_progress,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_totals["total_loss"] / max(batches, 1)
        metrics["train_event_loss"] = running_totals["event_loss"] / max(batches, 1)
        if "approval_loss" in running_totals:
            metrics["train_approval_loss"] = running_totals["approval_loss"] / max(batches, 1)
        if "disapproval_loss" in running_totals:
            metrics["train_disapproval_loss"] = running_totals["disapproval_loss"] / max(batches, 1)
        history.append(metrics)

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )

        polarity_segment_f1 = polarity_validation_score(metrics, "segment_macro_f1")
        polarity_event_f1 = polarity_validation_score(metrics, "event_f1")
        relevant_event_f1 = validation_score(metrics, "event_f1")
        if polarity_segment_f1 >= best_polarity_segment_f1:
            best_polarity_segment_f1 = polarity_segment_f1
            save_checkpoint(
                output_dir / "best_polarity_segment_f1.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
        if polarity_event_f1 >= best_polarity_event_f1:
            best_polarity_event_f1 = polarity_event_f1
            save_checkpoint(
                output_dir / "best_polarity_event_f1.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
        if relevant_event_f1 >= best_relevant_event_f1:
            best_relevant_event_f1 = relevant_event_f1
            save_checkpoint(
                output_dir / "best_relevant_event_f1.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
        strong = metrics.get("strong")
        strong_event = None if strong is None else strong.get("event", strong)
        if strong_event is not None and "segment_macro_precision" not in strong_event:
            strong_event = None
        strong_approval = None if strong is None else strong.get("approval")
        strong_disapproval = None if strong is None else strong.get("disapproval")

        print(
            " | ".join(
                [
                    f"epoch {epoch}/{total_epochs}",
                    f"train_loss={metrics['train_loss']:.4f}",
                    f"polarity_segment_f1={polarity_segment_f1:.4f}",
                    f"polarity_event_f1={polarity_event_f1:.4f}",
                    f"relevant_event_f1={relevant_event_f1:.4f}",
                ]
            )
        )
        if wandb_run is not None:
            wandb_run.log(wandb_validation_payload(metrics), step=epoch)

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_polarity_segment_f1"] = best_polarity_segment_f1
        wandb_run.summary["best_polarity_event_f1"] = best_polarity_event_f1
        wandb_run.summary["best_relevant_event_f1"] = best_relevant_event_f1
        for checkpoint_name in (
            "last.pt",
            "best_polarity_segment_f1.pt",
            "best_polarity_event_f1.pt",
            "best_relevant_event_f1.pt",
        ):
            checkpoint_path = output_dir / checkpoint_name
            if checkpoint_path.exists():
                wandb_run.save(str(checkpoint_path), policy="now")
        wandb_run.finish()


if __name__ == "__main__":
    main()
