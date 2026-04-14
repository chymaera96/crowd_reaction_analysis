#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crowd_reaction.data import WeakChunkDataset, collate_batch, read_strong_events, speech_durations_from_records
from crowd_reaction.eval import collect_strong_predictions, evaluate_strong, evaluate_weak
from crowd_reaction.model import CrowdReactionModel, mmm_bag_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crowd reaction SED with frozen BEATs and MMM MIL loss")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def build_dataloader(config: dict[str, Any], metadata_csv: str, shuffle: bool) -> DataLoader:
    dataset = WeakChunkDataset(
        metadata_csv,
        sample_rate=int(config["sample_rate"]),
        chunk_sec=float(config["chunk_sec"]),
        instance_sec=float(config["instance_sec"]),
        num_classes=int(config["num_classes"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_batch,
    )


def evaluate_epoch(
    model: CrowdReactionModel,
    val_loader: DataLoader,
    *,
    strong_events_path: str | None,
    instance_sec: float,
    num_classes: int,
    threshold: float,
    event_collar_sec: float,
    event_offset_ratio: float,
    device: torch.device,
) -> dict[str, Any]:
    weak_targets, weak_probs, chunk_predictions = collect_strong_predictions(model, val_loader, device=device)
    weak_metrics = evaluate_weak(weak_targets, weak_probs, threshold=threshold)

    strong_metrics: dict[str, Any] | None = None
    if strong_events_path:
        strong_events = read_strong_events(strong_events_path)
        speech_durations = speech_durations_from_records(val_loader.dataset.records)
        strong_metrics = evaluate_strong(
            chunk_predictions,
            strong_events,
            num_classes=num_classes,
            instance_sec=instance_sec,
            speech_durations=speech_durations,
            threshold=threshold,
            event_collar_sec=event_collar_sec,
            event_offset_ratio=event_offset_ratio,
        )

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


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=full_key))
        elif isinstance(value, list):
            continue
        elif value is None:
            continue
        elif isinstance(value, (int, float)):
            flattened[full_key] = value
    return flattened


def init_wandb(config: dict[str, Any], output_dir: Path):
    wandb_config = config.get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return None

    if wandb_config.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_config["mode"])
    if wandb_config.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_config["project"]))
    if wandb_config.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(wandb_config["entity"]))

    wandb = _import_wandb()
    run = wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("run_name"),
        tags=wandb_config.get("tags"),
        notes=wandb_config.get("notes"),
        dir=str(output_dir),
        config=config,
    )
    return run


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb(config, output_dir)

    train_loader = build_dataloader(config["train"], config["train"]["weak_metadata_csv"], shuffle=True)
    val_loader = build_dataloader(config["val"], config["val"]["weak_metadata_csv"], shuffle=False)

    model = CrowdReactionModel(
        num_classes=int(config["model"]["num_classes"]),
        beats_checkpoint_path=config["model"]["beats_checkpoint_path"],
        head_hidden_dim=int(config["model"].get("head_hidden_dim", 256)),
        head_dropout=float(config["model"].get("head_dropout", 0.1)),
        sample_rate=int(config["train"]["sample_rate"]),
        chunk_sec=float(config["train"]["chunk_sec"]),
        instance_sec=float(config["train"]["instance_sec"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )

    class_weights = config["loss"].get("class_weights")
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    best_weak = float("-inf")
    best_strong = float("-inf")
    history = []

    for epoch in range(1, int(config["trainer"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            instances = batch["instances"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(instances=instances)
            loss = mmm_bag_loss(logits, labels, class_weights=class_weights_tensor)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            batches += 1

        model.eval()
        metrics = evaluate_epoch(
            model,
            val_loader,
            strong_events_path=config["val"].get("strong_metadata_csv"),
            instance_sec=float(config["val"]["instance_sec"]),
            num_classes=int(config["model"]["num_classes"]),
            threshold=float(config["val"].get("threshold", 0.5)),
            event_collar_sec=float(config["val"].get("event_collar_sec", config["val"]["instance_sec"])),
            event_offset_ratio=float(config["val"].get("event_offset_ratio", 0.2)),
            device=device,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(batches, 1)
        history.append(metrics)

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )

        weak_score = float(metrics["weak"]["macro_average_precision"])
        if weak_score >= best_weak:
            best_weak = weak_score
            save_checkpoint(
                output_dir / "best_weak.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )

        strong = metrics.get("strong")
        strong_score = float(strong["segment_macro_f1"]) if strong is not None else float("-inf")
        if strong_score >= best_strong:
            best_strong = strong_score
            save_checkpoint(
                output_dir / "best_strong.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": metrics["train_loss"],
                    "weak_macro_auroc": metrics["weak"]["macro_auroc"],
                    "weak_macro_ap": metrics["weak"]["macro_average_precision"],
                    "weak_macro_f1": metrics["weak"]["macro_f1"],
                    "strong_segment_macro_f1": None if strong is None else strong["segment_macro_f1"],
                    "strong_event_f1": None if strong is None else strong["event_f1"],
                }
            )
        )
        if wandb_run is not None:
            wandb_run.log(flatten_metrics(metrics), step=epoch)

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_weak_macro_average_precision"] = best_weak
        if best_strong != float("-inf"):
            wandb_run.summary["best_strong_segment_macro_f1"] = best_strong
        wandb_run.save(str(output_dir / "history.json"), policy="now")
        for checkpoint_name in ("last.pt", "best_weak.pt", "best_strong.pt"):
            checkpoint_path = output_dir / checkpoint_name
            if checkpoint_path.exists():
                wandb_run.save(str(checkpoint_path), policy="now")
        wandb_run.finish()


if __name__ == "__main__":
    main()
