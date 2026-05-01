"""Crowd reaction sound event detection package."""

from .data import SplitDatasets, StrongEvent, WeakBagTargets, WeakChunkDataset, build_split_records, weak_row_to_targets
from .eval import evaluate_multitask_weak, evaluate_strong, evaluate_weak
from .model import CrowdReactionModel, DummyFeatureExtractor, FrozenWav2Vec2FeatureExtractor, MultiTaskOutputs, mmm_bag_loss, mmm_bag_loss_from_probs

__all__ = [
    "SplitDatasets",
    "CrowdReactionModel",
    "DummyFeatureExtractor",
    "FrozenWav2Vec2FeatureExtractor",
    "MultiTaskOutputs",
    "StrongEvent",
    "WeakBagTargets",
    "WeakChunkDataset",
    "build_split_records",
    "weak_row_to_targets",
    "evaluate_multitask_weak",
    "evaluate_strong",
    "evaluate_weak",
    "mmm_bag_loss",
    "mmm_bag_loss_from_probs",
]
