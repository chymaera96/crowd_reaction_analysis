"""Crowd reaction sound event detection package."""

from .data import SplitDatasets, StrongEvent, WeakChunkDataset, build_split_records
from .eval import evaluate_strong, evaluate_weak
from .model import CrowdReactionModel, DummyFeatureExtractor, mmm_bag_loss

__all__ = [
    "SplitDatasets",
    "CrowdReactionModel",
    "DummyFeatureExtractor",
    "StrongEvent",
    "WeakChunkDataset",
    "build_split_records",
    "evaluate_strong",
    "evaluate_weak",
    "mmm_bag_loss",
]
