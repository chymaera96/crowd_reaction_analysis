"""Crowd reaction sound event detection package."""

from .data import StrongEvent, WeakChunkDataset, read_strong_events, read_weak_metadata
from .eval import evaluate_strong, evaluate_weak
from .model import CrowdReactionModel, DummyFeatureExtractor, mmm_bag_loss

__all__ = [
    "CrowdReactionModel",
    "DummyFeatureExtractor",
    "StrongEvent",
    "WeakChunkDataset",
    "evaluate_strong",
    "evaluate_weak",
    "mmm_bag_loss",
    "read_strong_events",
    "read_weak_metadata",
]
