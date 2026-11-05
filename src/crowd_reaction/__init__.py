"""Crowd reaction classifiers and single-recording annotation API."""
from importlib import import_module

__version__ = "0.2.0"
_EXPORTS = {'SplitDatasets': 'data',
 'CrowdReactionModel': 'model',
 'DummyFeatureExtractor': 'model',
 'FrozenWav2Vec2FeatureExtractor': 'model',
 'MultiTaskOutputs': 'model',
 'Wav2Vec2LayerScalarMix': 'model',
 'StrongEvent': 'data',
 'WeakBagTargets': 'data',
 'WeakChunkDataset': 'data',
 'build_split_records': 'data',
 'weak_row_to_targets': 'data',
 'evaluate_multitask_weak': 'eval',
 'evaluate_strong': 'eval',
 'evaluate_weak': 'eval',
 'mmm_bag_loss': 'model',
 'mmm_bag_loss_from_probs': 'model',
 'CrowdDetector': 'api',
 'InferenceResult': 'api',
 'run_audio_inference': 'api',
 'region_mean_score': 'api',
 'write_predicted_segments_csv': 'api',
 'write_scores_json': 'api',
 'plot_inference_result': 'api',
 'get_weights': 'weights'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
