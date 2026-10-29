from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from crowd_reaction.model import CrowdReactionModel, encoder_configuration, load_checkpoint_model


def _roundtrip(tmp_path, config, encoder_config=None):
    torch.manual_seed(0)
    original = CrowdReactionModel(**config['model'], encoder_config=encoder_config,
                                  **config['data'], tasks_config=config['tasks']).eval()
    checkpoint = tmp_path / 'model.pt'
    torch.save({'model_state_dict': original.state_dict(), 'config': config,
                'encoder_config': encoder_configuration(original)}, checkpoint)
    loaded = load_checkpoint_model(config, str(checkpoint), torch.device('cpu'))
    samples = round(config['data']['sample_rate'] * config['data']['chunk_sec'])
    waveform = torch.randn(1, samples)
    with torch.no_grad():
        expected = original(waveform=waveform)
        actual = loaded(waveform=waveform)
    assert loaded.enabled_tasks == original.enabled_tasks
    for task in expected.instance_logits:
        torch.testing.assert_close(actual.instance_logits[task], expected.instance_logits[task])
    return checkpoint, loaded


def test_cnn_tcn_checkpoint_restores_event_only_heads_and_rejects_wrong_time_bins(tmp_path):
    config = {'model': {'encoder_type': 'cnn_tcn', 'head_hidden_dim': 8},
              'data': {'sample_rate': 16000, 'chunk_sec': 1.28, 'instance_sec': 0.04},
              'tasks': {'approval': {'enabled': False}, 'disapproval': {'enabled': False}}}
    checkpoint, loaded = _roundtrip(tmp_path, config)
    assert list(loaded.heads) == ['event']
    supplied = copy.deepcopy(config)
    supplied['tasks'] = {}
    assert load_checkpoint_model(supplied, str(checkpoint), torch.device('cpu')).enabled_tasks == ('event',)
    supplied['data']['instance_sec'] = 0.5
    with pytest.raises(ValueError, match='instance_sec'):
        load_checkpoint_model(supplied, str(checkpoint), torch.device('cpu'))


def test_beats_classifier_loads_without_external_encoder_file(tmp_path):
    encoder = {'input_patch_size': 16, 'embed_dim': 8, 'encoder_embed_dim': 8,
               'encoder_ffn_embed_dim': 16, 'encoder_layers': 1, 'encoder_attention_heads': 2,
               'conv_pos': 8, 'conv_pos_groups': 2, 'deep_norm': True,
               'relative_position_embedding': True, 'num_buckets': 32, 'max_distance': 64,
               'gru_rel_pos': True, 'dropout': 0., 'attention_dropout': 0., 'encoder_layerdrop': 0.}
    config = {'model': {'encoder_type': 'beats', 'beats_checkpoint_path': str(tmp_path/'missing.pt'),
                         'head_hidden_dim': 8},
              'data': {'sample_rate': 16000, 'chunk_sec': 1., 'instance_sec': 1.}, 'tasks': {}}
    _roundtrip(tmp_path, config, encoder)


def test_wav2vec2_classifier_loads_without_pretrained_download(tmp_path, monkeypatch):
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    def forbidden(*args, **kwargs):
        raise AssertionError('Checkpoint loading must not download pretrained weights')
    monkeypatch.setattr(Wav2Vec2Model, 'from_pretrained', forbidden)
    encoder = Wav2Vec2Config(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,8), conv_stride=(2,2), conv_kernel=(3,3),
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2).to_dict()
    config = {'model': {'encoder_type': 'wav2vec2', 'wav2vec2_model_name': 'unused',
                        'wav2vec2_layer_index': 1, 'head_hidden_dim': 8},
              'data': {'sample_rate': 16000, 'chunk_sec': 0.128, 'instance_sec': 0.064}, 'tasks': {}}
    _roundtrip(tmp_path, config, encoder)
