from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from crowd_reaction.model import CrowdReactionModel, linear_softmax_pool, linear_softmax_bag_loss
from scripts.train import checkpoint_selection_scores, compute_multitask_loss


@pytest.mark.parametrize('architecture', ['cnn', 'tcn', 'gru'])
def test_cnn_encoders_preserve_time_and_train_the_frontend(architecture):
    model = CrowdReactionModel(encoder_type={'cnn': 'cnn', 'tcn': 'cnn_tcn', 'gru': 'cnn_gru'}[architecture], head_hidden_dim=64,
                              chunk_sec=1.28, instance_sec=0.04)
    output = model(waveform=torch.randn(2, 20480))
    assert all(logits.shape == (2, 32, 1) for logits in output.instance_logits.values())
    assert 50_000 < sum(p.numel() for p in model.parameters()) < 300_000
    sum(logits.square().mean() for logits in output.instance_logits.values()).backward()
    convolution = model.feature_extractor.cnn[0][0]
    assert convolution.weight.requires_grad
    assert convolution.weight.grad is not None and convolution.weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in model.parameters())


def test_linear_softmax_pool_and_masked_loss():
    probabilities = torch.tensor([[[0.1], [0.8]], [[0.3], [0.4]]], requires_grad=True)
    pooled = linear_softmax_pool(probabilities)
    assert pooled[0, 0].item() == pytest.approx(0.65 / 0.9)
    loss = linear_softmax_bag_loss(probabilities, torch.ones(2, 1), bag_mask=torch.tensor([1., 0.]))
    assert loss.item() == pytest.approx(-torch.log(pooled[0, 0]).item())
    loss.backward()
    assert probabilities.grad[0].abs().sum() > 0
    assert probabilities.grad[1].abs().sum() == 0
    assert torch.isfinite(linear_softmax_pool(torch.zeros(2, 3, 1))).all()
    assert linear_softmax_bag_loss(probabilities.detach(), torch.ones(2, 1),
                                   bag_mask=torch.zeros(2)).item() == 0


def test_joint_linear_softmax_loss_and_event_only_selection():
    logits = {task: torch.zeros(2, 10, 1, requires_grad=True)
              for task in ('event', 'approval', 'disapproval')}
    targets = {f'{task}_{kind}': (torch.ones(2, 1) if kind == 'target' else torch.ones(2))
               for task in logits for kind in ('target', 'mask')}
    loss, values = compute_multitask_loss(SimpleNamespace(instance_logits=logits), targets,
        loss_config={'type': 'linear_softmax', 'conditional_attribute_loss': True,
                     'lambda_approval': 0.5, 'lambda_disapproval': 0.5}, task_class_weights={})
    assert set(values) == {'total_loss', 'event_loss', 'approval_loss', 'disapproval_loss'}
    loss.backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in logits.values())
    scores = checkpoint_selection_scores({'strong': {'event_f1': 0.2, 'segment_macro_f1': 0.7},
                                          'weak': {'event': {'macro_average_precision': 0.8}}}, ('event',))
    assert scores == {'best_relevant_event_f1.pt': 0.2, 'best_event_segment_f1.pt': 0.7,
                      'best_weak_event_auprc.pt': 0.8}


def test_event_only_cnn_tcn_model_has_no_polarity_parameters():
    model = CrowdReactionModel(encoder_type='cnn_tcn', chunk_sec=1.28, instance_sec=0.04,
                              head_hidden_dim=64, mil_pooling='linear_softmax',
                              tasks_config={'approval': {'enabled': False}, 'disapproval': {'enabled': False}})
    output = model(waveform=torch.zeros(1, 20480))
    assert list(model.heads) == ['event']
    assert list(output.instance_logits) == ['event']
    assert torch.isfinite(output.bag_probabilities['event']).all()
