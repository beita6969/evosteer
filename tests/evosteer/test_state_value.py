import pytest

from skillev.policy.state_value import StateValueHeads


def test_heads_initialize_zero_flow_residual_and_frozen_encoding():
    torch = pytest.importorskip("torch")
    heads = StateValueHeads(encoding_dim=5, num_task_types=3)
    features = torch.randn(4, 30, requires_grad=True)
    encoding = torch.randn(4, 5, requires_grad=True)
    residual = heads.flow_residual(encoding, features)
    assert torch.equal(residual, torch.zeros(4))
    (residual - 1).square().mean().backward()
    assert encoding.grad is None
    assert features.grad is None
    assert heads.residual_head[-1].bias.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in heads.runtime_head.parameters())
    assert all(parameter.grad is None for parameter in heads.outcome_heads.parameters())


def test_runtime_value_is_bounded_independent_of_encoder_and_has_supervised_gradients():
    torch = pytest.importorskip("torch")
    heads = StateValueHeads(encoding_dim=5, num_task_types=3)
    features = torch.randn(4, 30)
    types = torch.tensor([0, 1, 2, 0])
    rewards = torch.tensor([0.0, 0.25, 1.0, 0.5], requires_grad=True)
    predictions = heads.runtime_value(features, types)
    assert predictions.shape == (4,)
    assert bool(((predictions > 0) & (predictions < 1)).all())
    heads.runtime_loss(features, types, rewards).backward()
    assert rewards.grad is None
    assert any(parameter.grad is not None for parameter in heads.runtime_head.parameters())
    assert all(parameter.grad is None for parameter in heads.residual_head.parameters())
    assert all(parameter.grad is None for parameter in heads.outcome_heads.parameters())


def test_two_outcome_heads_are_independent_diagnostics_with_detached_inputs_and_targets():
    torch = pytest.importorskip("torch")
    heads = StateValueHeads(encoding_dim=5, num_task_types=3)
    features = torch.randn(4, 30, requires_grad=True)
    encoding = torch.randn(4, 5, requires_grad=True)
    rewards = torch.tensor([0.0, 0.25, 1.0, 0.5], requires_grad=True)
    predictions = heads.outcome_values(encoding, features)
    assert predictions.shape == (4, 2)
    assert bool(((predictions > 0) & (predictions < 1)).all())
    assert not torch.equal(predictions[:, 0], predictions[:, 1])
    first_parameters = {id(p) for p in heads.outcome_heads[0].parameters()}
    assert not first_parameters.intersection(id(p) for p in heads.outcome_heads[1].parameters())
    losses = heads.outcome_losses(encoding, features, rewards)
    torch.testing.assert_close(losses, (predictions - rewards.detach()[:, None]).square().mean(0))
    losses.mean().backward()
    assert rewards.grad is None
    assert encoding.grad is None
    assert features.grad is None
    assert all(parameter.grad is None for parameter in heads.runtime_head.parameters())
    assert all(parameter.grad is None for parameter in heads.residual_head.parameters())
    for head in heads.outcome_heads:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in head.parameters()
        )


def test_head_state_dict_roundtrip_and_invalid_inputs():
    torch = pytest.importorskip("torch")
    heads = StateValueHeads(encoding_dim=5, num_task_types=3)
    features = torch.randn(2, 30)
    types = torch.tensor([0, 2])
    restored = StateValueHeads(encoding_dim=5, num_task_types=3)
    restored.load_state_dict(heads.state_dict())
    assert torch.equal(
        restored.runtime_value(features, types), heads.runtime_value(features, types)
    )
    encoding = torch.randn(2, 5)
    assert torch.equal(
        restored.outcome_values(encoding, features), heads.outcome_values(encoding, features)
    )
    with pytest.raises(ValueError):
        heads.runtime_value(features, torch.tensor([0, 3]))
    with pytest.raises(ValueError):
        heads.runtime_value(torch.zeros(2, 31), types)
    with pytest.raises(ValueError):
        heads.runtime_loss(features, types, torch.tensor([0.0, float("nan")]))
    with pytest.raises(ValueError):
        heads.runtime_value(torch.zeros(0, 30), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError):
        heads.outcome_losses(encoding, features, torch.tensor([0.0, 1.5]))
    with pytest.raises(ValueError):
        heads.outcome_losses(encoding, features, torch.zeros(2, 1))
