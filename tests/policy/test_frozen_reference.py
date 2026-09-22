import torch

from skillev.policy.interface import AdapterRole
from skillev.policy.reference import QwenFrozenForwardReference
from skillev.scoring.edge_plan import PreparedEdge
from skillev.training.stability import categorical_reference_kl


def test_frozen_reference_preserves_actor_parameters_and_restores(backbone, tmp_path):
    reference = QwenFrozenForwardReference.capture(
        backbone, reference_id="accepted-initial-f@1", maximum_context_tokens=32
    )
    edge = PreparedEdge(1, AdapterRole.FORWARD_POLICY, (4, 5), (6, 7))
    before = {
        name: value.detach().clone()
        for name, value in backbone.named_trainable_parameters().items()
    }
    current, frozen = reference.logits(edge)
    assert current.shape == frozen.shape
    assert categorical_reference_kl(current, frozen).abs().max() < 1e-6
    assert not frozen.requires_grad
    assert all(
        torch.equal(value, before[name])
        for name, value in backbone.named_trainable_parameters().items()
    )
    reference.save(tmp_path / "reference")
    restored = QwenFrozenForwardReference.load(
        backbone, tmp_path / "reference", expected_reference_id="accepted-initial-f@1"
    )
    assert torch.equal(restored.logits(edge)[1], frozen)


def test_frozen_reference_full_logits_backward_only_updates_current_forward(backbone):
    reference = QwenFrozenForwardReference.capture(
        backbone, reference_id="initial@1", maximum_context_tokens=32
    )
    groups = backbone.parameter_groups()
    with torch.no_grad():
        for parameter in groups.forward:
            parameter.add_(0.05)
    current, frozen = reference.logits(PreparedEdge(1, AdapterRole.FORWARD_POLICY, (4, 5), (6, 7)))
    loss = categorical_reference_kl(current, frozen).mean()
    loss.backward()
    assert float(loss) > 0
    assert sum(float(p.grad.abs().sum()) for p in groups.forward if p.grad is not None) > 0
    assert all(p.grad is None for p in (*groups.backward, *groups.z_head))
