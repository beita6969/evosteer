import math

import pytest
import torch
from torch import nn

from skillev.policy.interface import PolicyParameterGroups
from skillev.policy.z_initialization import ZInitializationSpec
from skillev.scoring.ttb_reference import ttb_reference
from skillev.training.config import OptimizerConfig
from skillev.training.stability import (
    PolicyStabilityConfig,
    categorical_reference_kl,
    clip_full_batch_groups,
)


def test_kl_zero_gradient_reference_sensitivity_and_detach():
    logits = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    zero = categorical_reference_kl(logits, reference, vocabulary_chunk_size=1).sum()
    zero.backward()
    assert abs(float(zero)) < 1e-12
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-12)
    assert reference.grad is None
    logits.grad = None
    shifted = torch.tensor([[0.8, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    loss = categorical_reference_kl(logits, shifted).sum()
    loss.backward()
    assert float(loss) > 0
    assert logits.grad.abs().sum() > 0.01
    assert shifted.grad is None
    logits.grad = None
    sample_ratio = logits.log_softmax(-1)[0, 0] - reference.detach().log_softmax(-1)[0, 0]
    sample_ratio.backward()
    assert logits.grad.abs().sum() > 0.01  # upstream counterexample, not KL


def test_masked_kl_shared_support_and_vocab_chunking():
    logits = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    reference = torch.randn_like(logits)
    support = torch.tensor([[True, False, True, False, False, True, False]]).expand_as(logits)
    actual = categorical_reference_kl(logits, reference, support=support, vocabulary_chunk_size=2)
    expected = categorical_reference_kl(logits[:, [0, 2, 5]], reference[:, [0, 2, 5]])
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[~support].any()


def test_z_constant_initialization_and_seeded_resets():
    module = nn.Sequential(nn.Linear(4, 3), nn.GELU(), nn.Linear(3, 1))
    spec = ZInitializationSpec("output-bias-log-epsilon@1", 0.1, 17)
    spec.initialize(module, seed=17)
    state = {k: v.clone() for k, v in module.state_dict().items()}
    assert torch.allclose(module(torch.randn(5, 4)), torch.full((5, 1), math.log(0.1)))
    spec.initialize(module, seed=17)
    assert all(torch.equal(state[k], value) for k, value in module.state_dict().items())
    assert ZInitializationSpec.from_value(spec.to_value()) == spec


def test_ttb_independent_formula_and_global_denominator():
    z = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    f = torch.tensor([-4.0, -2.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-6.0, -1.0], dtype=torch.float64, requires_grad=True)
    oracle = ttb_reference(
        log_z=0.3,
        forward_sums=(-4.0, -2.0),
        backward_sums=(-6.0, -1.0),
        token_counts=(2, 1),
        reward=0.5,
        epsilon=0.1,
        beta=1.2,
        batch_size=28,
    )
    delta = (
        z
        + (f / torch.tensor([2, 1])).sum()
        - 1.2 * math.log(0.6)
        - (b / torch.tensor([2, 1])).sum()
    )
    loss = delta.square() / (2**2 * 28)
    loss.backward()
    assert float(loss) == pytest.approx(oracle.loss)
    assert float(z.grad) == pytest.approx(oracle.z_derivative)
    assert f.grad.tolist() == pytest.approx(oracle.forward_sum_derivatives)
    assert b.grad.tolist() == pytest.approx(oracle.backward_sum_derivatives)
    with pytest.raises(ValueError):
        ttb_reference(
            log_z=0,
            forward_sums=(1,),
            backward_sums=(),
            token_counts=(1,),
            reward=0,
            epsilon=0.1,
            beta=1,
        )


def test_clip_only_after_complete_sum_and_condition_roundtrip():
    f, b, z = (nn.Parameter(torch.zeros(1)) for _ in range(3))
    f.grad, b.grad, z.grad = (
        torch.tensor([2.0]) + torch.tensor([4.0]),
        torch.tensor([4.0]),
        torch.tensor([0.0]),
    )
    config = PolicyStabilityConfig(forward_max_norm=3)
    report = clip_full_batch_groups(PolicyParameterGroups((f,), (b,), (z,)), config)
    assert report["forward"]["pre_clip_norm"] == 6
    assert f.grad.item() == 3
    assert b.grad.item() == 4
    optimizer = OptimizerConfig(
        1e-4,
        1e-3,
        format="skillev-optimizer-config@4",
        backward_learning_rate=2e-4,
        stability=config,
    )
    assert OptimizerConfig.from_value(optimizer.to_value()) == optimizer


def test_interface_rollout_config_roundtrip_and_defaults():
    from skillev.runtime import BudgetVector
    from skillev.training.config import PolicyRolloutConfig

    legacy = PolicyRolloutConfig(
        0,
        1,
        8,
        8,
        BudgetVector(
            input_tokens=8192,
            output_tokens=16,
            model_calls=2,
            agent_turns=1,
            tool_calls=1,
            wall_time_milliseconds=1000,
        ),
    )
    assert PolicyRolloutConfig.from_value(legacy.to_value()) == legacy
    from dataclasses import replace

    candidate = replace(
        legacy,
        format="skillev-policy-rollout@7",
        phase_context=True,
        action_wire="native-single-tool-call@1",
        skill_exposure="catalog-then-read@1",
    )
    assert PolicyRolloutConfig.from_value(candidate.to_value()) == candidate
    assert candidate.context_assembler(maximum_h0_tokens=4096).action_wire == candidate.action_wire
    with pytest.raises(ValueError):
        replace(legacy, action_wire="native-single-tool-call@1")


def test_complete_kernel_clipping_diagnostics_roundtrip(make_training_harness):
    import asyncio
    from dataclasses import replace

    from skillev.contracts.ttb_source_events import TrainingStepReportValue
    from tests.training.test_v3_training import _step_context

    baseline = make_training_harness()
    optimizer = replace(
        baseline.config.optimizer,
        format="skillev-optimizer-config@4",
        stability=PolicyStabilityConfig(forward_max_norm=0.001, z_max_norm=0.001),
    )
    harness = make_training_harness(config=replace(baseline.config, optimizer=optimizer))
    report = asyncio.run(harness.loop.run_one(_step_context(1)))
    diagnostics = report.optimization_diagnostics
    assert diagnostics is not None
    assert diagnostics["stability_loss"] == 0
    assert diagnostics["total_loss"] == diagnostics["ttb_loss"]
    assert 0 < diagnostics["gradient_groups"]["forward"]["scale"] <= 1
    assert TrainingStepReportValue.from_value(report.to_value()) == report
    assert harness.loop.optimizer_step == 1
