"""Real CPU optimizer transitions, without altering the optimizer algorithm."""

import asyncio
import copy
import json
from dataclasses import replace

import pytest
import torch

from skillev.contracts import TrainingStepReportValue
from skillev.policy import PolicyParameterGroups
from skillev.training.optimizer_observation import OptimizerTransitionObservation


def _groups():
    return PolicyParameterGroups(
        forward=(torch.nn.Parameter(torch.tensor([1.0, 2.0])),),
        backward=(torch.nn.Parameter(torch.tensor([3.0])),),
        z_head=(torch.nn.Parameter(torch.zeros(1)),),
    )


def _params(groups):
    return (*groups.forward, *groups.backward, *groups.z_head)


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, list):
        for left, right in zip(a, b, strict=True):
            _equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("warm", [False, True])
def test_observation_preserves_real_adam_parameters_gradients_state_and_rng(warm):
    groups = _groups()
    params = _params(groups)
    optimizer = torch.optim.Adam(params, lr=0.01)
    for p in params:
        p.grad = torch.full_like(p, 0.25)
    if warm:
        optimizer.step()
    # A no-gradient component is still observed, but must not create Adam state.
    params[1].grad = None
    plain = tuple(torch.nn.Parameter(p.detach().clone()) for p in params)
    control = torch.optim.Adam(plain, lr=0.01)
    control.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for source, target in zip(params, plain, strict=True):
        target.grad = source.grad.clone() if source.grad is not None else None
    gradients = [None if p.grad is None else p.grad.clone() for p in params]
    state_before = copy.deepcopy(optimizer.state_dict())
    before = [p.detach().double().clone() for p in params]
    rng = torch.random.get_rng_state().clone()
    observed = OptimizerTransitionObservation.capture(optimizer, groups)
    _equal(optimizer.state_dict(), state_before)
    optimizer.step()
    state_after = copy.deepcopy(optimizer.state_dict())
    result = observed.finish()
    _equal(optimizer.state_dict(), state_after)
    control.step()
    _equal(optimizer.state_dict(), control.state_dict())
    assert torch.equal(torch.random.get_rng_state(), rng)
    for p, expected, gradient in zip(params, plain, gradients, strict=True):
        assert torch.equal(p, expected)
        _equal(p.grad, gradient)
    json.dumps(result, allow_nan=False)
    for name, p, b in zip(("forward", "backward", "z"), params, before, strict=True):
        group = result["components"][name]
        assert group["parameters"]["update_l2"] == pytest.approx(
            float(torch.linalg.vector_norm(p.detach().double() - b))
        )
        assert group["parameters"]["after_nonfinite_count"] == 0
    b = result["components"]["backward"]
    assert b["parameters"]["no_gradient_tensor_count"] == 1
    assert b["parameters"]["update_l2"] == 0
    assert b["adam"]["after"]["step_min"] == (1 if warm else None)
    assert result["components"]["forward"]["adam"]["after"]["step_max"] == (2 if warm else 1)
    if not warm:
        assert result["components"]["z"]["parameters"]["relative_update_l2"] is None
        assert result["components"]["z"]["parameters"]["update_l2"] > 0
        assert b["adam"]["after"]["moments"]["exp_avg"]["nonfinite_count"] is None


def test_nonfinite_observations_are_counts_and_null_not_nan_or_fake_zero():
    groups = _groups()
    optimizer = torch.optim.Adam(_params(groups))
    p = groups.forward[0]
    optimizer.state[p] = {
        "step": torch.tensor(float("nan")),
        "exp_avg": torch.tensor([float("nan"), 0.0]),
        "exp_avg_sq": torch.tensor([0.0, float("inf")]),
    }
    with torch.no_grad():
        p[0] = float("inf")
    result = OptimizerTransitionObservation.capture(optimizer, groups).finish()
    f = result["components"]["forward"]
    assert f["parameters"]["before_nonfinite_count"] == 1
    assert f["parameters"]["after_l2"] is None
    assert f["parameters"]["update_l2"] is None
    assert f["parameters"]["relative_update_l2"] is None
    assert f["adam"]["before"]["step_invalid_count"] == 1
    assert f["adam"]["after"]["moments"]["exp_avg_sq"]["nonfinite_count"] == 1
    json.dumps(result, allow_nan=False)


def test_non_adam_is_explicitly_unavailable_and_frozen_weights_not_copied():
    groups = _groups()
    frozen = torch.nn.Parameter(torch.tensor([float("nan")]), requires_grad=False)
    groups = replace(groups, forward=(*groups.forward, frozen))
    optimizer = torch.optim.SGD(_params(groups), lr=0.1)
    report = OptimizerTransitionObservation.capture(optimizer, groups).finish()
    f = report["components"]["forward"]
    assert f["parameters"]["tensor_count"] == 1
    assert f["parameters"]["after_nonfinite_count"] == 0
    assert f["adam"]["before"] is None
    assert f["adam"]["after"] is None


@pytest.mark.parametrize("stability", [False, True])
def test_enabled_loop_reports_one_real_step_with_optional_stability(
    make_training_harness, stability
):
    from skillev.training.config import PolicyStabilityConfig
    from tests.training.test_v3_training import _step_context

    harness = make_training_harness()
    if stability:
        config = replace(
            harness.config,
            optimizer=replace(
                harness.config.optimizer,
                format="skillev-optimizer-config@4",
                stability=PolicyStabilityConfig(forward_max_norm=0.001, z_max_norm=0.001),
            ),
        )
        harness = make_training_harness(config=config)
    harness.loop.enable_update_observation()
    report = asyncio.run(harness.loop.run_one(_step_context(1)))
    assert report.format == "skillev-training-step-report@5"
    assert TrainingStepReportValue.from_value(report.to_value()) == report
    assert harness.loop.optimizer_step == 1
    assert report.optimization_diagnostics is not None
    decomposition = report.optimization_diagnostics["ttb_decomposition"]
    assert decomposition["optimizer_step"] == 1
    assert decomposition["sampled_policy_step"] == 0
    assert decomposition["trajectories"]
    assert ("stability_loss" in report.optimization_diagnostics) == stability
    for component in report.optimizer_transition["components"].values():
        assert component["parameters"]["update_l2"] > 0
        assert component["adam"]["after"]["step_min"] == 1
        assert component["adam"]["after"]["step_max"] == 1


def test_report_old_wire_shapes_and_new_transition_evidence():
    report = TrainingStepReportValue(
        optimizer_step=1,
        batch_id="public-fixture",
        torch_batch_loss=1,
        audited_batch_loss=1,
        mean_reward=0.5,
        grad_norm_forward=1,
        grad_norm_backward=1,
        grad_norm_z=1,
        forward_adapter_version="f1",
        backward_adapter_version="b1",
        z_version="z1",
        started_at="2026-09-13T00:00:00Z",
        completed_at="2026-09-13T00:00:01Z",
    )
    for old in (
        report,
        replace(
            report, format="skillev-training-step-report@4", optimization_diagnostics={"loss": 1}
        ),
    ):
        value = old.to_value()
        assert "optimizer_transition" not in value
        assert TrainingStepReportValue.from_value(value).to_value() == value
    assert "optimization_diagnostics" not in report.to_value()
    with pytest.raises(ValueError):
        replace(report, optimizer_transition={"observed": True})
    with pytest.raises(ValueError):
        replace(
            report,
            format="skillev-training-step-report@5",
            optimizer_transition={"bad": float("nan")},
        )


def test_learning_report_preserves_actual_optimizer_evidence():
    from skillev.training.learning_behavior_report import committed_learning_report
    from tests.training.test_metrics_contract import event

    groups = _groups()
    optimizer = torch.optim.Adam(_params(groups))
    for p in _params(groups):
        p.grad = torch.ones_like(p)
    capture = OptimizerTransitionObservation.capture(optimizer, groups)
    optimizer.step()
    transition = capture.finish()
    source = event()
    source["payload"]["report"] = {"optimizer_transition": transition}
    before = copy.deepcopy(source)
    report = committed_learning_report(source, condition_id="fixture")
    assert source == before
    assert report["optimizer_transition"] == transition
    for name, component in transition["components"].items():
        assert report["component_parameter_changes"][name] == component["parameters"]
        assert report["optimizer_state_anomalies"][name] == component["adam"]
    assert not any("Adam state require" in item for item in report["uncovered"])


def test_default_loop_never_captures_transition(make_training_harness, monkeypatch):
    from tests.training.test_v3_training import _step_context

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled observation must not inspect parameters")

    monkeypatch.setattr(OptimizerTransitionObservation, "capture", forbidden)
    harness = make_training_harness()
    report = asyncio.run(harness.loop.run_one(_step_context(1)))
    assert report.format == "skillev-training-step-report@3"
    assert report.optimizer_transition is None
    assert "optimizer_transition" not in report.to_value()
