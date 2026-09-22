import math

import pytest

from skillev.scoring.anchor_tb import anchor_tb_loss, anchor_tb_scalar, log_terminal_reward


def test_reward_transform_endpoints_continuous_and_extreme_beta():
    assert log_terminal_reward(0.0) == 0.0
    assert log_terminal_reward(1.0) == 1.0
    assert log_terminal_reward(0.25) == pytest.approx(math.log1p(math.expm1(1) * 0.25))
    assert log_terminal_reward(0.5, 1000) == pytest.approx(1000 - math.log(2))
    for reward, beta in ((-0.1, 1), (1.1, 1), (0.5, 0), (float("nan"), 1)):
        with pytest.raises(ValueError):
            log_terminal_reward(reward, beta)


def test_explicit_oracle_has_distinct_action_credit_and_trainable_root():
    result = anchor_tb_scalar([0.2, -0.1], [0.4, 0.7, 999.0], reward=0.0)
    assert [d for _, _, d in result.segment_residuals] == pytest.approx([-0.1, 0.5, 0.6])
    assert result.loss == pytest.approx((0.01 + 0.25 + 0.36) / 3)
    assert result.action_coefficients == pytest.approx((2 * 0.4 / 3, 2 * 1.1 / 3))
    assert result.state_coefficients == pytest.approx((2 * 0.4 / 3, 2 * 0.7 / 3, 0))
    assert result.state_flows[-1] == 0.0


def test_hard_root_remains_an_explicit_ablation():
    default = anchor_tb_scalar([0.2, -0.1], [0.4, 0.7, 999.0], reward=0.0)
    hard = anchor_tb_scalar([0.2, -0.1], [0.4, 0.7, 999.0], reward=0.0, hard_anchor_root=True)
    assert default.loss == hard.loss
    assert default.state_coefficients[0] != 0.0
    assert hard.state_coefficients[0] == 0.0


def test_scalar_action_coefficients_match_finite_difference():
    ratios = [0.23, -0.61, 0.08]
    flows = [0.45, 0.6, 0.39, 0.2]
    result = anchor_tb_scalar(ratios, flows, reward=0.75)
    h = 1e-6
    for index, derivative in enumerate(result.action_coefficients):
        plus, minus = ratios.copy(), ratios.copy()
        plus[index] += h
        minus[index] -= h
        observed = (
            anchor_tb_scalar(plus, flows, reward=0.75).loss
            - anchor_tb_scalar(minus, flows, reward=0.75).loss
        ) / (2 * h)
        assert derivative == pytest.approx(observed, abs=1e-9)
    for index, derivative in enumerate(result.state_coefficients):
        plus, minus = flows.copy(), flows.copy()
        plus[index] += h
        minus[index] -= h
        observed = (
            anchor_tb_scalar(ratios, plus, reward=0.75).loss
            - anchor_tb_scalar(ratios, minus, reward=0.75).loss
        ) / (2 * h)
        assert derivative == pytest.approx(observed, abs=1e-9)


def test_invalid_shape_and_nonfinite_inputs_are_rejected():
    for ratios, flows in (([], [0]), ([0.0], [0.0]), ([math.inf], [0, 0])):
        with pytest.raises(ValueError):
            anchor_tb_scalar(ratios, flows, reward=0)


def test_torch_explicit_and_efficient_values_and_gradients_equal_scalar_oracle():
    torch = pytest.importorskip("torch")
    ratios = torch.tensor([0.2, -0.5, 0.3, -0.1], dtype=torch.float64, requires_grad=True)
    flows = torch.tensor([0.6, 0.8, 0.1, 0.3, 42.0], dtype=torch.float64, requires_grad=True)
    expected = anchor_tb_scalar(ratios.tolist(), flows.tolist(), reward=0.7)
    for implementation in ("explicit", "efficient"):
        loss = anchor_tb_loss(ratios, flows, reward=0.7, implementation=implementation)
        dr, du = torch.autograd.grad(loss, (ratios, flows))
        assert loss.item() == pytest.approx(expected.loss, abs=1e-13)
        assert dr.tolist() == pytest.approx(expected.action_coefficients, abs=1e-13)
        assert du.tolist() == pytest.approx(expected.state_coefficients, abs=1e-13)
        assert du[0] != 0
        assert du[-1] == 0


def test_torch_root_override_and_frozen_reference_have_no_gradients():
    torch = pytest.importorskip("torch")
    current = torch.tensor([0.3, -0.1], requires_grad=True)
    reference = torch.tensor([-0.2, -0.2], requires_grad=True)
    flows = torch.tensor([100.0, 0.2, -100.0], requires_grad=True)
    loss = anchor_tb_loss(current - reference.detach(), flows, reward=1.0, root_anchor=0.7)
    loss.backward()
    assert reference.grad is None
    assert flows.grad[0].item() == 0.0
    assert flows.grad[-1].item() == 0.0
    assert current.grad is not None
