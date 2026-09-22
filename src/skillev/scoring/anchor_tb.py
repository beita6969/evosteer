"""Reference-relative, uniformly weighted subtrajectory balance for EvoSteer.

Inputs are *sequence sums* of selected-token log probabilities, not token means.
The root follows the same measured-plus-residual parameterization as every
nonterminal state. Only the observed terminal reward transform is fixed by
default. A hard root is available as an explicitly selected ablation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def log_terminal_reward(reward: float, beta: float = 1.0) -> float:
    """Return log(1 + (exp(beta) - 1) * reward), including exact endpoints."""
    reward = _finite(reward, "reward")
    beta = _finite(beta, "beta")
    if not 0 <= reward <= 1 or beta <= 0:
        raise ValueError("reward must lie in [0, 1] and beta must be positive")
    if reward == 0:
        return 0.0
    if reward == 1:
        return beta
    if beta < 700:
        return math.log1p(math.expm1(beta) * reward)
    # Stable equivalent for finite beta values whose exponential overflows.
    return beta + math.log(reward + (1 - reward) * math.exp(-beta))


@dataclass(frozen=True, slots=True)
class AnchorTBResult:
    loss: float
    segment_residuals: tuple[tuple[int, int, float], ...]
    action_coefficients: tuple[float, ...]
    state_coefficients: tuple[float, ...]
    state_flows: tuple[float, ...]


def anchor_tb_scalar(
    action_log_ratios: Sequence[float],
    state_flows: Sequence[float],
    *,
    reward: float,
    beta: float = 1.0,
    root_anchor: float | None = None,
    hard_anchor_root: bool = False,
) -> AnchorTBResult:
    """Explicit-segment oracle; root_anchor overrides the root as a hard ablation."""
    ratios = tuple(_finite(v, "action log ratio") for v in action_log_ratios)
    flows = [_finite(v, "state flow") for v in state_flows]
    horizon = len(ratios)
    if horizon < 1 or len(flows) != horizon + 1:
        raise ValueError("T actions require T+1 state flows and T must be positive")
    if type(hard_anchor_root) is not bool:
        raise ValueError("hard_anchor_root must be boolean")
    if root_anchor is not None:
        flows[0] = _finite(root_anchor, "root anchor")
    flows[-1] = log_terminal_reward(reward, beta)
    count = horizon * (horizon + 1) // 2
    residuals: list[tuple[int, int, float]] = []
    action_gradients = [0.0] * horizon
    state_gradients = [0.0] * (horizon + 1)
    for i in range(horizon):
        for j in range(i + 1, horizon + 1):
            residual = flows[i] + math.fsum(ratios[i:j]) - flows[j]
            residuals.append((i, j, residual))
            coefficient = 2 * residual / count
            state_gradients[i] += coefficient
            state_gradients[j] -= coefficient
            for t in range(i, j):
                action_gradients[t] += coefficient
    if hard_anchor_root or root_anchor is not None:
        state_gradients[0] = 0.0
    state_gradients[-1] = 0.0
    loss = math.fsum(delta * delta for _, _, delta in residuals) / count
    if not all(math.isfinite(v) for v in (loss, *action_gradients, *state_gradients)):
        raise ValueError("AnchorTB arithmetic overflowed")
    return AnchorTBResult(
        loss, tuple(residuals), tuple(action_gradients), tuple(state_gradients), tuple(flows)
    )


def anchor_tb_loss(
    action_log_ratios: Tensor,
    state_flows: Tensor,
    *,
    reward: float,
    beta: float = 1.0,
    root_anchor: float | None = None,
    hard_anchor_root: bool = False,
    implementation: str = "efficient",
) -> Tensor:
    """Exact all-subtrajectory loss, with a trainable root and fixed terminal flow.

    ``hard_anchor_root=True`` detaches the supplied root; ``root_anchor`` replaces
    it by a fixed scalar. Both are ablation controls, not Eq. (8)'s default.
    """
    import torch

    if (
        not isinstance(action_log_ratios, torch.Tensor)
        or not isinstance(state_flows, torch.Tensor)
        or action_log_ratios.ndim != 1
        or state_flows.ndim != 1
        or not action_log_ratios.is_floating_point()
        or not state_flows.is_floating_point()
        or action_log_ratios.device != state_flows.device
        or action_log_ratios.dtype != state_flows.dtype
    ):
        raise ValueError("ratios and flows must be matching floating one-dimensional tensors")
    horizon = action_log_ratios.numel()
    if horizon < 1 or state_flows.numel() != horizon + 1:
        raise ValueError("T actions require T+1 state flows and T must be positive")
    if type(hard_anchor_root) is not bool:
        raise ValueError("hard_anchor_root must be boolean")
    if not bool(torch.isfinite(action_log_ratios).all()) or not bool(
        torch.isfinite(state_flows).all()
    ):
        raise ValueError("AnchorTB inputs must be finite")
    if root_anchor is None:
        root = state_flows[:1].detach() if hard_anchor_root else state_flows[:1]
    else:
        root = state_flows.new_tensor([_finite(root_anchor, "root anchor")])
    terminal = state_flows.new_tensor([log_terminal_reward(reward, beta)])
    flows = torch.cat((root, state_flows[1:-1], terminal))
    prefix = torch.cat((action_log_ratios.new_zeros(1), action_log_ratios.cumsum(0)))
    offsets = flows - prefix
    count = horizon * (horizon + 1) // 2
    if implementation == "explicit":
        loss = torch.stack(
            [
                (offsets[i] - offsets[j]).square()
                for i in range(horizon)
                for j in range(i + 1, horizon + 1)
            ]
        ).mean()
    elif implementation == "efficient":
        centered = offsets - offsets.mean()
        loss = centered.square().sum() * ((horizon + 1) / count)
    else:
        raise ValueError("implementation must be explicit or efficient")
    if not bool(torch.isfinite(loss)):
        raise ValueError("AnchorTB loss overflowed")
    return loss
