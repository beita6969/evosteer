"""Small deterministic parity oracle for upstream SkillFlow TTB scalars."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UpstreamTTBScalars:
    raw_residual: float
    normalized_residual: float
    loss: float
    theta_scale_per_token: float
    phi_scale_per_token: float
    z_scale: float


def compute_upstream_ttb_scalars(
    *,
    log_z: float,
    forward_logprob: float,
    backward_logprob: float,
    reward: float,
    beta: float,
    effective_steps: int,
    action_token_count: int,
    batch_size: int,
    epsilon_min: float,
) -> UpstreamTTBScalars:
    """Mirror the scalar path in ``GFlowNetTrainer._gradient_accumulation_step``."""

    if effective_steps < 1 or action_token_count < 1 or batch_size < 1:
        raise ValueError("upstream TTB normalizers must be positive")
    if not 0.0 < epsilon_min <= 1.0 or beta <= 0.0:
        raise ValueError("upstream TTB reward controls are invalid")
    values = (log_z, forward_logprob, backward_logprob, reward, beta, epsilon_min)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("upstream TTB inputs must be finite")
    log_reward = math.log(max(reward, epsilon_min))
    raw = log_z + forward_logprob - beta * log_reward - backward_logprob
    normalized = max(-10.0, min(10.0, raw / effective_steps))
    z_scale = 2.0 * normalized / (effective_steps * batch_size)
    theta = z_scale / action_token_count
    return UpstreamTTBScalars(
        raw_residual=raw,
        normalized_residual=normalized,
        loss=normalized * normalized,
        theta_scale_per_token=theta,
        phi_scale_per_token=-theta,
        z_scale=z_scale,
    )


__all__ = ["UpstreamTTBScalars", "compute_upstream_ttb_scalars"]
