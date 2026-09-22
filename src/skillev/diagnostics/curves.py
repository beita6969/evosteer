"""Windowed training-curve metrics without monotonicity assumptions."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class TrainingCurvePoint:
    optimizer_step: int
    mean_reward: float
    success_rate: float
    completion_rate: float
    parse_invalid_rate: float
    horizon_rate: float
    tool_success_rate: float
    ttb_loss: float
    mean_abs_residual: float
    gradient_norm: float
    kl_value: float | None
    policy_entropy: float | None
    effective_sample_size: float | None
    active_skill_count: int
    invoked_skill_count: int
    infrastructure_failure_count: int
    fixed_validation_reward: float | None = None
    fixed_validation_success_rate: float | None = None
    median_abs_residual: float | None = None
    p95_abs_residual: float | None = None
    parameter_delta_norm: float | None = None
    retrieval_hit_rate: float | None = None
    skill_success_rate: float | None = None
    domain_rewards: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.optimizer_step < 0:
            raise ValueError("optimizer step cannot be negative")
        for definition in fields(self):
            value = getattr(self, definition.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("curve values must be finite")
        if (
            min(
                self.active_skill_count,
                self.invoked_skill_count,
                self.infrastructure_failure_count,
            )
            < 0
        ):
            raise ValueError("curve counts cannot be negative")
        if self.domain_rewards is not None and (
            not self.domain_rewards
            or any(
                not name.strip() or not math.isfinite(value)
                for name, value in self.domain_rewards.items()
            )
        ):
            raise ValueError("domain rewards must be a non-empty finite mapping")

    @property
    def is_formal(self) -> bool:
        return self.infrastructure_failure_count == 0


def ewma(values: tuple[float, ...], *, alpha: float) -> tuple[float, ...]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    output: list[float] = []
    current: float | None = None
    for value in values:
        if not math.isfinite(value):
            raise ValueError("curve value must be finite")
        current = value if current is None else alpha * value + (1.0 - alpha) * current
        output.append(current)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class CurveSanityPolicy:
    window: int = 4
    allowed_reward_drop: float = 0.05
    allowed_validation_drop: float = 0.05
    maximum_gradient_norm: float = 1_000.0
    maximum_residual_growth_ratio: float = 2.0
    minimum_completion_rate: float = 0.05
    maximum_kl_value: float | None = None
    minimum_policy_entropy: float | None = None

    def validate(self, points: tuple[TrainingCurvePoint, ...]) -> None:
        if not points:
            raise ValueError("curve cannot be empty")
        if any(not point.is_formal for point in points):
            raise ValueError("formal curve contains infrastructure failures")
        if all(point.mean_reward == 0.0 for point in points):
            raise ValueError("reward remained zero for the full curve")
        if all(point.completion_rate == 0.0 for point in points):
            raise ValueError("completion remained zero for the full curve")
        if min(point.completion_rate for point in points) < self.minimum_completion_rate:
            raise ValueError("completion collapsed below the frozen floor")
        if all(point.gradient_norm <= 0.0 for point in points):
            raise ValueError("gradient norm remained zero for the full curve")
        if max(point.gradient_norm for point in points) > self.maximum_gradient_norm:
            raise ValueError("gradient norm exceeded the frozen safety ceiling")
        if all(point.parameter_delta_norm is not None for point in points) and all(
            (point.parameter_delta_norm or 0.0) <= 0.0 for point in points
        ):
            raise ValueError("parameter delta remained zero for the full curve")
        width = min(self.window, len(points))
        first = sum(item.mean_reward for item in points[:width]) / width
        last = sum(item.mean_reward for item in points[-width:]) / width
        if last < first - self.allowed_reward_drop:
            raise ValueError("last-window reward regressed beyond the frozen tolerance")
        first_residual = sum(item.mean_abs_residual for item in points[:width]) / width
        last_residual = sum(item.mean_abs_residual for item in points[-width:]) / width
        if last_residual > max(first_residual, 1e-12) * self.maximum_residual_growth_ratio:
            raise ValueError("residual grew beyond the frozen ratio")
        if all(item.fixed_validation_reward is not None for item in points):
            first_validation = (
                sum(item.fixed_validation_reward or 0.0 for item in points[:width]) / width
            )
            last_validation = (
                sum(item.fixed_validation_reward or 0.0 for item in points[-width:]) / width
            )
            if last_validation < first_validation - self.allowed_validation_drop:
                raise ValueError("fixed validation reward regressed beyond tolerance")
        if all(item.domain_rewards is not None for item in points):
            domains = set(points[0].domain_rewards or {})
            if any(set(item.domain_rewards or {}) != domains for item in points):
                raise ValueError("domain reward identities changed across the curve")
            for domain in domains:
                if all((item.domain_rewards or {})[domain] == 0.0 for item in points):
                    raise ValueError(f"domain reward remained zero: {domain}")
            if not any(
                (points[-1].domain_rewards or {})[domain] > (points[0].domain_rewards or {})[domain]
                for domain in domains
            ):
                raise ValueError("no domain reward improved across the controlled curve")
        if self.maximum_kl_value is not None and any(
            item.kl_value is not None and item.kl_value > self.maximum_kl_value for item in points
        ):
            raise ValueError("KL exceeded the preregistered safety range")
        if self.minimum_policy_entropy is not None and any(
            item.policy_entropy is not None and item.policy_entropy < self.minimum_policy_entropy
            for item in points
        ):
            raise ValueError("policy entropy fell below the preregistered safety range")


__all__ = ["CurveSanityPolicy", "TrainingCurvePoint", "ewma"]
