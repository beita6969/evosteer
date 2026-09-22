"""Answer-free identity and paired-comparison contracts for Step-0 evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from skillev.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class ArchitectureIdentityReceipt:
    method_id: str
    architecture_family: str
    architecture_lineage: str
    architecture_components: tuple[str, ...]
    controller_id: str
    optimizer_steps: int
    forward_adapter_active: bool
    backward_adapter_active: bool
    posterior_active: bool
    calibration_active: bool
    operator_active: bool
    seed_library_id: str
    seed_skill_ids: tuple[str, ...]
    retrieval_policy_id: str
    initial_context_profile: str
    reasoning_authority: str
    reasoning_contract_resolved: bool
    action_policy_authority: str
    inference_arm: dict[str, object] | None = None
    intervention_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        for name in (
            "method_id",
            "architecture_family",
            "architecture_lineage",
            "controller_id",
            "seed_library_id",
            "retrieval_policy_id",
            "initial_context_profile",
            "reasoning_authority",
            "action_policy_authority",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.architecture_components or len(set(self.architecture_components)) != len(
            self.architecture_components
        ):
            raise ValueError("architecture components must be non-empty and unique")
        required_components = {
            "seeded-skill-controller",
            "trajectory-balance-gflownet",
            "beta-bernoulli-lcb-calibration",
            "operator-driven-skill-evolution",
        }
        if not required_components.issubset(self.architecture_components):
            raise ValueError("Step-0 identity must name the complete BayesianImprove architecture")
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be a non-negative integer")
        if type(self.reasoning_contract_resolved) is not bool:
            raise TypeError("reasoning contract flag must be boolean")
        if not self.reasoning_contract_resolved:
            raise ValueError("Step-0 cannot publish with an unresolved reasoning contract")
        frozen_authorities = {
            "frozen-evaluation-configured",
            "frozen-deterministic",
            "frozen-benchmark-matched",
            "frozen-benchmark-conditioned-mixed",
        }
        trained_authorities = {
            "trained-forward-policy-configured",
            "trained-forward-policy-deterministic",
            "trained-forward-policy-benchmark-matched",
            "trained-forward-policy-benchmark-conditioned-mixed",
        }
        if self.optimizer_steps == 0:
            if self.reasoning_authority not in frozen_authorities:
                raise ValueError("unscored Step-0 reasoning must use a frozen authority")
            if any(
                (
                    self.forward_adapter_active,
                    self.backward_adapter_active,
                    self.posterior_active,
                    self.calibration_active,
                    self.operator_active,
                )
            ):
                raise ValueError("Step-0 cannot claim trained BayesianImprove components")
        else:
            if not self.forward_adapter_active:
                raise ValueError("trained inference must activate the learned forward adapter")
            if self.reasoning_authority not in trained_authorities:
                raise ValueError("trained inference must name the learned forward policy")
        if len(set(self.seed_skill_ids)) != len(self.seed_skill_ids):
            raise ValueError("seed skill identities must be unique")

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "action_policy_authority": self.action_policy_authority,
            "architecture_components": list(self.architecture_components),
            "architecture_family": self.architecture_family,
            "architecture_lineage": self.architecture_lineage,
            "backward_adapter_active": self.backward_adapter_active,
            "calibration_active": self.calibration_active,
            "controller_id": self.controller_id,
            "forward_adapter_active": self.forward_adapter_active,
            "initial_context_profile": self.initial_context_profile,
            "method_id": self.method_id,
            "operator_active": self.operator_active,
            "optimizer_steps": self.optimizer_steps,
            "posterior_active": self.posterior_active,
            "reasoning_authority": self.reasoning_authority,
            "reasoning_contract_resolved": self.reasoning_contract_resolved,
            "retrieval_policy_id": self.retrieval_policy_id,
            "seed_library_id": self.seed_library_id,
            "seed_skill_ids": list(self.seed_skill_ids),
        }
        if self.inference_arm is not None:
            from skillev.contracts import normalize_json

            value["inference_arm"] = normalize_json(self.inference_arm)
            value["intervention_counts"] = normalize_json(self.intervention_counts)
        return value


@dataclass(frozen=True, slots=True)
class PairedArchitectureComparison:
    pair_id: str
    benchmark: str
    task_ids: tuple[str, ...]
    model_service_receipt_id: str
    environment_receipt_id: str | None
    evaluator_receipt_id: str
    backbone_condition_id: str
    architecture_condition_id: str
    scorer_condition_equal: bool
    panel_equal: bool

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "benchmark",
            "model_service_receipt_id",
            "evaluator_receipt_id",
            "backbone_condition_id",
            "architecture_condition_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("paired task IDs must be non-empty and unique")
        if not self.scorer_condition_equal or not self.panel_equal:
            raise ValueError("a paired comparison requires the same panel and scorer")


@dataclass(frozen=True, slots=True)
class PairedOutcomeCounts:
    both_pass: int
    backbone_only_pass: int
    step0_only_pass: int
    both_fail: int

    def __post_init__(self) -> None:
        if min(self.both_pass, self.backbone_only_pass, self.step0_only_pass, self.both_fail) < 0:
            raise ValueError("paired outcome counts cannot be negative")

    @property
    def total(self) -> int:
        return self.both_pass + self.backbone_only_pass + self.step0_only_pass + self.both_fail


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    """One headline metric under the strict Step-0 non-regression rule."""

    metric: str
    backbone_percent: float
    step0_percent: float
    maximum_regression_pp: float = 7.0

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("paired metric name must be non-empty")
        for name in ("backbone_percent", "step0_percent", "maximum_regression_pp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.maximum_regression_pp <= 0:
            raise ValueError("maximum regression must be positive")

    @property
    def delta_pp(self) -> float:
        return self.step0_percent - self.backbone_percent

    @property
    def passes_non_regression(self) -> bool:
        # The gate is deliberately strict: exactly -7 pp is a failure.
        return self.delta_pp > -self.maximum_regression_pp


@dataclass(frozen=True, slots=True)
class StrictBackboneImprovementTarget:
    """Owner-authoritative Step-0 target relative to the current backbone score."""

    metric: str
    backbone_percent: float
    step0_percent: float

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("target metric name must be non-empty")
        for name in ("backbone_percent", "step0_percent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 100.0:
                raise ValueError(f"{name} must be a percentage in [0, 100]")

    @property
    def delta_pp(self) -> float:
        return self.step0_percent - self.backbone_percent

    @property
    def passes(self) -> bool:
        # Equality is not an improvement: every Step-0 target is strictly greater-than.
        return self.delta_pp > 0.0


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    delta: float
    lower: float
    upper: float
    confidence: float
    resamples: int

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1 or self.resamples < 1:
            raise ValueError("paired bootstrap configuration is invalid")
        if any(not math.isfinite(value) for value in (self.delta, self.lower, self.upper)):
            raise ValueError("paired bootstrap values must be finite")
        if not self.lower <= self.upper:
            raise ValueError("paired bootstrap interval is reversed")


def paired_outcome_counts(
    backbone: tuple[bool, ...],
    step0: tuple[bool, ...],
) -> PairedOutcomeCounts:
    if not backbone or len(backbone) != len(step0):
        raise ValueError("paired outcomes require equal non-empty populations")
    pairs = tuple(zip(backbone, step0, strict=True))
    return PairedOutcomeCounts(
        both_pass=sum(left and right for left, right in pairs),
        backbone_only_pass=sum(left and not right for left, right in pairs),
        step0_only_pass=sum(not left and right for left, right in pairs),
        both_fail=sum(not left and not right for left, right in pairs),
    )


def paired_bootstrap_interval(
    backbone: tuple[float, ...],
    step0: tuple[float, ...],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapInterval:
    """Bootstrap the mean within-record Step-0 minus backbone difference."""

    if not backbone or len(backbone) != len(step0):
        raise ValueError("paired metrics require equal non-empty populations")
    if not 0 < confidence < 1 or type(resamples) is not int or resamples < 1:
        raise ValueError("paired bootstrap configuration is invalid")
    differences = tuple(right - left for left, right in zip(backbone, step0, strict=True))
    if any(isinstance(value, bool) or not math.isfinite(value) for value in differences):
        raise ValueError("paired metric values must be finite")
    # Deterministic statistical resampling; this is not a security boundary.
    generator = random.Random(seed)  # noqa: S311
    size = len(differences)
    estimates = sorted(
        sum(differences[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = min(int(tail * resamples), resamples - 1)
    upper_index = min(int((1.0 - tail) * resamples), resamples - 1)
    return PairedBootstrapInterval(
        delta=sum(differences) / size,
        lower=estimates[lower_index],
        upper=estimates[upper_index],
        confidence=confidence,
        resamples=resamples,
    )


def mcnemar_exact_two_sided(counts: PairedOutcomeCounts) -> float:
    """Return the exact two-sided binomial McNemar p-value."""

    if not isinstance(counts, PairedOutcomeCounts):
        raise TypeError("McNemar input must be paired outcome counts")
    discordant = counts.backbone_only_pass + counts.step0_only_pass
    if discordant == 0:
        return 1.0
    smaller = min(counts.backbone_only_pass, counts.step0_only_pass)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * tail))


__all__ = [
    "ArchitectureIdentityReceipt",
    "PairedArchitectureComparison",
    "PairedBootstrapInterval",
    "PairedMetricComparison",
    "PairedOutcomeCounts",
    "StrictBackboneImprovementTarget",
    "mcnemar_exact_two_sided",
    "paired_bootstrap_interval",
    "paired_outcome_counts",
]
