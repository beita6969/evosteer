"""Same-budget aggregate comparisons for preregistered ablation arms."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._statistics import BinomialRate, DifferenceEstimate, MeanEstimate
from .protocol import FIXED_SEED, AblationArm


@dataclass(frozen=True, slots=True)
class BenchmarkAggregateOutcome:
    benchmark_id: str
    sample_count: int
    reward_sum: float
    reward_squared_sum: float
    success_count: int

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if type(self.success_count) is not int or not 0 <= self.success_count <= self.sample_count:
            raise ValueError("success_count must lie in [0, sample_count]")
        for field_name in ("reward_sum", "reward_squared_sum"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be finite")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, normalized)
        if self.reward_sum > self.sample_count or self.reward_squared_sum > self.sample_count:
            raise ValueError("reward moments exceed [0, 1] support")
        if self.reward_squared_sum + 1e-12 < self.reward_sum**2 / self.sample_count:
            raise ValueError("reward squared sum violates the moment lower bound")


@dataclass(frozen=True, slots=True)
class ExperimentArmAggregate:
    """Identity-free result envelope for one fixed training/evaluation arm."""

    protocol_hash: str
    arm: AblationArm
    seed: int
    initial_checkpoint_hash: str
    initial_library_version: str
    initial_skill_library_state_hash: str
    budget_snapshot_hash: str
    training_task_ids: tuple[str, ...]
    training_rollout_count: int
    optimizer_step_count: int
    outcomes: tuple[BenchmarkAggregateOutcome, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_hash",
            "initial_checkpoint_hash",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "budget_snapshot_hash",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not isinstance(self.arm, AblationArm):
            raise TypeError("arm must be an AblationArm")
        if self.seed != FIXED_SEED:
            raise ValueError("ablation result must use the preregistered single seed")
        if type(self.training_rollout_count) is not int or self.training_rollout_count <= 0:
            raise ValueError("training_rollout_count must be positive")
        if len(self.training_task_ids) != self.training_rollout_count:
            raise ValueError("training task order must contain one ID per rollout")
        if any(
            type(task_id) is not str or not task_id.strip() for task_id in self.training_task_ids
        ):
            raise ValueError("training task IDs must be non-empty text")
        if len(set(self.training_task_ids)) != len(self.training_task_ids):
            raise ValueError("training task population cannot repeat a task ID")
        if type(self.optimizer_step_count) is not int or self.optimizer_step_count <= 0:
            raise ValueError("optimizer_step_count must be positive")
        if not self.outcomes or any(
            not isinstance(outcome, BenchmarkAggregateOutcome) for outcome in self.outcomes
        ):
            raise ValueError("outcomes must contain benchmark aggregate outcomes")
        benchmark_ids = tuple(outcome.benchmark_id for outcome in self.outcomes)
        if benchmark_ids != tuple(sorted(benchmark_ids)) or len(set(benchmark_ids)) != len(
            benchmark_ids
        ):
            raise ValueError("benchmark outcomes must be unique and sorted")


@dataclass(frozen=True, slots=True)
class BenchmarkArmDifference:
    benchmark_id: str
    reference_reward: MeanEstimate
    treatment_reward: MeanEstimate
    reward_difference: DifferenceEstimate
    reference_success: BinomialRate
    treatment_success: BinomialRate
    success_rate_difference: float


@dataclass(frozen=True, slots=True)
class SameBudgetArmComparison:
    reference_arm: AblationArm
    treatment_arm: AblationArm
    training_rollout_count: int
    optimizer_step_count: int
    benchmarks: tuple[BenchmarkArmDifference, ...]


def _mean(outcome: BenchmarkAggregateOutcome) -> MeanEstimate:
    return MeanEstimate.from_moments(
        count=outcome.sample_count,
        total=outcome.reward_sum,
        squared_total=outcome.reward_squared_sum,
    )


def compare_same_budget_arms(
    reference: ExperimentArmAggregate,
    treatment: ExperimentArmAggregate,
) -> SameBudgetArmComparison:
    """Compare two arms only when provenance and training budget are identical."""

    if not isinstance(reference, ExperimentArmAggregate) or not isinstance(
        treatment, ExperimentArmAggregate
    ):
        raise TypeError("ablation comparison requires ExperimentArmAggregate values")
    for field_name in (
        "protocol_hash",
        "seed",
        "initial_checkpoint_hash",
        "initial_library_version",
        "initial_skill_library_state_hash",
        "budget_snapshot_hash",
        "training_task_ids",
        "training_rollout_count",
        "optimizer_step_count",
    ):
        if getattr(reference, field_name) != getattr(treatment, field_name):
            raise ValueError(f"ablation arms differ in {field_name}")
    if reference.arm is not AblationArm.FULL:
        raise ValueError("reference arm must be the preregistered full method")
    if treatment.arm is AblationArm.FULL:
        raise ValueError("treatment arm must be a non-full ablation")
    reference_by_id = {outcome.benchmark_id: outcome for outcome in reference.outcomes}
    treatment_by_id = {outcome.benchmark_id: outcome for outcome in treatment.outcomes}
    if set(reference_by_id) != set(treatment_by_id):
        raise ValueError("ablation arms must report the same benchmark aggregates")

    differences: list[BenchmarkArmDifference] = []
    for benchmark_id in sorted(reference_by_id):
        baseline = reference_by_id[benchmark_id]
        ablated = treatment_by_id[benchmark_id]
        baseline_mean = _mean(baseline)
        ablated_mean = _mean(ablated)
        baseline_success = BinomialRate.from_counts(
            baseline.success_count,
            baseline.sample_count,
        )
        ablated_success = BinomialRate.from_counts(
            ablated.success_count,
            ablated.sample_count,
        )
        differences.append(
            BenchmarkArmDifference(
                benchmark_id=benchmark_id,
                reference_reward=baseline_mean,
                treatment_reward=ablated_mean,
                reward_difference=DifferenceEstimate.between(ablated_mean, baseline_mean),
                reference_success=baseline_success,
                treatment_success=ablated_success,
                success_rate_difference=ablated_success.value - baseline_success.value,
            )
        )
    return SameBudgetArmComparison(
        reference_arm=reference.arm,
        treatment_arm=treatment.arm,
        training_rollout_count=reference.training_rollout_count,
        optimizer_step_count=reference.optimizer_step_count,
        benchmarks=tuple(differences),
    )


__all__ = [
    "BenchmarkAggregateOutcome",
    "BenchmarkArmDifference",
    "ExperimentArmAggregate",
    "SameBudgetArmComparison",
    "compare_same_budget_arms",
]
