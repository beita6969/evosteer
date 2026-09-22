from __future__ import annotations

import pytest

from skillev.experiments.comparison import (
    BenchmarkAggregateOutcome,
    ExperimentArmAggregate,
    compare_same_budget_arms,
)
from skillev.experiments.protocol import FIXED_SEED, AblationArm


def _run(
    arm: AblationArm,
    *,
    rollouts: int = 100,
) -> ExperimentArmAggregate:
    return ExperimentArmAggregate(
        protocol_hash="protocol-hash",
        arm=arm,
        seed=FIXED_SEED,
        initial_checkpoint_hash="checkpoint-hash",
        initial_library_version="library-v0",
        initial_skill_library_state_hash="library-state-v0",
        budget_snapshot_hash="budget-hash",
        training_task_ids=tuple(f"task-{index:03d}" for index in range(rollouts)),
        training_rollout_count=rollouts,
        optimizer_step_count=10,
        outcomes=(
            BenchmarkAggregateOutcome(
                benchmark_id="hotpotqa",
                sample_count=4,
                reward_sum=2.0 if arm is AblationArm.FULL else 1.0,
                reward_squared_sum=2.0 if arm is AblationArm.FULL else 1.0,
                success_count=2 if arm is AblationArm.FULL else 1,
            ),
        ),
    )


def test_compare_same_budget_arms_reports_aggregate_differences() -> None:
    comparison = compare_same_budget_arms(
        _run(AblationArm.FULL),
        _run(AblationArm.SKILLFLOW_DISABLED),
    )

    assert comparison.training_rollout_count == 100
    assert comparison.optimizer_step_count == 10
    assert len(comparison.benchmarks) == 1
    metric = comparison.benchmarks[0]
    assert metric.reward_difference.value == -0.25
    assert metric.success_rate_difference == -0.25


def test_compare_same_budget_arms_rejects_budget_mismatch() -> None:
    with pytest.raises(ValueError):
        compare_same_budget_arms(
            _run(AblationArm.FULL),
            _run(AblationArm.NO_FLOW_WEIGHTING, rollouts=99),
        )


def test_compare_same_budget_arms_rejects_task_order_mismatch() -> None:
    treatment = _run(AblationArm.NO_FLOW_WEIGHTING)
    with pytest.raises(ValueError):
        compare_same_budget_arms(
            _run(AblationArm.FULL),
            ExperimentArmAggregate(
                protocol_hash=treatment.protocol_hash,
                arm=treatment.arm,
                seed=treatment.seed,
                initial_checkpoint_hash=treatment.initial_checkpoint_hash,
                initial_library_version=treatment.initial_library_version,
                initial_skill_library_state_hash=treatment.initial_skill_library_state_hash,
                budget_snapshot_hash=treatment.budget_snapshot_hash,
                training_task_ids=tuple(reversed(treatment.training_task_ids)),
                training_rollout_count=treatment.training_rollout_count,
                optimizer_step_count=treatment.optimizer_step_count,
                outcomes=treatment.outcomes,
            ),
        )


def test_compare_same_budget_arms_requires_full_reference() -> None:
    with pytest.raises(ValueError):
        compare_same_budget_arms(
            _run(AblationArm.CAPPED_FLOW_WEIGHT),
            _run(AblationArm.CLIPPED_IMPORTANCE),
        )
