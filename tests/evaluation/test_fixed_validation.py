import asyncio

import pytest

from skillev.evaluation.fixed_validation import (
    FixedValidationCase,
    FixedValidationOutcome,
    run_fixed_validation,
)


def test_fixed_validation_aggregates_without_mutating_state() -> None:
    state = {"optimizer_step": 2, "library_version": "v1"}

    async def evaluate(case: FixedValidationCase) -> FixedValidationOutcome:
        return FixedValidationOutcome(case.task_id, case.benchmark, 0.5, True, False, False, False)

    report = asyncio.run(
        run_fixed_validation(
            checkpoint_id="checkpoint-2",
            cases=(FixedValidationCase("a", "qa"), FixedValidationCase("b", "code")),
            evaluate=evaluate,
            mutation_state=lambda: state,
        )
    )
    assert report.macro_mean == 0.5
    assert report.infrastructure_failure_count == 0


def test_fixed_validation_rejects_mutation() -> None:
    state = {"optimizer_step": 2}

    async def evaluate(case: FixedValidationCase) -> FixedValidationOutcome:
        state["optimizer_step"] += 1
        return FixedValidationOutcome(case.task_id, case.benchmark, 0.0, False, True, False, False)

    with pytest.raises(RuntimeError):
        asyncio.run(
            run_fixed_validation(
                checkpoint_id="checkpoint-2",
                cases=(FixedValidationCase("a", "qa"),),
                evaluate=evaluate,
                mutation_state=lambda: state,
            )
        )
