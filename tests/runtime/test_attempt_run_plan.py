from __future__ import annotations

import pytest

from skillev.runtime import (
    AttemptRunCursorState,
    AttemptRunProgress,
    ExactAttemptRunPlan,
    RunSlotKind,
)


def test_run_plan_has_fixed_cross_arm_step_count() -> None:
    plan = ExactAttemptRunPlan(
        phase_search_steps=3,
        closure_steps=1,
        maximum_cycles=2,
    )

    assert plan.total_training_steps == 4
    assert tuple(plan.slot_kind(index) for index in range(1, 5)) == (
        RunSlotKind.PHASE_SEARCH,
        RunSlotKind.PHASE_SEARCH,
        RunSlotKind.PHASE_SEARCH,
        RunSlotKind.CLOSURE,
    )
    assert ExactAttemptRunPlan.from_value(plan.to_value()) == plan


def test_run_cursor_cannot_cross_plan_or_cycle_bound() -> None:
    plan = ExactAttemptRunPlan(
        phase_search_steps=1,
        closure_steps=1,
        maximum_cycles=1,
    )
    cursor = AttemptRunCursorState.fresh(plan)
    cursor = cursor.after_training_step(plan).after_cycle(plan, action_count=2)
    cursor = cursor.after_training_step(plan)

    assert cursor.completed_training_steps == 2
    assert cursor.committed_cycles == 1
    assert cursor.committed_actions == 2
    with pytest.raises(RuntimeError):
        cursor.after_training_step(plan)
    with pytest.raises(RuntimeError):
        cursor.after_cycle(plan, action_count=1)


def test_plan_requires_a_real_closure_and_reports_only_remaining_capacity() -> None:
    plan = ExactAttemptRunPlan(
        phase_search_steps=2,
        closure_steps=2,
        maximum_cycles=2,
    )
    cursor = AttemptRunCursorState.fresh(plan).after_training_step(plan)

    assert plan.remaining_from(cursor).training_steps == 3
    assert plan.remaining_from(cursor).possible_cycles == 2
    after_cycle = cursor.after_cycle(plan, action_count=1)
    assert plan.remaining_from(after_cycle).possible_cycles == 1


def test_progress_requires_previewed_one_cycle_advance() -> None:
    plan = ExactAttemptRunPlan(
        phase_search_steps=1,
        closure_steps=1,
        maximum_cycles=1,
    )
    progress = AttemptRunProgress.fresh(plan)
    progress.commit_training_step()
    next_state = progress.preview_cycle(action_count=1)
    progress.commit_cycle(next_state)

    assert progress.state == next_state
    with pytest.raises(ValueError):
        progress.commit_cycle(next_state)
