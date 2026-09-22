from __future__ import annotations

import pytest

from skillev.runtime import BudgetExceededError, BudgetVector
from skillev.training import FixedAttemptBudgetPlan


def _plan(*, maximum_cycles: int) -> FixedAttemptBudgetPlan:
    return FixedAttemptBudgetPlan(
        batch_count=2,
        batch_size=3,
        max_turns=2,
        reasoning_call_maximum=BudgetVector(
            input_tokens=1,
            output_tokens=2,
            model_calls=1,
        ),
        action_call_maximum=BudgetVector(
            input_tokens=3,
            output_tokens=4,
            model_calls=1,
            agent_turns=1,
        ),
        tool_call_maximum=BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=5,
        ),
        maximum_cycles=maximum_cycles,
        phi_per_cycle_maximum=BudgetVector(
            input_tokens=7,
            output_tokens=11,
            model_calls=2,
        ),
    )


@pytest.mark.parametrize("maximum_cycles", [0, 1, 2])
def test_fixed_attempt_budget_includes_each_predeclared_phi_cycle(
    maximum_cycles: int,
) -> None:
    plan = _plan(maximum_cycles=maximum_cycles)
    rollout_count = plan.batch_count * plan.batch_size
    generation_count = rollout_count * plan.max_turns
    base = (
        plan.reasoning_call_maximum.scale(generation_count)
        .add(plan.action_call_maximum.scale(generation_count))
        .add(plan.tool_call_maximum.scale(generation_count))
    )
    required = base.add(plan.phi_per_cycle_maximum.scale(maximum_cycles))

    assert plan.required() == required
    plan.validate_against(required)
    with pytest.raises(BudgetExceededError):
        plan.validate_against(
            BudgetVector(
                input_tokens=required.input_tokens - 1,
                output_tokens=required.output_tokens,
                model_calls=required.model_calls,
                agent_turns=required.agent_turns,
                tool_calls=required.tool_calls,
                wall_time_milliseconds=required.wall_time_milliseconds,
            )
        )
