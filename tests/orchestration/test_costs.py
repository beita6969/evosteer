from __future__ import annotations

from skillev.orchestration.costs import CostRecord, aggregate_costs
from skillev.runtime import BudgetVector


def test_costs_keep_logical_and_physical_usage_separate() -> None:
    retried = CostRecord.from_physical_attempts(
        logical=BudgetVector(
            input_tokens=5,
            output_tokens=2,
            model_calls=1,
            agent_turns=1,
        ),
        physical_attempts=(
            BudgetVector(input_tokens=5, output_tokens=2, model_calls=1, agent_turns=1),
            BudgetVector(input_tokens=3, model_calls=1),
        ),
    )
    direct = CostRecord(
        logical=BudgetVector(tool_calls=1, wall_time_milliseconds=4),
        physical=BudgetVector(tool_calls=1, wall_time_milliseconds=7),
    )

    summary = aggregate_costs((retried, direct))

    assert summary.record_count == 2
    assert summary.logical == BudgetVector(
        input_tokens=5,
        output_tokens=2,
        model_calls=1,
        agent_turns=1,
        tool_calls=1,
        wall_time_milliseconds=4,
    )
    assert summary.physical == BudgetVector(
        input_tokens=8,
        output_tokens=2,
        model_calls=2,
        agent_turns=1,
        tool_calls=1,
        wall_time_milliseconds=7,
    )


def test_empty_cost_collection_has_zero_vector_totals() -> None:
    summary = aggregate_costs(())

    assert summary.record_count == 0
    assert summary.logical == BudgetVector()
    assert summary.physical == BudgetVector()
