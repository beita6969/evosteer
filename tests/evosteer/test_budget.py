"""Combined token caps constrain both live reservations and restored charges."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from skillev.orchestration.actions import GraphAction, GraphActionKind
from skillev.orchestration.budget import EvoBudgetLedger, can_reserve
from skillev.orchestration.execution import GraphRuntime
from skillev.orchestration.graph import NodeExecutionResult, RoleSpec
from skillev.runtime.budget_ledger import BudgetExceededError, BudgetLedger
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector

CAP = BudgetVector(input_tokens=100, output_tokens=100, model_calls=20, agent_turns=20)


def ledger(total=10):
    return EvoBudgetLedger(run_id="run", attempt_id="attempt", cap=CAP, total_token_cap=total)


def reservation(identity, input_tokens, output_tokens):
    return BudgetReservation(
        identity,
        "run",
        "attempt",
        "producer",
        BudgetVector(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def test_outstanding_reservations_and_exact_settlements_share_one_token_cap():
    budget = ledger()
    budget.reserve(reservation("controller", 4, 3))
    assert budget.total_tokens_available == 3
    assert not budget.can_reserve(BudgetVector(input_tokens=2, output_tokens=2))
    with pytest.raises(BudgetExceededError, match="combined"):
        budget.reserve(reservation("node", 2, 2))
    assert len(budget.entries) == 1
    budget.settle(BudgetSettlement("controller", BudgetVector(input_tokens=2, output_tokens=1)))
    assert budget.total_tokens_available == 7
    budget.reserve(reservation("node", 3, 4))
    budget.settle(BudgetSettlement("node", BudgetVector(input_tokens=3, output_tokens=4)))
    assert budget.total_tokens_available == 0
    assert budget.settled.input_tokens + budget.settled.output_tokens == 10
    budget.assert_fully_settled()


def test_per_dimension_limits_remain_effective():
    budget = ledger(1000)
    assert not budget.can_reserve(BudgetVector(input_tokens=101))
    with pytest.raises(BudgetExceededError):
        budget.reserve(reservation("too-many-input", 101, 0))
    assert budget.total_tokens_available == 1000


def test_zero_settlement_releases_unused_maximum_without_inventing_usage():
    budget = ledger()
    budget.reserve(reservation("not-dispatched", 5, 5))
    budget.settle(BudgetSettlement("not-dispatched", BudgetVector()))
    assert budget.settled == BudgetVector()
    assert budget.total_tokens_available == 10


def test_combined_check_and_parent_reservation_are_atomic_under_concurrency():
    budget = ledger()

    def attempt(index):
        try:
            budget.reserve(reservation(str(index), 4, 2))
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as workers:
        outcomes = tuple(workers.map(attempt, range(16)))
    assert sum(outcomes) == 1
    assert len(budget.entries) == 1
    assert budget.total_tokens_available == 4


def test_restored_charges_cannot_bypass_combined_cap_and_rejection_is_atomic():
    source = ledger(20)
    source.reserve(reservation("old", 7, 6))
    source.settle(BudgetSettlement("old", BudgetVector(input_tokens=7, output_tokens=6)))
    target = ledger(10)
    with pytest.raises(BudgetExceededError, match="combined"):
        target.restore_completed(source.entries)
    assert target.entries == ()
    assert target.settled == BudgetVector()
    target = ledger(13)
    target.restore_completed(source.entries)
    assert target.total_tokens_available == 0
    assert target.entries == source.entries


class Executor:
    frozen_identity = "budget-test"
    calls = 0

    async def execute(self, request):
        self.calls += 1
        return NodeExecutionResult(
            "output",
            BudgetVector(
                input_tokens=3,
                output_tokens=3,
                model_calls=1,
                agent_turns=1,
            ),
        )


def runtime(budget, executor):
    return GraphRuntime(
        task_prompt="task",
        roles=(
            RoleSpec(
                "solver",
                "solve",
                BudgetVector(
                    input_tokens=3,
                    output_tokens=3,
                    model_calls=1,
                    agent_turns=1,
                ),
            ),
        ),
        skills={},
        executor=executor,
        ledger=budget,
        evaluator=lambda _: 0.5,
    )


def test_graph_mask_checks_combined_cap_and_existing_output_can_still_stop():
    async def scenario():
        budget = ledger(10)
        executor = Executor()
        graph = runtime(budget, executor)
        await graph.apply(GraphAction(GraphActionKind.ADD_AGENT, node_id="n0", role_id="solver"))
        kinds = {action.kind for action in graph.legal_actions()}
        assert GraphActionKind.ADD_AGENT not in kinds
        assert GraphActionKind.RERUN_AGENT not in kinds
        assert graph.public_state()["total_token_budget"] == {"cap": 10, "available": 4}
        await graph.apply(GraphAction(GraphActionKind.SET_OUTPUT, node_id="n0"))
        await graph.apply(GraphAction(GraphActionKind.STOP))
        assert graph.reward == 0.5
        assert executor.calls == 1

    asyncio.run(scenario())


def test_snapshot_restoration_requires_the_same_aggregate_cap():
    budget = ledger(10)
    executor = Executor()
    graph = runtime(budget, executor)
    snapshot = graph.snapshot()
    restored = GraphRuntime.from_snapshot(
        snapshot, executor=executor, ledger=ledger(10), evaluator=lambda _: 0.5
    )
    assert restored.public_state() == graph.public_state()
    with pytest.raises(ValueError, match="exact saved ledger"):
        GraphRuntime.from_snapshot(
            snapshot, executor=executor, ledger=ledger(20), evaluator=lambda _: 0.5
        )


def test_legacy_ledger_keeps_independent_dimension_semantics():
    legacy = BudgetLedger(run_id="run", attempt_id="attempt", cap=CAP)
    assert can_reserve(legacy, BudgetVector(input_tokens=100, output_tokens=100))
    assert not can_reserve(ledger(), BudgetVector(input_tokens=6, output_tokens=6))


@pytest.mark.parametrize("value", [True, -1, 1.5, None])
def test_invalid_combined_cap_rejected(value):
    with pytest.raises(ValueError):
        ledger(value)
