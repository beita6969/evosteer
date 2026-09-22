"""Behavioral checks for real online graph effects without model services."""

import asyncio
from dataclasses import replace

import pytest

from skillev.contracts.canonical import stable_hash
from skillev.orchestration.actions import GraphAction
from skillev.orchestration.actions import GraphActionKind as K
from skillev.orchestration.evosteer_features import FEATURE_NAMES
from skillev.orchestration.execution import (
    GraphRuntime,
    IllegalGraphActionError,
    RuntimePoisonedError,
)
from skillev.orchestration.graph import NodeExecutionResult, RoleSpec
from skillev.runtime.budget_ledger import BudgetExceededError, BudgetLedger
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector

MAXIMUM = BudgetVector(
    input_tokens=10,
    output_tokens=10,
    model_calls=1,
    agent_turns=1,
    tool_calls=2,
    wall_time_milliseconds=10,
)
ACTUAL = BudgetVector(
    input_tokens=2,
    output_tokens=3,
    model_calls=1,
    agent_turns=1,
    tool_calls=1,
    wall_time_milliseconds=2,
)


class Executor:
    frozen_identity = "scripted-frozen-executor@1"

    def __init__(self, usage=ACTUAL):
        self.requests = []
        self.usage = usage
        self.fail = False
        self.mutate_identity = False

    async def execute(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("world may already have advanced")
        if self.mutate_identity:
            self.frozen_identity = "changed"
        correct = (
            request.role.role_id == "verifier"
            or request.skills
            or any(message["body"] == "correct" for message in request.messages)
        )
        return NodeExecutionResult("correct" if correct else "wrong", self.usage)


class Evaluator:
    def __init__(self):
        self.outputs = []

    async def __call__(self, output):
        self.outputs.append(output)
        return float(output == "correct")


def build(*, executor=None, cap=None, max_nodes=4, max_actions=16):
    executor = executor or Executor()
    evaluator = Evaluator()
    ledger = BudgetLedger(
        run_id="test-run", attempt_id="test-attempt", cap=cap or MAXIMUM.scale(20)
    )
    runtime = GraphRuntime(
        task_prompt="Produce the correct result.",
        roles=(RoleSpec("solver", "Solve", MAXIMUM), RoleSpec("verifier", "Check", MAXIMUM)),
        skills={"repair": "Use the correct result.", "check": "Check your result."},
        executor=executor,
        evaluator=evaluator,
        ledger=ledger,
        max_nodes=max_nodes,
        max_actions=max_actions,
        runtime_id="episode-1",
        seed=123,
    )
    return runtime, executor, evaluator, ledger


def run(coroutine):
    return asyncio.run(coroutine)


def add(node_id, role_id="solver", skill_id=None):
    return GraphAction(K.ADD_AGENT, node_id=node_id, role_id=role_id, skill_id=skill_id)


@pytest.mark.parametrize(
    "action",
    [
        add("n0"),
        add("n0", skill_id="repair"),
        GraphAction(K.ADD_EDGE, source_id="n0", target_id="n1", protocol="feedback"),
        GraphAction(K.BIND_SKILL, node_id="n0", skill_id="repair"),
        GraphAction(K.SET_OUTPUT, node_id="n0"),
        GraphAction(K.RERUN_AGENT, node_id="n0"),
        GraphAction(K.DROP_AGENT, node_id="n0"),
        GraphAction(K.STOP),
    ],
)
def test_action_roundtrip_is_canonical(action):
    assert GraphAction.from_value(action.to_value()) == action
    assert GraphAction.from_value(action.to_value()).text == action.text


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "STOP", "node_id": "n0"},
        {"kind": "STOP", "node_id": None},
        {"kind": "ADD_AGENT", "node_id": "n0"},
        {"kind": "ADD_EDGE", "source_id": "n0", "target_id": "n0", "protocol": "feedback"},
        {"kind": "ADD_EDGE", "source_id": "n0", "target_id": "n1", "protocol": "recursive"},
        {"kind": "STOP", "unused": True},
        {"kind": "stop"},
    ],
)
def test_action_rejects_ambiguous_or_invalid_wire(value):
    with pytest.raises((ValueError, TypeError)):
        GraphAction.from_value(value)


def test_feedback_repair_changes_final_output_and_keeps_exact_history():
    async def scenario():
        runtime, executor, evaluator, ledger = build()
        await runtime.apply(add("n0"))
        assert runtime.graph.nodes[0].output == "wrong"
        await runtime.apply(add("n1", "verifier"))
        receipt = await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n1", target_id="n0", protocol="feedback")
        )
        assert len(executor.requests) == 2  # Feedback queues a message, does not execute.
        assert receipt.observation["delivered_message"]["source_output_version"] == 1
        receipt = await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n0"))
        assert receipt.observation["output_changed"] is True
        assert executor.requests[-1].previous_output == "wrong"
        assert executor.requests[-1].messages[0]["body"] == "correct"
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        result = await runtime.apply(GraphAction(K.STOP))
        assert result.reward == 1.0
        assert runtime.output == "correct"
        assert runtime.stopped
        assert evaluator.outputs == ["correct"]
        assert runtime.history[0]["observation"]["node_result"]["output"] == "wrong"
        assert runtime.graph.nodes[0].output_version == 2
        assert ledger.settled == ACTUAL.scale(3)
        assert runtime.node_usage == ledger.settled
        ledger.assert_fully_settled()
        with pytest.raises(IllegalGraphActionError):
            await runtime.apply(GraphAction(K.STOP))
        assert evaluator.outputs == ["correct"]

    run(scenario())


def test_revise_executes_once_even_with_a_feedback_cycle():
    async def scenario():
        runtime, executor, _, _ = build()
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1", "verifier"))
        await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n0", target_id="n1", protocol="feedback")
        )
        await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n1", target_id="n0", protocol="revise")
        )
        assert len(executor.requests) == 3
        assert runtime.graph.nodes[0].output == "correct"
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n1"))
        assert executor.requests[-1].messages[0]["source_output_version"] == 2
        assert len(executor.requests) == 4

    run(scenario())


def test_binding_executes_immediately_and_first_agent_can_receive_candidate():
    async def scenario():
        runtime, executor, _, _ = build()
        await runtime.apply(add("n0"))
        await runtime.apply(GraphAction(K.BIND_SKILL, node_id="n0", skill_id="repair"))
        assert len(executor.requests) == 2
        assert executor.requests[-1].skills == (("repair", "Use the correct result."),)
        assert runtime.graph.nodes[0].output == "correct"
        with pytest.raises(IllegalGraphActionError):
            await runtime.apply(GraphAction(K.BIND_SKILL, node_id="n0", skill_id="repair"))
        await runtime.apply(add("n1", skill_id="check"))
        assert executor.requests[-1].skills == (("check", "Check your result."),)
        assert executor.requests[-1].previous_output is None

    run(scenario())


def test_drop_preserves_history_no_refund_and_never_reuses_node_id():
    async def scenario():
        runtime, _, _, ledger = build(max_nodes=2)
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1"))
        await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n0", target_id="n1", protocol="feedback")
        )
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        with pytest.raises(IllegalGraphActionError):
            await runtime.apply(GraphAction(K.DROP_AGENT, node_id="n0"))
        before = ledger.settled
        await runtime.apply(GraphAction(K.DROP_AGENT, node_id="n1"))
        assert runtime.graph.edges == ()
        assert len(runtime.history) == 5
        assert ledger.settled == before
        assert all(a.node_id == "n2" for a in runtime.legal_actions() if a.kind is K.ADD_AGENT)
        await runtime.apply(add("n2"))
        assert "n1" in str(runtime.history[1])

    run(scenario())


def test_invalid_state_action_has_no_effect():
    async def scenario():
        runtime, executor, _, ledger = build()
        before = runtime.state_id
        for action in (
            GraphAction(K.STOP),
            add("n7"),
            GraphAction(K.RERUN_AGENT, node_id="missing"),
        ):
            with pytest.raises(IllegalGraphActionError):
                await runtime.apply(action)
        assert runtime.state_id == before
        assert not executor.requests
        assert not ledger.entries

    run(scenario())


def test_horizon_reserves_output_selection_and_stop():
    async def scenario():
        runtime, _, _, _ = build(max_actions=3)
        await runtime.apply(add("n0"))
        assert runtime.legal_actions() == (GraphAction(K.SET_OUTPUT, node_id="n0"),)
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        assert runtime.legal_actions() == (GraphAction(K.STOP),)
        await runtime.apply(GraphAction(K.STOP))
        assert runtime.legal_actions() == ()

    run(scenario())


def test_node_budget_exhaustion_still_allows_selection_and_stop():
    async def scenario():
        runtime, _, evaluator, ledger = build(executor=Executor(MAXIMUM), cap=MAXIMUM)
        await runtime.apply(add("n0"))
        assert not any(
            a.kind in {K.ADD_AGENT, K.BIND_SKILL, K.RERUN_AGENT} for a in runtime.legal_actions()
        )
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        await runtime.apply(GraphAction(K.STOP))
        assert ledger.settled == MAXIMUM
        assert evaluator.outputs == ["wrong"]

    run(scenario())


def test_deleting_last_node_cannot_strand_horizon_or_execution_budget():
    async def scenario():
        short, _, _, _ = build(max_actions=4)
        await short.apply(add("n0"))
        assert GraphAction(K.DROP_AGENT, node_id="n0") not in short.legal_actions()
        exhausted, _, _, _ = build(executor=Executor(MAXIMUM), cap=MAXIMUM)
        await exhausted.apply(add("n0"))
        assert GraphAction(K.DROP_AGENT, node_id="n0") not in exhausted.legal_actions()
        enough, _, _, _ = build(max_actions=5)
        await enough.apply(add("n0"))
        await enough.apply(GraphAction(K.DROP_AGENT, node_id="n0"))
        await enough.apply(add("n1"))
        await enough.apply(GraphAction(K.SET_OUTPUT, node_id="n1"))
        await enough.apply(GraphAction(K.STOP))
        assert enough.stopped

    run(scenario())


def test_features_use_public_state_only_and_are_owned_copies():
    async def scenario():
        runtime, _, _, _ = build()
        assert len(FEATURE_NAMES) == len(runtime.features()) == 30
        await runtime.apply(add("n0"))
        features = runtime.features()
        assert all(isinstance(value, float) and 0 <= value <= 1 for value in features)
        by_name = dict(zip(FEATURE_NAMES, features, strict=True))
        assert by_name["node_count_saturated"] == 0.5
        assert by_name["last_add_agent"] == 1.0
        state = runtime.public_state()
        state["history"][0]["observation"]["node_result"]["output"] = "tampered"
        history = runtime.history
        history[0]["action"]["node_id"] = "tampered"
        assert runtime.graph.nodes[0].output == "wrong"
        assert runtime.history[0]["action"]["node_id"] == "n0"
        assert runtime.features() == features

    run(scenario())


def test_snapshot_restores_exact_history_seed_and_does_not_recharge():
    async def scenario():
        runtime, executor, evaluator, ledger = build()
        await runtime.apply(add("n0"))
        # Model orchestration charges belong to the same ledger, not node_usage.
        controller = BudgetVector(input_tokens=1, output_tokens=1, model_calls=1)
        ledger.reserve(
            BudgetReservation(
                "controller", ledger.run_id, ledger.attempt_id, "episode-1", controller
            )
        )
        ledger.settle(BudgetSettlement("controller", controller))
        snapshot = runtime.snapshot()
        restored = GraphRuntime.from_snapshot(
            snapshot, executor=executor, ledger=ledger, evaluator=evaluator
        )
        assert restored.state_id == runtime.state_id
        assert len(executor.requests) == 1
        assert ledger.settled == ACTUAL.add(controller)
        assert restored.node_usage == ACTUAL
        await restored.apply(GraphAction(K.BIND_SKILL, node_id="n0", skill_id="repair"))
        assert executor.requests[-1].execution_index == 2
        await restored.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        await restored.apply(GraphAction(K.STOP))
        final = restored.snapshot()
        stopped = GraphRuntime.from_snapshot(
            final, executor=executor, ledger=ledger, evaluator=evaluator
        )
        assert stopped.stopped
        assert stopped.reward == 1.0
        assert evaluator.outputs == ["correct"]

    run(scenario())


def test_snapshot_rejects_tampering_and_missing_ledger():
    async def scenario():
        runtime, executor, evaluator, ledger = build()
        await runtime.apply(add("n0"))
        snapshot = runtime.snapshot()
        snapshot["graph"]["nodes"][0]["output"] = "changed"
        with pytest.raises(ValueError, match="identity"):
            GraphRuntime.from_snapshot(
                snapshot, executor=executor, ledger=ledger, evaluator=evaluator
            )
        empty = BudgetLedger(run_id=ledger.run_id, attempt_id=ledger.attempt_id, cap=ledger.cap)
        with pytest.raises(ValueError, match="ledger"):
            GraphRuntime.from_snapshot(
                runtime.snapshot(), executor=executor, ledger=empty, evaluator=evaluator
            )

    run(scenario())


def test_snapshot_recomputes_graph_from_history_even_with_recomputed_hash():
    async def scenario():
        runtime, executor, evaluator, ledger = build()
        await runtime.apply(add("n0"))
        snapshot = runtime.snapshot()
        snapshot["graph"]["nodes"][0]["output"] = "changed"
        snapshot["snapshot_hash"] = stable_hash(
            {key: item for key, item in snapshot.items() if key != "snapshot_hash"}
        )
        with pytest.raises(ValueError, match="disagree"):
            GraphRuntime.from_snapshot(
                snapshot, executor=executor, ledger=ledger, evaluator=evaluator
            )

    run(scenario())


def test_empty_model_response_is_a_valid_failed_node_outcome():
    class EmptyExecutor(Executor):
        async def execute(self, request):
            self.requests.append(request)
            return NodeExecutionResult("", ACTUAL)

    async def scenario():
        runtime, executor, evaluator, ledger = build(executor=EmptyExecutor())
        await runtime.apply(add("n0"))
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        await runtime.apply(GraphAction(K.STOP))
        assert runtime.stopped
        assert runtime.reward == 0.0
        assert evaluator.outputs == [""]
        restored = GraphRuntime.from_snapshot(
            runtime.snapshot(), executor=executor, ledger=ledger, evaluator=evaluator
        )
        assert restored.stopped
        assert restored.output == ""
        assert not restored.poisoned

    run(scenario())


def test_external_failure_poisoned_runtime_never_retries_or_claims_rollback():
    async def scenario():
        runtime, executor, _, ledger = build()
        executor.fail = True
        with pytest.raises(RuntimeError, match="world may"):
            await runtime.apply(add("n0"))
        assert runtime.poisoned
        assert runtime.legal_actions() == ()
        assert len(executor.requests) == 1
        assert runtime.history[-1]["observation"]["status"] == "infrastructure-failure"
        with pytest.raises(RuntimePoisonedError):
            await runtime.apply(add("n0"))
        with pytest.raises(RuntimePoisonedError):
            runtime.snapshot()
        with pytest.raises(RuntimeError, match="unsettled"):
            ledger.assert_fully_settled()

    run(scenario())


def test_budget_admission_race_is_rejected_before_effect_without_poisoning():
    async def scenario():
        runtime, executor, _, ledger = build()
        original = ledger.reserve

        def rejected(reservation):
            raise BudgetExceededError("capacity changed before dispatch")

        ledger.reserve = rejected
        with pytest.raises(IllegalGraphActionError, match="before dispatch"):
            await runtime.apply(add("n0"))
        assert not runtime.poisoned
        assert not runtime.history
        assert not executor.requests
        ledger.reserve = original
        await runtime.apply(add("n0"))
        assert len(executor.requests) == 1

    run(scenario())


def test_cancelled_external_execution_cannot_be_resumed_as_if_nothing_happened():
    class CancelExecutor(Executor):
        async def execute(self, request):
            self.requests.append(request)
            raise asyncio.CancelledError()

    async def scenario():
        runtime, executor, _, ledger = build(executor=CancelExecutor())
        with pytest.raises(asyncio.CancelledError):
            await runtime.apply(add("n0"))
        assert runtime.poisoned
        assert len(executor.requests) == 1
        with pytest.raises(RuntimeError, match="unsettled"):
            ledger.assert_fully_settled()

    run(scenario())


def test_concurrent_actions_are_serialized_and_revalidated():
    class YieldExecutor(Executor):
        async def execute(self, request):
            await asyncio.sleep(0)
            return await super().execute(request)

    async def scenario():
        runtime, executor, _, _ = build(executor=YieldExecutor())
        results = await asyncio.gather(
            runtime.apply(add("n0")),
            runtime.apply(add("n0")),
            return_exceptions=True,
        )
        assert sum(isinstance(result, IllegalGraphActionError) for result in results) == 1
        assert len(executor.requests) == 1
        assert len(runtime.history) == 1
        assert not runtime.poisoned

    run(scenario())


def test_changed_executor_identity_is_rejected_and_measured_usage_is_settled():
    async def scenario():
        runtime, executor, _, ledger = build()
        executor.mutate_identity = True
        with pytest.raises(RuntimeError, match="identity changed"):
            await runtime.apply(add("n0"))
        assert runtime.poisoned
        assert ledger.settled == ACTUAL
        ledger.assert_fully_settled()

    run(scenario())


def test_executor_overcharge_poisoned_instead_of_underreporting():
    async def scenario():
        runtime, _, _, _ = build(executor=Executor(replace(ACTUAL, model_calls=2)))
        with pytest.raises(RuntimeError, match="usage exceeds"):
            await runtime.apply(add("n0"))
        assert runtime.poisoned

    run(scenario())


def test_evaluator_failure_is_not_a_zero_reward_or_repeatable_call():
    async def scenario():
        runtime, _, _, _ = build()
        calls = []

        async def failing(output):
            calls.append(output)
            raise RuntimeError("grading service failed")

        runtime._evaluator = failing
        await runtime.apply(add("n0"))
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        with pytest.raises(RuntimeError, match="grading"):
            await runtime.apply(GraphAction(K.STOP))
        assert runtime.poisoned
        assert runtime.reward is None
        assert calls == ["wrong"]
        with pytest.raises(RuntimePoisonedError):
            await runtime.apply(GraphAction(K.STOP))
        assert calls == ["wrong"]

    run(scenario())


def test_default_runtime_has_no_node_or_action_ceiling_and_restores_long_histories():
    async def scenario():
        runtime, executor, evaluator, ledger = build(
            max_nodes=None, max_actions=None, cap=MAXIMUM.scale(100)
        )
        for index in range(10):
            await runtime.apply(add(f"n{index}"))
        for _ in range(20):
            await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n0"))
        assert len(runtime.graph.nodes) == 10
        assert len(runtime.history) == 30
        assert runtime.public_state()["actions_remaining"] is None
        assert runtime.public_state()["debug_limits"] == {"max_nodes": None, "max_actions": None}
        assert all(action.kind is not K.STOP for action in runtime.legal_actions())
        restored = GraphRuntime.from_snapshot(
            runtime.snapshot(), executor=executor, ledger=ledger, evaluator=evaluator
        )
        assert restored.public_state() == runtime.public_state()
        await restored.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        await restored.apply(GraphAction(K.STOP))
        assert restored.stopped
        assert len(executor.requests) == 30
        assert ledger.settled == ACTUAL.scale(30)

    run(scenario())


def test_unbounded_runtime_still_masks_exhausted_node_calls_and_allows_legal_stop():
    async def scenario():
        runtime, _, _, _ = build(
            max_nodes=None, max_actions=None, executor=Executor(MAXIMUM), cap=MAXIMUM
        )
        await runtime.apply(add("n0"))
        assert runtime.legal_actions() == (GraphAction(K.SET_OUTPUT, node_id="n0"),)
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        assert runtime.legal_actions() == (GraphAction(K.STOP),)
        await runtime.apply(GraphAction(K.STOP))
        assert runtime.stopped

    run(scenario())


def test_all_thirty_schema_two_coordinates_respond_to_observed_execution():
    class ObservedExecutor(Executor):
        def public_environment_state(self):
            count = len(self.requests)
            return {
                "enabled": True,
                "steps_per_phase": 8,
                "max_phases": 12,
                "steps_completed": count * 8,
                "phase_index": count,
                "environment_terminal": count >= 6,
                "last_outcome": None
                if count == 0
                else {
                    "status": "tool_error" if count == 5 else "success",
                },
            }

        async def execute(self, request):
            self.requests.append(request)
            count = len(self.requests)
            output = ("draft", "correct", "correct", "revised", "observed error", "revised")[
                count - 1
            ]
            return NodeExecutionResult(
                output,
                ACTUAL,
                {
                    "execution_outcome": {
                        "status": "failed" if count == 5 else "answered",
                        "answer_present": count != 5,
                        "failure_kind": "tool_error" if count == 5 else None,
                    }
                },
            )

    async def scenario():
        runtime, _, _, _ = build(executor=ObservedExecutor(), max_nodes=None, max_actions=None)
        vectors = [runtime.features()]
        actions = (
            add("n0"),
            add("n1", "verifier"),
            GraphAction(K.BIND_SKILL, node_id="n0", skill_id="repair"),
            GraphAction(K.ADD_EDGE, source_id="n0", target_id="n1", protocol="feedback"),
            GraphAction(K.RERUN_AGENT, node_id="n1"),
            GraphAction(K.ADD_EDGE, source_id="n1", target_id="n0", protocol="revise"),
            GraphAction(K.RERUN_AGENT, node_id="n0"),
            GraphAction(K.SET_OUTPUT, node_id="n0"),
            GraphAction(K.DROP_AGENT, node_id="n1"),
            GraphAction(K.STOP),
        )
        for action in actions:
            receipt = await runtime.apply(action)
            assert receipt.features == runtime.features()
            vectors.append(receipt.features)
        for index, name in enumerate(FEATURE_NAMES):
            assert len({vector[index] for vector in vectors}) > 1, name
        assert all(0.0 <= value <= 1.0 for vector in vectors for value in vector)
        before = runtime.features()
        runtime._reward = 0.123
        assert runtime.features() == before

    run(scenario())


def test_runtime_snapshot_requires_observed_environment_state_to_match():
    class ObservedExecutor(Executor):
        step = 0

        def public_environment_state(self):
            return {"steps_completed": self.step}

    executor = ObservedExecutor()
    runtime, _, evaluator, ledger = build(executor=executor, max_nodes=None, max_actions=None)
    snapshot = runtime.snapshot()
    executor.step = 1
    with pytest.raises(ValueError, match="same observed external"):
        GraphRuntime.from_snapshot(snapshot, executor=executor, ledger=ledger, evaluator=evaluator)


def test_paper_skill_menu_exposes_slots_and_execution_features_without_inline_skill_text():
    async def scenario():
        runtime, executor, _, _ = build()
        initial = runtime.public_state()
        # A slot carries its identifier, hash and label (name/trigger), never the body.
        assert all(
            set(slot) <= {"skill_id", "content_hash", "name", "trigger"}
            for slot in initial["skill_menu"]
        )
        assert initial["feature_names"] == list(FEATURE_NAMES)
        assert initial["execution_features"] == list(runtime.features())
        # Schema 4: answers are counted from the output text that states one.
        assert initial["feature_version"] == "evosteer-public-features@4"
        await runtime.apply(add("n0", skill_id="repair"))
        assert executor.requests[0].skills == (("repair", "Use the correct result."),)
        assert runtime.history[0]["observation"]["node_request"]["skills"] == [
            {"skill_id": "repair", "body": "Use the correct result."}
        ]
        menu = runtime.public_state()["skill_menu"]
        assert all(set(slot) <= {"skill_id", "content_hash", "name", "trigger"} for slot in menu)
        # The procedure text itself never reaches the orchestrator's state.
        assert all("Use the correct result." not in str(slot.values()) for slot in menu)

    run(scenario())
