"""CPU graph rollouts at controller budget boundaries and prompt projection."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from skillev.contracts.canonical import canonical_json
from skillev.contracts.evosteer import EvoTask
from skillev.orchestration.actions import GraphAction, GraphActionKind
from skillev.orchestration.execution import GraphRuntime
from skillev.orchestration.graph import NodeExecutionResult, RoleSpec
from skillev.rollout.evosteer import actor_projection, collect_episode
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetVector


class NoInferencePolicy:
    """Real action text/token paths; any unexpected stochastic call fails."""

    actor_id = "boundary-actor"
    reference_id = "boundary-reference"
    context_window = 100_000
    max_action_tokens = 256

    def __init__(self):
        self.prompts = []
        self.supports = []
        self.calls = 0

    def menu(self, actions):
        texts = tuple(canonical_json(action) for action in actions)
        self.supports.append(texts)
        return SimpleNamespace(
            actions=texts,
            paths=tuple((*tuple(text.encode()), 256) for text in texts),
        )

    def encode_prompt(self, text):
        self.prompts.append(text)
        return tuple(text.encode())

    def sample(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("deterministic budget finalization must not call a model")


class PublicExecutor:
    frozen_identity = "frozen-boundary-executor"

    def __init__(self):
        self.calls = []

    async def execute(self, request):
        self.calls.append(request)
        return NodeExecutionResult(
            f"observed-answer-{request.node_id}",
            BudgetVector(input_tokens=10, output_tokens=2, model_calls=1, agent_turns=1),
            metadata={
                "reservation_ids": ["paired_treatment-INTERNAL-reservation"],
                "local_budget_entries": [{"invocation_id": "paired_control-INTERNAL-invocation"}],
                "local_turns": [
                    {
                        "prompt": "NODE-ONLY-SKILL-PROMPT",
                        "reservation_id": "natural_reference-INTERNAL-reservation",
                        "response": "observed-local-response",
                    }
                ],
            },
        )


def runtime_fixture(*, model_calls, max_nodes=2, controller_time=1_000_000, roles=1):
    ledger = BudgetLedger(
        run_id="boundary-run",
        attempt_id="boundary-attempt",
        cap=BudgetVector(
            input_tokens=200_000,
            output_tokens=20_000,
            model_calls=model_calls,
            agent_turns=100,
            wall_time_milliseconds=controller_time,
        ),
    )
    maximum = BudgetVector(
        input_tokens=50,
        output_tokens=20,
        model_calls=1,
        agent_turns=1,
        wall_time_milliseconds=100,
    )
    executor = PublicExecutor()
    evaluated = []

    def evaluate(output):
        evaluated.append(output)
        return 0.75

    runtime = GraphRuntime(
        task_prompt="PUBLIC-TASK-PROMPT",
        roles=tuple(
            RoleSpec(f"solver-{index}", "Solve the public task.", maximum) for index in range(roles)
        ),
        skills={},
        executor=executor,
        ledger=ledger,
        evaluator=evaluate,
        max_nodes=max_nodes,
        max_actions=12,
        runtime_id="current-INTERNAL-runtime",
    )
    return runtime, ledger, executor, evaluated


async def add_node(runtime, index):
    await runtime.apply(
        GraphAction(
            GraphActionKind.ADD_AGENT,
            node_id=f"n{index}",
            role_id="solver-0",
        )
    )


async def collect(runtime, ledger, policy):
    return await collect_episode(
        policy=policy,
        runtime=runtime,
        ledger=ledger,
        task=EvoTask("boundary-task", "math", "PUBLIC-TASK-PROMPT"),
        sample_id="current-INTERNAL-sample",
        batch_id="batch",
        source="current",
        menu_id="menu",
        value_snapshot_id="value",
        statistics_context="statistics",
        value_function=lambda features, family: 0.5,
        seed=4,
    )


def actions(trajectory):
    return [json.loads(record.action_json) for record in trajectory.decisions]


def test_controller_reservation_reduces_support_to_singleton_and_settles_zero_actual_usage():
    runtime, ledger, executor, evaluated = runtime_fixture(model_calls=2, max_nodes=1)
    policy = NoInferencePolicy()

    async def run():
        await add_node(runtime, 0)
        await runtime.apply(GraphAction(GraphActionKind.SET_OUTPUT, node_id="n0"))
        assert len(runtime.legal_actions()) > 1
        return await collect(runtime, ledger, policy)

    result = asyncio.run(run())
    assert actions(result) == [{"kind": "STOP"}]
    assert len(policy.supports[0]) > 1
    assert len(result.decisions[0].legal_token_paths) == 1
    controller = [
        entry for entry in ledger.entries if ":controller:" in entry.reservation.reservation_id
    ]
    assert len(controller) == 1
    assert controller[0].reservation.maximum.model_calls == 1
    assert controller[0].settlement.actual == BudgetVector()
    assert ledger.reserved == BudgetVector()
    assert ledger.settled.model_calls == len(executor.calls) == 1
    assert policy.calls == 0
    assert evaluated == ["observed-answer-n0"]


@pytest.mark.parametrize("selected_output", [None, "n1"])
def test_exhausted_controller_budget_finalizes_existing_nodes_without_inference(selected_output):
    runtime, ledger, executor, evaluated = runtime_fixture(model_calls=2)
    policy = NoInferencePolicy()

    async def run():
        await add_node(runtime, 0)
        await add_node(runtime, 1)
        if selected_output is not None:
            await runtime.apply(GraphAction(GraphActionKind.SET_OUTPUT, node_id=selected_output))
        assert len(runtime.legal_actions()) > 1
        return await collect(runtime, ledger, policy)

    result = asyncio.run(run())
    expected = [] if selected_output is not None else [{"kind": "SET_OUTPUT", "node_id": "n0"}]
    assert actions(result) == [*expected, {"kind": "STOP"}]
    assert all(len(record.legal_token_paths) == 1 for record in result.decisions)
    assert all(
        json.loads(record.state_json)["controller_budget_finalization"]
        for record in result.decisions
    )
    assert policy.calls == 0
    assert ledger.settled.model_calls == len(executor.calls) == 2
    assert ledger.reserved == BudgetVector()
    assert not any(":controller:" in entry.reservation.reservation_id for entry in ledger.entries)
    output_node = selected_output or "n0"
    assert result.output == f"observed-answer-{output_node}"
    assert result.reward == 0.75
    assert evaluated == [result.output]


@pytest.mark.parametrize(
    ("model_calls", "controller_time", "message"),
    [
        (0, 1_000_000, "no legal graph continuation"),
        (1, 100, "insufficient initial budget"),
    ],
)
def test_no_node_and_insufficient_budget_fails_without_fabricating_output_or_reward(
    model_calls,
    controller_time,
    message,
):
    runtime, ledger, executor, evaluated = runtime_fixture(
        model_calls=model_calls,
        controller_time=controller_time,
        roles=2,
    )
    policy = NoInferencePolicy()
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(collect(runtime, ledger, policy))
    assert runtime.output is None
    assert runtime.reward is None
    assert not runtime.stopped
    assert not runtime.history
    assert not executor.calls
    assert not evaluated
    assert policy.calls == 0
    assert not ledger.entries


def test_controller_prompt_hides_operational_source_ids_and_node_local_prompt_but_keeps_evidence():
    runtime, ledger, _, _ = runtime_fixture(model_calls=1, max_nodes=1)
    policy = NoInferencePolicy()

    async def run():
        await add_node(runtime, 0)
        return await collect(runtime, ledger, policy)

    result = asyncio.run(run())
    # Full audit state retains execution provenance; policy receives its projection.
    assert "paired_treatment-INTERNAL-reservation" in result.terminal_state_json
    assert "NODE-ONLY-SKILL-PROMPT" in result.terminal_state_json
    for prompt in policy.prompts:
        assert "INTERNAL" not in prompt
        assert "NODE-ONLY-SKILL-PROMPT" not in prompt
        assert "reservation_id" not in prompt
        assert "local_budget_entries" not in prompt
        assert "PUBLIC-TASK-PROMPT" in prompt
        assert "observed-answer-n0" in prompt
        assert "observed-local-response" in prompt


def test_projection_removes_nested_internal_ids_independently_of_source_label():
    def material(source):
        return {
            "runtime_id": f"{source}:runtime",
            "task_prompt": "public task",
            "history": [
                {
                    "before_state_id": source,
                    "after_state_id": source,
                    "observation": {
                        "reservation_id": f"{source}:node-call",
                        "node_request": {"skills": "node-only procedure"},
                        "node_result": {
                            "output": "public result",
                            "metadata": {
                                "session_id": source,
                                "local_turns": [
                                    {
                                        "prompt": "private local prompt",
                                        "response": "public response",
                                    }
                                ],
                            },
                        },
                    },
                }
            ],
        }

    expected = actor_projection(material("current"))
    for source in ("natural_reference", "paired_treatment", "paired_control"):
        assert actor_projection(material(source)) == expected
    assert "public task" in canonical_json(expected)
    assert "public response" in canonical_json(expected)
    assert "private local prompt" not in canonical_json(expected)
