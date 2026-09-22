"""Frozen role tools share real public side effects, never evaluator authority."""

import asyncio
import json
from dataclasses import replace

import pytest

from skillev.contracts.canonical import canonical_json
from skillev.orchestration.actions import GraphAction
from skillev.orchestration.actions import GraphActionKind as K
from skillev.orchestration.execution import GraphRuntime
from skillev.orchestration.graph import NodeExecutionRequest, RoleSpec
from skillev.orchestration.tool_executor import FrozenToolExecutor, ToolExecutorPoisonedError
from skillev.rollout.action_surface import (
    ACTION_SURFACE_FORMAT_V2,
    ActionSurface,
    ArgumentFieldSpec,
    ArgumentType,
    TerminalMode,
    ToolActionSpecV2,
)
from skillev.runtime.bounded_agent import EnvironmentMethodFailedError
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetVector
from skillev.runtime.execution import EnvironmentObservation

CAP = BudgetVector(
    input_tokens=500,
    output_tokens=500,
    model_calls=6,
    agent_turns=6,
    tool_calls=6,
    wall_time_milliseconds=1000,
)
TOOL_MAXIMUM = BudgetVector(tool_calls=1, wall_time_milliseconds=10)
TOOL_ACTUAL = BudgetVector(tool_calls=1, wall_time_milliseconds=2)


def complete(text):
    return canonical_json(
        {"kind": "complete", "name": "complete", "arguments": {"value": {"answer": text}}}
    )


def tool(name, **arguments):
    return canonical_json(
        {"kind": "tool", "resource_id": "world", "name": name, "arguments": arguments}
    )


def skill(skill_id="repair", **arguments):
    return canonical_json(
        {
            "kind": "skill",
            "resource_id": "skill-runtime",
            "name": "invoke",
            "skill_id": skill_id,
            "arguments": arguments,
        }
    )


class Model:
    reference_id = "frozen-scripted-policy@1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def frozen_text(self, text, **kwargs):
        self.requests.append((text, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value, 20, 10


class World:
    environment_id = "synthetic-shared-world@1"
    task_family = "synthetic-interactive"

    def __init__(self):
        self.count = 0
        self.calls = []
        self.fail = None
        self.terminal_at = None

    async def execute(self, action, *, step_index):
        assert not self.calls or step_index > self.calls[-1][1]
        self.calls.append((action, step_index))
        if action.name == "increment":
            self.count += action.arguments["amount"]
        if self.fail is not None:
            raise self.fail
        terminal = self.terminal_at is not None and self.count >= self.terminal_at
        return EnvironmentObservation(
            {"count": self.count},
            "success",
            terminal=terminal,
            terminal_submission={"private_route": "never-in-a-model-prompt"} if terminal else None,
            budget_usage=TOOL_ACTUAL,
        )

    def validate_completion(self, value):
        raise AssertionError("task completion/evaluation must not be consulted by node COMPLETE")


def surface():
    return ActionSurface(
        TerminalMode.ENVIRONMENT,
        tools=(
            ToolActionSpecV2(
                "world",
                "increment",
                {"amount": ArgumentFieldSpec(ArgumentType.INTEGER, True)},
                {"amount": 1},
            ),
            ToolActionSpecV2("world", "read", {}, {}),
        ),
        format=ACTION_SURFACE_FORMAT_V2,
    )


def build(responses, *, world=None, max_turns=4, permissions=None):
    model = Model(responses)
    world = world or World()
    executor = FrozenToolExecutor(
        model,
        world,
        surface(),
        role_permissions=permissions
        or {
            "solver": (("world", "increment"), ("world", "read"), ("skill-runtime", "invoke")),
            "verifier": (("world", "read"),),
        },
        max_turns=max_turns,
        tool_call_maximum=TOOL_MAXIMUM,
        initial_observation={"count": 0},
        clock=lambda: 0.0,
    )
    return executor, model, world


def request(index=1, *, node="n0", role="solver", cap=CAP, skills=()):
    return NodeExecutionRequest(
        "team-1",
        node,
        RoleSpec(role, "Use your permitted public tools.", cap),
        "Increment the shared counter.",
        skills,
        (),
        None,
        index,
        987,
    )


def run(coroutine):
    return asyncio.run(coroutine)


def test_two_nodes_and_rerun_share_side_effects_and_monotone_steps():
    async def scenario():
        executor, model, world = build(
            [
                tool("increment", amount=1),
                complete("counter is 1"),
                tool("read"),
                complete("verified 1"),
                tool("increment", amount=1),
                complete("counter is 2"),
            ]
        )
        first = await executor.execute(request())
        second = await executor.execute(request(2, node="n1", role="verifier"))
        third = await executor.execute(replace(request(3), previous_output=first.output))
        assert [first.output, second.output, third.output] == [
            "counter is 1",
            "verified 1",
            "counter is 2",
        ]
        assert world.count == 2
        assert [step for _, step in world.calls] == [1, 2, 3]
        assert executor.environment_step_index == 3
        assert json.loads(model.requests[2][0])["shared_environment_history"][0]["observation"][
            "public_value"
        ] == {"count": 1}
        assert json.loads(model.requests[4][0])["previous_node_output"] == "counter is 1"
        assert first.usage == BudgetVector(
            input_tokens=40,
            output_tokens=20,
            model_calls=2,
            agent_turns=2,
            tool_calls=1,
            wall_time_milliseconds=2,
        )
        assert first.metadata["completed"]
        assert first.metadata["task_success"] is None

    run(scenario())


def test_graph_runtime_charges_aggregate_once_and_only_team_stop_evaluates():
    async def scenario():
        executor, _, world = build([tool("increment", amount=1), complete("1")])
        ledger = BudgetLedger(run_id="run", attempt_id="attempt", cap=CAP.scale(2))
        evaluations = []

        async def evaluate(output):
            evaluations.append(output)
            return 1.0

        runtime = GraphRuntime(
            task_prompt="Increment once.",
            roles=(RoleSpec("solver", "Act", CAP),),
            skills={},
            executor=executor,
            ledger=ledger,
            evaluator=evaluate,
            runtime_id="team-1",
            max_actions=3,
        )
        receipt = await runtime.apply(GraphAction(K.ADD_AGENT, node_id="n0", role_id="solver"))
        assert evaluations == []
        assert world.count == 1
        assert len(ledger.entries) == 1  # Inner local reservations are not global charges.
        assert ledger.settled == BudgetVector.from_value(
            receipt.observation["node_result"]["usage"]
        )
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        await runtime.apply(GraphAction(K.STOP))
        assert evaluations == ["1"]

    run(scenario())


def test_read_only_role_cannot_write_even_with_a_well_formed_action():
    async def scenario():
        executor, model, world = build(
            [tool("increment", amount=9), tool("read"), complete("unchanged")]
        )
        result = await executor.execute(request(role="verifier"))
        assert world.count == 0
        assert [action.name for action, _ in world.calls] == ["read"]
        assert result.usage.model_calls == 3
        assert result.usage.tool_calls == 1
        assert result.metadata["turns"][0]["observation"]["public_value"] == {
            "error": "role_tool_forbidden"
        }
        instructions = json.loads(model.requests[0][0])["action_instructions"]
        assert "increment" not in str(instructions)

    run(scenario())


@pytest.mark.parametrize("bad", ["", "not an action", '{"kind":', tool("increment", amount="many")])
def test_malformed_or_empty_model_output_is_observed_failure_not_infrastructure(bad):
    async def scenario():
        executor, _, world = build([bad], max_turns=1)
        result = await executor.execute(request())
        assert result.metadata["termination"] == "turn-limit"
        assert not result.metadata["completed"]
        assert not world.calls
        assert result.usage.tool_calls == 0
        assert result.usage.model_calls == 1
        assert result.metadata["turns"][0]["raw_text"] == bad
        assert not executor.poisoned
        if bad in {"", "not an action", '{"kind":'}:
            assert result.output == bad

    run(scenario())


def test_bound_skill_read_is_advice_and_has_real_read_cost_without_world_step():
    async def scenario():
        executor, _, world = build([skill(), complete("used advice")])
        result = await executor.execute(request(skills=(("repair", "Use the public read tool."),)))
        assert result.output == "used advice"
        assert not world.calls
        assert result.usage.tool_calls == 1
        assert executor.environment_step_index == 0
        observation = result.metadata["turns"][0]["observation"]
        assert observation["public_value"]["content"] == "Use the public read tool."
        assert observation["public_value"]["environment_action_executed"] is False

    run(scenario())


def test_unbound_skill_and_bad_skill_arguments_cannot_run():
    async def scenario():
        executor, _, world = build([skill("other"), skill("repair", bad=True), complete("")])
        result = await executor.execute(request(skills=(("repair", "Advice"),)))
        assert result.output == ""
        assert result.metadata["completed"]
        assert result.usage.tool_calls == 0
        assert not world.calls

    run(scenario())


def test_local_model_budget_stops_with_last_observation_without_faking_completion():
    async def scenario():
        executor, model, world = build([tool("increment", amount=1), complete("must not run")])
        result = await executor.execute(request(cap=replace(CAP, model_calls=1, agent_turns=1)))
        assert len(model.requests) == 1
        assert world.count == 1
        assert result.output == canonical_json({"count": 1})
        assert result.metadata["termination"] == "budget-limit"
        assert not result.metadata["completed"]

    run(scenario())


def test_tool_budget_denial_is_observation_and_does_not_touch_world():
    async def scenario():
        executor, _, world = build([tool("increment", amount=1), complete("no tool budget")])
        result = await executor.execute(request(cap=replace(CAP, tool_calls=0)))
        assert not world.calls
        assert result.usage.tool_calls == 0
        assert result.metadata["turns"][0]["observation"]["public_value"] == {
            "error": "node_tool_budget_exhausted"
        }

    run(scenario())


def test_environment_terminal_does_not_call_grader_or_expose_submission_payload():
    async def scenario():
        world = World()
        world.terminal_at = 1
        executor, model, _ = build(
            [tool("increment", amount=1), tool("read"), complete("done")], world=world
        )
        result = await executor.execute(request())
        assert world.count == 1
        assert len(world.calls) == 1
        assert executor.environment_terminal
        assert result.metadata["turns"][1]["observation"]["public_value"] == {
            "error": "environment_already_terminal"
        }
        assert "never-in-a-model-prompt" not in str(model.requests)
        assert "never-in-a-model-prompt" not in str(result.metadata)
        assert result.metadata["task_success"] is None

    run(scenario())


def test_measured_environment_failure_remains_a_costed_public_observation():
    async def scenario():
        world = World()
        world.fail = EnvironmentMethodFailedError(
            budget_usage=TOOL_ACTUAL, public_error_code="native-refused"
        )
        executor, _, _ = build([tool("increment", amount=1), complete("tool refused")], world=world)
        result = await executor.execute(request())
        assert world.count == 1
        assert result.usage.tool_calls == 1
        assert result.metadata["turns"][0]["observation"]["observation_status"] == "tool_error"
        assert not executor.poisoned

    run(scenario())


def test_unknown_environment_failure_poisoned_and_does_not_reset_side_effect():
    async def scenario():
        world = World()
        world.fail = RuntimeError("unknown usage after write")
        executor, model, _ = build([tool("increment", amount=1)], world=world)
        with pytest.raises(RuntimeError, match="unknown usage"):
            await executor.execute(request())
        assert world.count == 1
        assert executor.poisoned
        with pytest.raises(ToolExecutorPoisonedError):
            await executor.execute(request(2))
        assert len(model.requests) == 1

    run(scenario())


def test_model_overcharge_is_infrastructure_error_never_underreported():
    async def scenario():
        executor, _, _ = build([complete("answer")])
        with pytest.raises(RuntimeError, match="usage exceeds"):
            await executor.execute(request(cap=replace(CAP, input_tokens=10)))
        assert executor.poisoned

    run(scenario())


def test_duplicate_invocation_and_cross_team_environment_reuse_rejected():
    async def scenario():
        executor, model, _ = build([complete("one")])
        await executor.execute(request())
        with pytest.raises(ValueError, match="twice"):
            await executor.execute(request())
        with pytest.raises(ValueError, match="different teams"):
            await executor.execute(replace(request(2), runtime_id="other-team"))
        assert len(model.requests) == 1

    run(scenario())


def test_role_permission_configuration_must_be_explicit_and_valid():
    model, world = Model([]), World()
    with pytest.raises(ValueError, match="unavailable"):
        FrozenToolExecutor(
            model, world, surface(), role_permissions={"solver": (("world", "invented"),)}
        )
    executor, model, _ = build([complete("no call")], permissions={"verifier": ()})
    with pytest.raises(ValueError, match="lacks"):
        run(executor.execute(request()))
    assert not model.requests


def test_default_phases_yield_after_eight_real_steps_and_stop_world_at_twelve_phases():
    async def scenario():
        model = Model(
            [tool("increment", amount=1)] * 96 + [tool("increment", amount=1), complete("96")]
        )
        world = World()
        executor = FrozenToolExecutor(
            model,
            world,
            surface(),
            role_permissions={"solver": (("world", "increment"),)},
            max_turns=20,
            tool_call_maximum=TOOL_MAXIMUM,
            clock=lambda: 0.0,
        )
        generous = CAP.scale(10)
        for phase in range(12):
            result = await executor.execute(request(phase + 1, cap=generous))
            assert len(result.metadata["turns"]) == 8
            assert result.usage.tool_calls == 8
            assert result.usage.model_calls == 8
            assert not result.metadata["completed"]
            assert result.metadata["execution_outcome"]["status"] == "yielded"
            assert result.metadata["execution_outcome"]["answer_present"] is False
            assert world.count == (phase + 1) * 8
            state = executor.public_environment_state()
            assert state["phase_index"] == phase + 1
            assert state["steps_in_phase"] == 0
            assert state["remaining_steps"] == 96 - world.count
            assert state["last_outcome"]["status"] == "success"
        assert result.metadata["termination"] == "environment-phase-budget"
        assert executor.phase_budget_exhausted
        assert not executor.environment_terminal
        assert state["phase_budget_exhausted"]
        final = await executor.execute(request(13, cap=generous))
        assert final.output == "96"
        assert final.metadata["completed"]
        assert final.usage.tool_calls == 0
        assert world.count == 96
        assert len(world.calls) == 96
        assert final.metadata["turns"][0]["observation"]["public_value"] == {
            "error": "environment_phase_budget_exhausted"
        }
        assert "validate_completion" not in executor.public_environment_state()

    run(scenario())


def test_phase_boundary_is_global_across_nodes_and_skill_reads_do_not_advance_it():
    async def scenario():
        executor, model, world = build(
            [tool("increment", amount=1)] * 4 + [skill()] + [tool("increment", amount=1)] * 4,
            max_turns=5,
        )
        first = await executor.execute(request(cap=CAP.scale(3), skills=(("repair", "advice"),)))
        assert len(first.metadata["turns"]) == 5
        assert first.metadata["termination"] == "turn-limit"
        assert first.metadata["execution_outcome"]["status"] == "yielded"
        assert executor.environment_step_index == 4
        second = await executor.execute(request(2, node="n1", cap=CAP.scale(3)))
        assert len(second.metadata["turns"]) == 4
        assert second.metadata["termination"] == "environment-phase-boundary"
        assert second.metadata["environment"]["phase_index"] == 1
        assert world.count == 8
        assert len(model.requests) == 9

    run(scenario())


def test_phase_protocol_can_be_explicitly_disabled_and_configuration_is_pinned():
    model = Model([tool("increment", amount=1)] * 9)
    world = World()
    executor = FrozenToolExecutor(
        model,
        world,
        surface(),
        role_permissions={"solver": (("world", "increment"),)},
        max_turns=9,
        phase_steps=None,
        max_phases=None,
        tool_call_maximum=TOOL_MAXIMUM,
        clock=lambda: 0.0,
    )
    result = run(executor.execute(request(cap=CAP.scale(3))))
    assert result.usage.tool_calls == 9
    assert executor.public_environment_state()["enabled"] is False
    assert executor.public_environment_state()["remaining_steps"] is None
    assert result.metadata["execution_outcome"]["status"] == "yielded"
    enabled = FrozenToolExecutor(
        Model([]),
        World(),
        surface(),
        role_permissions={"solver": (("world", "increment"),)},
        max_turns=9,
        tool_call_maximum=TOOL_MAXIMUM,
        clock=lambda: 0.0,
    )
    assert enabled.frozen_identity != executor.frozen_identity


def test_measured_environment_error_advances_dispatch_phase_without_claiming_success():
    async def scenario():
        world = World()
        world.fail = EnvironmentMethodFailedError(
            public_error_code="public_failure", budget_usage=TOOL_ACTUAL
        )
        executor = FrozenToolExecutor(
            Model([tool("increment", amount=1)]),
            world,
            surface(),
            role_permissions={"solver": (("world", "increment"),)},
            phase_steps=1,
            max_phases=1,
            tool_call_maximum=TOOL_MAXIMUM,
            clock=lambda: 0.0,
        )
        result = await executor.execute(request())
        assert world.count == 1
        assert result.usage.tool_calls == 1
        assert result.metadata["execution_outcome"]["status"] == "failed"
        assert result.metadata["environment"]["last_outcome"]["status"] == "tool_error"
        assert result.metadata["task_success"] is None
        assert executor.phase_budget_exhausted
        assert not executor.environment_terminal

    run(scenario())


@pytest.mark.parametrize(("steps", "phases"), [(None, 12), (8, None), (0, 12), (8, 0), (True, 12)])
def test_invalid_phase_configuration_rejected(steps, phases):
    with pytest.raises(ValueError, match="phase_steps/max_phases"):
        FrozenToolExecutor(
            Model([]),
            World(),
            surface(),
            role_permissions={"solver": ()},
            phase_steps=steps,
            max_phases=phases,
        )


def test_default_local_turn_count_is_resource_bound_and_can_reach_full_paper_phase():
    model = Model([tool("increment", amount=1)] * 8)
    world = World()
    executor = FrozenToolExecutor(
        model,
        world,
        surface(),
        role_permissions={"solver": (("world", "increment"),)},
        tool_call_maximum=TOOL_MAXIMUM,
        clock=lambda: 0.0,
    )
    result = run(executor.execute(request(cap=CAP.scale(3))))
    assert len(result.metadata["turns"]) == 8
    assert result.metadata["termination"] == "environment-phase-boundary"
    assert world.count == 8
    assert not result.metadata["completed"]


def test_default_local_turn_count_yields_on_real_resource_exhaustion():
    model = Model([tool("increment", amount=1)] * 6)
    world = World()
    executor = FrozenToolExecutor(
        model,
        world,
        surface(),
        role_permissions={"solver": (("world", "increment"),)},
        tool_call_maximum=TOOL_MAXIMUM,
        clock=lambda: 0.0,
    )
    result = run(executor.execute(request()))
    assert result.usage.model_calls == 6
    assert result.metadata["termination"] == "budget-limit"
    assert result.metadata["execution_outcome"]["status"] == "yielded"
    assert world.count == 6


def test_unknown_external_failure_is_not_counted_as_an_observed_completed_step():
    world = World()
    world.fail = OSError("unknown after mutation")
    executor, _, _ = build([tool("increment", amount=1)], world=world)
    with pytest.raises(OSError):
        run(executor.execute(request()))
    state = executor.public_environment_state()
    assert world.count == 1
    assert state["dispatches_started"] == 1
    assert state["steps_completed"] == 0
    assert state["last_outcome"] is None
    assert state["poisoned"]
