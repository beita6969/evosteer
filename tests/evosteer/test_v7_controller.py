"""Controller view: each node output once, head/tail caps, first value change, instruction."""

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
from skillev.rollout.evosteer import (
    CONTROLLER_OUTPUT_HEAD_CHARS,
    CONTROLLER_OUTPUT_TAIL_CHARS,
    HISTORY_OUTPUT_REF,
    LIVE_OUTPUT_REF,
    actor_projection,
    collect_episode,
    controller_projection,
    elide_output,
)
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetVector

FILLER = "reasoning step; " * 400  # 6400 characters, like a long AIME completion


def long_output(node_id, execution_index):
    return f"HEAD-{node_id}-{execution_index}|{FILLER}|TAIL-{node_id}-{execution_index}"


class LongOutputExecutor:
    """Every execution returns a distinct long text with identifiable head and tail."""

    frozen_identity = "frozen-long-output-executor"

    def __init__(self):
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return NodeExecutionResult(
            long_output(request.node_id, request.execution_index),
            BudgetVector(input_tokens=10, output_tokens=2, model_calls=1, agent_turns=1),
        )


def runtime_fixture():
    maximum = BudgetVector(
        input_tokens=50, output_tokens=20, model_calls=1, agent_turns=1, wall_time_milliseconds=10
    )
    ledger = BudgetLedger(
        run_id="controller-run",
        attempt_id="controller-attempt",
        cap=BudgetVector(
            input_tokens=100_000_000,
            output_tokens=1_000_000,
            model_calls=100,
            agent_turns=100,
            wall_time_milliseconds=100_000_000,
        ),
    )
    executor = LongOutputExecutor()
    runtime = GraphRuntime(
        task_prompt="PUBLIC-TASK-PROMPT",
        roles=(
            RoleSpec("solver", "Solve the public task.", maximum),
            RoleSpec("verifier", "Check the available draft.", maximum),
        ),
        skills={},
        executor=executor,
        ledger=ledger,
        evaluator=lambda output: 1.0,
        runtime_id="controller-INTERNAL-runtime",
    )
    return runtime, ledger, executor


def add(node_id, role_id):
    return GraphAction(GraphActionKind.ADD_AGENT, node_id=node_id, role_id=role_id)


def edge(source_id, target_id, protocol):
    return GraphAction(
        GraphActionKind.ADD_EDGE, source_id=source_id, target_id=target_id, protocol=protocol
    )


async def apply_all(runtime, actions):
    for action in actions:
        await runtime.apply(action)


def observations(view):
    return [event["observation"] for event in view["history"]]


def test_elided_output_keeps_exact_head_and_tail_with_an_explicit_marker():
    limit = CONTROLLER_OUTPUT_HEAD_CHARS + CONTROLLER_OUTPUT_TAIL_CHARS
    assert elide_output("x" * limit) == "x" * limit
    assert elide_output("") == ""
    text = "h" * CONTROLLER_OUTPUT_HEAD_CHARS + "MIDDLE" + "t" * CONTROLLER_OUTPUT_TAIL_CHARS
    assert elide_output(text) == (
        "h" * CONTROLLER_OUTPUT_HEAD_CHARS
        + "[... 6 characters omitted ...]"
        + "t" * CONTROLLER_OUTPUT_TAIL_CHARS
    )
    long = long_output("n0", 1)
    shown = elide_output(long)
    assert shown.startswith(long[:CONTROLLER_OUTPUT_HEAD_CHARS])
    assert shown.endswith(long[-CONTROLLER_OUTPUT_TAIL_CHARS:])
    omitted = len(long) - CONTROLLER_OUTPUT_HEAD_CHARS - CONTROLLER_OUTPUT_TAIL_CHARS
    assert f"[... {omitted} characters omitted ...]" in shown
    with pytest.raises(TypeError):
        elide_output(None)


def test_each_output_version_is_rendered_once_and_every_copy_names_its_location():
    runtime, _, _ = runtime_fixture()
    asyncio.run(
        apply_all(
            runtime,
            (
                add("n0", "solver"),  # execution 1: n0 v1
                add("n1", "verifier"),  # execution 2: n1 v1
                edge("n1", "n0", "feedback"),  # carries n1 v1
                GraphAction(GraphActionKind.RERUN_AGENT, node_id="n0"),  # execution 3: n0 v2
                edge("n0", "n1", "revise"),  # carries n0 v2; execution 4: n1 v2
                GraphAction(GraphActionKind.SET_OUTPUT, node_id="n0"),
            ),
        )
    )
    state = runtime.public_state()
    before = canonical_json(state)
    view = controller_projection(state)
    assert canonical_json(state) == before  # the full public state is not modified
    text = canonical_json(view)
    # The full state repeats outputs; the view shows each produced version once.
    for node_id, index in (("n0", 1), ("n1", 2), ("n0", 3), ("n1", 4)):
        assert before.count(f"HEAD-{node_id}-{index}|") >= 2
        assert text.count(f"HEAD-{node_id}-{index}|") == 1
        assert text.count(f"|TAIL-{node_id}-{index}") == 1
    assert FILLER not in text
    nodes = {node["node_id"]: node for node in view["graph"]["nodes"]}
    assert nodes["n0"]["output"] == elide_output(long_output("n0", 3))
    assert nodes["n1"]["output"] == elide_output(long_output("n1", 4))
    first, second, feedback, rerun, revise, selected = observations(view)
    # Overwritten versions exist only in the event that executed them.
    assert first["node_result"]["output"] == elide_output(long_output("n0", 1))
    assert second["node_result"]["output"] == elide_output(long_output("n1", 2))
    assert "output_ref" not in first["node_result"]
    # Live versions are referenced, with usage and metadata still visible.
    for event, node_id, version in ((rerun, "n0", 2), (revise, "n1", 2)):
        assert (event["node_id"], event["output_version"]) == (node_id, version)
        assert event["node_result"]["output_ref"] == LIVE_OUTPUT_REF
        assert "output" not in event["node_result"]
        assert set(event["node_result"]) == {"output_ref", "usage", "metadata"}
    assert feedback["delivered_message"]["body_ref"] == HISTORY_OUTPUT_REF
    assert feedback["delivered_message"]["source_output_version"] == 1
    assert revise["delivered_message"]["body_ref"] == LIVE_OUTPUT_REF
    assert revise["delivered_message"]["source_output_version"] == 2
    assert all("body" not in event["delivered_message"] for event in (feedback, revise))
    assert selected == {"selected_output_node": "n0"}
    # Operational IDs are still removed exactly as by the actor projection.
    assert "INTERNAL" not in text
    assert "reservation_id" not in text
    assert "node_request" not in text


def test_dropped_node_keeps_its_only_rendering_in_history():
    runtime, _, _ = runtime_fixture()
    asyncio.run(
        apply_all(
            runtime,
            (
                add("n0", "solver"),
                add("n1", "verifier"),
                edge("n0", "n1", "feedback"),
                GraphAction(GraphActionKind.SET_OUTPUT, node_id="n1"),
                GraphAction(GraphActionKind.DROP_AGENT, node_id="n0"),
            ),
        )
    )
    view = controller_projection(runtime.public_state())
    assert [node["node_id"] for node in view["graph"]["nodes"]] == ["n1"]
    first, second, feedback, _, dropped = observations(view)
    assert first["node_result"]["output"] == elide_output(long_output("n0", 1))
    assert second["node_result"]["output_ref"] == LIVE_OUTPUT_REF
    assert feedback["delivered_message"]["body_ref"] == HISTORY_OUTPUT_REF
    assert dropped == {"dropped_node": "n0"}
    assert canonical_json(view).count("HEAD-n0-1|") == 1


def test_short_outputs_are_unchanged_and_unmatched_bodies_stay_inline():
    state = {
        "graph": {
            "nodes": [{"node_id": "n0", "role_id": "solver", "output": "4", "output_version": 1}],
            "edges": [],
            "output_node_id": None,
        },
        "history": [
            {"observation": {"node_id": "n0", "output_version": 1, "node_result": {"output": "4"}}},
            {
                "observation": {
                    "delivered_message": {
                        "source_id": "n9",
                        "source_output_version": 1,
                        "body": "not rendered anywhere else",
                    }
                }
            },
        ],
    }
    view = controller_projection(state)
    assert view["graph"]["nodes"][0]["output"] == "4"
    assert view["history"][0]["observation"]["node_result"] == {"output_ref": LIVE_OUTPUT_REF}
    assert view["history"][1]["observation"]["delivered_message"]["body"] == (
        "not rendered anywhere else"
    )
    # A mismatched live text is never replaced by a reference.
    changed = json.loads(canonical_json(state))
    changed["history"][0]["observation"]["node_result"]["output"] = "5"
    assert controller_projection(changed)["history"][0]["observation"]["node_result"] == {
        "output": "5"
    }
    with pytest.raises(ValueError, match="full public graph state"):
        controller_projection({"history": []})
    with pytest.raises(ValueError, match="full public graph state"):
        controller_projection([])


def test_projection_is_deterministic_and_independent_of_serialization():
    runtime, _, _ = runtime_fixture()
    asyncio.run(apply_all(runtime, (add("n0", "solver"), add("n1", "verifier"))))
    state = runtime.public_state()
    first = canonical_json(controller_projection(state))
    assert canonical_json(controller_projection(state)) == first
    assert canonical_json(controller_projection(json.loads(canonical_json(state)))) == first
    assert canonical_json(controller_projection(runtime.public_state())) == first


class ScriptedPolicy:
    """Exact action text paths; stochastic choices follow a fixed public script."""

    actor_id = "scripted-actor"
    reference_id = "scripted-reference"
    context_window = 10_000_000
    max_action_tokens = 4096

    def __init__(self, script):
        self.script = [canonical_json(action.to_value()) for action in script]
        self.prompts = []

    def menu(self, actions):
        texts = tuple(canonical_json(action) for action in actions)
        return SimpleNamespace(
            actions=texts,
            paths=tuple((*tuple(text.encode()), 256) for text in texts),
        )

    def encode_prompt(self, text):
        self.prompts.append(text)
        return tuple(text.encode())

    def sample(self, prompt_ids, menu, *, reference, seed):
        return menu.paths[menu.actions.index(self.script.pop(0))]


def collect(runtime, ledger, policy, value_function):
    return asyncio.run(
        collect_episode(
            policy=policy,
            runtime=runtime,
            ledger=ledger,
            task=EvoTask("controller-task", "math", "PUBLIC-TASK-PROMPT"),
            sample_id="current-INTERNAL-sample",
            batch_id="batch",
            source="current",
            menu_id="menu",
            value_snapshot_id="value",
            statistics_context="statistics",
            value_function=value_function,
            seed=3,
        )
    )


def rendered_prompts(trajectory):
    return [json.loads(bytes(record.prompt_ids).decode()) for record in trajectory.decisions]


def test_recorded_state_stays_full_while_each_prompt_shows_outputs_once():
    runtime, ledger, executor = runtime_fixture()
    script = (
        add("n0", "solver"),
        add("n1", "verifier"),
        GraphAction(GraphActionKind.RERUN_AGENT, node_id="n0"),
        GraphAction(GraphActionKind.SET_OUTPUT, node_id="n0"),
        GraphAction(GraphActionKind.STOP),
    )
    policy = ScriptedPolicy(script)
    trajectory = collect(runtime, ledger, policy, lambda features, family: 0.5)
    assert [json.loads(r.action_json) for r in trajectory.decisions] == [
        action.to_value() for action in script
    ]
    # The executor and the evaluator still receive complete texts.
    assert executor.requests[2].node_id == "n0"
    assert executor.requests[2].previous_output == long_output("n0", 1)
    assert trajectory.output == long_output("n0", 3)
    for record, rendered in zip(trajectory.decisions, rendered_prompts(trajectory), strict=True):
        # The recorded state is the full public state; the prompt is its projection.
        state = json.loads(record.state_json)
        assert rendered["execution"] == controller_projection(state)
        assert FILLER not in canonical_json(rendered)
    last_state = json.loads(trajectory.decisions[-1].state_json)
    assert last_state["history"][0]["observation"]["node_result"]["output"] == long_output("n0", 1)
    assert last_state["history"][0]["observation"]["node_request"]["task_prompt"]
    assert last_state["graph"]["nodes"][0]["output"] == long_output("n0", 3)
    last_prompt = policy.prompts[-1]
    # n0 v2 and n1 v1 in graph.nodes, the overwritten n0 v1 in history: three
    # capped renderings, where the old prompt repeated five full outputs.
    assert last_prompt.count("characters omitted ...]") == 3
    old_prompt = canonical_json(
        {**json.loads(last_prompt), "execution": actor_projection(last_state)}
    )
    assert old_prompt.count(FILLER) == 5
    assert len(last_prompt) < len(old_prompt) // 3


def test_first_decision_value_change_is_zero_then_tracks_the_previous_estimate():
    runtime, ledger, _ = runtime_fixture()
    script = (
        add("n0", "solver"),
        GraphAction(GraphActionKind.RERUN_AGENT, node_id="n0"),
        GraphAction(GraphActionKind.SET_OUTPUT, node_id="n0"),
        GraphAction(GraphActionKind.STOP),
    )

    def value(features, family):
        # Feature 4 counts executed actions, so the estimate moves every step.
        assert family == "math"
        return 0.2 + 0.5 * features[4]

    trajectory = collect(runtime, ledger, ScriptedPolicy(script), value)
    prompts = rendered_prompts(trajectory)
    values = [record.value_estimate for record in trajectory.decisions]
    assert values[0] == pytest.approx(0.2)
    assert [prompt["reference_value"] for prompt in prompts] == values
    assert prompts[0]["reference_value_change"] == 0.0
    for index in range(1, len(prompts)):
        assert prompts[index]["reference_value_change"] == pytest.approx(
            values[index] - values[index - 1]
        )
    assert prompts[1]["reference_value_change"] == pytest.approx(0.25)


def test_orchestrator_instruction_states_action_effects_without_strategy():
    pytest.importorskip("torch")
    from tests.evosteer.test_application import application

    policy = application(candidate=False).policy
    text = canonical_json({"task_family": "synthetic", "legal_actions": [{"kind": "STOP"}]})
    ids = policy.encode_prompt(text)
    assert policy.encode_prompt(text) == ids
    messages = json.loads(policy.tokenizer.decode(ids))
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == text
    system = messages[0]["content"]
    # The prompt names what each action does and never when to use it: team
    # composition is left to training.
    for advice in ("then stop", "select the output node", "use it when", "should", "repair"):
        assert advice not in system, advice
    for phrase in (
        "ADD_AGENT adds a node with a role and optional skill and executes it",
        "a node whose role is not primary_role_id also receives the current draft",
        "RERUN_AGENT executes a node again with its previous output and inbound messages",
        "ADD_EDGE feedback delivers the source output at the target's next execution",
        "ADD_EDGE revise executes the target at once",
        "BIND_SKILL adds a skill to a node and executes it again",
        "DROP_AGENT removes a node",
        "SET_OUTPUT selects the node whose output is scored",
        "STOP ends the episode and scores that output",
        "reference_value_change",
        "output_ref",
    ):
        assert phrase in system, phrase
    assert system.endswith("Return only the action; no separate reasoning text.")
    assert policy.configuration_id == application(candidate=False).policy.configuration_id
