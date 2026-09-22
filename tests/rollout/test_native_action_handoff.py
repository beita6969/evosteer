"""Observed handoff failures, using synthetic task-independent inputs only."""

import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.contracts.action_wire import NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE
from skillev.policy.phase_context import (
    NATIVE_ACTION_INSTRUCTION,
    TYPED_PUBLIC_HISTORY,
    PhaseContextSpec,
    native_action_instruction,
    phase_chat_messages,
)
from skillev.rollout.action_contract import ActionContract
from skillev.rollout.action_surface import ActionSurface, TerminalMode, ToolActionSpec
from skillev.rollout.native_wire import NativeToolWire
from skillev.runtime import ActionParseStatus
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
    render_reasoning_prefix,
)
from tests.rollout.test_native_tool_wire import call, contract


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("interactive", [False, True])
def test_handoff_describes_only_actual_functions_and_keeps_reasoning_private_to_forward(
    typed, interactive
):
    surface = (
        ActionContract.freeze(
            ActionSurface(
                tools=(ToolActionSpec("world", "act", {"command": "string"}, {"command": "look"}),),
                terminal_mode=TerminalMode.ENVIRONMENT,
            ),
            retrieved_skill_ids=("skill-a",),
            active_skill_ids=("skill-a",),
        )
        if interactive
        else contract()
    )
    spec = PhaseContextSpec(
        action_wire=NATIVE_TOOL_HANDOFF_WIRE,
        tools_json=canonical_json(list(surface.to_native_tools())),
        reasoning_tool_catalog=True,
        history_format=TYPED_PUBLIC_HISTORY if typed else None,
        max_turns=8 if typed else None,
    )
    initial = spec.wrap("Synthetic public task\n")
    forward = render_forward_prefix_from_parts(initial, (), 1, "unique current draft")
    backward = render_hindsight_prefix_from_parts(initial, (), 1, "unique real observation")
    messages, tools = phase_chat_messages(forward.text)
    hindsight, backward_tools = phase_chat_messages(backward.text)
    reminder = native_action_instruction(spec)
    assert tools == backward_tools == json.loads(spec.tools_json)
    for function in (tool["function"] for tool in tools):
        assert function["name"] in reminder
        assert all(field in reminder for field in function["parameters"]["properties"])
    assert ("submit_answer" in reminder) is not interactive
    assert ("submit_answer" in messages[0]["content"]) is not interactive
    assert "skill_id" in reminder
    assert "not callable function names" in reminder
    assert "unique current draft" in json.dumps(messages)
    assert "unique current draft" not in json.dumps(hindsight)
    assert "unique real observation" in json.dumps(hindsight)
    assert reminder in messages[-1]["content"]
    reasoning, reasoning_tools = phase_chat_messages(render_reasoning_prefix(initial, (), 1).text)
    assert reasoning_tools is None
    assert reminder not in reasoning[0]["content"]
    assert native_action_instruction(replace(spec, action_wire=NATIVE_TOOL_CARRIER_WIRE)) == (
        NATIVE_ACTION_INSTRUCTION
    )


def test_braced_non_json_commentary_does_not_hide_one_real_call():
    text = '{(public note: a description) "not a JSON carrier"}\n' + call(
        "submit_answer", answer="synthetic response"
    )
    old = NativeToolWire(contract(), format_version=NATIVE_TOOL_CARRIER_WIRE)
    repaired = NativeToolWire(contract(), format_version=NATIVE_TOOL_HANDOFF_WIRE)
    assert old.parse(text).status is ActionParseStatus.PARSE_ERROR
    parsed = repaired.parse(text)
    assert parsed.status is ActionParseStatus.VALID
    assert parsed.action.arguments == {"value": {"answer": "synthetic response"}}


@pytest.mark.parametrize(
    "text",
    [
        "A complete conversational answer without a function call.",
        "Action: `go to room 1`",
        call("skill-a", skill_id="skill-a"),
        call("submit_answer", answer="one") + call("submit_answer", answer="two"),
        '{"answer":"first"}\n' + call("submit_answer", answer="second"),
        '{"answer":"unfinished\n' + call("submit_answer", answer="second"),
        call("submit_answer", answer="first") + '\n{"answer":"unfinished',
        "<think>draft\n" + call("submit_answer", answer="not submitted"),
        call("submit_answer", answer="unfinished")[:-4],
        '{"answer":"one","answer":"two"}',
    ],
)
def test_handoff_never_relabels_prose_guesses_function_names_or_selects_candidates(text):
    assert (
        NativeToolWire(contract(), format_version=NATIVE_TOOL_HANDOFF_WIRE).parse(text).status
        is not ActionParseStatus.VALID
    )


@pytest.mark.parametrize("fenced", [False, True])
def test_json_parameter_delimiters_are_data_not_extra_calls(fenced):
    value = {"answer": 'def f():\n    return "<tool_call>literal</tool_call>"\n'}
    text = json.dumps(value)
    if fenced:
        text = "A submission:\n```json\n" + text + "\n```\n"
    parsed = NativeToolWire(contract(), format_version=NATIVE_TOOL_HANDOFF_WIRE).parse(text)
    assert parsed.status is ActionParseStatus.VALID
    assert parsed.action.arguments == {"value": value}
