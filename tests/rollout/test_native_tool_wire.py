"""Synthetic public examples, never licensed benchmark payloads."""

import json

import pytest

from skillev.rollout.action_contract import ActionContract
from skillev.rollout.action_surface import (
    ActionSurface,
    CompletionSpec,
    TerminalMode,
    ToolActionSpec,
)
from skillev.rollout.native_wire import NativeToolWire
from skillev.runtime import ActionKind, ActionParseStatus


def contract():
    return ActionContract.freeze(
        ActionSurface(
            tools=(ToolActionSpec("python", "execute", {"code": "string"}, {"code": "pass"}),),
            terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
            completion=CompletionSpec({"answer": "string"}, {"answer": "example"}),
        ),
        retrieved_skill_ids=("skill-a",),
        active_skill_ids=("skill-a",),
    )


def call(name, **arguments):
    params = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def test_exact_native_code_and_submission():
    wire = NativeToolWire(contract())
    code = 'def f():\n    return "中文\\n\\\\"\n'
    result = wire.parse(call("execute", code=code))
    assert result.action.arguments == {"code": code}
    submitted = wire.parse(call("submit_answer", answer="42"))
    assert submitted.action.kind is ActionKind.COMPLETE
    assert submitted.action.arguments == {"value": {"answer": "42"}}
    assert contract().validate(submitted.action, validate_completion=lambda value: True).completion
    assert json.dumps(contract().to_native_tools())


@pytest.mark.parametrize(
    "text",
    [
        "I think the answer is 42",
        call("submit_answer", answer="42") * 2,
        call("unknown", answer="42"),
        call("read_skill", skill_id="missing"),
        call("submit_answer", answer="42")[:-4],
        "Reasoning: 42\n" + call("submit_answer", answer="42"),
        '<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call>',
    ],
)
def test_no_repairs_or_candidate_selection(text):
    assert NativeToolWire(contract()).parse(text).status is not ActionParseStatus.VALID


def test_read_only_visible_skill():
    result = NativeToolWire(contract()).parse(call("read_skill", skill_id="skill-a"))
    assert result.action.skill_id == "skill-a"
    assert contract().invoked_skill_ids(result.action) == ("skill-a",)
