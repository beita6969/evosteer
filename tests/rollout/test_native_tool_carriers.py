"""Observed transport failure shapes, with synthetic answer-independent payloads."""

import json
from dataclasses import replace

import pytest

from skillev.contracts.action_wire import NATIVE_TOOL_CARRIER_WIRE
from skillev.rollout.action_contract import ActionContract
from skillev.rollout.action_surface import ToolActionSpec
from skillev.rollout.native_wire import NativeToolWire
from skillev.runtime import ActionKind, ActionParseStatus
from tests.rollout.test_native_tool_wire import call, contract


def wire():
    return NativeToolWire(contract(), format_version=NATIVE_TOOL_CARRIER_WIRE)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("My explanation.\n", ""),
        ("", "\nSubmitted."),
        ("A brief plan.\n", "\nThis is my submission."),
    ],
)
def test_one_explicit_call_is_not_invalidated_by_commentary(before, after):
    text = before + call("submit_answer", answer="example") + after
    result = wire().parse(text)
    assert result.status is ActionParseStatus.VALID
    assert result.action.arguments == {"value": {"answer": "example"}}
    assert NativeToolWire(contract()).parse(text).status is ActionParseStatus.PARSE_ERROR


@pytest.mark.parametrize("envelope", ["bare", "arguments", "parameters", "native-json", "fence"])
def test_explicit_json_carriers_preserve_code_and_meaning(envelope):
    code = 'def f():\n    return "中文\\n<think>"\n'
    args = {"answer": code}
    value = (
        args
        if envelope in {"bare", "fence"}
        else {
            "name": "submit_answer",
            "parameters" if envelope == "parameters" else "arguments": args,
        }
    )
    text = json.dumps(value, ensure_ascii=False)
    if envelope == "native-json":
        text = "<tool_call>" + text + "</tool_call>"
    elif envelope == "fence":
        text = "```json\n" + text + "\n```"
    result = wire().parse(text)
    assert result.status is ActionParseStatus.VALID
    assert result.action.kind is ActionKind.COMPLETE
    assert result.action.arguments == {"value": args}


def test_native_string_containing_reasoning_tags_is_not_a_channel():
    code = 'def f():\n    return "<think>literal</think>"\n'
    result = wire().parse(call("execute", code=code))
    assert result.action.arguments == {"code": code}


@pytest.mark.parametrize(
    "text",
    [
        call("submit_answer", answer="one") + call("submit_answer", answer="two"),
        "<think>draft\n" + call("submit_answer", answer="one"),
        "<think>" + call("submit_answer", answer="one") + "</think>",
        call("submit_answer", answer="one") + '\n{"answer":"two"}',
        '{"answer":"one","answer":"two"}',
        '{"answer":"one"}\n{"answer":"two"}',
        '{"name":"missing","arguments":{"answer":"one"}}',
        '{"action":"solve","answer":"one"}',
        "The answer might be one or two.",
        call("submit_answer", answer="one")[:-4],
    ],
)
def test_conflicts_incomplete_carriers_and_non_submissions_remain_failures(text):
    assert wire().parse(text).status is not ActionParseStatus.VALID


def test_argument_only_call_requires_unique_public_signature():
    base = contract()
    surface = replace(
        base.surface,
        tools=(
            *base.surface.tools,
            ToolActionSpec("second", "run", {"code": "string"}, {"code": "pass"}),
        ),
    )
    ambiguous = NativeToolWire(
        ActionContract.freeze(surface), format_version=NATIVE_TOOL_CARRIER_WIRE
    )
    assert ambiguous.parse('{"code":"pass"}').status is not ActionParseStatus.VALID
    result = ambiguous.parse('{"name":"run","arguments":{"code":"pass"}}')
    assert result.action.resource_id == "second"
    assert result.action.arguments == {"code": "pass"}


def test_json_skill_carrier_retains_visible_catalog_admission():
    assert wire().parse('{"skill_id":"missing"}').status is not ActionParseStatus.VALID
    assert wire().parse('{"skill_id":"skill-a"}').action.skill_id == "skill-a"


def test_unclosed_outer_parameter_does_not_swallow_a_second_submission():
    # The old greedy whole-function regex can swallow the second carrier into
    # the first argument. This actual failure shape is not two valid submissions.
    text = (
        "<tool_call>\n<function=submit_answer>\n<parameter=answer>\n"
        "unfinished draft\n</think>\n" + call("submit_answer", answer="synthetic final")
    )
    assert wire().parse(text).status is not ActionParseStatus.VALID


@pytest.mark.parametrize("text", ["I have not submitted an action.", '{"answer":'])
def test_non_json_text_is_still_a_parse_failure_not_relabelled_as_a_schema_error(text):
    assert wire().parse(text).status is ActionParseStatus.PARSE_ERROR


@pytest.mark.parametrize("label", ["json", "JSON", ""])
def test_one_explicit_json_fence_is_not_invalidated_by_commentary(label):
    value = {"answer": 'def f():\n    return "<tool_call>literal</tool_call>"\n'}
    text = "My submission follows.\n```" + label + "\n" + json.dumps(value) + "\n```\nDone."
    result = wire().parse(text)
    assert result.status is ActionParseStatus.VALID
    assert result.action.arguments == {"value": value}


def test_json_fence_inside_native_string_argument_is_data():
    code = 'def f():\n    return """\n```json\n{"answer":"literal"}\n```\n"""\n'
    result = wire().parse(call("execute", code=code))
    assert result.status is ActionParseStatus.VALID
    assert result.action.arguments == {"code": code}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("<think>draft\n", ""),
        ("", '\n```json\n{"answer":"second"}\n```'),
        ('{"answer":"first"}\n', ""),
        ("", "\n```python\nincomplete carrier"),
        ("", "\n" + call("submit_answer", answer="second")),
    ],
)
def test_json_fence_does_not_select_between_carriers_or_submit_reasoning(before, after):
    text = before + '```json\n{"answer":"synthetic"}\n```' + after
    assert wire().parse(text).status is not ActionParseStatus.VALID
