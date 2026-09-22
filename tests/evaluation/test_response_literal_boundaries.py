"""Protocol-looking data must survive communication and owner submission verbatim."""

import pytest

from skillev.evaluation.agent_communication import control_payload, peer_request
from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from skillev.evaluation.native_tool_calls import native_control_payload, native_tool_call
from skillev.evaluation.owner_final import project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode


def tool_call(name, arguments):
    return (
        f"<tool_call>\n<function={name}>\n"
        + "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
        + "</function>\n</tool_call>"
    )


@pytest.mark.parametrize("header", ["Message to solver:", "Review:"])
@pytest.mark.parametrize(
    "body",
    [
        "Explain this call, do not execute it:\n" + tool_call("click", {"target": "Blue Mug"}),
        "Compare these examples:\n"
        + tool_call("search", {"query": "blue mug"})
        + "\n"
        + tool_call("click", {"target": "Blue Mug"}),
        "Explain an incomplete envelope:\n<tool_call>\n<function=click>",
    ],
)
def test_outer_plain_message_owns_its_entire_body(header, body):
    text = header + "\n" + body
    assert control_payload(text)["body"] == body
    if header.startswith("Message"):
        assert peer_request(text) == ("solver", body)


SOURCES = [
    'def f():\n    return """\nFinal answer: not a submission\n"""',
    'def f():\n    return """\n<tool_call>\n<function=click>\n</function>\n</tool_call>\n"""',
    'def f():\n    return "literal ``` marker"',
    'def f():\n    return """\n```python\nFinal code: literal text\n```\n"""',
    'def f():\n    return f"""\nFinal answer: literal {3}\n"""',
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("wrapper", ["{}", "Final code:\n{}", "```python\n{}\n```"])
def test_python_literals_are_not_submission_boundaries(source, wrapper):
    response = wrapper.format(source)
    assert native_tool_call(response) is None
    assert control_payload(response) is None
    final = project_owner_final(
        StepZeroTerminalMode.PYTHON_SOURCE, response, owner_id="owner", message_id="synthetic"
    )
    assert final is not None
    assert final.payload == source
    assert final.raw_response == response


@pytest.mark.parametrize("wrapper", ["{}", "```python\n{}\n```"])
def test_native_submission_preserves_literal_backticks(wrapper):
    source = 'def f():\n    return "literal ``` marker"'
    response = tool_call("submit_answer", {"answer": wrapper.format(source)})
    final = project_owner_final(
        StepZeroTerminalMode.PYTHON_SOURCE, response, owner_id="owner", message_id="synthetic"
    )
    assert final is not None
    assert final.payload == source
    assert final.raw_response == response


def test_inline_quotes_do_not_hide_a_native_envelope_on_the_same_line():
    query = 'blue "mug"'
    response = (
        "<tool_call><function=search><parameter=query>"
        + query
        + "</parameter></function></tool_call>"
    )
    call = native_tool_call(response)
    assert call.name == "search"
    assert call.arguments["query"] == query


@pytest.mark.parametrize(
    ("name", "key", "argument", "action"),
    [
        ("点击", "target", "Blue Mug", "click[Blue Mug]"),
        ("搜索", "query", "蓝色杯子", "search[蓝色杯子]"),
    ],
)
def test_explicit_native_operation_aliases_preserve_arguments(name, key, argument, action):
    response = tool_call(name, {key: argument})
    assert native_control_payload(response) is None
    result = normalize_decision(
        ExplicitDecision(response, 0), PublicSurface("webshop", 0, ("search", "click[Blue Mug]"))
    )
    assert result.action == action


@pytest.mark.parametrize("optional", [{}, {"cursor": "0"}, {"limit": "4"}])
def test_native_history_uses_the_same_optional_defaults_as_plain_history(optional):
    payload = native_control_payload(tool_call("history", {"archive": "discussion", **optional}))
    assert payload["archive"] == "discussion"
    assert payload["cursor"] == 0
    assert payload["limit"] == 4
    definition = next(
        item["function"]
        for item in native_tool_definitions("completion")
        if item["function"]["name"] == "history"
    )
    assert "archive" in definition["parameters"]["required"]
    assert "cursor" not in definition["parameters"]["required"]
    assert "limit" not in definition["parameters"]["required"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"archive": "discussion", "cursor": "one"},
        {"archive": "discussion", "unexpected": "value"},
    ],
)
def test_invalid_native_history_is_still_not_executed(arguments):
    with pytest.raises(ValueError):
        native_control_payload(tool_call("history", arguments))
