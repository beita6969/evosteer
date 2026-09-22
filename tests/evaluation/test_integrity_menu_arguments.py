"""The model may copy one public API call into its corresponding literal argument."""

import json

import pytest

from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from tests.evaluation.test_integrity_native_tool_calls import call


@pytest.mark.parametrize(
    "representation",
    [
        lambda name, key, value: call(name, **{key: value}),
        lambda name, key, value: f"{name}({key}={value!r})",
        lambda name, key, value: json.dumps(
            {"kind": "tool", "resource_id": "webshop", "name": name, "arguments": {key: value}}
        ),
    ],
)
@pytest.mark.parametrize(
    ("name", "key", "argument", "expected"),
    [
        ("click", "target", "click[Next >]", "click[Next >]"),
        ("click", "target", "点击[Next >]", "click[Next >]"),
        ("search", "query", "search[Blue Mug]", "search[Blue Mug]"),
    ],
)
def test_one_same_operation_wrapper_preserves_the_complete_literal_call(
    representation, name, key, argument, expected
):
    surface = PublicSurface("webshop", 2, ("click[Next >]", "search"))
    result = normalize_decision(ExplicitDecision(representation(name, key, argument), 2), surface)
    assert result.action == expected
    assert result.error is None


@pytest.mark.parametrize(
    "argument",
    [
        "click[Absent]",
        "click[next >]",
        'click["Next >"]',
        "search[Next >]",
        "click[click[Next >]]",
        "click[Next >]\nclick[Other]",
        "I could click[Next >]",
    ],
)
def test_transport_does_not_select_repair_or_infer_the_argument(argument):
    surface = PublicSurface("webshop", 2, ("click[Next >]", "click[Other]"))
    assert (
        normalize_decision(ExplicitDecision(call("click", target=argument), 2), surface).action
        is None
    )


def test_copied_menu_call_still_uses_the_current_state_and_operation():
    text = call("click", target="click[Next >]")
    current = PublicSurface("webshop", 3, ("click[Next >]",))
    assert normalize_decision(ExplicitDecision(text, 2), current).action is None
    assert (
        normalize_decision(
            ExplicitDecision(call("search", query="search[Blue Mug]"), 3), current
        ).action
        is None
    )
