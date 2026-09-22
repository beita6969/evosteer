"""A unique standalone API call needs no redundant Action: label."""

import json

import pytest

from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("click[Next >]", "click[Next >]"),
        ("`click[Next >]`", "click[Next >]"),
        ("点击[Next >]", "click[Next >]"),
        ('click("Next >")', "click[Next >]"),
        ("search[blue mug]", "search[blue mug]"),
    ],
)
def test_discussion_preserves_one_standalone_literal_call(call, expected):
    surface = PublicSurface("webshop", 3, ("click[Next >]", "search"))
    decision = ExplicitDecision("I will inspect the public page.\n\n" + call, 3)
    assert normalize_decision(decision, surface).action == expected


@pytest.mark.parametrize(
    "text",
    [
        "I could click[Next >], but have not decided.",
        "Example:\nclick[Next >]",
        "Do not execute this:\nclick[Next >]",
        "Example:\n```text\nclick[Next >]\n```",
        "The two possibilities are:\nclick[Next >]\nclick[Absent]",
        "The two possibilities are:\nclick[Absent]\nclick[Next >]",
        "click[Next >]\nAction: click[Next >]",
        "I could choose this:\nclick[Next >]\nI have not decided.",
        'I will use this literal label:\nclick["Next >"]',
    ],
)
def test_discussion_does_not_choose_a_call_or_rewrite_its_literal_target(text):
    surface = PublicSurface("webshop", 3, ("click[Next >]",))
    assert normalize_decision(ExplicitDecision(text, 3), surface).action is None


def test_standalone_call_is_not_selected_by_whether_its_target_is_available():
    text = "I will inspect a public item.\n\nclick[Absent]"
    surface = PublicSurface("webshop", 3, ("click[Next >]",))
    rejected = normalize_decision(ExplicitDecision(text, 3), surface)
    assert rejected.action is None
    other_surface = PublicSurface("webshop", 3, ("click[Absent]",))
    assert normalize_decision(ExplicitDecision(text, 3), other_surface).action == "click[Absent]"


def test_standalone_public_search_uses_the_literal_query():
    surface = PublicSurface("trivia-search", 3, ("search",))
    decision = ExplicitDecision("I will retrieve public documents.\n\nsearch[blue mug]", 3)
    assert normalize_decision(decision, surface).action == "search[blue mug]"


@pytest.mark.parametrize("indent", [None, 2])
def test_one_standalone_json_call_after_prose_preserves_literal_arguments(indent):
    call = json.dumps(
        {
            "kind": "tool",
            "resource_id": "webshop",
            "name": "search",
            "arguments": {"query": "Blue Mug"},
        },
        indent=indent,
    )
    decision = ExplicitDecision("I will query the catalogue.\n" + call, 3)
    surface = PublicSurface("webshop", 3, ("search",))
    assert normalize_decision(decision, surface).action == "search[Blue Mug]"


@pytest.mark.parametrize(
    "wrap",
    [
        lambda c: "Example:\n" + c,
        lambda c: "Do not execute this:\n" + c,
        lambda c: "Discussion:\n```json\n" + c + "\n```",
        lambda c: "Discussion:\n" + c + "\n" + c,
        lambda c: "Discussion:\n" + c + " trailing prose",
        lambda c: "Discussion:\n" + c + "\nI have not decided.",
        lambda c: "click[Absent]\n" + c,
        lambda c: c + "\nclick[Absent]",
    ],
)
def test_json_in_discussion_is_not_selected_from_multiple_or_unsubmitted_calls(wrap):
    call = json.dumps(
        {
            "kind": "tool",
            "resource_id": "webshop",
            "name": "click",
            "arguments": {"target": "Red Mug"},
        }
    )
    surface = PublicSurface("webshop", 3, ("click[Red Mug]",))
    assert normalize_decision(ExplicitDecision(wrap(call), 3), surface).action is None
