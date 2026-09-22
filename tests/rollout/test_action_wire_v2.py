from __future__ import annotations

import json

import pytest

from skillev.rollout import ActionParseStatus, StructuredJsonActionCodec
from skillev.runtime import ActionKind


@pytest.mark.parametrize(
    ("wire", "kind"),
    [
        (
            {"arguments": {"value": {"answer": "Paris"}}, "kind": "complete", "name": "complete"},
            ActionKind.COMPLETE,
        ),
        (
            {
                "arguments": {"query": "mouse"},
                "kind": "tool",
                "name": "search",
                "resource_id": "webshop",
            },
            ActionKind.TOOL,
        ),
        (
            {
                "arguments": {},
                "kind": "skill",
                "name": "invoke",
                "resource_id": "skill-runtime",
                "skill_id": "skill-1",
            },
            ActionKind.SKILL,
        ),
    ],
)
def test_v2_discriminated_wire_canonicalizes_without_null_fields(
    wire: dict[str, object], kind: ActionKind
) -> None:
    result = StructuredJsonActionCodec().parse(json.dumps(wire))
    assert result.status is ActionParseStatus.VALID
    assert result.action is not None
    assert result.action.kind is kind


def test_v1_historical_wire_remains_readable() -> None:
    wire = {
        "arguments": {"value": {"answer": "Paris"}},
        "kind": "complete",
        "name": "complete",
        "resource_id": None,
        "skill_id": None,
    }
    assert StructuredJsonActionCodec().parse(json.dumps(wire)).status is ActionParseStatus.VALID


@pytest.mark.parametrize(
    "wire",
    [
        {"arguments": {}, "kind": "complete", "name": "complete", "resource_id": None},
        {"arguments": {}, "kind": "tool", "name": "x"},
        {"arguments": {}, "kind": "complete", "name": "complete", "extra": True},
    ],
)
def test_v2_rejects_mixed_or_extra_field_sets(wire: dict[str, object]) -> None:
    assert (
        StructuredJsonActionCodec().parse(json.dumps(wire)).status
        is ActionParseStatus.SCHEMA_INVALID
    )


def test_codec_does_not_extract_json_from_prose() -> None:
    result = StructuredJsonActionCodec().parse(
        'answer: {"arguments":{},"kind":"complete","name":"complete"}'
    )
    assert result.status is ActionParseStatus.PARSE_ERROR


@pytest.mark.parametrize(
    "wrapper", ["{}", "Action: {}", "```json\n{}\n```", "Action:\n```\n{}\n```"]
)
def test_action_transport_wrappers_preserve_the_explicit_action(wrapper) -> None:
    wire = {"arguments": {"value": {"answer": "text"}}, "kind": "complete", "name": "complete"}
    codec = StructuredJsonActionCodec()
    action = json.dumps(wire)
    assert codec.parse(wrapper.format(action)).action == codec.parse(action).action


@pytest.mark.parametrize(("name", "canonical"), [("点击", "click"), ("搜索", "search")])
def test_webshop_operation_alias_never_changes_arguments_or_another_resource(
    name, canonical
) -> None:
    wire = {
        "arguments": {"target": "点击 me"},
        "kind": "tool",
        "name": name,
        "resource_id": "webshop",
    }
    codec = StructuredJsonActionCodec()
    action = codec.parse(json.dumps(wire)).action
    assert action is not None
    assert action.name == canonical
    assert action.arguments == wire["arguments"]
    other = codec.parse(json.dumps({**wire, "resource_id": "another-environment"})).action
    assert other is not None
    assert other.name == name


@pytest.mark.parametrize(
    "wrapper", ["{}\n{}", "```json\n{}\n{}\n```", "```json\n{}\n```\nExplanation"]
)
def test_action_codec_does_not_choose_between_actions_or_drop_trailing_content(wrapper) -> None:
    action = '{"arguments":{},"kind":"complete","name":"complete"}'
    result = StructuredJsonActionCodec().parse(wrapper.format(action, action))
    assert result.action is None
