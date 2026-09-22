"""Exact token decoding and the structured JSON action wire codec."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol, cast

from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.action_text import action_payload
from skillev.contracts.action_wire import NATIVE_TOOL_WIRES
from skillev.runtime import (
    ActionKind,
    ActionParseResult,
    ActionParseStatus,
    StructuredAction,
)

from .errors import RolloutBoundaryError
from .generator import RolloutTokenizerProtocol


@dataclass(frozen=True, slots=True)
class DecodedSegment:
    text: str
    token_ids: tuple[int, ...]


def decode_reasoning_segment(
    tokenizer: RolloutTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> DecodedSegment:
    """Decode reasoning exactly; it is condition text rather than a scored span."""

    if not isinstance(token_ids, tuple) or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise RolloutBoundaryError("reasoning token IDs are invalid")
    text = tokenizer.decode(token_ids)
    if not isinstance(text, str):
        raise RolloutBoundaryError("tokenizer.decode must return text")
    return DecodedSegment(text=text, token_ids=token_ids)


def decode_action_segment(
    tokenizer: RolloutTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> DecodedSegment:
    """Project a sampled action span to text without retokenizing it.

    Tokenizer decoding is generally many-to-one: distinct sampled token spans
    can decode to the same text.  The sampled IDs therefore remain authoritative
    for scoring and training, while this deterministic decode supplies the text
    used by the action parser, environment, and trajectory history.
    """

    if not isinstance(token_ids, tuple) or not token_ids:
        raise RolloutBoundaryError("action token IDs must be non-empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise RolloutBoundaryError("action token IDs are invalid")
    text = tokenizer.decode(token_ids)
    if not isinstance(text, str) or not text:
        raise RolloutBoundaryError("decoded action text must be non-empty")
    return DecodedSegment(text=text, token_ids=token_ids)


class ActionCodec(Protocol):
    @property
    def format_version(self) -> str: ...

    def parse(self, action_text: str) -> ActionParseResult: ...


class StructuredJsonActionCodec:
    """Parse one explicit action, including harmless transport wrappers.

    V3 retains v1/v2 objects, an optional Action header, and one enclosing JSON
    fence. It never searches prose for a candidate or changes arguments. The
    rollout keeps the original text and sampled token IDs for TTB scoring.
    """

    format_version = "structured-action-json@3"

    def parse(self, action_text: str) -> ActionParseResult:
        if not isinstance(action_text, str):
            raise ValueError("action_text must be text")
        try:
            raw = cast(object, json.loads(action_payload(action_text)))
        except (json.JSONDecodeError, TypeError, ValueError):
            return ActionParseResult(
                status=ActionParseStatus.PARSE_ERROR,
                action=None,
                public_error_code="action_not_json",
            )
        try:
            normalized = normalize_json(raw)
            if not isinstance(normalized, dict):
                raise ValueError("structured action must be an object")
            action = _structured_action_from_wire(normalized)
            if action.kind is ActionKind.TOOL and action.resource_id == "webshop":
                # The same explicit operation aliases accepted by the native
                # evaluation adapter; never translate or choose its target.
                name = {"点击": "click", "搜索": "search"}.get(action.name, action.name)
                action = replace(action, name=name)
        except (KeyError, TypeError, ValueError):
            return ActionParseResult(
                status=ActionParseStatus.SCHEMA_INVALID,
                action=None,
                public_error_code="action_schema_invalid",
            )
        return ActionParseResult(
            status=ActionParseStatus.VALID,
            action=action,
            public_error_code=None,
        )


def _structured_action_from_wire(value: dict[str, JsonValue]) -> StructuredAction:
    """Canonicalize one exact v1 or v2 action object."""

    fields = set(value)
    if fields == {"arguments", "kind", "name", "resource_id", "skill_id"}:
        return StructuredAction.from_value(value)
    raw_kind = value.get("kind")
    if type(raw_kind) is not str:
        raise TypeError("action kind must be text")
    kind = ActionKind(raw_kind)
    expected = {
        ActionKind.TOOL: {"arguments", "kind", "name", "resource_id"},
        ActionKind.SKILL: {"arguments", "kind", "name", "resource_id", "skill_id"},
        ActionKind.COMPLETE: {"arguments", "kind", "name"},
    }[kind]
    if fields != expected:
        raise ValueError("v2 action has an incompatible field set")
    name = value["name"]
    if type(name) is not str:
        raise TypeError("action name must be text")
    arguments = normalize_json(value["arguments"])
    resource_id = value.get("resource_id")
    skill_id = value.get("skill_id")
    if resource_id is not None and type(resource_id) is not str:
        raise TypeError("action resource_id must be text")
    if skill_id is not None and type(skill_id) is not str:
        raise TypeError("action skill_id must be text")
    return StructuredAction(
        kind=kind,
        name=name,
        arguments=arguments,
        resource_id=resource_id,
        skill_id=skill_id,
    )


__all__ = [
    "ActionCodec",
    "ActionParseResult",
    "ActionParseStatus",
    "DecodedSegment",
    "StructuredJsonActionCodec",
    "decode_action_segment",
    "decode_reasoning_segment",
]


def codec_for_initial_meta(meta: Mapping[str, JsonValue]) -> ActionCodec:
    """Rebuild the declared representation for metrics and restored evidence."""
    wire = meta.get("action_wire", "structured-action-json@3")
    if wire == "structured-action-json@3":
        return StructuredJsonActionCodec()
    if not isinstance(wire, str) or wire not in NATIVE_TOOL_WIRES:
        raise ValueError("unsupported persisted action wire")
    from .action_contract import ActionContract
    from .action_surface import ActionSurface
    from .native_wire import NativeToolWire

    contract = meta.get("action_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("surface"), dict):
        raise ValueError("native wire metadata lacks its public contract")
    retrieved, active = contract.get("retrieved_skill_ids"), contract.get("active_skill_ids")
    if (
        not isinstance(retrieved, list)
        or not isinstance(active, list)
        or any(not isinstance(sid, str) for sid in [*retrieved, *active])
    ):
        raise ValueError("native contract lacks its exact skill identity")
    return NativeToolWire(
        ActionContract.freeze(
            ActionSurface.from_value(contract["surface"]),
            retrieved_skill_ids=tuple(cast(list[str], retrieved)),
            active_skill_ids=tuple(cast(list[str], active)),
        ),
        format_version=wire,
    )
