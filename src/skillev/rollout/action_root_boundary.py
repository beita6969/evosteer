"""Token-boundary termination for one strict rollout action object."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

ACTION_JSON_ROOT_BOUNDARY_VERSION: Final = "action-json-root@1"


class _Decoder(Protocol):
    def decode(self, token_ids: tuple[int, ...]) -> str: ...


@dataclass(frozen=True, slots=True)
class ActionBoundaryResult:
    content_token_ids: tuple[int, ...]
    matched: bool


class ActionRootScanner:
    """Incremental lexical scan, with a strict decode only at a root candidate.

    Production fast tokenizers supply a UTF-8-aware DecodeStream. Other decoders
    retain the exact general prefix oracle; no prefix-stability assumption is
    imposed on arbitrary test/custom tokenizers.
    """

    def __init__(self, tokenizer: _Decoder) -> None:
        self.tokenizer = tokenizer
        factory = getattr(tokenizer, "new_decode_stream", None)
        self.decode_next: Callable[[int], str | None] | None = (
            None if factory is None else factory()
        )
        self.ids: list[int] = []
        self.stack: list[str] = []
        self.started = False
        self.in_string = False
        self.escaped = False
        self.invalid = False
        self.matched = False

    def push(self, token_id: int) -> bool:
        if type(token_id) is not int or token_id < 0:
            raise ValueError("action boundary token ID is invalid")
        if self.matched:
            return True
        self.ids.append(token_id)
        if self.invalid:
            return False
        if self.decode_next is None:
            self.matched = _action_json_root_is_complete(self.tokenizer.decode(tuple(self.ids)))
            return self.matched
        chunk = self.decode_next(token_id)
        if chunk is None:
            return False
        for character in chunk:
            if not self.started:
                if character.isspace():
                    continue
                if character != "{":
                    self.invalid = True
                    return False
                self.started = True
                self.stack.append("}")
            elif self.in_string:
                if self.escaped:
                    self.escaped = False
                elif character == "\\":
                    self.escaped = True
                elif character == '"':
                    self.in_string = False
            elif not self.stack:
                if not character.isspace():
                    self.invalid = True
                    return False
            elif character == '"':
                self.in_string = True
            elif character in "{[":
                self.stack.append("}" if character == "{" else "]")
            elif character in "}]":
                if character != self.stack.pop():
                    self.invalid = True
                    return False
        if self.started and not self.stack:
            # Never slice a token containing both a close brace and prose.
            self.matched = _action_json_root_is_complete(self.tokenizer.decode(tuple(self.ids)))
        return self.matched


def _action_json_root_is_complete(text: str) -> bool:
    if not isinstance(text, str):
        raise TypeError("action JSON boundary requires text")
    start = next((index for index, character in enumerate(text) if not character.isspace()), None)
    if start is None or text[start] != "{":
        return False
    try:
        value, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and not text[end:].strip()


def apply_action_json_root_boundary(
    tokenizer: _Decoder,
    token_ids: tuple[int, ...],
) -> ActionBoundaryResult:
    """Return the earliest token prefix that is exactly one JSON object.

    No text is sliced. Leading prose, an object plus trailing non-whitespace
    inside the same token, or an incomplete root remains untouched and is
    rejected later by the strict codec.
    """

    if not isinstance(token_ids, tuple) or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise ValueError("action boundary token IDs are invalid")
    scanner = ActionRootScanner(tokenizer)
    for end, token_id in enumerate(token_ids, 1):
        if scanner.push(token_id):
            return ActionBoundaryResult(content_token_ids=token_ids[:end], matched=True)
    return ActionBoundaryResult(content_token_ids=token_ids, matched=False)


__all__ = [
    "ACTION_JSON_ROOT_BOUNDARY_VERSION",
    "ActionBoundaryResult",
    "ActionRootScanner",
    "apply_action_json_root_boundary",
]
