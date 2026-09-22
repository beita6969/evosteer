"""Generation-time boundary for one strict authoring JSON object.

Frozen-base skill authoring is a single sampled completion.  This module does
not extract, repair, or reinterpret a completed response: it merely recognizes
the first token boundary at which the generated prefix is already one valid
JSON object (with optional surrounding JSON whitespace).  The backend stops
there, before sampling another token that could turn an otherwise valid object
into commentary.
"""

from __future__ import annotations

import json
from typing import Final

AUTHORING_JSON_ROOT_BOUNDARY_VERSION: Final = "authoring-json-root@1"


def authoring_json_root_is_complete(text: str) -> bool:
    """Return whether *text* is exactly one completed JSON object.

    The recognizer intentionally accepts only an object root because the
    authoring contract requires ``{"drafts": ...}``.  It never slices the
    input: a token that contains non-whitespace after the closing brace leaves
    the completion unrecognized and the normal strict schema path rejects it.
    """

    if not isinstance(text, str):
        raise TypeError("authoring JSON boundary requires text")
    start = _first_non_whitespace(text)
    if start is None or text[start] != "{":
        return False
    root_end = _first_object_end(text, start=start)
    if root_end is None or any(not character.isspace() for character in text[root_end:]):
        return False
    try:
        value, parsed_end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and parsed_end == root_end


def _first_non_whitespace(text: str) -> int | None:
    for index, character in enumerate(text):
        if not character.isspace():
            return index
    return None


def _first_object_end(text: str, *, start: int) -> int | None:
    """Locate the first lexically balanced root without altering *text*."""

    expected_closers: list[str] = ["}"]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            expected_closers.append("}")
        elif character == "[":
            expected_closers.append("]")
        elif character in "}]":
            if not expected_closers or character != expected_closers[-1]:
                return None
            expected_closers.pop()
            if not expected_closers:
                return index + 1
    return None


__all__ = [
    "AUTHORING_JSON_ROOT_BOUNDARY_VERSION",
    "authoring_json_root_is_complete",
]
