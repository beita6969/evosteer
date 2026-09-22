"""Historical message parsing and current control-error feedback; no live peer calls."""

from __future__ import annotations

import json
import re
from typing import Any

from .native_tool_calls import native_control_payload, native_tool_call

# Historical wire values remain readable; they are not live capabilities.
PEERS = ("solver", "researcher")


def control_payload(text: str) -> dict[str, Any] | None:
    """Decode explicit control headers without putting prose inside JSON strings.

    Everything after the header newline is the verbatim body, including LaTeX,
    source indentation, quotes and newlines. Ordinary answers are not searched
    for embedded commands. JSON remains supported for existing callers.
    """
    raw = text.lstrip()
    header = re.match(
        r"(?:Message to ([a-zA-Z][a-zA-Z0-9_-]*)|Review):[ \t]*(?:\r?\n)?", raw, flags=re.I
    )
    if header is not None:
        body = raw[header.end() :]
        if not body.strip():
            raise ValueError("a control header requires a message body")
        return (
            {"kind": "message", "recipient": header[1].casefold(), "body": body}
            if header[1] is not None
            else {"kind": "review", "body": body}
        )
    # The explicit outer message owns all following text. Native-call examples
    # inside its prose/code body are data, not a competing submission channel.
    native = native_control_payload(text)
    if native is not None:
        return native
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        # This happened in real AIME requests: math backslashes/newlines broke
        # JSON, and an intended peer message was silently treated as a final.
        if re.match(r'^\s*\{\s*"kind"\s*:\s*"(?:message|review)"', text):
            raise ValueError("declared control message has invalid JSON encoding") from error
        return None
    return value if isinstance(value, dict) else None


def peer_request(text: str) -> tuple[str, str] | None:
    value = control_payload(text)
    if value is None or value.get("kind") != "message":
        return None
    recipient, body = value.get("recipient"), value.get("body")
    if recipient not in PEERS or not isinstance(body, str) or not body.strip():
        raise ValueError("message recipient or body is invalid")
    return recipient, body


def control_failure_feedback(
    text: str, *, direct_submission: str, finish_reason: str, action_mode: str | None = None
) -> str:
    """Describe the attempted protocol, without redirecting an answer to a peer."""
    try:
        native = native_tool_call(text) is not None
    except ValueError:
        native = True  # An explicit but malformed native envelope was attempted.
    message = (
        "That native tool call was not decoded; no tool ran and no message or submission "
        "was accepted. Use an available function with complete arguments and closing tags. "
        if native
        else "That control message was not decoded or delivered. "
        "Use a complete control message from the available interface. "
    )
    if finish_reason == "length":
        message += "This generation ended at its output-token limit. "
    if native and action_mode == "alfworld":
        message += (
            "For household actions, use function act with parameter command containing "
            "your complete native command; command words are not separate function names. "
        )
    return message + f"You can also submit {direct_submission} directly."
