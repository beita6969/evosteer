"""An explicit owner stop, independent of the simulator's score or success."""

from __future__ import annotations

import re

from .native_tool_calls import native_tool_call

OWNER_FINISH_PROFILE = "owner-finish-cooperative-deadline@1"
LEGACY_TERMINATION_PROFILE = "environment-terminal-hard-deadline@1"
OWNER_FINISH_INSTRUCTION = (
    "You may end the episode by calling finish() with no arguments, or by replying "
    "Finish on its own line as your entire submission. Use this when you decide "
    "to stop, including when you believe the task is complete. This stops at the "
    "current environment state; it does not claim success or change the score."
)


def owner_finish_requested(text: str) -> bool:
    """Recognize one literal stop carrier, never infer completion from prose."""
    if re.fullmatch(r"finish(?:\(\))?[.!]?", text.strip(), re.I):
        return True
    try:
        call = native_tool_call(text)
    except ValueError:
        return False
    return call is not None and call.name == "finish" and not call.arguments
