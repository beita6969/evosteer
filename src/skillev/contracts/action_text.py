"""The shared action-wire transport projection; never a scored-token rewrite."""

from __future__ import annotations

import re


def action_payload(text: str) -> str:
    """Unwrap only the optional Action header and one whole JSON fence.

    Live execution and persisted invocation admission must read the same action.
    Do not search prose, remove trailing content, repair JSON or change arguments.
    Original text/token IDs remain authoritative for trajectory scoring.
    """
    payload = text.strip()
    if payload.startswith("Action:"):
        payload = payload.removeprefix("Action:").strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", payload, re.I | re.S)
    return fence[1] if fence is not None else payload
