"""Read-only reset evidence carried by a committed native reward, not model input."""

from __future__ import annotations

from .canonical import canonical_json, normalize_json

RESET_BINDING_FORMAT = "training-reset-binding@1"
STATIC_RESET_KIND = "static-no-environment-reset"
ALFWORLD_RESET_KIND = "alfworld-observed-reset"
_STATIC_DOMAINS = frozenset(
    {"hotpotqa", "triviaqa", "aime-2026", "healthbench", "mbpp-plus", "humaneval"}
)


def reset_binding_comparison(value: object, *, benchmark: str) -> tuple[str, str] | None:
    """Return full canonical equality material and a short kind, never a new ID.

    This reads committed observations; it neither performs a reset nor upgrades
    missing/partial historical evidence to a matched initial state.
    """
    if not isinstance(value, dict) or value.get("format") != RESET_BINDING_FORMAT:
        return None
    public = value.get("public_task")
    if not isinstance(public, dict) or not isinstance(public.get("query"), str):
        return None
    if "public_context" not in public or "task_id" in public:
        return None
    kind = value.get("kind")
    reset = value.get("observed_reset")
    if kind == STATIC_RESET_KIND:
        if benchmark not in _STATIC_DOMAINS or reset is not None:
            return None
    elif kind == ALFWORLD_RESET_KIND:
        if benchmark != "alfworld" or not isinstance(reset, dict):
            return None
        if any(
            not isinstance(reset.get(key), str) or not reset[key]
            for key in ("game_id", "instruction_text", "observation_text")
        ):
            return None
        if type(reset.get("seed")) is not int or type(reset.get("max_steps")) is not int:
            return None
        if not isinstance(reset.get("admissible_commands"), list):
            return None
    else:
        return None
    return canonical_json(normalize_json(value)), kind
