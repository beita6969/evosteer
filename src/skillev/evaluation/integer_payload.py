"""Answer-blind scalar decoding shared by every AIME submission envelope."""

from __future__ import annotations

import re

INTEGER_SCALAR = r"(?:[0-9]{1,3}|\\boxed\{\s*[0-9]{1,3}\s*\})"
INTEGER_PAYLOAD = rf"{INTEGER_SCALAR}[ \t]*\.?(?:\s+{INTEGER_SCALAR}[ \t]*\.?)*"
_ALTERNATIVE_SCALARS = (
    rf"{INTEGER_SCALAR}[ \t]*\.?"
    rf"(?:\s*(?:,|;|/|\bor\b|\band\b|和|或|、)\s*{INTEGER_SCALAR}[ \t]*\.?)+"
)


def is_integer_payload(payload: str) -> bool:
    """Recognize scalar notation, including conflicting repeated declarations."""
    return re.fullmatch(INTEGER_PAYLOAD, payload.strip()) is not None


def parse_explicit_integer_payload(payload: str) -> int | None:
    """Decode one value, not a choice among values; punctuation is transport only."""
    if not is_integer_payload(payload):
        return None
    values = {int(number) for number in re.findall(r"[0-9]+", payload)}
    return next(iter(values)) if len(values) == 1 else None


def is_integer_alternatives(payload: str) -> bool:
    """A scalar list must not disappear into the legacy boxed-answer fallback."""
    return re.fullmatch(_ALTERNATIVE_SCALARS, payload.strip(), re.I) is not None
