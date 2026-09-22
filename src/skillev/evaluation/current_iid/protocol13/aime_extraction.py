"""Strict, target-independent AIME final-answer extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class AIMEExtractionReason(StrEnum):
    BOXED = "boxed"
    EXPLICIT_FINAL = "explicit-final"
    FINAL_LINE = "final-line"
    EMPTY = "empty"
    CONFLICTING = "conflicting"
    INCOMPLETE_BOX = "incomplete-box"
    INVALID_INTEGER = "invalid-integer"
    OUT_OF_RANGE = "out-of-range"


@dataclass(frozen=True, slots=True)
class AIMEExtraction:
    value: int | None
    reason: AIMEExtractionReason


def all_complete_boxed_values(text: str) -> tuple[str, ...]:
    results: list[str] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return tuple(results)
        body_start = start + len(marker)
        depth = 1
        index = body_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[body_start : index - 1].strip())
            cursor = index
        else:
            cursor = body_start


def strict_aime_integer(value: str) -> int | None:
    cleaned = value.strip()
    if re.fullmatch(r"[0-9]{1,3}", cleaned) is None:
        return None
    integer = int(cleaned)
    return integer if 0 <= integer <= 999 else None


def extract_aime_boxed(text: str) -> AIMEExtraction:
    values = all_complete_boxed_values(text)
    if not values:
        return AIMEExtraction(
            None,
            AIMEExtractionReason.INCOMPLETE_BOX
            if r"\boxed{" in text
            else AIMEExtractionReason.EMPTY,
        )
    integer = strict_aime_integer(values[-1])
    if integer is None:
        reason = (
            AIMEExtractionReason.OUT_OF_RANGE
            if values[-1].strip().isdigit() and int(values[-1].strip()) > 999
            else AIMEExtractionReason.INVALID_INTEGER
        )
        return AIMEExtraction(None, reason)
    return AIMEExtraction(integer, AIMEExtractionReason.BOXED)


def extract_aime_explicit_final(text: str) -> AIMEExtraction:
    matches = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer)\s*[:=]\s*([0-9]{1,4})\s*$",
        text,
    )
    if not matches:
        return AIMEExtraction(None, AIMEExtractionReason.EMPTY)
    values = tuple(strict_aime_integer(item) for item in matches)
    valid = tuple(value for value in values if value is not None)
    if len(set(valid)) > 1:
        return AIMEExtraction(None, AIMEExtractionReason.CONFLICTING)
    if not valid:
        return AIMEExtraction(None, AIMEExtractionReason.OUT_OF_RANGE)
    return AIMEExtraction(valid[-1], AIMEExtractionReason.EXPLICIT_FINAL)


def extract_aime_final_line(text: str) -> AIMEExtraction:
    lines = tuple(item.strip() for item in text.splitlines() if item.strip())
    if not lines:
        return AIMEExtraction(None, AIMEExtractionReason.EMPTY)
    value = strict_aime_integer(lines[-1])
    return AIMEExtraction(
        value,
        AIMEExtractionReason.FINAL_LINE
        if value is not None
        else AIMEExtractionReason.INVALID_INTEGER,
    )


def extract_aime_source_final(text: str) -> AIMEExtraction:
    """Extract the source-parity final without scanning arbitrary reasoning numbers.

    The last complete boxed value has priority. If no complete box exists, only an
    explicit ``Final answer:``/``Answer:`` line is eligible. A malformed complete
    box remains a definitive invalid response rather than silently selecting a
    different number from the derivation.
    """

    boxed = all_complete_boxed_values(text)
    if boxed:
        value = strict_aime_integer(boxed[-1])
        if value is None:
            reason = (
                AIMEExtractionReason.OUT_OF_RANGE
                if boxed[-1].strip().isdigit() and int(boxed[-1].strip()) > 999
                else AIMEExtractionReason.INVALID_INTEGER
            )
            return AIMEExtraction(None, reason)
        return AIMEExtraction(value, AIMEExtractionReason.BOXED)
    return extract_aime_explicit_final(text)


__all__ = [
    "AIMEExtraction",
    "AIMEExtractionReason",
    "all_complete_boxed_values",
    "extract_aime_boxed",
    "extract_aime_explicit_final",
    "extract_aime_final_line",
    "extract_aime_source_final",
    "strict_aime_integer",
]
