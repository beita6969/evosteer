"""Owner-designated payloads, distinct from discussion and legacy whole-text parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .integer_payload import INTEGER_SCALAR
from .integer_payload import parse_explicit_integer_payload as parse_explicit_integer_payload
from .native_tool_calls import native_tool_call
from .python_payload import PYTHON_CARRIER_VERSION
from .response_syntax import protocol_lines
from .step0_completion import StepZeroTerminalMode, project_terminal_candidate


@dataclass(frozen=True, slots=True)
class FinalSubmission:
    owner_id: str
    message_id: str
    raw_response: str
    payload: str
    payload_type: str
    projection_id: str


_FINAL_LABEL = r"(?:Final(?: answer| response| code)?|Terminal payload)"


def _short_answer_header(
    line: str, label_pattern: str = _FINAL_LABEL
) -> tuple[int, int | None] | None:
    """Recognize Markdown around a declared field, not bold text elsewhere."""
    prefix = r"^[ \t]*(?:\#{1,6}[ \t]+)?"
    plain = re.match(prefix + label_pattern + r"[ \t]*:[ \t]*", line, re.I)
    if plain:
        return plain.end(), None
    for emphasis in (r"\*\*", "__", r"\*", "_"):
        label = re.match(
            prefix + emphasis + label_pattern + rf"[ \t]*(?::{emphasis}|{emphasis}:)[ \t]*",
            line,
            re.I,
        )
        if label:
            return label.end(), None
        whole = re.fullmatch(
            prefix
            + emphasis
            + label_pattern
            + r"[ \t]*:[ \t]*(?P<body>.*?)"
            + emphasis
            + r"[ \t]*(?:\r?\n)?",
            line,
            re.I,
        )
        if whole:
            return whole.start("body"), whole.end("body")
    return None


def _final_fields(
    text: str, *, markdown: bool = False, label_pattern: str = _FINAL_LABEL
) -> tuple[str, ...]:
    """Only top-level final headers delimit a payload, never strings inside code."""
    offsets: list[tuple[int, int, int | None]] = []
    for offset, line in protocol_lines(text):
        if markdown:
            bounds = _short_answer_header(line, label_pattern)
            if bounds is not None:
                start, end = bounds
                offsets.append((offset, offset + start, offset + end if end is not None else None))
            continue
        header = re.match(r"^" + label_pattern + r":[ \t]*", line, re.I)
        if header:
            offsets.append((offset, offset + header.end(), None))
    return tuple(
        text[
            start : end
            if end is not None
            else offsets[index + 1][0]
            if index + 1 < len(offsets)
            else len(text)
        ]
        for index, (_, start, end) in enumerate(offsets)
    )


def _short_answer_fields(text: str) -> tuple[str, ...]:
    # Specific final declarations take precedence over intermediate Answer:
    # labels. This is a public field convention, not reference-based selection.
    for label in (_FINAL_LABEL, r"Short answer", r"Answer"):
        fields = _final_fields(text, markdown=True, label_pattern=label)
        if fields:
            return fields
    return ()


def _declared_short_answer(fields: tuple[str, ...]) -> str | None:
    # As with the existing Final answer: wire, the first nonempty field line
    # is the short answer; following explanation is not part of that line.
    values = {field.strip().splitlines()[0].strip() if field.strip() else None for field in fields}
    return next(iter(values)) if len(values) == 1 and None not in values else None


def _python_payload(payload: str) -> str | None:
    # The same answer-blind decoder applies to direct and enveloped source.
    # Backticks inside Python strings are data, not a second program.
    return project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, payload)


def _aime_field_value(payload: str) -> int | None:
    """Read the declared scalar field, not its following explanatory prose.

    Adjacent scalar lines still belong to the declaration and must agree.
    Once prose begins, intermediate quantities in that explanation are not new
    final declarations. A subsequent explicit final header is parsed separately.
    """
    complete = parse_explicit_integer_payload(payload)
    if complete is not None:
        return complete
    values = set()
    while payload.strip():
        leading = re.match(rf"\s*({INTEGER_SCALAR}[ \t]*\.?)[ \t]*(?:\r?\n|$)", payload)
        if leading is None:
            if re.match(r"\s*(?:[0-9+-]|\\boxed|\b(?:or|and)\b)", payload, re.I):
                return None
            break
        values.add(parse_explicit_integer_payload(leading[1]))
        payload = payload[leading.end() :]
    return next(iter(values)) if len(values) == 1 and None not in values else None


def project_owner_final(
    mode: StepZeroTerminalMode, text: str, *, owner_id: str, message_id: str
) -> FinalSubmission | None:
    if not text.strip():
        return None
    try:
        native = native_tool_call(text)
    except ValueError:
        return None
    if native is not None:
        if native.name != "submit_answer" or set(native.arguments) != {"answer"}:
            return None
        payload = native.arguments["answer"]
        if mode is StepZeroTerminalMode.AIME_INTEGER:
            integer = parse_explicit_integer_payload(payload)
            projected = rf"\boxed{{{integer}}}" if integer is not None else None
        elif mode is StepZeroTerminalMode.PYTHON_SOURCE:
            projected = _python_payload(payload)
        elif mode is StepZeroTerminalMode.SHORT_ANSWER:
            fields = _short_answer_fields(payload)
            projected = _declared_short_answer(fields) if fields else payload.strip() or None
        else:
            projected = project_terminal_candidate(mode, payload)
        if projected is None:
            return None
        return FinalSubmission(
            owner_id,
            message_id,
            text,
            projected,
            mode.value,
            "owner-native-integer-projection@3"
            if mode is StepZeroTerminalMode.AIME_INTEGER
            else "owner-native-" + PYTHON_CARRIER_VERSION
            if mode is StepZeroTerminalMode.PYTHON_SOURCE
            else "owner-native-short-answer-fields@5"
            if mode is StepZeroTerminalMode.SHORT_ANSWER
            else "owner-native-tool-projection@1",
        )
    # A clinical/natural-language response remains the entire owner's response.
    fields = (
        ()
        if mode is StepZeroTerminalMode.NATURAL_LANGUAGE
        else _short_answer_fields(text)
        if mode is StepZeroTerminalMode.SHORT_ANSWER
        else _final_fields(
            text,
            markdown=mode is StepZeroTerminalMode.PYTHON_SOURCE,
        )
    )
    if fields and mode is StepZeroTerminalMode.AIME_INTEGER:
        values = {_aime_field_value(payload) for payload in fields}
        integer = next(iter(values)) if len(values) == 1 and None not in values else None
        projected = rf"\boxed{{{integer}}}" if integer is not None else None
        projection_id = "owner-declared-integer-field@5"
    elif fields and mode is StepZeroTerminalMode.SHORT_ANSWER:
        projected = _declared_short_answer(fields)
        projection_id = "owner-explicit-final-projection@5"
    elif len(fields) > 1 and mode is not StepZeroTerminalMode.PYTHON_SOURCE:
        return None
    elif not fields:
        # The top-level scanner already found no short-answer declaration.
        # Do not let the legacy regex extract an Answer: example from code.
        projected = (
            text.strip()
            if mode is StepZeroTerminalMode.SHORT_ANSWER
            else project_terminal_candidate(mode, text)
        )
        projection_id = (
            "owner-direct-" + PYTHON_CARRIER_VERSION
            if mode is StepZeroTerminalMode.PYTHON_SOURCE
            else "owner-unlabelled-short-answer@5"
            if mode is StepZeroTerminalMode.SHORT_ANSWER
            else "owner-legacy-compatible-projection@4"
            if mode is StepZeroTerminalMode.AIME_INTEGER
            else "owner-legacy-compatible-projection@3"
        )
    else:
        # Code follows the fixed final-block/final-declaration convention,
        # never a correctness-dependent choice between the owner's snippets.
        payload = fields[-1]
        projection_id = (
            "owner-declared-" + PYTHON_CARRIER_VERSION
            if mode is StepZeroTerminalMode.PYTHON_SOURCE
            else "owner-explicit-final-projection@2"
        )
        if mode is StepZeroTerminalMode.PYTHON_SOURCE:
            projected = _python_payload(payload)
        else:
            projected = project_terminal_candidate(mode, payload)
    if projected is None:
        return None
    return FinalSubmission(owner_id, message_id, text, projected, mode.value, projection_id)
