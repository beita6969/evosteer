"""Answer-blind, contract-specific parsers for direct benchmark responses."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from ..response_syntax import protocol_lines


class ParseStatus(StrEnum):
    EXTRACTED = "extracted"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"


class ParseReason(StrEnum):
    EXPLICIT_FINAL = "explicit-final"
    FINAL_LINE = "final-line"
    JSON_ANSWER_FIELD = "json-answer-field"
    CODE_FENCE = "code-fence"
    UNIFIED_DIFF = "unified-diff"
    BOXED_ANSWER = "boxed-answer"
    EMPTY_RESPONSE = "empty-response"
    CONFLICTING_FINALS = "conflicting-finals"
    INVALID_RANGE = "invalid-range"
    INVALID_FORMAT = "invalid-format"
    TRUNCATED_BEFORE_FINAL = "truncated-before-final"


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    value: str | None
    status: ParseStatus
    reason: ParseReason = ParseReason.INVALID_FORMAT
    candidate_count: int = 0


@dataclass(frozen=True, slots=True)
class ParsedVisibleReAct:
    """Public structured-memory fields plus the independently parsed native action."""

    memory: str | None
    thought: str | None
    action: ParsedResponse

    @property
    def memory_present(self) -> bool:
        return self.memory is not None


def canonical_native_action(action: str) -> str:
    """Normalize only explicit API verb aliases, never guess or change an argument.

    This boundary is shared by skill and no-skill native runners. In particular,
    a translated WebShop verb is not a different shopping decision.
    """

    match = re.fullmatch(r"(click|search|点击|搜索)\[([^\[\]\r\n]+)\]", action.strip(), re.I)
    if match is None:
        return action
    verb = {"点击": "click", "搜索": "search"}.get(match[1].casefold(), match[1].casefold())
    return f"{verb}[{match[2]}]"


_FINAL_MARKER = re.compile(r"(?im)^\s*(?:final answer|answer)\s*:\s*(.+?)\s*$")
_BOX_START = re.compile(r"\\boxed\s*\{")
_CHOICE_MARKER = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer|correct\s+answer)\s*(?:is|[:=])?\s*"
    r"\(?([A-D])\)?\s*[.]?\s*$"
)
_CHOICE_ONLY = re.compile(r"(?i)^\s*\(?([A-D])\)?\s*[.]?\s*$")
_AIME_EXPLICIT = re.compile(r"(?im)^\s*(?:final answer|answer)\s*[:=]\s*\$?([0-9]{1,4})\$?\s*$")
_AIME_FINAL_LINE = re.compile(r"^\s*\$?([0-9]{1,4})\$?\s*$")
_ACTION_MARKER = re.compile(r"(?im)^\s*action\s*:\s*(.+?)\s*$")
_REACT_SECTION = re.compile(r"(?im)^\s*(memory|thought|action)\s*:\s*")
_DIFF_HEADER = re.compile(r"(?m)^diff --git a/(.+?) b/(.+?)$")
_DIFF_FILE_HEADERS = re.compile(r"(?m)^--- .+\n\+\+\+ .+$")
_DIFF_HUNK_HEADER = re.compile(r"(?m)^@@ (?:-[^\n]+ \+[^\n]+ )?@@")
_FENCE_LINE = re.compile(r"\s*```\s*([^`]*)\s*")


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()


def _empty(reason: ParseReason = ParseReason.EMPTY_RESPONSE) -> ParsedResponse:
    return ParsedResponse(None, ParseStatus.EMPTY, reason, 0)


def _unique(candidates: list[str], reason: ParseReason) -> ParsedResponse:
    unique = tuple(dict.fromkeys(item.strip() for item in candidates if item.strip()))
    if not unique:
        return _empty()
    if len(unique) > 1:
        return ParsedResponse(
            None, ParseStatus.AMBIGUOUS, ParseReason.CONFLICTING_FINALS, len(unique)
        )
    return ParsedResponse(unique[0], ParseStatus.EXTRACTED, reason, 1)


def parse_short_answer(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    if not visible:
        return _empty()
    explicit = [match.group(1).strip() for match in _FINAL_MARKER.finditer(visible)]
    if explicit:
        return _unique(explicit, ParseReason.EXPLICIT_FINAL)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if len(lines) == 1:
        return ParsedResponse(lines[0], ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)
    return _empty(ParseReason.INVALID_FORMAT)


def parse_short_answer_last_line(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    explicit = [match.group(1).strip() for match in _FINAL_MARKER.finditer(visible)]
    if explicit:
        return _unique(explicit, ParseReason.EXPLICIT_FINAL)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if not lines:
        return _empty()
    return ParsedResponse(lines[-1], ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)


def parse_multiple_choice_label(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    candidates = [match.group(1).upper() for match in _CHOICE_MARKER.finditer(visible)]
    if candidates:
        return _unique(candidates, ParseReason.EXPLICIT_FINAL)
    lines = [line for line in visible.splitlines() if line.strip()]
    if lines and (match := _CHOICE_ONLY.fullmatch(lines[-1])):
        return ParsedResponse(
            match.group(1).upper(), ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1
        )
    return _empty(ParseReason.INVALID_FORMAT)


def parse_multiple_choice_label_v2(text: str) -> ParsedResponse:
    """Public final-field syntax, including Markdown and a single labelled option.

    The legacy parser remains available for historical conditions. No option
    text or reference answer is consulted to decide which label was submitted.
    """
    return _parse_multiple_choice_label(text, structured_explanations=False)


def parse_multiple_choice_label_v3(text: str) -> ParsedResponse:
    """Also accept final headings and a leading choice with an Explanation field.

    These are literal submission carriers, not searches through reasoning for a
    plausible answer. Conflicting declared choices remain invalid. Historical
    v2 callers keep their original behavior.
    """
    return _parse_multiple_choice_label(text, structured_explanations=True)


def _parse_multiple_choice_label(text: str, *, structured_explanations: bool) -> ParsedResponse:
    lines = [
        re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", line).strip()
        for _, line in protocol_lines(_strip_thinking(text))
        if line.strip()
    ]

    def label(value: str) -> str | None:
        named_option = re.match(r"^option\s+", value.strip(), re.I) is not None
        value = re.sub(r"^option\s+", "", value.strip(), flags=re.I)
        if match := _CHOICE_ONLY.fullmatch(value):
            return match[1].upper()
        if structured_explanations and named_option:
            if re.search(r"\b(?:or|and)\s+(?:option\s+)?[A-D]\b", value, re.I):
                return None
            if match := re.fullmatch(r"([A-D])\s+(?!or\b|and\b).+", value, re.I):
                return match[1].upper()
        # A single labelled option is an explicit submission, not a scan for
        # letters in a rationale. Lists of alternative options remain invalid.
        if len(re.findall(r"(?:^|\s)[A-D][.)]\s+", value, re.I)) != 1:
            return None
        match = re.fullmatch(r"([A-D])[.)]\s+(?!or\b|and\b).+", value, re.I)
        return match[1].upper() if match else None

    declarations: list[str | None] = []
    if (
        structured_explanations
        and len(lines) >= 2
        and re.match(r"^Explanation\s*:", lines[1], re.I)
    ):
        declarations.append(label(lines[0]))
    for index, line in enumerate(lines):
        match = re.match(
            r"^(?:\#{1,6}\s+)?(?:final\s+answer|answer|correct\s+answer)"
            r"\b\s*(?:is\b|[:=])?\s*(.*)$",
            line,
            re.I,
        )
        if match:
            value = match[1]
            if structured_explanations and not value:
                # A heading is not itself a conflicting answer. If its next
                # line explicitly names a choice, bind that line to the heading.
                following = lines[index + 1] if index + 1 < len(lines) else ""
                if not re.match(
                    r"^(?:option\s+[A-D]\b|\(?[A-D]\)?(?:[.)]|\s*$|\s+(?:or|and)\b))",
                    following,
                    re.I,
                ):
                    continue
                value = following
            declarations.append(label(value))
    if declarations:
        if any(value is None for value in declarations):
            return _empty(ParseReason.INVALID_FORMAT)
        return _unique(
            [value for value in declarations if value is not None], ParseReason.EXPLICIT_FINAL
        )
    if len(lines) == 1 and (value := label(lines[0])):
        return ParsedResponse(value, ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)
    if lines and (match := _CHOICE_ONLY.fullmatch(lines[-1])):
        return ParsedResponse(match[1].upper(), ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)
    return _empty(ParseReason.INVALID_FORMAT)


def parse_multiple_choice_json(text: str) -> ParsedResponse:
    visible = _strip_thinking(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)```", visible, flags=re.I | re.S)
    if fenced:
        visible = fenced.group(1).strip()
    try:
        value = json.loads(visible)
    except json.JSONDecodeError:
        return _empty(ParseReason.INVALID_FORMAT)
    if type(value) is not dict or set(value) != {"answer"}:
        return _empty(ParseReason.INVALID_FORMAT)
    answer = value["answer"]
    if answer not in {"A", "B", "C", "D"}:
        return _empty(ParseReason.INVALID_FORMAT)
    return ParsedResponse(answer, ParseStatus.EXTRACTED, ParseReason.JSON_ANSWER_FIELD, 1)


parse_multiple_choice = parse_multiple_choice_label


def parse_aime_integer(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    explicit = [match.group(1) for match in _AIME_EXPLICIT.finditer(visible)]
    unique = tuple(dict.fromkeys(explicit))
    if len(unique) > 1:
        return ParsedResponse(
            None, ParseStatus.AMBIGUOUS, ParseReason.CONFLICTING_FINALS, len(unique)
        )
    if len(unique) == 1:
        return _aime_value(unique[0], ParseReason.EXPLICIT_FINAL)
    lines = [line for line in visible.splitlines() if line.strip()]
    if not lines:
        return _empty()
    match = _AIME_FINAL_LINE.fullmatch(lines[-1])
    if match is None:
        return _empty(ParseReason.INVALID_FORMAT)
    return _aime_value(match.group(1), ParseReason.FINAL_LINE)


def parse_aime_boxed_integer(text: str) -> ParsedResponse:
    """Parse the last complete boxed AIME integer without float coercion."""

    visible = _strip_thinking(text)
    values: list[str] = []
    for start in _BOX_START.finditer(visible):
        depth = 1
        cursor = start.end()
        while cursor < len(visible) and depth:
            if visible[cursor] == "{":
                depth += 1
            elif visible[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            values.append(visible[start.end() : cursor - 1].strip())
    if not values:
        return _empty(ParseReason.INVALID_FORMAT)
    value = values[-1]
    if re.fullmatch(r"[0-9]{1,3}", value) is None:
        return _empty(ParseReason.INVALID_FORMAT)
    return ParsedResponse(str(int(value)), ParseStatus.EXTRACTED, ParseReason.BOXED_ANSWER, 1)


def parse_aime_source_integer(text: str) -> ParsedResponse:
    """Prefer the last complete box, otherwise require an explicit final-answer line."""

    visible = _strip_thinking(text)
    values: list[str] = []
    for start in _BOX_START.finditer(visible):
        depth = 1
        cursor = start.end()
        while cursor < len(visible) and depth:
            depth += (visible[cursor] == "{") - (visible[cursor] == "}")
            cursor += 1
        if depth == 0:
            values.append(visible[start.end() : cursor - 1].strip())
    if values:
        if re.fullmatch(r"[0-9]{1,3}", values[-1]) is None:
            return _empty(ParseReason.INVALID_FORMAT)
        return ParsedResponse(
            str(int(values[-1])), ParseStatus.EXTRACTED, ParseReason.BOXED_ANSWER, 1
        )
    explicit = [match.group(1) for match in _AIME_EXPLICIT.finditer(visible)]
    unique = tuple(dict.fromkeys(explicit))
    if len(unique) > 1:
        return ParsedResponse(
            None, ParseStatus.AMBIGUOUS, ParseReason.CONFLICTING_FINALS, len(unique)
        )
    if not unique:
        return _empty(ParseReason.INVALID_FORMAT)
    return _aime_value(unique[0], ParseReason.EXPLICIT_FINAL)


def _aime_value(value: str, reason: ParseReason) -> ParsedResponse:
    numeric = int(value)
    if numeric > 999:
        return _empty(ParseReason.INVALID_RANGE)
    return ParsedResponse(str(numeric), ParseStatus.EXTRACTED, reason, 1)


def parse_boxed_math(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    matches: list[str] = []
    for start in _BOX_START.finditer(visible):
        depth = 1
        index = start.end()
        while index < len(visible) and depth:
            depth += (visible[index] == "{") - (visible[index] == "}")
            index += 1
        if depth == 0:
            matches.append(visible[start.end() : index - 1].strip())
    if matches:
        return ParsedResponse(matches[-1], ParseStatus.EXTRACTED, ParseReason.BOXED_ANSWER, 1)
    return parse_short_answer(visible)


def parse_python_source(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", visible, flags=re.I | re.S)
    candidates = tuple(dict.fromkeys(value.strip() for value in fenced if value.strip()))
    if len(candidates) > 1:
        return ParsedResponse(
            None,
            ParseStatus.AMBIGUOUS,
            ParseReason.CONFLICTING_FINALS,
            len(candidates),
        )
    value = candidates[0] if candidates else visible.strip()
    if not value:
        return _empty()
    return ParsedResponse(
        value,
        ParseStatus.EXTRACTED,
        ParseReason.CODE_FENCE if fenced else ParseReason.FINAL_LINE,
        1,
    )


def _parse_unified_diff_v1(text: str) -> ParsedResponse:
    visible = _strip_thinking(text).strip()
    fenced = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", visible, flags=re.I | re.S)
    candidate = fenced[-1].strip() if fenced else visible
    if not candidate:
        return _empty()
    if "\x00" in candidate or _DIFF_HEADER.search(candidate) is None:
        return _empty(ParseReason.INVALID_FORMAT)
    return ParsedResponse(candidate, ParseStatus.EXTRACTED, ParseReason.UNIFIED_DIFF, 1)


def _fenced_blocks(text: str) -> tuple[str, ...]:
    """Return closed Markdown fence bodies without pairing a closing fence as an opener."""

    blocks: list[str] = []
    body: list[str] | None = None
    for line in text.splitlines():
        marker = _FENCE_LINE.fullmatch(line)
        if body is None:
            if marker is not None:
                body = []
            continue
        if marker is not None and not marker.group(1).strip():
            blocks.append("\n".join(body).strip())
            body = None
        else:
            body.append(line)
    return tuple(blocks)


def _is_unified_diff(candidate: str) -> bool:
    if not candidate or "\x00" in candidate:
        return False
    if _DIFF_HEADER.search(candidate) is not None:
        return True
    return (
        _DIFF_FILE_HEADERS.search(candidate) is not None
        and _DIFF_HUNK_HEADER.search(candidate) is not None
    )


def parse_unified_diff(text: str) -> ParsedResponse:
    """Extract one unambiguous unified diff from a model response.

    Version 2 deliberately treats the Markdown language label as presentation metadata:
    the patch body, rather than a possibly mistaken ``python``/``diff`` label, determines
    whether a fenced block is a diff. Repeated identical blocks collapse to one candidate,
    while distinct patch blocks are ambiguous rather than silently selecting the last one.
    """

    visible = _strip_thinking(text).strip()
    if not visible:
        return _empty()
    candidates = tuple(
        dict.fromkeys(block for block in _fenced_blocks(visible) if _is_unified_diff(block))
    )
    if not candidates and _is_unified_diff(visible):
        candidates = (visible,)
    if not candidates:
        return _empty(ParseReason.INVALID_FORMAT)
    if len(candidates) > 1:
        return ParsedResponse(
            None,
            ParseStatus.AMBIGUOUS,
            ParseReason.CONFLICTING_FINALS,
            len(candidates),
        )
    return ParsedResponse(candidates[0], ParseStatus.EXTRACTED, ParseReason.UNIFIED_DIFF, 1)


def parse_native_action(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    actions = [match.group(1).strip() for match in _ACTION_MARKER.finditer(visible)]
    if actions:
        return _unique(actions, ParseReason.EXPLICIT_FINAL)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if len(lines) == 1 and not lines[0].startswith(("{", "[")):
        return ParsedResponse(lines[0], ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)
    if not lines:
        return _empty()
    return ParsedResponse(None, ParseStatus.AMBIGUOUS, ParseReason.INVALID_FORMAT, len(lines))


def parse_visible_react(text: str) -> ParsedVisibleReAct:
    """Extract visible Memory/Thought sections without making either an action gate.

    The action retains the ordinary answer-blind native-action contract.  A missing
    memory or thought is diagnostic only so the interactive runner can carry the
    previous cumulative memory forward without inventing or correcting an action.
    Repeated structured sections are treated as absent rather than selecting one
    opportunistically; the action parser remains authoritative for validity.
    """

    visible = _strip_thinking(text)
    matches = tuple(_REACT_SECTION.finditer(visible))
    sections: dict[str, list[str]] = {"memory": [], "thought": [], "action": []}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(visible)
        value = visible[match.end() : end].strip()
        if value:
            sections[match.group(1).casefold()].append(value)

    def unique_section(name: str) -> str | None:
        values = tuple(dict.fromkeys(sections[name]))
        return values[0] if len(values) == 1 else None

    action_sections = tuple(dict.fromkeys(sections["action"]))
    if len(action_sections) > 1:
        action = ParsedResponse(
            None,
            ParseStatus.AMBIGUOUS,
            ParseReason.CONFLICTING_FINALS,
            len(action_sections),
        )
    elif len(action_sections) == 1:
        action_lines = tuple(
            line.strip() for line in action_sections[0].splitlines() if line.strip()
        )
        action = (
            ParsedResponse(
                action_lines[0],
                ParseStatus.EXTRACTED,
                ParseReason.EXPLICIT_FINAL,
                1,
            )
            if len(action_lines) == 1 and not action_lines[0].startswith(("{", "["))
            else ParsedResponse(
                None,
                ParseStatus.AMBIGUOUS,
                ParseReason.INVALID_FORMAT,
                len(action_lines),
            )
        )
    else:
        action = parse_native_action(visible)

    return ParsedVisibleReAct(
        memory=unique_section("memory"),
        thought=unique_section("thought"),
        action=action,
    )


def parse_webshop_source_action_strict(text: str) -> ParsedResponse:
    """Accept exactly one bare SkillFlow WebShop action and no explanatory wrapper."""

    visible = _strip_thinking(text)
    if re.fullmatch(r"search\[[^\[\]\n]+\]|click\[[^\[\]\n]+\]", visible) is None:
        return _empty(ParseReason.INVALID_FORMAT)
    return ParsedResponse(visible, ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)


def parse_alfworld_source_action_strict(text: str) -> ParsedResponse:
    """Accept one bare line; the environment performs exact admissible-list validation."""

    visible = _strip_thinking(text)
    if not visible or "\n" in visible or visible.lower().startswith("action:"):
        return _empty(ParseReason.INVALID_FORMAT)
    if visible.startswith(("{", "[", "<")):
        return _empty(ParseReason.INVALID_FORMAT)
    return ParsedResponse(visible, ParseStatus.EXTRACTED, ParseReason.FINAL_LINE, 1)


Parser = Callable[[str], ParsedResponse]
PARSER_REGISTRY: dict[str, Parser] = {
    "short-answer-final@2": parse_short_answer,
    "short-answer-last-line@1": parse_short_answer_last_line,
    "aime-integer-final@2": parse_aime_integer,
    "aime-boxed-integer@1": parse_aime_boxed_integer,
    "aime-integer-final@3": parse_aime_source_integer,
    "multiple-choice-label@1": parse_multiple_choice_label,
    "multiple-choice-json@1": parse_multiple_choice_json,
    "boxed-math@1": parse_boxed_math,
    "python-source@1": parse_python_source,
    "python-source@2": parse_python_source,
    "python-source@3": parse_python_source,
    "unified-diff@1": _parse_unified_diff_v1,
    "unified-diff@2": parse_unified_diff,
    "workspace-git-diff@1": parse_unified_diff,
    "native-action@2": parse_native_action,
    "native-action-memory-v5@1": lambda text: parse_visible_react(text).action,
    "native-action-memory-v6@1": lambda text: parse_visible_react(text).action,
    "webshop-source-action-strict@1": parse_webshop_source_action_strict,
    "alfworld-source-action-strict@1": parse_alfworld_source_action_strict,
}
