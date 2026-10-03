from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from skillev.runtime.contracts import BudgetVector

FEATURE_VERSION = "evosteer-public-features@5"
FEATURE_NAMES = (
    "node_count_saturated",
    "edge_density",
    "role_diversity",
    "bound_skill_fraction",
    "action_count_saturated",
    "last_add_agent",
    "last_add_edge",
    "last_bind_skill",
    "last_set_output",
    "last_rerun_agent",
    "tool_used_fraction",
    "python_clean_fraction",
    "repair_attempted",
    "repair_output_changed",
    "last_execution_failed",
    "answered_fraction",
    "selected_output",
    "answer_agreement",
    "last_output_changed",
    "communication_delivered",
    "answer_supported_by_tools",
    "environment_step_fraction",
    "environment_phase_fraction",
    "environment_terminal",
    "environment_call_succeeded",
    "environment_call_failed",
    "total_token_budget_used",
    "tool_budget_used",
    "time_budget_used",
    "model_budget_used",
)
assert len(FEATURE_NAMES) == 30
_FAILURE_STATUSES = frozenset(
    {
        "parse_error",
        "schema_invalid",
        "tool_error",
        "timeout",
        "infrastructure-failure",
        "failed",
    }
)
_NODE_FAILURE_KINDS = _FAILURE_STATUSES | {"truncated"}
_FENCE_LINE = re.compile(r"^[ \t]*(?:```|~~~)[^\n]*$", re.MULTILINE)
_BOXED = "\\boxed{"
_ANSWER_CUE = re.compile(r"(?:final answer|answer)\s*(?:is\s*:?|:)\s*([^\n]+)", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+•‣]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])[ \t]+\S")
_HEADING = re.compile(r"^#{1,6}[ \t]+\S")
_EMPHASIS_TITLE = re.compile(r"^(?P<marks>\*{1,3}|_{1,3})(?P<title>[^*_\n]{1,120})(?P=marks)$")
_EMPHASIS = "*_`#~ \t"
_INLINE_ITEM = re.compile(r"(?<![\w.])\d{1,3}[.)][ \t]+\S")
_FUNCTION_DEF = re.compile(r"\b(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _final_boxed(text: str) -> str | None:
    end = len(text)
    while (start := text.rfind(_BOXED, 0, end)) >= 0:
        begin, depth = start + len(_BOXED), 0
        for index in range(begin, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                if depth == 0:
                    return text[begin:index]
                depth -= 1
        end = start
    return None


def _cued_answer(text: str) -> str | None:
    matches = list(_ANSWER_CUE.finditer(text))
    if not matches:
        return None
    value = matches[-1].group(1).strip().strip("*_`\"'").strip()
    value = value.rstrip(".").strip().strip("*_`\"'").strip()
    return value or None


def _largest_code_block(text: str) -> str | None:
    blocks = [match.group(1).strip() for match in _CODE_BLOCK.finditer(text)]
    return max(blocks, key=len) if blocks else None


def normalized_answer(text: str) -> str:
    value = _final_boxed(text)
    if value is None:
        value = _cued_answer(text)
    if value is None:
        value = _largest_code_block(text)
    if value is None:
        value = text
    value = _FENCE_LINE.sub("", value).strip()
    wrapped = re.fullmatch(r"(`+)([^`]*)\1", value)
    if wrapped:
        value = wrapped.group(2)
    return " ".join(value.casefold().split())


def _lead_in(line: str) -> bool:
    return line.strip().strip(_EMPHASIS).endswith(":")


def _title(line: str) -> bool:
    stripped = line.strip()
    emphasized = _EMPHASIS_TITLE.match(stripped)
    return bool(
        _HEADING.match(stripped)
        or _lead_in(stripped)
        or (emphasized and not emphasized.group("title").rstrip().endswith((".", "!", "?")))
    )


def _inline_list(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        marks = [match.start() for match in _INLINE_ITEM.finditer(line)]
        head = line[: marks[0]].rstrip() if marks else line
        if len(marks) < 3 or not _lead_in(head):
            lines.append(line)
            continue
        bounds = [*marks, len(line)]
        pieces = [head, *(line[start:end] for start, end in pairwise(bounds))]
        lines.extend(piece for piece in pieces if piece.strip())
    return "\n".join(lines)


def _enumeration_only(text: str) -> bool:
    items = titles = 0
    for line in _inline_list(text).splitlines():
        if not line.strip():
            continue
        if _LIST_ITEM.match(line):
            items += 1
            continue
        if _title(line):
            titles += 1
            continue
        if items and line[:1].isspace():
            continue
        return False
    return items >= 2 and titles >= 1


def answer_present(text: str, *, code: bool = False) -> bool:
    body = text.strip()
    if not body:
        return False
    if code:
        return _FUNCTION_DEF.search(body) is not None
    if _final_boxed(body) is not None or _CODE_BLOCK.search(body) is not None:
        return True
    if _cued_answer(body) is not None:
        return True
    return not _enumeration_only(body)


_EVIDENCE_MIN = 3
_EVIDENCE_MAX = 120
_NON_ANSWERS = frozenset(
    {"n/a", "na", "none", "unknown", "unanswerable", "not found", "no answer", "not available"}
)


def _tool_calls(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    transcript = _mapping(result.get("metadata")).get("tool_transcript")
    if not isinstance(transcript, Sequence) or isinstance(transcript, str):
        return []
    return [entry for entry in map(_mapping, transcript) if entry.get("kind") == "call"]


def stated_answers(
    nodes: Sequence[Mapping[str, Any]], latest_results: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    answers: list[str] = []
    for node in nodes:
        node_metadata = _mapping(latest_results.get(node["node_id"], {}).get("metadata"))
        node_outcome = _mapping(node_metadata.get("execution_outcome"))
        reported = node_outcome.get("answer_present", bool(node["output"].strip()))
        if reported and answer_present(node["output"]):
            answers.append(node["output"])
    return answers


def tool_evidence(
    nodes: Sequence[Mapping[str, Any]],
    latest_results: Mapping[str, Mapping[str, Any]],
    answers: Sequence[str],
) -> tuple[float, float, float]:
    used = 0
    for node in nodes:
        metadata = _mapping(latest_results.get(node["node_id"], {}).get("metadata"))
        used += int(_mapping(metadata.get("tool_use")).get("calls") or 0) > 0
    retrieved: list[str] = []
    python_calls = clean = 0
    for result in latest_results.values():
        for call in _tool_calls(result):
            if call.get("tool") == "search":
                retrieved.append(" ".join(str(call.get("result") or "").casefold().split()))
            elif call.get("tool") == "python":
                record = _mapping(call.get("record"))
                python_calls += 1
                clean += record.get("exit_status") == 0 and not record.get("timed_out")
    text = "\n".join(retrieved)
    supported = 0
    for answer in answers:
        key = normalized_answer(answer)
        if (
            _EVIDENCE_MIN <= len(key) <= _EVIDENCE_MAX
            and key not in _NON_ANSWERS
            and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text)
        ):
            supported += 1

    def share(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return share(used, len(nodes)), share(supported, len(answers)), share(clean, python_calls)


def extract_features(
    *,
    graph: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    max_nodes: int | None,
    max_actions: int | None,
    role_count: int,
    skill_count: int,
    used: BudgetVector,
    cap: BudgetVector,
    total_token_cap: int | None = None,
    environment: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    del max_nodes, max_actions
    nodes = list(graph["nodes"])
    edges = list(graph["edges"])
    last_event = history[-1] if history else {}
    action = _mapping(last_event.get("action"))
    observation = _mapping(last_event.get("observation"))
    result = _mapping(observation.get("node_result"))
    metadata = _mapping(result.get("metadata"))
    outcome = _mapping(metadata.get("execution_outcome"))
    request = _mapping(observation.get("node_request"))
    latest_results: dict[str, Mapping[str, Any]] = {}
    for event in history:
        event_observation = _mapping(event.get("observation"))
        if "node_result" in event_observation:
            latest_results[str(event_observation["node_id"])] = _mapping(
                event_observation["node_result"]
            )
    live = {str(node["node_id"]) for node in nodes}
    latest_results = {k: v for k, v in latest_results.items() if k in live}
    answers = stated_answers(nodes, latest_results)
    tools_used, supported, python_clean = tool_evidence(nodes, latest_results, answers)
    pairs = len(answers) * (len(answers) - 1) // 2
    normalized = [normalized_answer(answer) for answer in answers]
    agreement = sum(
        bool(normalized[i]) and normalized[i] == normalized[j]
        for i in range(len(normalized))
        for j in range(i)
    )
    kind = action.get("kind")
    repaired = kind in {"RERUN_AGENT", "BIND_SKILL"} or (
        kind == "ADD_EDGE" and action.get("protocol") == "revise"
    )
    changed = bool(observation.get("output_changed", False))
    delivered = bool(observation.get("delivered_message")) or bool(request.get("messages"))
    failed = outcome.get("status") == "failed" or outcome.get("failure_kind") in _NODE_FAILURE_KINDS
    failed = failed or observation.get("status") in _FAILURE_STATUSES
    world = environment or {}
    phase_enabled = world.get("enabled") is True
    phase_cap = int(world.get("max_phases") or 0)
    step_cap = phase_cap * int(world.get("steps_per_phase") or 0)
    last_world_outcome = _mapping(world.get("last_outcome"))
    world_status = last_world_outcome.get("status")

    def ratio(numerator: int | float, denominator: int | float) -> float:
        return min(1.0, max(0.0, float(numerator / denominator))) if denominator else 0.0

    token_cap = cap.input_tokens + cap.output_tokens if total_token_cap is None else total_token_cap
    values = [
        ratio(len(nodes), len(nodes) + 1),
        ratio(len(edges), len(nodes) * (len(nodes) - 1)),
        ratio(len({node["role_id"] for node in nodes}), role_count),
        ratio(sum(len(node["skill_ids"]) for node in nodes), len(nodes) * max(1, skill_count)),
        ratio(len(history), len(history) + 1),
        *[
            float(kind == name)
            for name in (
                "ADD_AGENT",
                "ADD_EDGE",
                "BIND_SKILL",
                "SET_OUTPUT",
                "RERUN_AGENT",
            )
        ],
        tools_used,
        python_clean,
        float(repaired),
        float(repaired and changed),
        float(failed),
        ratio(len(answers), len(nodes)),
        float(graph["output_node_id"] is not None),
        ratio(agreement, pairs),
        float(changed),
        float(delivered),
        supported,
        ratio(int(world.get("steps_completed") or 0), step_cap) if phase_enabled else 0.0,
        ratio(int(world.get("phase_index") or 0), phase_cap) if phase_enabled else 0.0,
        float(world.get("environment_terminal") is True),
        float(world_status == "success"),
        float(world_status in _FAILURE_STATUSES),
        ratio(used.input_tokens + used.output_tokens, token_cap),
        ratio(used.tool_calls, cap.tool_calls),
        ratio(used.wall_time_milliseconds, cap.wall_time_milliseconds),
        ratio(used.model_calls, cap.model_calls),
    ]
    return tuple(values)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "answer_present",
    "extract_features",
    "normalized_answer",
    "stated_answers",
    "tool_evidence",
]
