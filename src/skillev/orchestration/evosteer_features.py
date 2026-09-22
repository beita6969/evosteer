"""Thirty measured public features covering the categories named in EvoSteer.

The paper specifies categories and width, not the thirty coordinate formulas.
Schema 4 is this implementation's explicit definition. It uses no evaluator
labels, learned judgments, token estimation, or additional model calls.
Schema 3 differed from 2 in two coordinates: answer agreement compares
normalized answers instead of raw text, and an executor-reported output-cap
truncation counts as a failed execution. Schema 4 changes one coordinate:
``answered_fraction`` counts a node only when its text actually states an
answer, so an output that is nothing but a plan or a list of facts to gather
no longer reads as an answer (see ``answer_present``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from skillev.runtime.contracts import BudgetVector

FEATURE_VERSION = "evosteer-public-features@4"
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
    "last_drop_agent",
    "last_stop",
    "repair_attempted",
    "repair_output_changed",
    "last_execution_failed",
    "answered_fraction",
    "selected_output",
    "answer_agreement",
    "last_output_changed",
    "communication_delivered",
    "communication_output_revised",
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
# Node-only: a cut completion is an execution failure, while an environment's
# own status vocabulary (environment_call_failed) is left unchanged.
_NODE_FAILURE_KINDS = _FAILURE_STATUSES | {"truncated"}
_FENCE_LINE = re.compile(r"^[ \t]*(?:```|~~~)[^\n]*$", re.MULTILINE)
_BOXED = "\\boxed{"
_ANSWER_CUE = re.compile(r"(?:final answer|answer)\s*(?:is\s*:?|:)\s*([^\n]+)", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# A list item: bullet, "1." / "1)" / "(1)", or "a." / "(b)".
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+•‣]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])[ \t]+\S")
_HEADING = re.compile(r"^#{1,6}[ \t]+\S")
# A whole line set in bold or italics, e.g. "**Facts Needed:**".
_EMPHASIS_TITLE = re.compile(r"^(?P<marks>\*{1,3}|_{1,3})(?P<title>[^*_\n]{1,120})(?P=marks)$")
_EMPHASIS = "*_`#~ \t"
# An enumeration written inline: " ... : 1. item 2. item 3. item".
_INLINE_ITEM = re.compile(r"(?<![\w.])\d{1,3}[.)][ \t]+\S")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _final_boxed(text: str) -> str | None:
    """Content of the last complete ``\\boxed{...}``; an unclosed box is no answer."""
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
    """The span after the last 'answer is' / 'answer:' cue, up to the line end."""
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
    """Formatting-insensitive answer text used only for agreement.

    The stated answer is, in order: the last complete ``\\boxed{}`` (the math
    graders' convention), the text after the last answer cue on its line (the
    short-answer graders' convention), the largest fenced code block (the code
    grader's convention), else the whole output. Markdown fence lines, backticks
    wrapping the whole answer, letter case and whitespace runs are presentation,
    not content.
    """
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
    """A line that only announces the list under it ("**Facts Needed:**")."""
    return line.strip().strip(_EMPHASIS).endswith(":")


def _title(line: str) -> bool:
    """A line that titles the list under it, rather than saying anything itself.

    A markdown heading ("### Step-by-Step Plan"), a line set entirely in bold or
    italics that is not a sentence ("**Plan**"), or a lead-in ending in a colon.
    """
    stripped = line.strip()
    emphasized = _EMPHASIS_TITLE.match(stripped)
    return bool(
        _HEADING.match(stripped)
        or _lead_in(stripped)
        or (emphasized and not emphasized.group("title").rstrip().endswith((".", "!", "?")))
    )


def _inline_list(text: str) -> str:
    """Put a list written inline ("Facts: 1. a 2. b 3. c") on separate lines."""
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
    """Whether the text is a titled list and nothing else.

    Three conditions, all structural: a title or lead-in introduces the list (an
    untitled list can itself be the answer, as for "name the three largest
    cities"), it has at least two items (a one-line answer written as a bullet
    is still an answer), and nothing outside the items says anything — an
    ordinary sentence anywhere, in particular a concluding one, is where an
    answer would be stated, so the reading stays on the conservative side.
    """
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
        if items and line[:1].isspace():  # a wrapped continuation of an item
            continue
        return False
    return items >= 2 and titles >= 1


def answer_present(text: str) -> bool:
    """Whether the output states an answer, read from the text alone.

    The graders' conventions are an answer: a complete ``\\boxed{}``, a fenced
    code block, or a span after an answer cue. Failing all three, an output that
    is only a titled enumeration — a step-by-step plan, a list of facts to
    gather — states no answer, while ordinary prose (a short answer such as
    "Lawrence County", a derivation ending in a sentence) does. Nothing here
    reads the node's role: a planning role that does answer counts, and a
    solving role that only plans does not.
    """
    body = text.strip()
    if not body:
        return False
    if _final_boxed(body) is not None or _CODE_BLOCK.search(body) is not None:
        return True
    if _cued_answer(body) is not None:
        return True
    return not _enumeration_only(body)


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
    """Counts use n/(n+1), so debug caps do not change the method's representation.

    Answers mean outputs that state an answer (``answer_present``), never
    correct ones: a finished local completion whose text carries an answer in
    one of the graders' forms, or prose that is not merely a plan or a list of
    facts. Missing completion metadata on a plain text executor uses its
    returned text alone.
    Agreement is equality of ``normalized_answer`` between nonempty answers; an
    answer that normalizes to nothing agrees with none. The last execution
    failed when its executor reported a failed status or a failure kind,
    including ``truncated`` at the output cap. Repair is rerun, bind,
    or an immediately executing revise edge; revision means a changed text result.
    Environment success/failure is the adapter's observed call status, never the
    terminal task score. Disabled phases and unobserved outcomes contribute zero.
    """
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
    answers: list[str] = []
    for node in nodes:
        node_metadata = _mapping(latest_results.get(node["node_id"], {}).get("metadata"))
        node_outcome = _mapping(node_metadata.get("execution_outcome"))
        # The executor knows whether its completion finished at all; the text
        # says whether that completion states an answer. Both must hold.
        reported = node_outcome.get("answer_present", bool(node["output"].strip()))
        if reported and answer_present(node["output"]):
            answers.append(node["output"])
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
    revised = (
        bool(request.get("messages")) and request.get("previous_output") is not None and changed
    )
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
                "DROP_AGENT",
                "STOP",
            )
        ],
        float(repaired),
        float(repaired and changed),
        float(failed),
        ratio(len(answers), len(nodes)),
        float(graph["output_node_id"] is not None),
        ratio(agreement, pairs),
        float(changed),
        float(delivered),
        float(revised),
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
]
