from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import re

from skillev.orchestration.evosteer_features import (
    _ANSWER_CUE,
    _CODE_BLOCK,
    _EVIDENCE_MAX,
    _EVIDENCE_MIN,
    _FUNCTION_DEF,
    _NODE_FAILURE_KINDS,
    _NON_ANSWERS,
    FEATURE_NAMES,
    _cued_answer,
    _final_boxed,
    _tool_calls,
    answer_present,
    normalized_answer,
    stated_answers,
)

DIAGNOSIS_SEPARATOR = "\n\nDiagnosis:\n"
CONTRIBUTION_THRESHOLD = 0.01
CONTRIBUTION_TOP = 3
HIGH_USE = 0.8
NOT_FITTED_LINE = "Feature contributions appear once the value head has been fitted."
NO_CONTRIBUTION_LINE = "No answer, tool, environment or failure feature moves the estimate by 0.01 or more."
NO_FEATURE_LINE = "No feature is present yet, so none contributes to the estimate."
NO_REAL_ANSWER_LINE = (
    "No contribution is listed: every output the answer features count is a refusal, a "
    "placeholder or a plan, and no other feature moves the estimate by 0.01 or more."
)

_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_BUDGET_PROBLEM = "little budget is left for further executions"
_ENVIRONMENT_REJECTED = "the environment rejected or could not perform the last action"
_NOT_COUNTED = "plans and fact lists are not counted as answers"
_NOT_COUNTED_ALL = "plans, fact lists, refusals and placeholders are not counted as answers"

PHRASES: dict[str, tuple[str, str | None, str | None]] = {
    "node_count_saturated": (
        "team size",
        None,
        "each extra node spends budget and adds an output that may conflict",
    ),
    "edge_density": ("edges between nodes", None, None),
    "role_diversity": ("number of distinct roles", None, None),
    "bound_skill_fraction": ("nodes bound to a skill", None, None),
    "action_count_saturated": ("number of actions taken", None, None),
    "last_add_agent": ("last action ADD_AGENT", None, None),
    "last_add_edge": ("last action ADD_EDGE", None, None),
    "last_bind_skill": ("last action BIND_SKILL", None, None),
    "last_set_output": ("last action SET_OUTPUT", None, None),
    "last_rerun_agent": ("last action RERUN_AGENT", None, None),
    "tool_used_fraction": ("nodes that called a tool", None, None),
    "python_clean_fraction": (
        "python runs that exited cleanly",
        "python runs exiting cleanly",
        "the code raised an error or timed out",
    ),
    "repair_attempted": ("last action re-executed a node", None, None),
    "repair_output_changed": (
        "the re-execution changed that node's output",
        None,
        "a changed output can be better or worse than the one it replaced",
    ),
    "last_execution_failed": (
        "the last execution failed or was cut at its output limit",
        None,
        "that node's output may be incomplete or lack its final answer",
    ),
    "answered_fraction": (
        "nodes whose output states an answer",
        "a node whose output states an answer",
        f"no node has stated an answer yet; {_NOT_COUNTED}",
    ),
    "selected_output": (
        "an output node is selected",
        "a selected output node",
        "STOP scores only the output node's output",
    ),
    "answer_agreement": (
        "answering nodes agree",
        "agreement between answering nodes",
        "the answering nodes state different answers, so they may not all be right",
    ),
    "last_output_changed": ("the last execution changed its node's output", None, None),
    "communication_delivered": ("the last execution received other nodes' outputs", None, None),
    "answer_supported_by_tools": (
        "answers found in the search results",
        "an answer found in the search results",
        "the stated answer does not appear in what search returned, so it may not be "
        "grounded in the evidence",
    ),
    "environment_step_fraction": (
        "environment steps used",
        None,
        "few environment steps are left",
    ),
    "environment_phase_fraction": ("environment phases used", None, None),
    "environment_terminal": (
        "the environment reports the task completed",
        "the environment reporting the task completed",
        "the environment task is not finished yet",
    ),
    "environment_call_succeeded": (
        "the last environment action succeeded",
        "a successful last environment action",
        _ENVIRONMENT_REJECTED,
    ),
    "environment_call_failed": (
        "the last environment action failed",
        None,
        _ENVIRONMENT_REJECTED,
    ),
    "total_token_budget_used": ("token budget used", None, _BUDGET_PROBLEM),
    "tool_budget_used": ("tool-call budget used", None, _BUDGET_PROBLEM),
    "time_budget_used": ("time budget used", None, _BUDGET_PROBLEM),
    "model_budget_used": ("model-call budget used", None, _BUDGET_PROBLEM),
}
assert set(PHRASES) == set(FEATURE_NAMES)

_SHARES: dict[str, str] = {
    "total_token_budget_used": "the token budget",
    "tool_budget_used": "the tool-call budget",
    "time_budget_used": "the time budget",
    "model_budget_used": "the model-call budget",
    "environment_step_fraction": "the environment steps",
    "environment_phase_fraction": "the environment phases",
}
_TOOL_OUTCOME_PHRASES: dict[str, tuple[str, str | None, str | None]] = {
    "environment_call_succeeded": ("the last tool call succeeded", None, None),
    "environment_call_failed": (
        "the last tool call failed",
        None,
        "the tool could not perform the last call",
    ),
}
_PYTHON_FAILED = ("the last python run exited with an error", "the code raised an error or timed out")
_TEXT_DIFFERS = (
    "the answering nodes' outputs differ as text; without a boxed or cued answer whole code "
    "blocks or outputs are compared, so equivalent answers can also differ"
)
_TEXT_DIFFERS_PLAIN = (
    "{subject}' outputs differ as text; without a boxed or cued answer whole outputs are "
    "compared, so equivalent answers can also differ"
)
_LONG_ANSWER = (
    "{subject} state different answers; one answer is a long sentence rather than a short "
    "answer, so it is compared as text"
)
_LONG_ANSWERS = (
    "{subject} state different answers; {count} answers are long sentences rather than short "
    "answers, so they are compared as text"
)
_DIFFERENT = "{subject} state different answers, so they may not all be right"
_CODE_FAMILIES = frozenset({"mbpp_plus"})
_INTEGER_FAMILIES = frozenset({"aime_2026"})
ALL_USED = 0.99
_BUDGET_GONE = "no budget is left"
_STEPS_GONE = "no environment steps are left"

REFUSAL, PLACEHOLDER, PLAN = "refusal", "placeholder", "plan"
_NO_ANSWER_NOTE = {
    REFUSAL: "states no answer (it says the information was not found)",
    PLACEHOLDER: "states no answer (a placeholder)",
    PLAN: "states no answer (a plan)",
}
_SOURCES = (
    r"(?:search(?:\s+results?)?|results?|passages?|facts|context|sources?|documents?|"
    r"texts?|information|data|evidence|articles?|snippets?)"
)
_REFUSAL = re.compile(
    "|".join(
        (
            rf"\b{_SOURCES}\s+(?:do|does|did)\s+not\s+(?:explicitly\s+|directly\s+|specifically\s+"
            r"|clearly\s+)?(?:contain|state|mention|provide|specify|identify|include|give|list|"
            r"say|name)\b",
            r"\b(?:cannot|can't|can\s+not|could\s+not|couldn't)\s+be\s+(?:determined|found|"
            r"identified)\b",
            r"\bunable\s+to\s+(?:determine|find|identify)\b",
            r"\bnot\s+(?:available|found|mentioned|provided|specified|stated)"
            rf"(?:\s*$|\s+(?:in|from|within|among)\b[^.;]*?\b{_SOURCES}\b)",
            r"\bno\s+information\b",
            r"^unknown(?:\s+(?:based\s+on|from|given|according\s+to|in)\b.*)?$",
        )
    ),
    re.IGNORECASE,
)
_REFUSAL_WHOLE_MAX = 200
_HEDGED = re.compile(
    r"\b(?:approximately|roughly|though|although|however|likely|probably|possibly|perhaps|"
    r"presumably|estimated?|notable|best\s+guess)\b|\*\*",
    re.IGNORECASE,
)
_PLACEHOLDER_BOX = re.compile(
    r"^(?:(?:your|final|short)\s+)?(?:plan(?:\s+(?:complete|completed|provided|formulated))?|"
    r"answer|final|result|\?+|\.{3}|…)?$",
    re.IGNORECASE,
)
_PLACEHOLDER_CUE = re.compile(
    r"^[<\[({]*\s*(?:(?:your|final|short)\s+)?(?:answer|\?+|\.{3}|…)\s*[>\])}]*$", re.IGNORECASE
)
_NUMBER_CUE = re.compile(r"^[\s$*(\\\[]*-?\d")
_PLAN_ITEM = re.compile(r"^\s*\d{1,2}[.)]\s+(?:\*\*)?([A-Za-z]+)")
_PLAN_VERBS = frozenset(
    {
        "identify", "search", "locate", "find", "determine", "verify", "confirm", "compare",
        "extract", "formulate", "provide", "output", "write", "summarize", "recall", "select",
        "scan", "look", "check", "retrieve", "gather", "list", "read", "review", "analyze",
        "analyse", "calculate", "compute", "state", "insert", "use", "consult", "cross-reference",
    }
)

_MISSABLE = frozenset(name for name, (_, missing, _) in PHRASES.items() if missing is not None)
_GOOD = _MISSABLE
_BAD = frozenset({"last_execution_failed", "environment_call_failed", *_SHARES})
_ANSWER_FACTS = frozenset({"answered_fraction", "answer_agreement", "answer_supported_by_tools"})
_TEXT_ONLY = frozenset(
    {"answered_fraction", "answer_agreement", "answer_supported_by_tools", "python_clean_fraction"}
)

BatchedValues = Callable[[Sequence[tuple[float, ...]]], Sequence[float]]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def estimate(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    rounded = round(value, 2)
    return f"{0.0 if rounded == 0 else rounded:+.2f}"


def _share(value: float) -> str:
    percent = 100 if value >= 1.0 else int(100 * value)
    return "under 1%" if percent == 0 else f"{percent}%"


def _count(fraction: float, total: int) -> int:
    return int(round(fraction * total))


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _checkable(answer: str) -> bool:
    key = normalized_answer(answer)
    return _EVIDENCE_MIN <= len(key) <= _EVIDENCE_MAX and key not in _NON_ANSWERS


def _extracted(answer: str) -> bool:
    return _marked(answer) and len(normalized_answer(answer)) <= _EVIDENCE_MAX


def _marked(answer: str) -> bool:
    return _final_boxed(answer) is not None or _cued_answer(answer) is not None


def _family(state: Mapping[str, Any]) -> str:
    parts = str(state.get("runtime_id") or "").split("/")
    return parts[1] if len(parts) > 2 else ""


def refusal(output: str) -> bool:
    if _CODE_BLOCK.search(output) or _FUNCTION_DEF.search(output):
        return False
    span = _final_boxed(output)
    if span is None:
        span = _cued_answer(output)
    if span is None:
        span = output.strip()
        if len(span) > _REFUSAL_WHOLE_MAX:
            return False
    span = " ".join(span.strip().strip("*_`\"'").rstrip(".").split())
    match = _REFUSAL.search(span)
    if match is None:
        return False
    prefix, suffix = span[: match.start()], span[match.end() :]
    if prefix.count("(") > prefix.count(")"):
        return False
    return _HEDGED.search(prefix) is None and _HEDGED.search(suffix) is None


def _box_text(content: str) -> str:
    content = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", content)
    content = content.replace("\\ ", " ").strip("{}$ \\.:")
    return " ".join(content.split())


def placeholder(output: str, family: str = "") -> bool:
    integer = family in _INTEGER_FAMILIES
    boxed = _final_boxed(output)
    if boxed is not None:
        content = _box_text(boxed)
        empty = _PLACEHOLDER_BOX.match(content) is not None or (
            integer and not any(char.isdigit() for char in content)
        )
        if not empty:
            return False
        cues = list(_ANSWER_CUE.finditer(output))
        if cues and cues[-1].start() > output.rfind("\\boxed{"):
            cued = _cued_answer(output)
            if cued is not None and _PLACEHOLDER_CUE.match(cued) is None:
                return integer and _NUMBER_CUE.match(cued) is None
        return True
    cued = _cued_answer(output)
    return cued is not None and _PLACEHOLDER_CUE.match(cued) is not None


def plan_only(output: str) -> bool:
    if _final_boxed(output) is not None or _cued_answer(output) is not None:
        return False
    if _CODE_BLOCK.search(output):
        return False
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    for line in lines:
        item = _PLAN_ITEM.match(line)
        if item is None or item.group(1).casefold() not in _PLAN_VERBS:
            return False
    return True


def no_answer_kind(output: str, family: str = "") -> str | None:
    if placeholder(output, family):
        return PLACEHOLDER
    if family in _CODE_FAMILIES:
        return None
    if refusal(output):
        return REFUSAL
    if plan_only(output):
        return PLAN
    return None


def pending_feedback(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    graph = _mapping(state.get("graph"))
    edges = graph.get("edges")
    history = state.get("history")
    events = [_mapping(event) for event in history] if isinstance(history, list) else []
    pending: list[tuple[str, str]] = []
    for edge in edges if isinstance(edges, list) else ():
        edge = _mapping(edge)
        if edge.get("protocol") != "feedback":
            continue
        source, target = str(edge.get("source_id")), str(edge.get("target_id"))
        added = None
        for index, event in enumerate(events):
            action = _mapping(event.get("action"))
            if (
                action.get("kind") == "ADD_EDGE"
                and action.get("protocol") == "feedback"
                and str(action.get("source_id")) == source
                and str(action.get("target_id")) == target
            ):
                added = index
        if added is None:
            continue
        executed = any(
            "node_result" in observation and str(observation.get("node_id")) == target
            for observation in (_mapping(event.get("observation")) for event in events[added + 1 :])
        )
        if not executed:
            pending.append((source, target))
    return pending


class _Facts:
    def __init__(self, features: Sequence[float], state: Mapping[str, Any]) -> None:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError("diagnosis needs the full public feature vector")
        self.x = {name: float(features[index]) for name, index in _INDEX.items()}
        self.family = _family(state)
        graph = _mapping(state.get("graph"))
        nodes = graph.get("nodes")
        nodes = [_mapping(item) for item in nodes] if isinstance(nodes, list) else []
        self.nodes = len(nodes)
        self.output_node = graph.get("output_node_id")
        self.answers = _count(self.x["answered_fraction"], self.nodes)
        latest: dict[str, Mapping[str, Any]] = {}
        last_result: Mapping[str, Any] | None = None
        history = state.get("history")
        for event in history if isinstance(history, list) else ():
            observation = _mapping(_mapping(event).get("observation"))
            if "node_result" in observation:
                latest[str(observation.get("node_id"))] = _mapping(observation["node_result"])
                last_result = latest[str(observation.get("node_id"))]
        live = {str(item.get("node_id")) for item in nodes}
        latest = {key: value for key, value in latest.items() if key in live}
        self.search_calls = self.python_calls = 0
        for result in latest.values():
            for call in _tool_calls(result):
                self.search_calls += call.get("tool") == "search"
                self.python_calls += call.get("tool") == "python"
        self.last_called = last_result is None or any(
            call.get("executed", True) is not False for call in _tool_calls(last_result)
        )
        structured = all(isinstance(item.get("output"), str) and "node_id" in item for item in nodes)
        environment = _mapping(state.get("environment"))
        self.environment = environment.get("enabled") is True
        answers: list[str] = []
        if structured:
            answers = stated_answers(nodes, latest)
        self.checkable_all = sum(_checkable(answer) for answer in answers)
        self.excluded: dict[str, int] = {}
        real: list[str] = []
        for answer in answers:
            kind = None if self.environment else no_answer_kind(answer, self.family)
            if kind is None:
                real.append(answer)
            else:
                self.excluded[kind] = self.excluded.get(kind, 0) + 1
        self.real = max(0, self.answers - sum(self.excluded.values()))
        self.checkable = sum(_checkable(answer) for answer in real)
        self.unmarked = sum(not _marked(answer) for answer in real)
        self.long = sum(
            _marked(answer) and len(normalized_answer(answer)) > _EVIDENCE_MAX for answer in real
        )
        keys = [normalized_answer(answer) for answer in real]
        self.real_pairs = len(keys) * (len(keys) - 1) // 2
        self.real_agreeing = sum(
            bool(keys[i]) and keys[i] == keys[j] for i in range(len(keys)) for j in range(i)
        )
        self.real_found = 0
        self.pending = pending_feedback(state)
        self.environment_call = self.environment and isinstance(
            environment.get("last_outcome"), Mapping
        )
        self.last_tool = _mapping(environment.get("last_outcome")).get("tool")
        self.environment_steps = int(environment.get("steps_completed") or 0)
        self.environment_cap = int(environment.get("max_phases") or 0) * int(
            environment.get("steps_per_phase") or 0
        )
        self.cap_reached = bool(self.environment_cap) and self.environment_steps >= self.environment_cap
        self.node_rows: list[str] = []
        self.answer_groups: list[list[str]] = []
        if not self.environment and structured:
            self._describe_nodes(nodes, latest, graph.get("output_node_id"))

    def _describe_nodes(
        self,
        nodes: Sequence[Mapping[str, Any]],
        latest: Mapping[str, Mapping[str, Any]],
        output_node: Any,
    ) -> None:
        retrieved = [
            " ".join(str(call.get("result") or "").casefold().split())
            for result in latest.values()
            for call in _tool_calls(result)
            if call.get("tool") == "search"
        ]
        text = "\n".join(retrieved)
        groups: dict[str, list[str]] = {}
        extracted_all = True
        for node in nodes:
            node_id = str(node["node_id"])
            result = latest.get(node_id, {})
            metadata = _mapping(result.get("metadata"))
            outcome = _mapping(metadata.get("execution_outcome"))
            reported = outcome.get("answer_present", bool(node["output"].strip()))
            answered = bool(reported and answer_present(node["output"]))
            kind = no_answer_kind(node["output"], self.family) if answered else None
            answered = answered and kind is None
            failed = (
                outcome.get("status") == "failed"
                or outcome.get("failure_kind") in _NODE_FAILURE_KINDS
            )
            parts = []
            if failed:
                parts.append("its last execution failed or was cut at its output limit")
            if kind is not None:
                parts.append(_NO_ANSWER_NOTE[kind])
            else:
                parts.append("states an answer" if answered else "states no answer")
            if answered and self.search_calls and _checkable(node["output"]):
                key = normalized_answer(node["output"])
                found = re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text) is not None
                self.real_found += found
                searched = any(call.get("tool") == "search" for call in _tool_calls(result))
                parts.append(
                    "its answer appears in the search results"
                    if found
                    else "its answer does not appear in the search results"
                    if searched
                    else "its latest execution did not search"
                )
            runs = [call for call in _tool_calls(result) if call.get("tool") == "python"]
            if runs:
                clean = sum(
                    _mapping(call.get("record")).get("exit_status") == 0
                    and not _mapping(call.get("record")).get("timed_out")
                    for call in runs
                )
                parts.append(f"{clean} of {len(runs)} python runs exited cleanly")
            label = f"{node_id} {node.get('role_id', '')}".strip()
            if node_id == output_node:
                label += " (output node)"
            self.node_rows.append(f"{label}: {', '.join(parts)}")
            if answered:
                extracted_all = extracted_all and _extracted(node["output"])
                groups.setdefault(normalized_answer(node["output"]), []).append(node_id)
        if extracted_all and sum(len(ids) for ids in groups.values()) >= 2:
            self.answer_groups = list(groups.values())

    def nodes_lines(self) -> list[str]:
        if not self.node_rows:
            return []
        lines = ["Nodes: " + "; ".join(self.node_rows) + "."]
        if len(self.answer_groups) == 1:
            lines.append("Answer groups: all answering nodes give the same answer.")
        elif self.answer_groups:
            lines.append(
                "Answer groups (nodes giving the same answer): "
                + " | ".join(", ".join(ids) for ids in self.answer_groups)
                + "."
            )
        return lines

    def environment_status(self) -> str:
        used = (
            f"{self.environment_steps} of {self.environment_cap} steps used"
            if self.environment_cap
            else f"{self.environment_steps} steps used"
        )
        if self.x["environment_terminal"] >= 1.0 and self.cap_reached:
            return f"the environment ended because all {self.environment_cap} steps were used"
        if self.x["environment_terminal"] >= 1.0:
            return f"the environment reports the task completed ({used})"
        return f"the environment task is not finished yet ({used})"

    def agreement(self) -> tuple[float, int, int]:
        if not self.excluded:
            pairs = self.answers * (self.answers - 1) // 2
            value = self.x["answer_agreement"]
            return value, int(round(value * pairs)), pairs
        pairs = self.real_pairs
        return (self.real_agreeing / pairs if pairs else 0.0), self.real_agreeing, pairs

    def rankable(self, name: str) -> bool:
        if self.environment and (name in _TEXT_ONLY or name == "tool_budget_used"):
            return False
        if name == "environment_terminal" and self.cap_reached:
            return False
        if not self.environment and name in _ANSWER_FACTS and self.real == 0:
            return False
        if name == "answer_agreement" and (self.real < 2 or self.agreement()[0] <= 0.0):
            return False
        if name in _GOOD:
            return True
        return name in _BAD and (name not in _SHARES or self.x[name] >= HIGH_USE)

    def _checkable_qualifier(self) -> str:
        return "" if self.checkable >= self.real else f" of {_EVIDENCE_MIN} to {_EVIDENCE_MAX} characters"

    def _excluded_note(self) -> str:
        if not self.excluded:
            return ""
        nouns = [_plural(self.excluded[kind], kind) for kind in (REFUSAL, PLACEHOLDER, PLAN) if kind in self.excluded]
        return f" ({', '.join(nouns)})"

    def _pending_facts(self) -> list[str]:
        by_target: dict[str, list[str]] = {}
        for source, target in self.pending:
            by_target.setdefault(target, []).append(f"{source}->{target}")
        return [
            f"feedback edge {edges[0]} waits for {target}'s next execution"
            if len(edges) == 1
            else f"feedback edges {', '.join(edges)} wait for {target}'s next execution"
            for target, edges in by_target.items()
        ]

    def team_line(self) -> str:
        x, n, a = self.x, self.nodes, self.real
        output = (
            f"output node {self.output_node} selected"
            if self.output_node is not None
            else "no output node selected"
        )
        if self.environment:
            facts = [_plural(n, "node"), output, self.environment_status()]
            if x["last_execution_failed"] >= 1.0:
                facts.append("the last execution failed or was cut at its output limit")
            if x["environment_call_succeeded"] >= 1.0:
                facts.append("last environment action succeeded")
            elif x["environment_call_failed"] >= 1.0:
                facts.append("last environment action failed")
            facts.extend(self._pending_facts())
            return "Team: " + ", ".join(facts) + "."
        facts = [
            _plural(n, "node"),
            f"{a} of {n} state an answer{self._excluded_note()}",
            output,
        ]
        used = _count(x["tool_used_fraction"], n)
        if used:
            facts.append(f"{used} of {n} called a tool")
        if self.search_calls and a and self.checkable:
            supported = (
                self.real_found if self.excluded else _count(x["answer_supported_by_tools"], a)
            )
            facts.append(
                f"{supported} of {a} answers appear in the search results"
                if self.checkable >= a
                else f"{supported} of {self.checkable} answers{self._checkable_qualifier()} "
                "appear in the search results"
            )
        if self.python_calls:
            clean = _count(x["python_clean_fraction"], self.python_calls)
            facts.append(
                f"{clean} of {self.python_calls} python runs in the nodes' latest executions "
                "exited cleanly"
            )
        if a >= 2:
            agreement = self.agreement()[0]
            facts.append(
                "answering nodes agree"
                if agreement >= 1.0
                else "answering nodes disagree"
                if agreement <= 0.0
                else "answering nodes partly agree"
            )
        if x["last_execution_failed"] >= 1.0:
            facts.append("the last execution failed or was cut at its output limit")
        if x["last_output_changed"] >= 1.0:
            facts.append("the last execution changed its node's output")
        facts.extend(self._pending_facts())
        return "Team: " + ", ".join(facts) + "."

    def target(self, name: str) -> float:
        if name == "answer_supported_by_tools" and self.answers:
            return min(1.0, self.checkable_all / self.answers)
        return 1.0

    def can_miss(self, name: str) -> bool:
        n, a = self.nodes, self.real
        conditions = {
            "answered_fraction": n >= 1,
            "selected_output": n >= 1,
            "answer_agreement": a >= 2 and self.agreement()[0] < 1.0,
            "answer_supported_by_tools": self.search_calls > 0 and a >= 1 and self.checkable >= 1,
            "python_clean_fraction": self.python_calls > 0,
            "environment_terminal": self.environment,
            "environment_call_succeeded": self.environment_call,
        }
        if self.environment and name in _TEXT_ONLY:
            return False
        return conditions.get(name, False) and self.x[name] < self.target(name) - 1e-9

    def _tool_outcome(self, name: str) -> tuple[str, str | None, str | None]:
        phrase, missing, problem = _TOOL_OUTCOME_PHRASES[name]
        python = self.last_tool == "python"
        if name == "environment_call_failed" and python:
            phrase, problem = _PYTHON_FAILED
        if not self.last_called:
            phrase = (
                "the most recent python run (in an earlier execution) exited with an error"
                if name == "environment_call_failed" and python
                else "the most recent tool call (in an earlier execution) failed"
                if name == "environment_call_failed"
                else "the most recent tool call (in an earlier execution) succeeded"
            )
        return phrase, missing, problem

    def phrases(self, name: str) -> tuple[str, str | None, str | None]:
        if not self.environment and name in _TOOL_OUTCOME_PHRASES:
            return self._tool_outcome(name)
        phrase, missing, problem = PHRASES[name]
        value = self.x[name]
        if name in _SHARES:
            phrase = f"{_share(value)} of {_SHARES[name]} used"
            problem = problem if value >= HIGH_USE else None
            if name == "environment_step_fraction" and self.environment_cap:
                if problem is not None and self.cap_reached:
                    problem = _STEPS_GONE
            elif problem is not None and value >= ALL_USED:
                problem = _STEPS_GONE if name == "environment_step_fraction" else _BUDGET_GONE
        elif name == "node_count_saturated" and self.nodes < 2:
            problem = None
        elif name == "selected_output" and self.environment:
            problem = None
        elif name == "answer_agreement":
            share, agreeing, pairs = self.agreement()
            if 0.0 < share < 1.0:
                phrase = f"some answering nodes agree ({agreeing} of {pairs} pairs)"
                missing = "all answering nodes agreeing"
            subject = "some answering nodes" if share > 0.0 else "the answering nodes"
            if self.unmarked:
                problem = (
                    _TEXT_DIFFERS.replace("the answering nodes", subject, 1)
                    if self.family in _CODE_FAMILIES
                    else _TEXT_DIFFERS_PLAIN.format(subject=subject)
                )
            elif self.long == 1:
                problem = _LONG_ANSWER.format(subject=subject)
            elif self.long:
                problem = _LONG_ANSWERS.format(subject=subject, count=self.long)
            else:
                problem = _DIFFERENT.format(subject=subject)
        return phrase, missing, problem

    def _not_counted(self) -> str:
        return _NOT_COUNTED_ALL if self.excluded else _NOT_COUNTED

    def missing(self, name: str) -> tuple[str, str | None]:
        phrase, missing, problem = self.phrases(name)
        value = self.x[name]
        if name == "answered_fraction" and self.real == 0:
            return (
                "a node whose output states an answer",
                f"no node has stated an answer yet; {self._not_counted()}",
            )
        if name == "answered_fraction" and value > 0.0:
            return (
                "every node's output stating an answer",
                f"some nodes have not stated an answer; {self._not_counted()}",
            )
        if name == "answer_supported_by_tools":
            q = self._checkable_qualifier()
            if value > 0.0:
                return (
                    f"every answer{q} found in the search results",
                    f"some stated answers{q} do not appear in what search returned, so they "
                    "may not be grounded in the evidence",
                )
            if self.checkable == 1:
                return (
                    f"an answer{q} found in the search results",
                    f"the stated answer{q} does not appear in what search returned, so it may "
                    "not be grounded in the evidence",
                )
            return (
                f"answers{q} found in the search results",
                f"no stated answer{q} appears in what search returned, so the answers may not "
                "be grounded in the evidence",
            )
        assert missing is not None
        return missing, problem


def contributions(
    features: Sequence[float], facts: _Facts, values: BatchedValues
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]] | None:
    base = tuple(float(item) for item in features)
    present = [name for name in FEATURE_NAMES if base[_INDEX[name]] > 0.0 and facts.rankable(name)]
    missing = [
        name for name in FEATURE_NAMES if name in _MISSABLE and facts.can_miss(name)
    ]
    if not present and not missing:
        return None
    batch = [base]
    for name in present:
        batch.append(base[: _INDEX[name]] + (0.0,) + base[_INDEX[name] + 1 :])
    for name in missing:
        batch.append(base[: _INDEX[name]] + (facts.target(name),) + base[_INDEX[name] + 1 :])
    result = [float(item) for item in values(batch)]
    if len(result) != len(batch):
        raise ValueError("batched value function returned the wrong number of values")
    reference = result[0]
    removed = result[1 : 1 + len(present)]
    added = result[1 + len(present) :]
    return (
        [(name, reference - value) for name, value in zip(present, removed, strict=True)],
        [(name, value - reference) for name, value in zip(missing, added, strict=True)],
    )


def contribution_lines(
    features: Sequence[float], facts: _Facts, values: BatchedValues
) -> list[str]:
    measured = contributions(features, facts, values)
    if measured is None:
        return [NO_FEATURE_LINE]
    present, missing = measured
    order = {name: index for index, name in enumerate(FEATURE_NAMES)}
    raising = sorted(
        (item for item in present if item[0] in _GOOD and item[1] >= CONTRIBUTION_THRESHOLD),
        key=lambda item: (-item[1], order[item[0]]),
    )[:CONTRIBUTION_TOP]
    lowering = sorted(
        (item for item in present if item[0] in _BAD and item[1] <= -CONTRIBUTION_THRESHOLD),
        key=lambda item: (item[1], order[item[0]]),
    )[:CONTRIBUTION_TOP]
    absent = sorted(
        (item for item in missing if item[1] >= CONTRIBUTION_THRESHOLD),
        key=lambda item: (-item[1], order[item[0]]),
    )[:CONTRIBUTION_TOP]
    lines = []
    if raising:
        lines.append(
            "Raising the estimate: "
            + "; ".join(f"{facts.phrases(name)[0]} ({signed(c)})" for name, c in raising)
            + "."
        )
    if lowering:
        parts = []
        for name, c in lowering:
            phrase, missing_phrase, problem = facts.phrases(name)
            shown = problem if missing_phrase is None else None
            parts.append(f"{phrase} ({signed(c)})" + (f": {shown}" if shown else ""))
        lines.append("Lowering the estimate: " + "; ".join(parts) + ".")
    if absent:
        parts = []
        for name, m in absent:
            phrase, problem = facts.missing(name)
            parts.append(f"{phrase} ({signed(m)})" + (f": {problem}" if problem else ""))
        lines.append("Missing, would raise the estimate: " + "; ".join(parts) + ".")
    if not lines and facts.excluded and facts.real == 0 and not facts.environment:
        return [NO_REAL_ANSWER_LINE]
    return lines or [NO_CONTRIBUTION_LINE]


def diagnosis_lines(
    features: Sequence[float],
    state: Mapping[str, Any],
    *,
    decision: int,
    value: float,
    start_value: float | None,
    previous_value: float | None,
    last_action: str | None,
    fitted: bool,
    values: BatchedValues,
) -> list[str]:
    facts = _Facts(features, state)
    if decision == 0:
        lines = [
            f"No node has executed yet. Estimated final score {estimate(value)} for tasks of "
            "this family."
        ]
    else:
        if start_value is None or previous_value is None or not last_action:
            raise ValueError("a later decision needs the start and previous estimates and action")
        shown, start, previous = (
            float(estimate(item)) for item in (value, start_value, previous_value)
        )
        lines = [
            f"Estimated final score {estimate(value)} (episode start {estimate(start_value)}; "
            f"{signed(shown - start)} since start; {signed(shown - previous)} "
            f"after your last action {last_action})."
        ]
        lines.append(facts.team_line())
        lines.extend(facts.nodes_lines())
    if not fitted:
        lines.append(NOT_FITTED_LINE)
    else:
        lines.extend(contribution_lines(features, facts, values))
    return lines


def diagnosis_text(*args: Any, **kwargs: Any) -> str:
    return DIAGNOSIS_SEPARATOR + "\n".join(diagnosis_lines(*args, **kwargs))


__all__ = [
    "CONTRIBUTION_THRESHOLD",
    "CONTRIBUTION_TOP",
    "DIAGNOSIS_SEPARATOR",
    "HIGH_USE",
    "NOT_FITTED_LINE",
    "NO_CONTRIBUTION_LINE",
    "NO_FEATURE_LINE",
    "NO_REAL_ANSWER_LINE",
    "PHRASES",
    "contribution_lines",
    "contributions",
    "diagnosis_lines",
    "diagnosis_text",
]
