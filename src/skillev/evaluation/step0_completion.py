"""Typed, answer-blind terminal and native-action wires for Step-0 evaluation."""

from __future__ import annotations

import re
from dataclasses import replace
from enum import StrEnum

from skillev.contracts import JsonValue
from skillev.rollout import GenerationPhase, RolloutGenerationRequest
from skillev.rollout.codec import StructuredJsonActionCodec
from skillev.rollout.evaluation_sglang import EvaluationGenerationConstraint
from skillev.runtime import ActionKind, ActionParseResult, ActionParseStatus, StructuredAction

from .direct_baseline.parsing import (
    ParseStatus,
    canonical_native_action,
    parse_short_answer,
)
from .integer_payload import (
    is_integer_alternatives,
    is_integer_payload,
    parse_explicit_integer_payload,
)
from .python_payload import decode_python_module
from .scienceworld_commands import TYPED_COMMANDS


class StepZeroTerminalMode(StrEnum):
    SHORT_ANSWER = "short-answer"
    AIME_INTEGER = "integer-0-999"
    NATURAL_LANGUAGE = "natural-language"
    PYTHON_SOURCE = "python-source"


def parse_native_tool_action(mode: str, text: str) -> ActionParseResult:
    """Accept the explicit JSON call or a single native call, without semantic repair."""

    parsed = StructuredJsonActionCodec().parse(text)
    if parsed.status is ActionParseStatus.VALID:
        action = parsed.action
        if action is not None and action.resource_id == "webshop":
            name = {"点击": "click", "搜索": "search"}.get(action.name, action.name)
            return replace(parsed, action=replace(action, name=name))
        return parsed
    native = text.strip()
    if native.startswith("Action:"):
        native = native.removeprefix("Action:").strip()
    if mode == "webshop":
        match = re.fullmatch(r"(click|search)\[([^\[\]\r\n]+)\]", canonical_native_action(native))
        if match is None:
            return parsed
        name = match[1]
        arguments = {"target" if name == "click" else "query": match[2]}
        resource = "webshop"
    elif mode == "alfworld":
        if not native or "\n" in native or "\r" in native:
            return parsed
        name, arguments, resource = "act", {"command": native}, "alfworld"
    elif mode == "trivia-search":
        match = re.fullmatch(r"search\[([^\[\]\r\n]+)\]", native)
        if match is None:
            return parsed
        name, arguments, resource = "search", {"query": match[1]}, "triviaqa"
    else:
        return parsed
    return ActionParseResult(
        ActionParseStatus.VALID,
        StructuredAction(ActionKind.TOOL, name, arguments, resource_id=resource),
        None,
    )


def terminal_constraint(mode: StepZeroTerminalMode) -> EvaluationGenerationConstraint:
    """Leave natural answers unconstrained; parse explicit native submissions later."""

    del mode
    return EvaluationGenerationConstraint()


def native_action_constraint(
    mode: str,
    *,
    available_actions: tuple[str, ...] = (),
    blocked_actions: tuple[str, ...] = (),
) -> EvaluationGenerationConstraint:
    """Return the executable JSON grammar for the *current* native surface.

    WebShop changes between a parameterized search box and a finite click
    surface, while ALFWorld exposes a finite set of admissible commands.  The
    Step-0 bridge used to advertise the union for every turn and reject stale
    choices only after generation.  Besides wasting the internal repair
    horizon, that let an episode's first action surface dominate all later
    turns.  When the environment supplies its public surface, constrain only
    to actions executable now; the empty-surface fallback remains useful for
    static fixtures and legacy callers.
    """

    if blocked_actions:
        raise ValueError("semantic action pruning belongs only to the legacy grammar")
    schemas: dict[str, JsonValue] = {
        "trivia-search": _tool_union(
            (_tool_schema("triviaqa", "search", {"query": _string_schema()}),)
        ),
        "webshop": _tool_union(
            (
                _tool_schema("webshop", "search", {"query": _string_schema()}),
                _tool_schema("webshop", "click", {"target": _string_schema()}),
            )
        ),
        "alfworld": _tool_union((_tool_schema("alfworld", "act", {"command": _string_schema()}),)),
        "scienceworld": _tool_union(
            (
                _tool_schema("scienceworld", "act", {"command": _string_schema()}),
                *(
                    _tool_schema("scienceworld", name, {"target": _string_schema()})
                    for name in TYPED_COMMANDS
                    if name in available_actions
                ),
            )
        ),
    }
    if mode == "webshop" and available_actions:
        blocked = {item.casefold() for item in blocked_actions}
        branches: list[dict[str, JsonValue]] = []
        exact_search_queries = tuple(
            action[len("search[") : -1]
            for action in available_actions
            if action.casefold().startswith("search[")
            and action.endswith("]")
            and action[len("search[") : -1].strip()
        )
        if "search" in available_actions:
            branches.append(_tool_schema("webshop", "search", {"query": _string_schema()}))
        elif exact_search_queries:
            branches.append(
                _tool_schema(
                    "webshop",
                    "search",
                    {"query": _string_enum_schema(exact_search_queries)},
                )
            )
        click_targets = tuple(
            action[len("click[") : -1]
            for action in available_actions
            if action.startswith("click[")
            and action.endswith("]")
            and action.casefold() not in blocked
        )
        if click_targets:
            branches.append(
                _tool_schema(
                    "webshop",
                    "click",
                    {"target": _string_enum_schema(click_targets)},
                )
            )
        if branches:
            return EvaluationGenerationConstraint(
                json_schema=_tool_union(tuple(branches)),
                apply_json_root_boundary=True,
            )
    if mode == "alfworld" and available_actions:
        commands = tuple(dict.fromkeys(action for action in available_actions if action.strip()))
        if commands:
            return EvaluationGenerationConstraint(
                json_schema=_tool_schema(
                    "alfworld",
                    "act",
                    {"command": _string_enum_schema(commands)},
                ),
                apply_json_root_boundary=True,
            )
    try:
        schema = schemas[mode]
    except KeyError as exc:
        raise ValueError("unsupported Step-0 native action mode") from exc
    return EvaluationGenerationConstraint(
        json_schema=schema,
        apply_json_root_boundary=True,
    )


def render_terminal_prefix(
    initial_context: str,
    *,
    reasoning_text: str,
    mode: StepZeroTerminalMode,
    public_question: str | None = None,
) -> str:
    """Render a terminal-only pass with no contradictory JSON-action instruction."""

    if not isinstance(initial_context, str) or not initial_context:
        raise ValueError("initial_context must be non-empty text")
    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
        raise ValueError("reasoning_text must be non-empty text")
    if public_question is not None and (
        type(public_question) is not str or not public_question.strip()
    ):
        raise ValueError("public_question must be non-empty text or null")
    instructions = {
        StepZeroTerminalMode.SHORT_ANSWER: ("Give a concise final answer to the question."),
        StepZeroTerminalMode.AIME_INTEGER: (
            "State your final integer from 0 through 999; a boxed integer is also accepted."
        ),
        StepZeroTerminalMode.NATURAL_LANGUAGE: (
            "Write the response to the conversation in natural language."
        ),
        StepZeroTerminalMode.PYTHON_SOURCE: (
            "Provide executable Python source implementing the requested function."
        ),
    }
    return (
        f"{initial_context}\n"
        f"Draft reasoning (may be revised):\n{reasoning_text}\n"
        f"Final response: {instructions[mode]}\n"
        "Terminal payload:\n"
    )


def _python_candidate(text: str) -> str | None:
    return decode_python_module(text)


def project_terminal_candidate(mode: StepZeroTerminalMode, text: str) -> str | None:
    """Project only the model's explicit payload into the benchmark wire."""

    if type(text) is not str or not text.strip():
        return None
    if mode is StepZeroTerminalMode.PYTHON_SOURCE:
        return _python_candidate(text)
    candidate = text.strip()
    if mode is StepZeroTerminalMode.AIME_INTEGER:
        if is_integer_payload(candidate):
            value = parse_explicit_integer_payload(candidate)
            return rf"\boxed{{{value}}}" if value is not None else None
        if is_integer_alternatives(candidate):
            return None
        finals = re.findall(r"\\boxed\{([^{}]*)\}", candidate)
        # A real owner response explained its calculation and then placed the
        # integer alone on the final line. Do not require the entire response
        # to contain only that integer, or discard this explicit final field.
        # An earlier number followed by discussion need not be a final answer.
        # Scalar-only lists were handled above without selecting their last value.
        last_line = candidate.splitlines()[-1].strip()
        if re.fullmatch(r"[0-9]{1,3}[ \t]*\.?", last_line):
            finals.append(last_line)
        finals.extend(
            re.findall(
                r"^(?:Final(?: answer)?|Answer|The answer is)\s*[:=]?\s*([0-9]+)\s*\.?\s*$",
                candidate,
                flags=re.I | re.M,
            )
        )
        values = {parse_explicit_integer_payload(value) for value in finals}
        if not values or None in values:
            return None
        return rf"\boxed{{{values.pop()}}}" if len(values) == 1 else None
    if mode is StepZeroTerminalMode.SHORT_ANSWER:
        parsed = parse_short_answer(candidate)
        if parsed.status is ParseStatus.AMBIGUOUS:
            return None
        # No format-only zero for multiline prose: the native EM/F1 scorer
        # evaluates it as-is, never searches the text for a reference answer.
        return parsed.value if parsed.value is not None else candidate
    return candidate


def terminal_submission_feedback(mode: StepZeroTerminalMode) -> str:
    """Describe an unsubmitted payload, without choosing or grading an answer."""
    payload = {
        StepZeroTerminalMode.PYTHON_SOURCE: "complete executable Python module",
        StepZeroTerminalMode.AIME_INTEGER: "final integer from 0 through 999",
        StepZeroTerminalMode.SHORT_ANSWER: "final short answer",
        StepZeroTerminalMode.NATURAL_LANGUAGE: "final response",
    }[mode]
    return (
        f"Your response was not submitted: the interface could not identify one {payload}. "
        f"Please send the {payload} you want to submit."
    )


def evaluation_request(
    *,
    input_ids: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
    decoding_profile_id: str,
    policy_snapshot_id: str,
    phase: GenerationPhase,
) -> RolloutGenerationRequest:
    return RolloutGenerationRequest(
        phase=phase,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        decoding_snapshot_id=decoding_profile_id,
        expected_policy_snapshot_id=policy_snapshot_id,
    )


def _string_schema() -> dict[str, JsonValue]:
    return {"minLength": 1, "type": "string"}


def _string_enum_schema(values: tuple[str, ...]) -> dict[str, JsonValue]:
    return {"enum": list(dict.fromkeys(values)), "type": "string"}


def _tool_schema(
    resource_id: str,
    name: str,
    arguments: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "additionalProperties": False,
        "properties": {
            "arguments": {
                "additionalProperties": False,
                "properties": arguments,
                "required": list(arguments),
                "type": "object",
            },
            "kind": {"const": "tool"},
            "name": {"const": name},
            "resource_id": {"const": resource_id},
        },
        "required": ["arguments", "kind", "name", "resource_id"],
        "type": "object",
    }


def _tool_union(schemas: tuple[dict[str, JsonValue], ...]) -> JsonValue:
    return schemas[0] if len(schemas) == 1 else {"oneOf": list(schemas)}


__all__ = [
    "StepZeroTerminalMode",
    "evaluation_request",
    "native_action_constraint",
    "project_terminal_candidate",
    "render_terminal_prefix",
    "terminal_constraint",
]
