"""Typed, answer-blind terminal and native-action wires for Step-0 evaluation."""

from __future__ import annotations

import re
from enum import StrEnum

from skillev.contracts import JsonValue
from skillev.rollout import GenerationPhase, RolloutGenerationRequest
from skillev.rollout.evaluation_sglang import EvaluationGenerationConstraint


class StepZeroTerminalMode(StrEnum):
    SHORT_ANSWER = "short-answer"
    AIME_INTEGER = "integer-0-999"
    NATURAL_LANGUAGE = "natural-language"
    PYTHON_SOURCE = "python-source"


def terminal_constraint(mode: StepZeroTerminalMode) -> EvaluationGenerationConstraint:
    """Return a syntax-only grammar where it is safe and benchmark-native."""

    if mode is StepZeroTerminalMode.AIME_INTEGER:
        # AIME submissions are integers in [0, 999], while the official answer
        # representation may retain leading zeroes (for example ``042``).
        return EvaluationGenerationConstraint(regex=r"[0-9]{1,3}")
    if mode is StepZeroTerminalMode.SHORT_ANSWER:
        return EvaluationGenerationConstraint(regex=r"[^\r\n]{1,512}")
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
    }
    if mode == "webshop" and available_actions:
        blocked = {item.casefold() for item in blocked_actions}
        has_result_return = any(item.casefold() == "click[< prev]" for item in available_actions)
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
            and not (has_result_return and action.casefold() == "click[back to search]")
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
        StepZeroTerminalMode.SHORT_ANSWER: (
            "Output one concise final answer on one line. Do not explain it and do not emit JSON."
        ),
        StepZeroTerminalMode.AIME_INTEGER: (
            "Output exactly one base-10 integer from 0 through 999. No box, prose, or punctuation."
        ),
        StepZeroTerminalMode.NATURAL_LANGUAGE: (
            "Write the complete benchmark-native natural-language response now. Do not emit JSON."
        ),
        StepZeroTerminalMode.PYTHON_SOURCE: (
            "Output only the complete executable Python source. Do not use markdown fences or JSON."
        ),
    }
    if mode is StepZeroTerminalMode.AIME_INTEGER:
        return (
            "The reasoning pass below is the sole authority for this terminal wire. "
            "Transcribe its last explicitly boxed integer; if none is boxed, transcribe its "
            "last explicit final integer. Do not solve the task again or introduce a different "
            "value.\n"
            f"Reasoning:\n{reasoning_text}\n"
            f"Pass contract: {instructions[mode]}\n"
            "Terminal payload:\n"
        )
    if mode is StepZeroTerminalMode.SHORT_ANSWER:
        question = (
            f"Public question:\n{public_question.strip()}\n" if public_question is not None else ""
        )
        return (
            "Act only as an answer-span transcriber. The reasoning pass below is the sole "
            "authority. If it contains `Answer hypothesis:`, copy only the value after that "
            "marker. Otherwise extract the shortest name, noun phrase, number, date, place, or "
            "type in its final conclusion that directly answers the public question. Do not "
            "copy a sentence, solve again, introduce an alternative, or add an explanation.\n"
            "Example reasoning: The requested capital is Alpha City, founded in 1900.\n"
            "Example terminal payload: Alpha City\n"
            "Example reasoning: The animal described is a river otter.\n"
            "Example terminal payload: river otter\n"
            f"{question}"
            f"Reasoning:\n{reasoning_text}\n"
            f"Pass contract: {instructions[mode]}\n"
            "Terminal payload:\n"
        )
    return (
        f"{initial_context}### Step 1\n"
        f"Reasoning:\n{reasoning_text}\n"
        f"Pass contract: {instructions[mode]}\n"
        "Terminal payload:\n"
    )


def project_terminal_candidate(mode: StepZeroTerminalMode, text: str) -> str | None:
    """Project only the model's explicit payload into the benchmark wire."""

    if type(text) is not str or not text.strip():
        return None
    candidate = text.strip()
    if mode is StepZeroTerminalMode.AIME_INTEGER:
        if re.fullmatch(r"[0-9]{1,3}", candidate) is None:
            return None
        return rf"\boxed{{{int(candidate)}}}"
    if mode is StepZeroTerminalMode.SHORT_ANSWER:
        if "\n" in candidate or "\r" in candidate or len(candidate) > 512:
            return None
        return candidate
    return candidate


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
