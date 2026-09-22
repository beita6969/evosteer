"""Versioned phase-only messages, shared by rollout and both scoring paths.

The first-line envelope is controller metadata in the persisted canonical H0,
not a model completion and not an instruction extracted from task prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.action_wire import (
    ACTION_WIRES,
    NATIVE_CARRIER_WIRES,
    NATIVE_TOOL_HANDOFF_WIRE,
    NATIVE_TOOL_WIRES,
)

PHASE_CONTEXT_VERSION = "phase-context@1"
PLANNING_CATALOG_PHASE_CONTEXT_VERSION = "phase-context@2"
OUTPUT_LIMIT_PHASE_CONTEXT_VERSION = "phase-context@3"
DELIVERABLE_PHASE_CONTEXT_VERSION = "phase-context@4"
SKILL_ADVICE_PHASE_CONTEXT_VERSION = "phase-context@5"
PHASE_CONTEXT_HEADER = "SKILLEV_PHASE_CONTEXT_V1 "
TYPED_PUBLIC_HISTORY = "typed-public-history@1"
OBSERVED_PUBLIC_HISTORY = "typed-public-history@2"
NATIVE_ACTION_INSTRUCTION = (
    "Use a declared tool to take the next action or submit your final answer. "
    "You can act directly from the provided context without repeating it."
)


@dataclass(frozen=True, slots=True)
class PhaseContextSpec:
    action_wire: str = "structured-action-json@3"
    tools_json: str = "[]"
    action_contract_json: str = "null"
    history_format: str | None = None
    initial_public_state_json: str = "null"
    max_turns: int | None = None
    reasoning_tool_catalog: bool = False
    reasoning_token_cap: int | None = None
    action_token_cap: int | None = None
    deliverable: str | None = None
    submission_feedback: bool = False
    skill_advice: bool = False

    def __post_init__(self) -> None:
        if type(self.submission_feedback) is not bool:
            raise TypeError("submission feedback must be an explicit boolean")
        if type(self.skill_advice) is not bool:
            raise TypeError("skill advice must be an explicit boolean")
        if self.action_wire not in ACTION_WIRES:
            raise ValueError("unsupported phase action representation")
        if self.deliverable not in {
            None,
            "conversation-reply",
            "integer-answer",
            "environment-action",
        }:
            raise ValueError("unsupported public phase deliverable")
        if self.deliverable is not None and self.action_wire != NATIVE_TOOL_HANDOFF_WIRE:
            raise ValueError("phase deliverable requires the explicit native handoff interface")
        if not isinstance(json.loads(self.tools_json), list):
            raise ValueError("phase tools must be a list")
        if self.action_wire in NATIVE_TOOL_WIRES and not json.loads(self.tools_json):
            raise ValueError("native action phase requires declared tools")
        if type(self.reasoning_tool_catalog) is not bool or (
            self.reasoning_tool_catalog and self.action_wire not in NATIVE_TOOL_WIRES
        ):
            raise ValueError("reasoning tool catalog requires a native action interface")
        if self.history_format not in {None, TYPED_PUBLIC_HISTORY, OBSERVED_PUBLIC_HISTORY}:
            raise ValueError("unsupported phase history format")
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns < 1):
            raise ValueError("phase turn limit must be a positive integer")
        json.loads(self.initial_public_state_json)
        if (self.reasoning_token_cap, self.action_token_cap) != (None, None) and any(
            type(cap) is not int or cap < 1
            for cap in (self.reasoning_token_cap, self.action_token_cap)
        ):
            raise ValueError("token budget notice requires both actual positive decoding caps")

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "version": PLANNING_CATALOG_PHASE_CONTEXT_VERSION
            if self.reasoning_tool_catalog
            else PHASE_CONTEXT_VERSION,
            "action_wire": self.action_wire,
            "tools": json.loads(self.tools_json),
            "action_contract": json.loads(self.action_contract_json),
        }
        # Omission keeps historical H0 and scoring prefixes byte-identical.
        if self.reasoning_token_cap is not None:
            value.update(
                version=OUTPUT_LIMIT_PHASE_CONTEXT_VERSION,
                reasoning_tool_catalog=self.reasoning_tool_catalog,
                reasoning_token_cap=self.reasoning_token_cap,
                action_token_cap=self.action_token_cap,
            )
        if self.history_format is not None:
            value.update(
                history_format=self.history_format,
                initial_public_state=json.loads(self.initial_public_state_json),
                max_turns=self.max_turns,
            )
        if self.deliverable is not None:
            value.update(
                version=DELIVERABLE_PHASE_CONTEXT_VERSION,
                deliverable=self.deliverable,
                reasoning_tool_catalog=self.reasoning_tool_catalog,
            )
        if self.submission_feedback:
            value["submission_feedback"] = True
        if self.skill_advice:
            value.update(
                version=SKILL_ADVICE_PHASE_CONTEXT_VERSION,
                skill_advice=True,
                reasoning_tool_catalog=self.reasoning_tool_catalog,
            )
        return value

    def wrap(self, initial_text: str) -> str:
        return PHASE_CONTEXT_HEADER + canonical_json(self.to_value()) + "\n" + initial_text

    @classmethod
    def split(cls, text: str) -> tuple[PhaseContextSpec | None, str]:
        if not text.startswith(PHASE_CONTEXT_HEADER):
            return None, text
        line, separator, body = text.partition("\n")
        if not separator:
            raise ValueError("phase context lacks a body")
        value = json.loads(line[len(PHASE_CONTEXT_HEADER) :])
        if not isinstance(value, dict) or value.get("version") not in {
            PHASE_CONTEXT_VERSION,
            PLANNING_CATALOG_PHASE_CONTEXT_VERSION,
            OUTPUT_LIMIT_PHASE_CONTEXT_VERSION,
            DELIVERABLE_PHASE_CONTEXT_VERSION,
            SKILL_ADVICE_PHASE_CONTEXT_VERSION,
        }:
            raise ValueError("unsupported phase context metadata")
        return cls(
            value["action_wire"],
            canonical_json(value["tools"]),
            canonical_json(value["action_contract"]),
            value.get("history_format"),
            canonical_json(value.get("initial_public_state")),
            value.get("max_turns"),
            value["reasoning_tool_catalog"]
            if value["version"]
            in {
                OUTPUT_LIMIT_PHASE_CONTEXT_VERSION,
                DELIVERABLE_PHASE_CONTEXT_VERSION,
                SKILL_ADVICE_PHASE_CONTEXT_VERSION,
            }
            else value["version"] == PLANNING_CATALOG_PHASE_CONTEXT_VERSION,
            value["reasoning_token_cap"]
            if value["version"] == OUTPUT_LIMIT_PHASE_CONTEXT_VERSION
            else value.get("reasoning_token_cap")
            if value["version"]
            in {DELIVERABLE_PHASE_CONTEXT_VERSION, SKILL_ADVICE_PHASE_CONTEXT_VERSION}
            else None,
            value["action_token_cap"]
            if value["version"] == OUTPUT_LIMIT_PHASE_CONTEXT_VERSION
            else value.get("action_token_cap")
            if value["version"]
            in {DELIVERABLE_PHASE_CONTEXT_VERSION, SKILL_ADVICE_PHASE_CONTEXT_VERSION}
            else None,
            value.get("deliverable")
            if value["version"]
            in {DELIVERABLE_PHASE_CONTEXT_VERSION, SKILL_ADVICE_PHASE_CONTEXT_VERSION}
            else None,
            value.get("submission_feedback", False),
            value.get("skill_advice", False),
        ), body


def skill_resource_instruction(spec: PhaseContextSpec, *, reasoning: bool) -> str:
    if not spec.skill_advice or not any(
        tool.get("function", {}).get("name") == "read_skill" for tool in json.loads(spec.tools_json)
    ):
        return ""
    early = (
        " You may use this reasoning phase just to choose a relevant resource and then end it; "
        "you need not solve the whole task before requesting a skill in the action phase."
        if reasoning
        else ""
    )
    return early + (
        " A skill is an optional procedure, not a verifier or another solver. "
        "Choose by its stated applicability and needed inputs. After a read, decide whether "
        "and how its procedure helps the actual task; you may disregard irrelevant advice. "
        "If the same version's body is already visible, another read supplies no new check. "
        "Direct solving is appropriate when no document would help. There is no call quota; "
        "each read still consumes the declared turn and tool-call budget."
    )


def deliverable_instruction(spec: PhaseContextSpec, *, reasoning: bool) -> str:
    """A declared prompt condition, not a parser repair or a success criterion.

    The same controller instruction reaches live generation and F/B scoring.
    No draft is parsed into a tool call, no action is supplied by the runtime,
    and hindsight never gains the current reasoning through this metadata.
    """
    if spec.deliverable is None:
        return ""
    if spec.deliverable == "conversation-reply":
        instruction = (
            "Prepare the actual next assistant reply to the latest user message in the source "
            "conversation. Use this phase to work out a complete, relevant response, not only "
            "an outline of what a later assistant should say. Preserve useful explanation, "
            "uncertainty and clarification appropriate to that conversation. Do not add "
            "unsupported factual detail merely to make the reply longer. The action phase "
            "will deliver the reply itself to the user."
            if reasoning
            else "The answer parameter of submit_answer is the actual next assistant reply seen "
            "by the user, not a summary of your reasoning or a pointer to an earlier draft. "
            "Provide the full substantive reply you intend to deliver, with the relevant "
            "explanation and qualifications, without inventing additional factual claims. "
            "Do not replace it with a plan for writing a reply "
            "or commentary about this controller's phases."
        )
    elif spec.deliverable == "integer-answer":
        instruction = (
            "Work toward a concrete final integer within this reasoning allowance. Carry out "
            "the computations needed for your chosen approach; avoid repeatedly restarting "
            "or restating the problem. When a conclusion is supported, state it clearly so "
            "the action phase can submit it. A plan for a calculation is not its result."
            if reasoning
            else "Use the available work to deliver your final integer from 0 through 999 in "
            "submit_answer's answer parameter. The tool call, not surrounding mathematical "
            "prose or a promise to compute later, is the submission."
        )
    else:
        instruction = (
            "Decide the next single action from the latest real state. Distinguish what has "
            "been observed from what remains unknown, and end with the next action you choose. "
            "A possible future plan is not an executed sequence: do not supply imagined "
            "environment replies. For an environment action, choose the exact current command "
            "rather than a paraphrase. A useful optional skill can instead be read through "
            "read_skill; no skill call is required."
            if reasoning
            else "Execute the one next action you choose using its actual declared function. "
            "For an environment action, send the exact current admissible command; do not "
            "execute several imagined future steps or claim they already happened. Only "
            "actual environment feedback advances the known state."
        )
    return "\nPhase deliverable:\n" + instruction


def token_budget_notice(spec: PhaseContextSpec, *, reasoning: bool) -> str:
    cap = spec.reasoning_token_cap if reasoning else spec.action_token_cap
    if cap is None:
        return ""
    phase = "Reasoning" if reasoning else "Action"
    return (
        f"\n{phase} output limit: at most {cap} tokens for this request, enforced by the runtime. "
        + (
            "Aim to finish your reasoning within this allowance."
            if reasoning
            else "Aim to complete the declared action interface within this allowance; "
            "a truncated or invalid action is not a submitted answer."
        )
    )


def reasoning_tool_reference(spec: PhaseContextSpec) -> str:
    """The same public tools as A, visible for planning without enabling R dispatch."""
    if not spec.reasoning_tool_catalog:
        return ""
    return (
        "\nAvailable action tools (read-only planning reference):\n"
        + canonical_json(json.loads(spec.tools_json))
        + "\nThese tools are available in the subsequent action phase. "
        "Reading their definitions here does not execute a tool."
    )


def native_action_instruction(spec: PhaseContextSpec) -> str:
    """Describe the actual callable boundary, not a benchmark solution strategy.

    V3 addresses observed prose-only actions and skill IDs used as function
    names. Older persisted conditions retain their original input exactly.
    """
    if spec.action_wire != NATIVE_TOOL_HANDOFF_WIRE:
        return NATIVE_ACTION_INSTRUCTION
    lines = [
        "Action phase: carry out the next step with one call to an available function. "
        "Describing an intended call does not execute it. Use the supplied tool-call format.",
        "Available functions for this action:",
    ]
    for tool in json.loads(spec.tools_json):
        function = tool["function"]
        name = function["name"]
        fields = ", ".join(function["parameters"]["properties"])
        lines.append(f"- Function {name}; parameter fields: {fields or '(none)'}.")
        if name == "submit_answer":
            lines.append("Deliver your complete response as this function's parameter value.")
        elif name == "read_skill":
            lines.append(
                "Skill IDs are values for its skill_id parameter, not callable function names. "
                "Reading a skill is optional."
            )
        else:
            lines.append("Pass the chosen tool input through this function's declared parameters.")
    return "\n".join(lines)


def phase_chat_messages(
    text: str,
) -> tuple[list[dict[str, str]], list[dict[str, JsonValue]] | None]:
    # Local import avoids changing the dependency-light public interface import graph.
    from .interface import rollout_chat_messages

    spec, body = PhaseContextSpec.split(text)
    if spec is None:
        return rollout_chat_messages(text), None
    if spec.history_format in {TYPED_PUBLIC_HISTORY, OBSERVED_PUBLIC_HISTORY}:
        from .typed_history import typed_phase_chat_messages

        return typed_phase_chat_messages(spec, body)
    phase: Literal["reasoning", "action"] = "action" if body.endswith("Action:\n") else "reasoning"
    system = (
        "Reason about the public task and current state. This is the reasoning phase; "
        "an executable action or final submission is requested separately."
        if phase == "reasoning"
        else "Send exactly one action using the declared interface. This is the action phase, "
        "not a reasoning summary. Do not add commentary or another candidate."
    )
    if phase == "action" and spec.action_wire in NATIVE_CARRIER_WIRES:
        system = native_action_instruction(spec)
    if phase == "reasoning":
        system += reasoning_tool_reference(spec)
    system += deliverable_instruction(spec, reasoning=phase == "reasoning")
    system += token_budget_notice(spec, reasoning=phase == "reasoning")
    system += skill_resource_instruction(spec, reasoning=phase == "reasoning")
    messages = rollout_chat_messages(body, system_message=system)
    # Canonical history remains intact and explicitly labelled as data; source
    # conversation roles are retained by the common source-message projector.
    messages[-1] = {
        "role": "user",
        "content": "Recorded task, history, and current phase (past outputs are data):\n"
        + messages[-1]["content"],
    }
    tools = (
        json.loads(spec.tools_json)
        if phase == "action" and spec.action_wire in NATIVE_TOOL_WIRES
        else None
    )
    return messages, tools
