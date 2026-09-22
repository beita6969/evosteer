"""Controller-owned history records for interactive reasoning/action phases.

This projection fixes an observed failure: model drafts invented the same
Action/Observation headings as real history. Strings are now values in typed
records, and only actual environment feedback refreshes the public state.
No generated action or reasoning is rewritten, selected, or discarded.
"""

from __future__ import annotations

import json
from typing import Literal

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.action_wire import NATIVE_CARRIER_WIRES, NATIVE_TOOL_WIRES
from skillev.contracts.ttb_trajectory import TrajectoryStep

from .phase_context import (
    OBSERVED_PUBLIC_HISTORY,
    TYPED_PUBLIC_HISTORY,
    PhaseContextSpec,
    deliverable_instruction,
    native_action_instruction,
    reasoning_tool_reference,
    skill_resource_instruction,
    token_budget_notice,
)

TYPED_HISTORY_HEADER = "SKILLEV_TYPED_HISTORY_V1\n"
_STATE_BOUNDARY = (
    "Reasoning drafts are the owner's unexecuted thoughts, not tool results. "
    "Only controller-provided execution feedback records what actually happened. "
    "Use the latest real observation and admissible commands when choosing an action. "
    "Execution status 'success' means the tool returned, not that the task was solved. "
    "The environment, not a claim in a draft, determines task completion."
)


def render_typed_phase(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    *,
    phase: Literal["reasoning", "forward", "hindsight"],
    instruction: str,
    current_text: str | None = None,
) -> str | None:
    """Render from typed controller inputs, never by parsing headings in prose."""
    spec, initial_body = PhaseContextSpec.split(initial_text)
    if spec is None or spec.history_format not in {TYPED_PUBLIC_HISTORY, OBSERVED_PUBLIC_HISTORY}:
        return None
    history: list[JsonValue] = [
        {
            "step": step.index,
            "owner_reasoning_draft": step.reasoning_text,
            "submitted_action": step.action_text,
            "execution_feedback": {
                "status": step.observation_status,
                "observation": step.observation_text,
            },
        }
        for step in previous_steps
    ]
    value: dict[str, JsonValue] = {
        "initial_text": initial_body,
        "history": history,
        "step": step_index,
        "phase": phase,
        "instruction": instruction,
    }
    if phase == "forward":
        value["owner_reasoning_draft"] = current_text
    elif phase == "hindsight":
        value["current_execution_observation"] = current_text
    return spec.wrap(TYPED_HISTORY_HEADER + canonical_json(value))


def _observed_state(
    observation: str, *, preserve_terminal: bool = False
) -> dict[str, JsonValue] | None:
    try:
        value = json.loads(observation)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    commands = value.get("admissible_commands")
    text = value.get("text", value.get("initial_observation"))
    if (
        not isinstance(commands, list)
        or any(not isinstance(command, str) for command in commands)
        or not isinstance(text, str)
    ):
        return None
    state: dict[str, JsonValue] = {"text": text, "admissible_commands": commands}
    if preserve_terminal:
        terminal = value.get("terminal")
        state["native_episode_terminal"] = terminal if type(terminal) is bool else None
        state["native_command"] = (
            value.get("command") if isinstance(value.get("command"), str) else None
        )
        state["native_max_steps"] = (
            value.get("max_steps") if type(value.get("max_steps")) is int else None
        )
    return state


def typed_phase_chat_messages(
    spec: PhaseContextSpec, body: str
) -> tuple[list[dict[str, str]], list[dict[str, JsonValue]] | None]:
    from .interface import rollout_chat_messages

    # H0 alone is also encoded (e.g. for the Z input and window size). Only the
    # controller renderer can place the history envelope at this exact boundary.
    value = (
        json.loads(body[len(TYPED_HISTORY_HEADER) :])
        if body.startswith(TYPED_HISTORY_HEADER)
        else None
    )
    phase = (
        value["phase"]
        if value is not None
        else ("forward" if body.endswith("Action:\n") else "reasoning")
    )
    system = (
        "Reason about the public task and current state. This is the reasoning phase; "
        "an executable action is requested separately. "
        if phase == "reasoning"
        else "Send exactly one action using the declared interface, without commentary. "
    ) + _STATE_BOUNDARY
    if phase != "reasoning" and spec.action_wire in NATIVE_CARRIER_WIRES:
        system = native_action_instruction(spec) + " " + _STATE_BOUNDARY
    if phase == "reasoning":
        system += reasoning_tool_reference(spec)
    system += deliverable_instruction(spec, reasoning=phase == "reasoning")
    system += token_budget_notice(spec, reasoning=phase == "reasoning")
    system += skill_resource_instruction(spec, reasoning=phase == "reasoning")
    messages = rollout_chat_messages(
        value["initial_text"] if value is not None else body, system_message=system
    )
    tools = (
        json.loads(spec.tools_json)
        if phase != "reasoning" and spec.action_wire in NATIVE_TOOL_WIRES
        else None
    )
    if value is None:
        return messages, tools

    observed_v2 = spec.history_format == OBSERVED_PUBLIC_HISTORY
    state = _observed_state(spec.initial_public_state_json, preserve_terminal=observed_v2)
    state_step = 0
    for record in value["history"]:
        # Use controller data messages, not assistant messages: Qwen's template
        # may remove text before </think> in past assistant messages. JSON keeps
        # the entire raw draft and its embedded mock headings as one string.
        messages.append(
            {"role": "user", "content": "Controller history record:\n" + canonical_json(record)}
        )
        observed = _observed_state(
            record["execution_feedback"]["observation"], preserve_terminal=observed_v2
        )
        if observed is not None:
            state, state_step = observed, record["step"]

    if phase == "forward":
        messages.append(
            {
                "role": "user",
                "content": "Current owner reasoning draft (not executed):\n"
                + canonical_json({"owner_reasoning_draft": value["owner_reasoning_draft"]}),
            }
        )
    elif phase == "hindsight":
        # B sees the actual current observation but never the current reasoning
        # or action. This is the same public hindsight condition as before.
        observation = value["current_execution_observation"]
        messages.append(
            {
                "role": "user",
                "content": "Current execution observation:\n"
                + canonical_json({"observation": observation}),
            }
        )
        observed = _observed_state(observation, preserve_terminal=observed_v2)
        if observed is not None:
            state, state_step = observed, value["step"]

    refresh: dict[str, JsonValue] = {
        "current_turn": value["step"],
        "latest_environment_observation_turn": state_step,
        "latest_environment_state": state,
    }
    if observed_v2:
        from .observed_history import history_facts

        refresh["controller_history_facts"] = history_facts(value["history"])
        initial_state = _observed_state(spec.initial_public_state_json, preserve_terminal=True)
        refresh["native_environment_max_steps"] = (
            initial_state.get("native_max_steps") if initial_state else None
        )
        refresh["budget_counting"] = (
            "Controller turns include advisory reads; native environment commands are separate."
        )
    if spec.max_turns is not None:
        refresh["controller_max_turns"] = spec.max_turns
        refresh["turns_remaining_including_current"] = max(0, spec.max_turns - value["step"] + 1)
    if value["history"]:
        last = value["history"][-1]
        refresh["previous_submitted_action"] = last["submitted_action"]
        refresh["previous_execution_feedback"] = last["execution_feedback"]
    messages.append(
        {
            "role": "user",
            "content": "Controller state for this call (not a model prediction):\n"
            + canonical_json(refresh)
            + "\n"
            + value["instruction"],
        }
    )
    return messages, tools
