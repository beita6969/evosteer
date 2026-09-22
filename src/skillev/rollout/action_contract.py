"""Frozen public action interface shared by H0 and execution.

This version preserves raw JSON sampling, legacy wire wrappers and current
admission behavior. It never selects an action from prose or corrects answers.
The environment remains authoritative for native completion-value validation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.contracts.skill_invocation import (
    SkillInvocationAdmissionError,
    canonical_invoked_skill_ids,
)
from skillev.runtime.contracts import ActionKind, StructuredAction

from .action_surface import ActionSurface, TerminalMode, ToolActionSpecV2


@dataclass(frozen=True, slots=True)
class AdmittedAction:
    action: StructuredAction
    error: tuple[str, str] | None = None
    invoked_skill_ids: tuple[str, ...] = ()
    completion: bool = False
    submission: JsonValue = None


@dataclass(frozen=True, slots=True)
class ActionContract:
    # JSON owns its contents: callers cannot mutate a nested surface after freezing.
    surface_json: str | None
    retrieved_skill_ids: tuple[str, ...] = ()
    active_skill_ids: tuple[str, ...] = ()

    @classmethod
    def freeze(
        cls,
        surface: ActionSurface | None,
        *,
        retrieved_skill_ids: tuple[str, ...] = (),
        active_skill_ids: tuple[str, ...] = (),
    ) -> ActionContract:
        return cls(
            None if surface is None else canonical_json(surface.to_value()),
            retrieved_skill_ids,
            active_skill_ids,
        )

    @property
    def surface(self) -> ActionSurface | None:
        return (
            None
            if self.surface_json is None
            else ActionSurface.from_value(json.loads(self.surface_json))
        )

    def render_public_instruction(self) -> tuple[str, ...]:
        surface = self.surface
        return () if surface is None else _surface_instruction(surface)

    def render_native_semantics(self) -> tuple[str, ...]:
        """Answer-free meaning, independent of any JSON/XML action carrier."""
        surface = self.surface
        if surface is None:
            raise ValueError("native semantics require an explicit surface")
        if surface.instructions is None and surface.public_instructions:
            raise ValueError("native semantics require an explicit semantic/wire partition")
        lines = list(surface.instructions.semantic if surface.instructions else ())
        if surface.terminal_mode is TerminalMode.ENVIRONMENT:
            lines.append(
                "The environment determines termination; separate answer submission is unavailable."
            )
        if surface.dynamic_choice_fields:
            lines.append(
                "Use current public choices from: " + ", ".join(surface.dynamic_choice_fields) + "."
            )
        return tuple(lines)

    def invoked_skill_ids(self, action: StructuredAction) -> tuple[str, ...]:
        return canonical_invoked_skill_ids(
            action_kind=action.kind.value,
            action_skill_id=action.skill_id,
            retrieved_skill_ids=self.retrieved_skill_ids,
            active_skill_ids=self.active_skill_ids,
        )

    def validate(
        self,
        action: StructuredAction,
        *,
        validate_completion: Callable[[JsonValue], bool],
    ) -> AdmittedAction:
        """Admission only: no environment execution, reward or answer selection.

        Completion validation uses the environment's existing public shape
        validator, not its scorer. Exceptions propagate as infrastructure errors.
        """
        action, error = self.admit_surface(action)
        if error is not None:
            return AdmittedAction(action, error)
        if action.kind is ActionKind.COMPLETE:
            arguments = action.arguments
            if not isinstance(arguments, dict) or "value" not in arguments:
                return AdmittedAction(action, ("schema_invalid", "invalid_completion"))
            submission = normalize_json(arguments["value"])
            if not validate_completion(submission):
                return AdmittedAction(action, ("schema_invalid", "invalid_completion"))
            return AdmittedAction(action, completion=True, submission=submission)
        try:
            skills = self.invoked_skill_ids(action)
        except SkillInvocationAdmissionError:
            return AdmittedAction(action, ("schema_invalid", "skill_not_available_in_h0"))
        return AdmittedAction(action, invoked_skill_ids=skills)

    def to_native_tools(
        self, *, public_action_semantics: bool = False, skill_read_description: str | None = None
    ) -> tuple[dict[str, JsonValue], ...]:
        """Derive the native candidate from the same execution contract."""
        from .native_wire import native_bindings

        return tuple(
            (
                replace(binding, description=skill_read_description)
                if skill_read_description is not None and binding.kind is ActionKind.SKILL
                else binding
            ).to_tool()
            for binding in native_bindings(self, public_action_semantics=public_action_semantics)
        )

    def to_scoring_metadata(self) -> dict[str, JsonValue]:
        return {
            "format": "raw-json-action-contract@1",
            "surface": None if self.surface_json is None else json.loads(self.surface_json),
            "retrieved_skill_ids": list(self.retrieved_skill_ids),
            "active_skill_ids": list(self.active_skill_ids),
            "probability": "raw-full-vocabulary",
        }

    def admit_surface(
        self,
        action: StructuredAction,
    ) -> tuple[StructuredAction, tuple[str, str] | None]:
        surface = self.surface
        if surface is None or action.kind is ActionKind.SKILL:
            return action, None
        if action.kind is ActionKind.COMPLETE:
            if (
                surface.terminal_mode.value != "explicit-completion"
                or surface.completion is None
                or action.name != "complete"
            ):
                return action, ("schema_invalid", "invalid_completion")
            return action, None
        resources = {tool.resource_id for tool in surface.tools}
        if action.resource_id not in resources:
            return action, ("tool_error", "unsupported_resource")
        matching_resource = tuple(
            tool for tool in surface.tools if tool.resource_id == action.resource_id
        )
        matching_tool = next(
            (tool for tool in matching_resource if tool.name == action.name),
            None,
        )
        if matching_tool is None:
            return action, ("tool_error", "unsupported_tool")
        from skillev.rollout.action_surface import ToolActionSpecV2, admit_arguments

        if isinstance(matching_tool, ToolActionSpecV2):
            admission = admit_arguments(action, matching_tool)
            if not admission.admitted:
                return action, ("schema_invalid", str(admission.error_code))
            normalized = cast(StructuredAction, admission.normalized_action)
            return normalized, None
        if not isinstance(action.arguments, dict) or set(action.arguments) != set(
            matching_tool.arguments_schema
        ):
            return action, ("schema_invalid", "invalid_arguments")
        for key, schema in matching_tool.arguments_schema.items():
            candidate = action.arguments[key]
            if (
                isinstance(schema, str)
                and "string" in schema
                and (type(candidate) is not str or not candidate.strip())
            ):
                return action, ("schema_invalid", "invalid_arguments")
        return action, None


def action_example(
    *,
    kind: str,
    name: str,
    arguments: JsonValue,
    resource_id: str | None,
    skill_id: str | None = None,
) -> str:
    value: dict[str, JsonValue] = {
        "arguments": arguments,
        "kind": kind,
        "name": name,
    }
    if resource_id is not None:
        value["resource_id"] = resource_id
    if skill_id is not None:
        value["skill_id"] = skill_id
    return canonical_json(value)


def completion_example(value: JsonValue) -> str:
    return action_example(
        kind="complete",
        name="complete",
        arguments={"value": value},
        resource_id=None,
    )


def _surface_instruction(surface: ActionSurface) -> tuple[str, ...]:
    lines: list[str] = []
    for instruction in surface.public_instructions:
        lines.append(instruction)
    for tool in surface.tools:
        schema = (
            {name: field.to_value() for name, field in tool.arguments.items()}
            if isinstance(tool, ToolActionSpecV2)
            else tool.arguments_schema
        )
        lines.extend(
            (
                f"Tool {tool.resource_id}.{tool.name} arguments schema: {canonical_json(schema)}",
                action_example(
                    kind="tool",
                    name=tool.name,
                    arguments=tool.example_arguments,
                    resource_id=tool.resource_id,
                ),
            )
        )
    if surface.completion is not None:
        lines.extend(
            (
                f"Completion value schema: {canonical_json(surface.completion.value_schema)}",
                completion_example(surface.completion.example_value),
            )
        )
    if surface.terminal_mode is TerminalMode.ENVIRONMENT:
        lines.append("The environment terminates the episode; kind=complete is forbidden.")
    if surface.dynamic_choice_fields:
        lines.append(
            "Use current public choices from: " + ", ".join(surface.dynamic_choice_fields) + "."
        )
    return tuple(lines)
