"""Model-visible action and per-task rollout-budget contracts.

The surface is answer-free.  It describes only actions that the public
environment can admit; evaluator truth and private routing identities never
belong here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast

from skillev.contracts import JsonValue, normalize_json

ACTION_SURFACE_FORMAT = "skillev-action-surface@1"
ACTION_SURFACE_FORMAT_V2 = "skillev-action-surface@2"
ACTION_SURFACE_FORMAT_V3 = "skillev-action-surface@3"
ROLLOUT_BUDGET_PROFILE_FORMAT = "skillev-rollout-budget-profile@1"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _object_value(value: object, *, label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


class TerminalMode(StrEnum):
    """How a valid trajectory reaches terminal evaluation."""

    EXPLICIT_COMPLETION = "explicit-completion"
    ENVIRONMENT = "environment-terminal"


class ArgumentType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class ArgumentFieldSpec:
    value_type: ArgumentType
    required: bool
    nullable: bool = False
    default: JsonValue = None
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if type(self.required) is not bool or type(self.nullable) is not bool:
            raise TypeError("argument required/nullable flags must be boolean")
        if self.required and self.default is not None:
            raise ValueError("required field cannot have an implicit default")
        if self.value_type is not ArgumentType.INTEGER and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("only integer fields may have numeric bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("argument bounds are inverted")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "default": self.default,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "nullable": self.nullable,
            "required": self.required,
            "value_type": self.value_type.value,
        }

    @classmethod
    def from_value(cls, value: object) -> ArgumentFieldSpec:
        data = _object(
            value,
            fields={"default", "maximum", "minimum", "nullable", "required", "value_type"},
            label="argument field spec",
        )
        if type(data["required"]) is not bool or type(data["nullable"]) is not bool:
            raise TypeError("argument required/nullable flags must be boolean")
        for bound in ("minimum", "maximum"):
            if data[bound] is not None and type(data[bound]) is not int:
                raise TypeError("argument bounds must be integers or null")
        return cls(
            value_type=ArgumentType(_text(data["value_type"], field="argument value type")),
            required=data["required"],
            nullable=data["nullable"],
            default=data["default"],
            minimum=cast(int | None, data["minimum"]),
            maximum=cast(int | None, data["maximum"]),
        )


@dataclass(frozen=True, slots=True)
class ToolActionSpecV2:
    resource_id: str
    name: str
    arguments: dict[str, ArgumentFieldSpec]
    example_arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.resource_id, field="tool resource_id")
        _text(self.name, field="tool name")
        if set(self.arguments) != set(self.example_arguments):
            raise ValueError("tool example keys must match the v2 argument schema")
        if any(
            not key.strip() or not isinstance(value, ArgumentFieldSpec)
            for key, value in self.arguments.items()
        ):
            raise ValueError("v2 argument schema is invalid")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arguments": {name: value.to_value() for name, value in self.arguments.items()},
            "example_arguments": self.example_arguments,
            "name": self.name,
            "resource_id": self.resource_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ToolActionSpecV2:
        data = _object(
            value,
            fields={"arguments", "example_arguments", "name", "resource_id"},
            label="v2 tool action spec",
        )
        arguments = _object_value(data["arguments"], label="v2 arguments")
        return cls(
            resource_id=_text(data["resource_id"], field="tool resource_id"),
            name=_text(data["name"], field="tool name"),
            arguments={name: ArgumentFieldSpec.from_value(raw) for name, raw in arguments.items()},
            example_arguments=_object_value(data["example_arguments"], label="example arguments"),
        )


@dataclass(frozen=True, slots=True)
class ActionAdmission:
    admitted: bool
    normalized_action: Any | None
    error_code: str | None


def _matches_type(value: JsonValue, expected: ArgumentType) -> bool:
    if expected is ArgumentType.STRING:
        return type(value) is str and bool(value.strip())
    if expected is ArgumentType.INTEGER:
        return type(value) is int
    return type(value) is bool


def admit_arguments(action: Any, spec: ToolActionSpecV2) -> ActionAdmission:
    if not isinstance(action.arguments, dict):
        return ActionAdmission(False, None, "invalid_arguments")
    unknown = set(action.arguments) - set(spec.arguments)
    if unknown:
        return ActionAdmission(False, None, "unknown_arguments")
    normalized: dict[str, JsonValue] = {}
    for name, field in spec.arguments.items():
        if name not in action.arguments:
            if field.required:
                return ActionAdmission(False, None, "missing_required_argument")
            normalized[name] = field.default
            continue
        value = action.arguments[name]
        if value is None and field.nullable:
            normalized[name] = None
            continue
        if not _matches_type(value, field.value_type):
            return ActionAdmission(False, None, "argument_type_mismatch")
        if field.value_type is ArgumentType.INTEGER:
            integer = int(value)
            if field.minimum is not None and integer < field.minimum:
                return ActionAdmission(False, None, "argument_out_of_range")
            if field.maximum is not None and integer > field.maximum:
                return ActionAdmission(False, None, "argument_out_of_range")
        normalized[name] = value
    return ActionAdmission(True, replace(action, arguments=normalize_json(normalized)), None)


@dataclass(frozen=True, slots=True)
class ToolActionSpec:
    resource_id: str
    name: str
    arguments_schema: dict[str, JsonValue]
    example_arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.resource_id, field="tool resource_id")
        _text(self.name, field="tool name")
        arguments_schema = normalize_json(self.arguments_schema)
        example_arguments = normalize_json(self.example_arguments)
        if not isinstance(arguments_schema, dict) or not isinstance(example_arguments, dict):
            raise ValueError("tool schemas and examples must be JSON objects")
        object.__setattr__(self, "arguments_schema", arguments_schema)
        object.__setattr__(self, "example_arguments", example_arguments)
        if set(self.arguments_schema) != set(self.example_arguments):
            raise ValueError("tool example keys must match the argument schema")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arguments_schema": self.arguments_schema,
            "example_arguments": self.example_arguments,
            "name": self.name,
            "resource_id": self.resource_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ToolActionSpec:
        data = _object(
            value,
            fields={"arguments_schema", "example_arguments", "name", "resource_id"},
            label="tool action spec",
        )
        return cls(
            resource_id=_text(data["resource_id"], field="tool resource_id"),
            name=_text(data["name"], field="tool name"),
            arguments_schema=_object_value(data["arguments_schema"], label="arguments schema"),
            example_arguments=_object_value(data["example_arguments"], label="example arguments"),
        )


@dataclass(frozen=True, slots=True)
class CompletionSpec:
    value_schema: dict[str, JsonValue]
    example_value: dict[str, JsonValue]

    def __post_init__(self) -> None:
        value_schema = normalize_json(self.value_schema)
        example_value = normalize_json(self.example_value)
        if not isinstance(value_schema, dict) or not isinstance(example_value, dict):
            raise ValueError("completion schema and example must be JSON objects")
        object.__setattr__(self, "value_schema", value_schema)
        object.__setattr__(self, "example_value", example_value)
        if set(self.value_schema) != set(self.example_value):
            raise ValueError("completion example keys must match the value schema")

    def to_value(self) -> dict[str, JsonValue]:
        return {"example_value": self.example_value, "value_schema": self.value_schema}

    @classmethod
    def from_value(cls, value: object) -> CompletionSpec:
        data = _object(
            value,
            fields={"example_value", "value_schema"},
            label="completion spec",
        )
        return cls(
            value_schema=_object_value(data["value_schema"], label="completion value schema"),
            example_value=_object_value(data["example_value"], label="completion example"),
        )


@dataclass(frozen=True, slots=True)
class PublicActionInstructions:
    """Explicit author-supplied semantics and legacy JSON carrier instructions."""

    semantic: tuple[str, ...] = ()
    json_wire: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for values in (self.semantic, self.json_wire):
            if not isinstance(values, tuple) or any(
                type(v) is not str or not v.strip() for v in values
            ):
                raise ValueError("public instructions must be immutable nonempty text entries")

    @property
    def legacy_json(self) -> tuple[str, ...]:
        return self.semantic + self.json_wire

    def to_value(self) -> dict[str, JsonValue]:
        return {"semantic": list(self.semantic), "json_wire": list(self.json_wire)}

    @classmethod
    def from_value(cls, value: object) -> PublicActionInstructions:
        data = _object(value, fields={"semantic", "json_wire"}, label="public action instructions")
        if not isinstance(data["semantic"], list) or not isinstance(data["json_wire"], list):
            raise TypeError("instruction partitions must be arrays")
        return cls(
            tuple(_text(v, field="semantic instruction") for v in data["semantic"]),
            tuple(_text(v, field="JSON wire instruction") for v in data["json_wire"]),
        )


@dataclass(frozen=True, slots=True)
class ActionSurface:
    """Exact action domain shown to the policy for one task."""

    terminal_mode: TerminalMode
    tools: tuple[ToolActionSpec | ToolActionSpecV2, ...] = ()
    completion: CompletionSpec | None = None
    dynamic_choice_fields: tuple[str, ...] = ()
    public_instructions: tuple[str, ...] = ()
    format: str = ACTION_SURFACE_FORMAT
    instructions: PublicActionInstructions | None = None

    def __post_init__(self) -> None:
        if self.format not in {
            ACTION_SURFACE_FORMAT,
            ACTION_SURFACE_FORMAT_V2,
            ACTION_SURFACE_FORMAT_V3,
        }:
            raise ValueError("unsupported action surface format")
        if self.format == ACTION_SURFACE_FORMAT_V3:
            if not isinstance(self.instructions, PublicActionInstructions):
                raise ValueError("surface v3 requires explicit semantic/wire instructions")
            if (
                self.public_instructions
                and self.public_instructions != self.instructions.legacy_json
            ):
                raise ValueError("legacy projection differs from the sole instruction contract")
            object.__setattr__(self, "public_instructions", self.instructions.legacy_json)
        elif self.instructions is not None:
            raise ValueError("instruction partitions require surface v3")
        if not isinstance(self.terminal_mode, TerminalMode):
            raise TypeError("action surface terminal_mode is invalid")
        if not isinstance(self.tools, tuple) or any(
            not isinstance(item, ToolActionSpec | ToolActionSpecV2) for item in self.tools
        ):
            raise TypeError("action surface tools are invalid")
        expected_type = (
            ToolActionSpecV2 if self.format == ACTION_SURFACE_FORMAT_V2 else ToolActionSpec
        )
        if any(not isinstance(item, expected_type) for item in self.tools):
            raise TypeError("action surface tool format differs from surface format")
        identities = tuple((item.resource_id, item.name) for item in self.tools)
        if len(set(identities)) != len(identities):
            raise ValueError("action surface tools must be unique")
        if self.completion is not None and not isinstance(self.completion, CompletionSpec):
            raise TypeError("action surface completion is invalid")
        if self.terminal_mode is TerminalMode.EXPLICIT_COMPLETION and self.completion is None:
            raise ValueError("explicit-completion surfaces require a completion spec")
        if self.terminal_mode is TerminalMode.ENVIRONMENT and self.completion is not None:
            raise ValueError("environment-terminal surfaces cannot expose completion")
        for field, values in (
            ("dynamic_choice_fields", self.dynamic_choice_fields),
            ("public_instructions", self.public_instructions),
        ):
            if not isinstance(values, tuple) or any(
                type(item) is not str or not item.strip() for item in values
            ):
                raise ValueError(f"{field} must be a tuple of non-empty text")
        if len(set(self.dynamic_choice_fields)) != len(self.dynamic_choice_fields):
            raise ValueError("dynamic choice fields must be unique")

    def to_value(self) -> dict[str, JsonValue]:
        instruction_value: dict[str, JsonValue]
        if self.instructions is not None:
            instruction_value = {"instructions": self.instructions.to_value()}
        else:
            instruction_value = {"public_instructions": list(self.public_instructions)}
        return {
            "completion": None if self.completion is None else self.completion.to_value(),
            "dynamic_choice_fields": list(self.dynamic_choice_fields),
            "format": self.format,
            **instruction_value,
            "terminal_mode": self.terminal_mode.value,
            "tools": [item.to_value() for item in self.tools],
        }

    @classmethod
    def from_value(cls, value: object) -> ActionSurface:
        partitioned = isinstance(value, dict) and value.get("format") == ACTION_SURFACE_FORMAT_V3
        data = _object(
            value,
            fields={
                "completion",
                "dynamic_choice_fields",
                "format",
                "instructions" if partitioned else "public_instructions",
                "terminal_mode",
                "tools",
            },
            label="action surface",
        )
        tools = data["tools"]
        choices = data["dynamic_choice_fields"]
        instructions = data.get("public_instructions", [])
        if (
            not isinstance(tools, list)
            or not isinstance(choices, list)
            or not isinstance(instructions, list)
        ):
            raise TypeError("action surface arrays are invalid")
        raw_completion = data["completion"]
        format_value = _text(data["format"], field="action surface format")
        return cls(
            terminal_mode=TerminalMode(_text(data["terminal_mode"], field="terminal_mode")),
            tools=tuple(
                (
                    ToolActionSpec.from_value(item)
                    if format_value != ACTION_SURFACE_FORMAT_V2
                    else ToolActionSpecV2.from_value(item)
                )
                for item in tools
            ),
            completion=(
                None if raw_completion is None else CompletionSpec.from_value(raw_completion)
            ),
            dynamic_choice_fields=tuple(
                _text(item, field="dynamic choice field") for item in choices
            ),
            public_instructions=tuple(
                _text(item, field="public instruction") for item in instructions
            ),
            format=format_value,
            instructions=PublicActionInstructions.from_value(data["instructions"])
            if partitioned
            else None,
        )


@dataclass(frozen=True, slots=True)
class RolloutBudgetProfile:
    """A preregistered per-task cap bounded by the global rollout envelope."""

    profile_id: str
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    format: str = ROLLOUT_BUDGET_PROFILE_FORMAT

    def __post_init__(self) -> None:
        _text(self.profile_id, field="budget profile_id")
        for field in ("max_turns", "max_reasoning_tokens", "max_action_tokens"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be positive")
        if self.format != ROLLOUT_BUDGET_PROFILE_FORMAT:
            raise ValueError("unsupported rollout budget profile format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "max_action_tokens": self.max_action_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_turns": self.max_turns,
            "profile_id": self.profile_id,
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutBudgetProfile:
        data = _object(
            value,
            fields={
                "format",
                "max_action_tokens",
                "max_reasoning_tokens",
                "max_turns",
                "profile_id",
            },
            label="rollout budget profile",
        )
        for field in ("max_turns", "max_reasoning_tokens", "max_action_tokens"):
            if type(data[field]) is not int:
                raise TypeError(f"{field} must be an integer")
        return cls(
            profile_id=_text(data["profile_id"], field="budget profile_id"),
            max_turns=data["max_turns"],  # type: ignore[arg-type]
            max_reasoning_tokens=data["max_reasoning_tokens"],  # type: ignore[arg-type]
            max_action_tokens=data["max_action_tokens"],  # type: ignore[arg-type]
            format=_text(data["format"], field="rollout budget profile format"),
        )


@dataclass(frozen=True, slots=True)
class ModelVisibleMessage:
    """One answer-free source message retained without role flattening."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("model-visible message role is unsupported")
        _text(self.content, field="model-visible message content")

    def to_value(self) -> dict[str, JsonValue]:
        return {"content": self.content, "role": self.role}

    @classmethod
    def from_value(cls, value: object) -> ModelVisibleMessage:
        data = _object(value, fields={"content", "role"}, label="model-visible message")
        return cls(
            role=_text(data["role"], field="model-visible message role"),
            content=_text(data["content"], field="model-visible message content"),
        )


__all__ = [
    "ACTION_SURFACE_FORMAT",
    "ACTION_SURFACE_FORMAT_V2",
    "ROLLOUT_BUDGET_PROFILE_FORMAT",
    "ActionAdmission",
    "ActionSurface",
    "ArgumentFieldSpec",
    "ArgumentType",
    "CompletionSpec",
    "ModelVisibleMessage",
    "RolloutBudgetProfile",
    "TerminalMode",
    "ToolActionSpec",
    "ToolActionSpecV2",
    "admit_arguments",
]
