"""Typed local tool interfaces used by the bounded agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast

from skillev.contracts.canonical import JsonValue, normalize_json

from .contracts import ActionKind, StructuredAction


@dataclass(frozen=True, slots=True)
class ToolRequest:
    action: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("Tool action cannot be empty")
        normalized = normalize_json(dict(self.arguments))
        if not isinstance(normalized, dict):
            raise TypeError("Tool arguments must form a JSON object")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(cast(dict[str, object], normalized)),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action": self.action,
            "arguments": normalize_json(dict(self.arguments)),
        }

    def to_structured_action(
        self,
        *,
        resource_id: str,
        skill_id: str | None = None,
    ) -> StructuredAction:
        return StructuredAction(
            kind=ActionKind.SKILL if skill_id is not None else ActionKind.TOOL,
            name=self.action,
            arguments=normalize_json(dict(self.arguments)),
            resource_id=resource_id,
            skill_id=skill_id,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    value: JsonValue
    completed: bool = True

    def __post_init__(self) -> None:
        if type(self.completed) is not bool:
            raise TypeError("Tool completion status must be boolean")
        if normalize_json(self.value) != self.value:
            raise ValueError("Tool result must be normalized JSON")


class ToolBackend(Protocol):
    def invoke(self, request: ToolRequest) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    resource_id: str
    backend: ToolBackend

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("Tool registration requires a resource ID")


class ToolRegistry:
    """A small immutable-at-execution resource registry."""

    def __init__(self, registrations: tuple[ToolRegistration, ...]) -> None:
        resources: dict[str, ToolBackend] = {}
        for registration in registrations:
            if registration.resource_id in resources:
                raise ValueError("Tool resource IDs must be unique")
            resources[registration.resource_id] = registration.backend
        self._resources: Mapping[str, ToolBackend] = MappingProxyType(resources)

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._resources))

    def invoke(self, resource_id: str, request: ToolRequest) -> ToolResult:
        try:
            backend = self._resources[resource_id]
        except KeyError as error:
            raise KeyError("Unknown tool resource") from error
        return backend.invoke(request)


@dataclass(slots=True)
class FakeTool:
    """A deterministic local tool for behavior tests and CPU smoke runs."""

    handlers: Mapping[str, Callable[[Mapping[str, object]], object]]
    calls: list[ToolRequest] = field(default_factory=list)

    def invoke(self, request: ToolRequest) -> ToolResult:
        try:
            handler = self.handlers[request.action]
        except KeyError as error:
            raise KeyError("Fake tool has no handler for the requested action") from error
        value = normalize_json(handler(request.arguments))
        self.calls.append(request)
        return ToolResult(value)
