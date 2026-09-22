"""Closed, canonical action space for online EvoSteer team construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillev.contracts.canonical import JsonValue, canonical_json


class GraphActionKind(StrEnum):
    ADD_AGENT = "ADD_AGENT"
    ADD_EDGE = "ADD_EDGE"
    BIND_SKILL = "BIND_SKILL"
    SET_OUTPUT = "SET_OUTPUT"
    RERUN_AGENT = "RERUN_AGENT"
    DROP_AGENT = "DROP_AGENT"
    STOP = "STOP"


# Do not reuse runtime.contracts.ActionKind: that is the old single-actor wire.
ActionKind = GraphActionKind
EDGE_PROTOCOLS = ("feedback", "revise")
_FIELDS = ("node_id", "role_id", "source_id", "target_id", "protocol", "skill_id")
_REQUIRED = {
    GraphActionKind.ADD_AGENT: {"node_id", "role_id"},
    GraphActionKind.ADD_EDGE: {"source_id", "target_id", "protocol"},
    GraphActionKind.BIND_SKILL: {"node_id", "skill_id"},
    GraphActionKind.SET_OUTPUT: {"node_id"},
    GraphActionKind.RERUN_AGENT: {"node_id"},
    GraphActionKind.DROP_AGENT: {"node_id"},
    GraphActionKind.STOP: set(),
}


@dataclass(frozen=True, slots=True)
class GraphAction:
    kind: GraphActionKind
    node_id: str | None = None
    role_id: str | None = None
    source_id: str | None = None
    target_id: str | None = None
    protocol: str | None = None
    skill_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GraphActionKind):
            raise TypeError("kind must be GraphActionKind")
        supplied = {name for name in _FIELDS if getattr(self, name) is not None}
        allowed = _REQUIRED[self.kind] | (
            {"skill_id"} if self.kind is GraphActionKind.ADD_AGENT else set()
        )
        if not _REQUIRED[self.kind] <= supplied <= allowed:
            raise ValueError("graph action has missing or unrelated fields")
        for name in supplied:
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        if self.protocol is not None and self.protocol not in EDGE_PROTOCOLS:
            raise ValueError("unsupported edge protocol")
        if self.kind is GraphActionKind.ADD_EDGE and self.source_id == self.target_id:
            raise ValueError("self edges are not part of the EvoSteer action surface")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind.value,
            **{name: getattr(self, name) for name in _FIELDS if getattr(self, name) is not None},
        }

    @property
    def text(self) -> str:
        """Exact finite action text used by sampling and both policy scorers."""
        return canonical_json(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> GraphAction:
        if not isinstance(value, dict) or "kind" not in value:
            raise ValueError("graph action must be an object containing kind")
        if set(value) - {"kind", *_FIELDS}:
            raise ValueError("unknown graph action fields")
        if type(value["kind"]) is not str:
            raise TypeError("graph action kind must be text")
        # Reject explicit nulls so each action has exactly one canonical wire form.
        if any(value[name] is None for name in value if name != "kind"):
            raise ValueError("optional graph action fields must be omitted, not null")
        return cls(
            GraphActionKind(value["kind"]), **{k: v for k, v in value.items() if k != "kind"}
        )


__all__ = ["EDGE_PROTOCOLS", "ActionKind", "GraphAction", "GraphActionKind"]
