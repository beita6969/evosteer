"""Immutable public graph/executor contracts; no policy or evaluator authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from skillev.contracts.canonical import JsonValue, normalize_json
from skillev.runtime.contracts import BudgetVector


def _text(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


@dataclass(frozen=True, slots=True)
class RoleSpec:
    role_id: str
    instruction: str
    model_maximum: BudgetVector = field(
        default_factory=lambda: BudgetVector(
            input_tokens=8192,
            output_tokens=2048,
            model_calls=1,
            agent_turns=1,
            tool_calls=0,
            wall_time_milliseconds=120_000,
        )
    )

    def __post_init__(self) -> None:
        _text(self.role_id, "role_id")
        _text(self.instruction, "role instruction")
        if not isinstance(self.model_maximum, BudgetVector):
            raise TypeError("role model_maximum must be BudgetVector")
        if self.model_maximum.model_calls < 1:
            raise ValueError("an executed role must reserve at least one model call")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "role_id": self.role_id,
            "instruction": self.instruction,
            "model_maximum": self.model_maximum.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> RoleSpec:
        if not isinstance(value, dict) or set(value) != {"role_id", "instruction", "model_maximum"}:
            raise ValueError("invalid role specification")
        return cls(
            value["role_id"], value["instruction"], BudgetVector.from_value(value["model_maximum"])
        )


@dataclass(frozen=True, slots=True)
class ProtocolEdge:
    source_id: str
    target_id: str
    protocol: str

    def to_value(self) -> dict[str, JsonValue]:
        return {"source_id": self.source_id, "target_id": self.target_id, "protocol": self.protocol}


@dataclass(frozen=True, slots=True)
class NodeExecutionRequest:
    runtime_id: str
    node_id: str
    role: RoleSpec
    task_prompt: str
    skills: tuple[tuple[str, str], ...]
    messages: tuple[dict[str, JsonValue], ...]
    previous_output: str | None
    execution_index: int
    seed: int

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "runtime_id": self.runtime_id,
            "node_id": self.node_id,
            "role": self.role.to_value(),
            "task_prompt": self.task_prompt,
            "skills": [{"skill_id": key, "body": body} for key, body in self.skills],
            "messages": list(self.messages),
            "previous_output": self.previous_output,
            "execution_index": self.execution_index,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class NodeExecutionResult:
    output: str
    usage: BudgetVector
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A sampled empty completion is still an observed model outcome.
        if type(self.output) is not str:
            raise TypeError("node output must be text")
        if not isinstance(self.usage, BudgetVector) or self.usage.model_calls < 1:
            raise ValueError("node execution requires exact positive model-call usage")
        metadata = normalize_json(self.metadata)
        if not isinstance(metadata, dict):
            raise TypeError("node metadata must be a public JSON object")
        object.__setattr__(self, "metadata", metadata)

    def to_value(self) -> dict[str, JsonValue]:
        return {"output": self.output, "usage": self.usage.to_value(), "metadata": self.metadata}


class NodeExecutor(Protocol):
    """An externally frozen role model; node tokens are transition observations."""

    @property
    def frozen_identity(self) -> str: ...

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult: ...


@dataclass(frozen=True, slots=True)
class NodeState:
    node_id: str
    role_id: str
    skill_ids: tuple[str, ...]
    output: str
    output_version: int

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "node_id": self.node_id,
            "role_id": self.role_id,
            "skill_ids": list(self.skill_ids),
            "output": self.output,
            "output_version": self.output_version,
        }


@dataclass(frozen=True, slots=True)
class GraphState:
    nodes: tuple[NodeState, ...]
    edges: tuple[ProtocolEdge, ...]
    output_node_id: str | None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "nodes": [node.to_value() for node in self.nodes],
            "edges": [edge.to_value() for edge in self.edges],
            "output_node_id": self.output_node_id,
        }


__all__ = [
    "GraphState",
    "NodeExecutionRequest",
    "NodeExecutionResult",
    "NodeExecutor",
    "NodeState",
    "ProtocolEdge",
    "RoleSpec",
]
