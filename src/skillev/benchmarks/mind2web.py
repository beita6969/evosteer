"""Answer-free Mind2Web step-prediction rollout environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from skillev.contracts import JsonValue, normalize_json
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    EnvironmentObservation,
    StructuredAction,
)

from .static import BenchmarkPublicItem

MIND2WEB_BENCHMARK_ID: Final = "mind2web"
MIND2WEB_OPERATIONS: Final = frozenset({"CLICK", "SELECT", "TYPE"})


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value:
        raise ValueError(f"{field} must be text without NUL")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class _SkillInvocation:
    step_index: int
    skill_id: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("Mind2Web skill invocation step must be positive")
        _text(self.skill_id, field="skill_id")


@dataclass(slots=True)
class Mind2WebCompletionEnvironment:
    """Completion-shape authority with no access to positive element labels."""

    item: BenchmarkPublicItem
    _skill_invocations: list[_SkillInvocation] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.item, BenchmarkPublicItem):
            raise TypeError("Mind2Web environment requires BenchmarkPublicItem")
        if self.item.benchmark_id != MIND2WEB_BENCHMARK_ID:
            raise ValueError("Mind2Web environment received another benchmark")

    @property
    def environment_id(self) -> str:
        return self.item.environment_id

    @property
    def task_family(self) -> str:
        return self.item.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if not isinstance(action, StructuredAction):
            raise TypeError("Mind2Web action must be StructuredAction")
        if type(step_index) is not int or step_index < 1:
            raise ValueError("Mind2Web step_index must be positive")
        if step_index != len(self._skill_invocations) + 1:
            raise ValueError("Mind2Web skill invocation steps must be contiguous")
        if action.kind is not ActionKind.SKILL or action.skill_id is None:
            raise EnvironmentMethodFailedError(
                budget_usage=BudgetVector(tool_calls=1),
                public_error_code="unsupported_mind2web_action",
            )
        invocation = _SkillInvocation(step_index, action.skill_id)
        self._skill_invocations.append(invocation)
        return _skill_observation(action.skill_id)

    def validate_completion(self, submission: JsonValue) -> bool:
        normalized = normalize_json(submission)
        if not isinstance(normalized, dict) or set(normalized) != {
            "backend_node_id",
            "operation",
            "value",
        }:
            return False
        operation = normalized["operation"]
        backend_node_id = normalized["backend_node_id"]
        value = normalized["value"]
        return (
            type(operation) is str
            and operation in MIND2WEB_OPERATIONS
            and type(backend_node_id) is str
            and bool(backend_node_id.strip())
            and type(value) is str
            and "\x00" not in value
        )


def _skill_observation(skill_id: str) -> EnvironmentObservation:
    return EnvironmentObservation(
        public_value={"status": "skill-invoked"},
        observation_status="success",
        invoked_skill_ids=(skill_id,),
        budget_usage=BudgetVector(tool_calls=1),
    )


__all__ = [
    "MIND2WEB_BENCHMARK_ID",
    "MIND2WEB_OPERATIONS",
    "Mind2WebCompletionEnvironment",
]
