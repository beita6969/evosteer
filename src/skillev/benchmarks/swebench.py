"""Dependency-light SWE-bench Verified workspace adapter.

The real repository container is injected through :class:`SWEWorkspaceBackend`.
This module never applies a patch or invokes a shell itself; it only projects a
small structured command language into an identity-pinned public workspace.
Hidden tests and verifier truth belong exclusively to the private-evaluation
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Protocol

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.rollout import RolloutTask
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentObservation,
    OrderedTaskCursorState,
    StructuredAction,
)

from .task_family import require_benchmark_task_family

SWEBENCH_VERIFIED_BENCHMARK_ID = "swe-bench-verified"
SWEBENCH_WORKSPACE_RESOURCE_ID = "swe-workspace"


def _text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} has invalid text")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _workspace_path(value: object, *, field_name: str, allow_root: bool) -> str:
    path = _text(value, field_name=field_name)
    if "\\" in path:
        raise ValueError(f"{field_name} must use POSIX separators")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field_name} must stay inside the workspace")
    if str(parsed) == "." and not allow_root:
        raise ValueError(f"{field_name} must name a workspace entry")
    return str(parsed)


@dataclass(frozen=True, slots=True)
class SWEBenchVerifiedPublicCase:
    """One public issue and exact repository/container identity."""

    dataset_revision: str
    split: str
    instance_id: str
    repo: str
    version: str
    base_commit: str
    problem_statement: str
    environment_image_id: str
    task_family: str
    max_steps: int

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_revision",
            "split",
            "instance_id",
            "repo",
            "version",
            "base_commit",
            "problem_statement",
            "environment_image_id",
            "task_family",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name=field_name),
            )
        require_benchmark_task_family(
            benchmark_id=SWEBENCH_VERIFIED_BENCHMARK_ID,
            task_family=self.task_family,
        )
        _positive_int(self.max_steps, field_name="max_steps")

    @property
    def task_id(self) -> str:
        return self.instance_id

    @property
    def environment_id(self) -> str:
        identity = stable_hash(
            {
                "base_commit": self.base_commit,
                "dataset_revision": self.dataset_revision,
                "environment_image_id": self.environment_image_id,
                "instance_id": self.instance_id,
                "repo": self.repo,
                "version": self.version,
            }
        )
        return f"benchmark:{SWEBENCH_VERIFIED_BENCHMARK_ID}:workspace:{identity}"

    def to_rollout_task(self) -> RolloutTask:
        return RolloutTask(
            task_id=self.task_id,
            environment_id=self.environment_id,
            task_family=self.task_family,
            context_id=f"repo:{self.repo}",
            query=self.problem_statement,
            available_tools=("read", "search", "submit_patch", "test", "write"),
            public_context={
                "base_commit": self.base_commit,
                "benchmark_id": SWEBENCH_VERIFIED_BENCHMARK_ID,
                "dataset_revision": self.dataset_revision,
                "environment_image_id": self.environment_image_id,
                "instance_id": self.instance_id,
                "max_steps": self.max_steps,
                "repo": self.repo,
                "split": self.split,
                "version": self.version,
                "tools": {
                    "read": {"arguments": ["path"]},
                    "search": {"arguments": ["path", "query"]},
                    "submit_patch": {"arguments": ["patch"]},
                    "test": {"arguments": ["target"]},
                    "write": {"arguments": ["path", "content"]},
                },
            },
        )


@dataclass(slots=True)
class OrderedSWEBenchVerifiedTaskProvider:
    cases: tuple[SWEBenchVerifiedPublicCase, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("SWE-bench task provider requires at least one case")
        if any(not isinstance(case, SWEBenchVerifiedPublicCase) for case in self.cases):
            raise TypeError("SWE-bench task provider cases have an incompatible type")
        task_ids = tuple(case.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("SWE-bench task identities must be unique")
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.cases):
            raise ValueError("SWE-bench task provider cursor is invalid")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.cases):
            raise RuntimeError("SWE-bench public curriculum is exhausted")
        task = self.cases[self.cursor].to_rollout_task()
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id=stable_hash(
                {"kind": "swe-bench-verified", "task_ids": [case.task_id for case in self.cases]}
            ),
            cursor=self.cursor,
        )


class SWECommandKind(StrEnum):
    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    TEST = "test"
    SUBMIT_PATCH = "submit_patch"


@dataclass(frozen=True, slots=True)
class SWEWorkspaceCommand:
    """Structured command interpreted by an injected repository backend."""

    kind: SWECommandKind
    arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SWECommandKind):
            raise TypeError("SWE workspace command kind is invalid")
        normalized = normalize_json(self.arguments)
        if not isinstance(normalized, dict) or normalized != self.arguments:
            raise TypeError("SWE workspace command arguments must be a JSON object")
        expected = {
            SWECommandKind.READ: {"path"},
            SWECommandKind.WRITE: {"content", "path"},
            SWECommandKind.SEARCH: {"path", "query"},
            SWECommandKind.TEST: {"target"},
            SWECommandKind.SUBMIT_PATCH: {"patch"},
        }[self.kind]
        if set(normalized) != expected:
            raise ValueError("SWE workspace command has an incompatible argument set")

        if self.kind in {SWECommandKind.READ, SWECommandKind.WRITE, SWECommandKind.SEARCH}:
            _workspace_path(
                normalized["path"],
                field_name="path",
                allow_root=self.kind is SWECommandKind.SEARCH,
            )
        if self.kind is SWECommandKind.WRITE:
            _text(normalized["content"], field_name="content", allow_empty=True)
        elif self.kind is SWECommandKind.SEARCH:
            _text(normalized["query"], field_name="query")
        elif self.kind is SWECommandKind.TEST:
            _text(normalized["target"], field_name="target")
        elif self.kind is SWECommandKind.SUBMIT_PATCH:
            _text(normalized["patch"], field_name="patch", allow_empty=True)

    def to_value(self) -> dict[str, JsonValue]:
        return {"arguments": self.arguments, "kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class SWEWorkspacePublicStep:
    """Public observation and measured cost of one workspace command."""

    public_observation: JsonValue
    terminal: bool
    budget_usage: BudgetVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_observation", normalize_json(self.public_observation))
        if type(self.terminal) is not bool:
            raise TypeError("SWE workspace terminal flag must be boolean")
        if not isinstance(self.budget_usage, BudgetVector):
            raise TypeError("SWE workspace budget usage must be a BudgetVector")


class SWEWorkspaceBackend(Protocol):
    """Narrow injection point for the real pinned repository container."""

    @property
    def instance_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    @property
    def max_steps(self) -> int: ...

    async def execute(
        self,
        command: SWEWorkspaceCommand,
        *,
        step_index: int,
    ) -> SWEWorkspacePublicStep: ...


@dataclass(frozen=True, slots=True)
class _ExecutionRecord:
    step_index: int
    action: StructuredAction
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        _positive_int(self.step_index, field_name="step_index")


@dataclass(slots=True)
class SWEBenchWorkspaceEnvironment:
    """Public, resumable workspace projection for one exact SWE-bench case."""

    case: SWEBenchVerifiedPublicCase
    backend: SWEWorkspaceBackend
    _records: list[_ExecutionRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.case, SWEBenchVerifiedPublicCase):
            raise TypeError("SWE-bench environment requires a public case")
        if self.backend.instance_id != self.case.instance_id:
            raise ValueError("SWE workspace backend belongs to another instance")
        if self.backend.environment_id != self.case.environment_id:
            raise ValueError("SWE workspace backend has another pinned environment identity")
        if self.backend.max_steps != self.case.max_steps:
            raise ValueError("SWE workspace backend uses another step limit")

    @property
    def environment_id(self) -> str:
        return self.case.environment_id

    @property
    def task_family(self) -> str:
        return self.case.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if not isinstance(action, StructuredAction):
            raise TypeError("SWE workspace action must be StructuredAction")
        _positive_int(step_index, field_name="step_index")
        if step_index != len(self._records) + 1:
            raise ValueError("SWE workspace execution steps must be contiguous")
        if step_index > self.case.max_steps:
            raise ValueError("SWE workspace execution exceeded its pinned step limit")
        if self._records and self._records[-1].observation.terminal:
            raise ValueError("SWE workspace cannot execute after patch submission")

        if action.kind is ActionKind.SKILL and action.skill_id is not None:
            observation = EnvironmentObservation(
                public_value={"status": "skill-invoked"},
                observation_status="success",
                invoked_skill_ids=(action.skill_id,),
                budget_usage=BudgetVector(tool_calls=1),
            )
        else:
            command = _command_from_action(action)
            if command is None:
                observation = EnvironmentObservation(
                    public_value={"error": "unsupported_swe_workspace_action"},
                    observation_status="tool_error",
                    budget_usage=BudgetVector(tool_calls=1),
                )
            else:
                public_step = await self.backend.execute(
                    command,
                    step_index=step_index,
                )
                if not isinstance(public_step, SWEWorkspacePublicStep):
                    raise TypeError("SWE workspace backend returned an incompatible step")
                if public_step.budget_usage.tool_calls != 1:
                    raise ValueError("SWE workspace backend must report one measured tool call")
                should_be_terminal = command.kind is SWECommandKind.SUBMIT_PATCH
                if public_step.terminal != should_be_terminal:
                    raise ValueError("SWE workspace backend returned an invalid terminal state")
                observation = EnvironmentObservation(
                    public_value=public_step.public_observation,
                    observation_status="success",
                    terminal_submission=(
                        {"patch": command.arguments["patch"]} if public_step.terminal else None
                    ),
                    terminal=public_step.terminal,
                    budget_usage=public_step.budget_usage,
                )

        self._records.append(
            _ExecutionRecord(
                step_index=step_index,
                action=action,
                observation=observation,
            )
        )
        return observation

    def validate_completion(self, submission: JsonValue) -> bool:
        del submission
        return False


def _command_from_action(action: StructuredAction) -> SWEWorkspaceCommand | None:
    if action.kind is not ActionKind.TOOL or action.resource_id != SWEBENCH_WORKSPACE_RESOURCE_ID:
        return None
    try:
        kind = SWECommandKind(action.name)
    except ValueError:
        return None
    if not isinstance(action.arguments, dict):
        return None
    try:
        return SWEWorkspaceCommand(kind, action.arguments)
    except (TypeError, ValueError):
        return None


__all__ = [
    "SWEBENCH_VERIFIED_BENCHMARK_ID",
    "SWEBENCH_WORKSPACE_RESOURCE_ID",
    "OrderedSWEBenchVerifiedTaskProvider",
    "SWEBenchVerifiedPublicCase",
    "SWEBenchWorkspaceEnvironment",
    "SWECommandKind",
    "SWEWorkspaceBackend",
    "SWEWorkspaceCommand",
    "SWEWorkspacePublicStep",
]
