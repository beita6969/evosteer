"""Dependency-light WebShop rollout adapter.

The official WebShop/Flask/search stack is deliberately injected behind
``WebShopEpisode``.  This module owns only model-visible task projection,
structured action translation, public observations, and resumable identity.
Reward and goal-matching state never cross this package boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
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
from skillev.runtime.skill_invocation import skill_invocation_observation

from .task_family import require_benchmark_task_family

WEBSHOP_BENCHMARK_ID = "webshop"
WEBSHOP_RESOURCE_ID = "webshop"


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class WebShopPublicItem:
    """One answer-free shopping instruction and pinned deployment identity."""

    dataset_revision: str
    environment_snapshot_id: str
    split: str
    task_id: str
    task_family: str
    query: str
    public_context: JsonValue

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_revision",
            "environment_snapshot_id",
            "split",
            "task_id",
            "task_family",
            "query",
        ):
            _text(getattr(self, field_name), field_name=field_name)
        require_benchmark_task_family(
            benchmark_id=WEBSHOP_BENCHMARK_ID,
            task_family=self.task_family,
        )
        object.__setattr__(self, "public_context", normalize_json(self.public_context))

    @property
    def environment_id(self) -> str:
        return (
            f"benchmark:{WEBSHOP_BENCHMARK_ID}@{self.dataset_revision}:"
            f"environment:{self.environment_snapshot_id}"
        )

    def to_rollout_task(self) -> RolloutTask:
        return RolloutTask(
            task_id=self.task_id,
            environment_id=self.environment_id,
            task_family=self.task_family,
            context_id=self.environment_snapshot_id,
            query=self.query,
            available_tools=("click", "purchase", "search"),
            public_context={
                "benchmark_id": WEBSHOP_BENCHMARK_ID,
                "dataset_revision": self.dataset_revision,
                "environment_snapshot_id": self.environment_snapshot_id,
                "payload": self.public_context,
                "split": self.split,
                "tools": {
                    "click": {"arguments": ["target"]},
                    "purchase": {"arguments": []},
                    "search": {"arguments": ["query"]},
                },
            },
        )


@dataclass(slots=True)
class OrderedWebShopTaskProvider:
    """Result-blind ordered curriculum over public WebShop tasks."""

    items: tuple[WebShopPublicItem, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("WebShop task provider requires at least one item")
        if any(not isinstance(item, WebShopPublicItem) for item in self.items):
            raise TypeError("WebShop task provider items have an incompatible type")
        task_ids = tuple(item.task_id for item in self.items)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("WebShop task identities must be unique")
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.items):
            raise ValueError("WebShop task provider cursor is invalid")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.items):
            raise RuntimeError("WebShop public curriculum is exhausted")
        task = self.items[self.cursor].to_rollout_task()
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id=stable_hash(
                {"kind": "webshop", "task_ids": [item.task_id for item in self.items]}
            ),
            cursor=self.cursor,
        )


class WebShopCommandKind(StrEnum):
    SEARCH = "search"
    CLICK = "click"
    PURCHASE = "purchase"


@dataclass(frozen=True, slots=True)
class WebShopCommand:
    """Official-environment-independent shopping command."""

    kind: WebShopCommandKind
    argument: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WebShopCommandKind):
            raise TypeError("WebShop command kind is invalid")
        if self.kind is WebShopCommandKind.PURCHASE:
            if self.argument is not None:
                raise ValueError("WebShop purchase command cannot carry an argument")
        else:
            _text(self.argument, field_name="WebShop command argument")

    def to_value(self) -> dict[str, JsonValue]:
        return {"argument": self.argument, "kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class WebShopPublicStep:
    """Public projection of one measured official WebShop step."""

    public_observation: JsonValue
    terminal: bool
    budget_usage: BudgetVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_observation", normalize_json(self.public_observation))
        if type(self.terminal) is not bool:
            raise TypeError("WebShop terminal flag must be boolean")
        if not isinstance(self.budget_usage, BudgetVector):
            raise TypeError("WebShop budget usage must be a BudgetVector")


class WebShopEpisode(Protocol):
    """Narrow public surface implemented by a private official deployment."""

    @property
    def task_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    async def execute(
        self,
        command: WebShopCommand,
        *,
        step_index: int,
    ) -> WebShopPublicStep: ...


@dataclass(frozen=True, slots=True)
class _ExecutionRecord:
    step_index: int
    action: StructuredAction
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("WebShop execution step must be positive")


@dataclass(slots=True)
class WebShopEnvironment:
    """Public rollout environment over one injected, identity-pinned episode."""

    item: WebShopPublicItem
    episode: WebShopEpisode
    _records: list[_ExecutionRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.item, WebShopPublicItem):
            raise TypeError("WebShop environment requires WebShopPublicItem")
        if self.episode.task_id != self.item.task_id:
            raise ValueError("WebShop episode belongs to another task")
        if self.episode.environment_id != self.item.environment_id:
            raise ValueError("WebShop episode belongs to another environment snapshot")

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
            raise TypeError("WebShop action must be StructuredAction")
        if type(step_index) is not int or step_index < 1:
            raise ValueError("WebShop step_index must be positive")
        if self._records and step_index <= self._records[-1].step_index:
            raise ValueError("WebShop execution steps must be strictly increasing")

        if action.kind is ActionKind.SKILL and action.skill_id is not None:
            observation = skill_invocation_observation(action.skill_id)
        else:
            command = _command_from_action(action)
            if command is None:
                observation = EnvironmentObservation(
                    public_value={"error": "unsupported_webshop_action"},
                    observation_status="tool_error",
                    budget_usage=BudgetVector(tool_calls=1),
                )
            else:
                # Do not translate official-environment exceptions into an
                # observation: an infrastructure failure is not agent data.
                public_step = await self.episode.execute(
                    command,
                    step_index=step_index,
                )
                if not isinstance(public_step, WebShopPublicStep):
                    raise TypeError("WebShop episode returned an incompatible step")
                if public_step.budget_usage.tool_calls != 1:
                    raise ValueError("WebShop episode must report exactly one measured tool call")
                observation = EnvironmentObservation(
                    public_value=public_step.public_observation,
                    observation_status="success",
                    terminal_submission=(
                        {"webshop_episode_terminal": True} if public_step.terminal else None
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


def _command_from_action(action: StructuredAction) -> WebShopCommand | None:
    if action.kind is not ActionKind.TOOL or action.resource_id != WEBSHOP_RESOURCE_ID:
        return None
    arguments = action.arguments
    if not isinstance(arguments, dict):
        return None
    if action.name == WebShopCommandKind.SEARCH.value and set(arguments) == {"query"}:
        query = arguments["query"]
        if type(query) is str and query.strip():
            return WebShopCommand(WebShopCommandKind.SEARCH, query)
    if action.name == WebShopCommandKind.CLICK.value and set(arguments) == {"target"}:
        target = arguments["target"]
        if type(target) is str and target.strip():
            return WebShopCommand(WebShopCommandKind.CLICK, target)
    if action.name == WebShopCommandKind.PURCHASE.value and not arguments:
        return WebShopCommand(WebShopCommandKind.PURCHASE, None)
    return None


__all__ = [
    "WEBSHOP_BENCHMARK_ID",
    "WEBSHOP_RESOURCE_ID",
    "OrderedWebShopTaskProvider",
    "WebShopCommand",
    "WebShopCommandKind",
    "WebShopEnvironment",
    "WebShopEpisode",
    "WebShopPublicItem",
    "WebShopPublicStep",
]
