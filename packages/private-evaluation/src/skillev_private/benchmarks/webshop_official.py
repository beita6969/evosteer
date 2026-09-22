"""Bridge from the dependency-light WebShop adapter to an official text env.

No official WebShop package is imported here.  Deployments inject a factory
implementing :class:`OfficialWebShopTextEnvFactory`; this bridge owns command
syntax, identity checks, public projection, and the private terminal-reward
view.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.webshop import (
    WebShopCommand,
    WebShopCommandKind,
    WebShopPublicStep,
)
from skillev.runtime import BudgetVector

from .webshop import (
    PrivateWebShopCase,
    PrivateWebShopEpisodeSession,
    WebShopRewardUnavailableError,
)


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _reward(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("official WebShop reward must be numeric")
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError("official WebShop reward must lie in [0, 1]")
    return reward


def _available_actions(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("official WebShop available actions must be a tuple")
    actions = tuple(_text(action, field_name="available action") for action in value)
    if len(set(actions)) != len(actions):
        raise ValueError("official WebShop available actions must be unique")
    return actions


@dataclass(frozen=True, slots=True)
class OfficialWebShopGoal:
    """Pinned private goal and the exact official session that owns it."""

    task_id: str
    environment_id: str
    goal_id: str
    session_id: str
    payload: object = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in ("task_id", "environment_id", "goal_id", "session_id"):
            _text(getattr(self, field_name), field_name=field_name)
        if self.payload is None:
            raise ValueError("official WebShop goal payload cannot be absent")


@dataclass(frozen=True, slots=True)
class OfficialWebShopStepResult:
    """Raw official result retained behind the private bridge."""

    observation_text: str
    reward: float
    terminal: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_text",
            _text(self.observation_text, field_name="official observation"),
        )
        object.__setattr__(self, "reward", _reward(self.reward))
        if type(self.terminal) is not bool:
            raise TypeError("official WebShop terminal flag must be boolean")


class OfficialWebShopTextEnv(Protocol):
    """Narrow surface supplied by an official WebShop deployment wrapper."""

    @property
    def goal_id(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def instruction_text(self) -> str: ...

    def reset(self, session_id: str) -> str: ...

    def step(self, action: str) -> OfficialWebShopStepResult: ...

    def get_available_actions(self) -> tuple[str, ...]: ...

    async def close(self) -> None: ...


class OfficialWebShopTextEnvFactory(Protocol):
    """Dependency injection point allowed to import the official package."""

    def create(self, goal: OfficialWebShopGoal) -> OfficialWebShopTextEnv: ...


@dataclass(frozen=True, slots=True)
class _ExecutionEntry:
    step_index: int
    command: WebShopCommand
    public_step: WebShopPublicStep

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("official WebShop replay step must be positive")


def _official_action(command: WebShopCommand) -> str:
    if command.kind is WebShopCommandKind.SEARCH:
        return f"search[{command.argument}]"
    if command.kind is WebShopCommandKind.CLICK:
        return f"click[{command.argument}]"
    return "click[buy now]"


def _public_step(
    *,
    observation_text: str,
    available_actions: tuple[str, ...],
    terminal: bool,
    budget_usage: BudgetVector,
) -> WebShopPublicStep:
    return WebShopPublicStep(
        public_observation={
            "available_actions": list(available_actions),
            "terminal": terminal,
            "text": observation_text,
        },
        terminal=terminal,
        budget_usage=budget_usage,
    )


@dataclass(slots=True)
class _OfficialEpisodeState:
    case: PrivateWebShopCase
    goal: OfficialWebShopGoal
    env_factory: OfficialWebShopTextEnvFactory
    env: OfficialWebShopTextEnv
    entries: list[_ExecutionEntry] = field(default_factory=list)
    terminal_reward: float | None = None

    @classmethod
    def create(
        cls,
        case: PrivateWebShopCase,
        goal: OfficialWebShopGoal,
        env_factory: OfficialWebShopTextEnvFactory,
    ) -> _OfficialEpisodeState:
        env = _create_pinned_env(case, goal, env_factory)
        return cls(case=case, goal=goal, env_factory=env_factory, env=env)

    async def execute(
        self,
        command: WebShopCommand,
        *,
        step_index: int,
    ) -> WebShopPublicStep:
        if self.entries and step_index <= self.entries[-1].step_index:
            raise ValueError("official WebShop steps must be strictly increasing")
        if self.terminal_reward is not None:
            raise ValueError("official WebShop episode already terminated")

        started_ns = time.perf_counter_ns()
        result = await asyncio.to_thread(self.env.step, _official_action(command))
        if not isinstance(result, OfficialWebShopStepResult):
            raise TypeError("official WebShop env returned an incompatible step")
        actions = _available_actions(await asyncio.to_thread(self.env.get_available_actions))
        elapsed_ns = time.perf_counter_ns() - started_ns
        budget = BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=(elapsed_ns + 999_999) // 1_000_000,
        )
        public_step = _public_step(
            observation_text=result.observation_text,
            available_actions=actions,
            terminal=result.terminal,
            budget_usage=budget,
        )
        entry = _ExecutionEntry(step_index, command, public_step)
        self.entries.append(entry)
        if result.terminal:
            self.terminal_reward = result.reward
        return public_step


def _create_pinned_env(
    case: PrivateWebShopCase,
    goal: OfficialWebShopGoal,
    env_factory: OfficialWebShopTextEnvFactory,
) -> OfficialWebShopTextEnv:
    env = env_factory.create(goal)
    if env.goal_id != goal.goal_id:
        raise ValueError("official WebShop env has another goal identity")
    if env.session_id != goal.session_id:
        raise ValueError("official WebShop env has another session identity")
    reset_observation = _text(env.reset(goal.session_id), field_name="official reset observation")
    if env.goal_id != goal.goal_id or env.session_id != goal.session_id:
        raise ValueError("official WebShop reset changed the pinned goal/session identity")
    if _text(env.instruction_text, field_name="official instruction") != case.public.query:
        raise ValueError("official WebShop instruction differs from the public query")
    reset_actions = _available_actions(env.get_available_actions())
    context = case.public.public_context
    if not isinstance(context, dict):
        raise ValueError("official WebShop task lacks its public reset projection")
    expected_observation = context.get("initial_observation")
    expected_actions = context.get("initial_available_actions")
    if expected_observation is None and expected_actions is None:
        return env
    if (
        type(expected_observation) is not str
        or not isinstance(expected_actions, list)
        or any(type(action) is not str for action in expected_actions)
        or reset_observation != expected_observation
        or reset_actions != tuple(expected_actions)
    ):
        raise ValueError("official WebShop reset differs from its model-visible projection")
    return env


@dataclass(frozen=True, slots=True)
class _OfficialEpisode:
    state: _OfficialEpisodeState = field(repr=False)

    @property
    def task_id(self) -> str:
        return self.state.case.public.task_id

    @property
    def environment_id(self) -> str:
        return self.state.case.public.environment_id

    async def execute(
        self,
        command: WebShopCommand,
        *,
        step_index: int,
    ) -> WebShopPublicStep:
        return await self.state.execute(command, step_index=step_index)


@dataclass(frozen=True, slots=True)
class _OfficialRewardView:
    state: _OfficialEpisodeState = field(repr=False)

    @property
    def task_id(self) -> str:
        return self.state.case.public.task_id

    @property
    def environment_id(self) -> str:
        return self.state.case.public.environment_id

    def final_reward(self) -> float:
        if self.state.terminal_reward is None:
            raise WebShopRewardUnavailableError(
                "official WebShop reward is unavailable before terminal state"
            )
        return self.state.terminal_reward


@dataclass(frozen=True, slots=True)
class OfficialWebShopEpisodeFactory:
    """Adapt an injected official text environment to the existing session API."""

    env_factory: OfficialWebShopTextEnvFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.env_factory, "create", None)):
            raise TypeError("official WebShop text environment factory must implement create")

    def create(self, case: PrivateWebShopCase) -> PrivateWebShopEpisodeSession:
        if not isinstance(case.private_goal, OfficialWebShopGoal):
            raise TypeError("official WebShop bridge requires OfficialWebShopGoal")
        goal = case.private_goal
        if goal.task_id != case.public.task_id:
            raise ValueError("official WebShop goal belongs to another task")
        if goal.environment_id != case.public.environment_id:
            raise ValueError("official WebShop goal belongs to another environment snapshot")
        state = _OfficialEpisodeState.create(case, goal, self.env_factory)
        return PrivateWebShopEpisodeSession(
            episode=_OfficialEpisode(state),
            reward_view=_OfficialRewardView(state),
            cleanup=state.env.close,
        )


__all__ = [
    "OfficialWebShopEpisodeFactory",
    "OfficialWebShopGoal",
    "OfficialWebShopStepResult",
    "OfficialWebShopTextEnv",
    "OfficialWebShopTextEnvFactory",
]
