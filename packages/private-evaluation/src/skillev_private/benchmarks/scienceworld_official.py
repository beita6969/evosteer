"""Strict bridge to an injected official ScienceWorld text environment.

The JVM/official Python package is never imported here.  A deployment wrapper
implements :class:`OfficialScienceWorldTextEnvFactory`; this module binds its
task variation, seed, and step limit to the public case while keeping every
native score behind the private outcome view.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.scienceworld import (
    ScienceWorldCommand,
    ScienceWorldPublicStep,
)
from skillev.runtime import BudgetVector

from .scienceworld import (
    PrivateScienceWorldCase,
    PrivateScienceWorldEpisodeSession,
    ScienceWorldOutcomeUnavailableError,
)


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("official ScienceWorld score must be numeric")
    score = float(value)
    if not math.isfinite(score) or score > 100.0:
        raise ValueError("official ScienceWorld score must be finite and at most 100")
    return score


@dataclass(frozen=True, slots=True)
class OfficialScienceWorldTask:
    """Exact private task variation loaded into the official simulator."""

    task_id: str
    environment_id: str
    task_name: str
    variation_index: int
    seed: int
    max_steps: int
    payload: object = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in ("task_id", "environment_id", "task_name"):
            _text(getattr(self, field_name), field_name=field_name)
        _non_negative_int(self.variation_index, field_name="variation_index")
        _non_negative_int(self.seed, field_name="seed")
        _positive_int(self.max_steps, field_name="max_steps")
        if self.payload is None:
            raise ValueError("official ScienceWorld task payload cannot be absent")


@dataclass(frozen=True, slots=True)
class OfficialScienceWorldStepResult:
    """Raw official transition retained exclusively in the private bridge."""

    observation_text: str
    score: float
    terminal: bool
    raw_score: float = field(init=False)
    public_state: dict[str, str] = field(default_factory=dict)
    native_moves: int | None = None
    private_diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_text",
            _text(self.observation_text, field_name="official ScienceWorld observation"),
        )
        raw_score = _score(self.score)
        object.__setattr__(self, "raw_score", raw_score)
        # Existing reward consumers remain bounded. Never discard the native
        # info["score"]: negative failure states are not a native zero score.
        object.__setattr__(self, "score", max(0.0, raw_score))
        if type(self.terminal) is not bool:
            raise TypeError("official ScienceWorld terminal flag must be boolean")
        if set(self.public_state) - {"current_look", "inventory", "task_description"} or any(
            not isinstance(value, str) for value in self.public_state.values()
        ):
            raise ValueError("ScienceWorld public state must use the observation whitelist")
        if self.native_moves is not None:
            _non_negative_int(self.native_moves, field_name="native moves")


class OfficialScienceWorldTextEnv(Protocol):
    """Narrow surface implemented by a wrapper around the official JVM env."""

    @property
    def task_name(self) -> str: ...

    @property
    def variation_index(self) -> int: ...

    @property
    def seed(self) -> int: ...

    @property
    def max_steps(self) -> int: ...

    @property
    def task_description(self) -> str: ...

    def reset(
        self,
        *,
        task_name: str,
        variation_index: int,
        seed: int,
        max_steps: int,
    ) -> str: ...

    def step(self, action: str) -> OfficialScienceWorldStepResult: ...

    async def close(self) -> None: ...


class OfficialScienceWorldTextEnvFactory(Protocol):
    """Injection point allowed to import and construct official ScienceWorld."""

    def create(self, task: OfficialScienceWorldTask) -> OfficialScienceWorldTextEnv: ...


@dataclass(frozen=True, slots=True)
class _ExecutionEntry:
    step_index: int
    command: ScienceWorldCommand
    public_step: ScienceWorldPublicStep

    def __post_init__(self) -> None:
        _positive_int(self.step_index, field_name="step_index")


def _public_step(
    *,
    observation_text: str,
    terminal: bool,
    budget_usage: BudgetVector,
) -> ScienceWorldPublicStep:
    return ScienceWorldPublicStep(
        public_observation={"terminal": terminal, "text": observation_text},
        terminal=terminal,
        budget_usage=budget_usage,
    )


@dataclass(slots=True)
class _OfficialEpisodeState:
    case: PrivateScienceWorldCase
    task: OfficialScienceWorldTask
    env_factory: OfficialScienceWorldTextEnvFactory
    env: OfficialScienceWorldTextEnv
    entries: list[_ExecutionEntry] = field(default_factory=list)
    terminal_score: float | None = None

    @classmethod
    def create(
        cls,
        case: PrivateScienceWorldCase,
        task: OfficialScienceWorldTask,
        env_factory: OfficialScienceWorldTextEnvFactory,
    ) -> _OfficialEpisodeState:
        env = _create_pinned_env(case, task, env_factory)
        return cls(case=case, task=task, env_factory=env_factory, env=env)

    async def execute(
        self,
        command: ScienceWorldCommand,
        *,
        step_index: int,
    ) -> ScienceWorldPublicStep:
        if step_index != len(self.entries) + 1:
            raise ValueError("official ScienceWorld steps must be contiguous")
        if step_index > self.task.max_steps:
            raise ValueError("official ScienceWorld exceeded its pinned step limit")
        if self.terminal_score is not None:
            raise ValueError("official ScienceWorld episode already terminated")

        started_ns = time.perf_counter_ns()
        result = self.env.step(command.text)
        elapsed_ns = time.perf_counter_ns() - started_ns
        if not isinstance(result, OfficialScienceWorldStepResult):
            raise TypeError("official ScienceWorld env returned an incompatible step")
        terminal = result.terminal or step_index == self.task.max_steps
        public_step = _public_step(
            observation_text=result.observation_text,
            terminal=terminal,
            budget_usage=BudgetVector(
                tool_calls=1,
                wall_time_milliseconds=(elapsed_ns + 999_999) // 1_000_000,
            ),
        )
        self.entries.append(_ExecutionEntry(step_index, command, public_step))
        if terminal:
            self.terminal_score = result.score
        return public_step


def _initial_observation(case: PrivateScienceWorldCase) -> str:
    context = case.public.public_context
    if not isinstance(context, dict) or "initial_observation" not in context:
        raise ValueError("official ScienceWorld public context requires initial_observation")
    return _text(context["initial_observation"], field_name="public initial observation")


def _create_pinned_env(
    case: PrivateScienceWorldCase,
    task: OfficialScienceWorldTask,
    env_factory: OfficialScienceWorldTextEnvFactory,
) -> OfficialScienceWorldTextEnv:
    env = env_factory.create(task)
    reset_observation = _text(
        env.reset(
            task_name=task.task_name,
            variation_index=task.variation_index,
            seed=task.seed,
            max_steps=task.max_steps,
        ),
        field_name="official ScienceWorld reset observation",
    )
    if env.task_name != task.task_name:
        raise ValueError("official ScienceWorld env has another task name")
    if env.variation_index != task.variation_index:
        raise ValueError("official ScienceWorld env has another task variation")
    if env.seed != task.seed:
        raise ValueError("official ScienceWorld env has another seed")
    if env.max_steps != task.max_steps:
        raise ValueError("official ScienceWorld env has another step limit")
    if _text(env.task_description, field_name="official task description") != case.public.query:
        raise ValueError("official ScienceWorld task description differs from public query")
    if reset_observation != _initial_observation(case):
        raise ValueError("official ScienceWorld reset observation differs from public context")
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

    @property
    def seed(self) -> int:
        return self.state.task.seed

    @property
    def max_steps(self) -> int:
        return self.state.task.max_steps

    async def execute(
        self,
        command: ScienceWorldCommand,
        *,
        step_index: int,
    ) -> ScienceWorldPublicStep:
        return await self.state.execute(command, step_index=step_index)


@dataclass(frozen=True, slots=True)
class _OfficialOutcomeView:
    state: _OfficialEpisodeState = field(repr=False)

    @property
    def task_id(self) -> str:
        return self.state.case.public.task_id

    @property
    def environment_id(self) -> str:
        return self.state.case.public.environment_id

    @property
    def seed(self) -> int:
        return self.state.task.seed

    def final_score(self) -> float:
        if self.state.terminal_score is None:
            raise ScienceWorldOutcomeUnavailableError(
                "official ScienceWorld score is unavailable before terminal state"
            )
        return self.state.terminal_score


@dataclass(frozen=True, slots=True)
class OfficialScienceWorldEpisodeFactory:
    """Adapt an official simulator factory to the existing private session API."""

    env_factory: OfficialScienceWorldTextEnvFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.env_factory, "create", None)):
            raise TypeError("official ScienceWorld env factory must implement create")

    def create(self, case: PrivateScienceWorldCase) -> PrivateScienceWorldEpisodeSession:
        if not isinstance(case.private_task, OfficialScienceWorldTask):
            raise TypeError("official ScienceWorld bridge requires OfficialScienceWorldTask")
        task = case.private_task
        if task.task_id != case.public.task_id:
            raise ValueError("official ScienceWorld task belongs to another public task")
        if task.environment_id != case.public.environment_id:
            raise ValueError("official ScienceWorld task belongs to another environment")
        if task.seed != case.public.seed:
            raise ValueError("official ScienceWorld task seed differs from public identity")
        if task.max_steps != case.public.max_steps:
            raise ValueError("official ScienceWorld task step limit differs from public identity")
        state = _OfficialEpisodeState.create(case, task, self.env_factory)
        return PrivateScienceWorldEpisodeSession(
            episode=_OfficialEpisode(state),
            outcome_view=_OfficialOutcomeView(state),
        )


__all__ = [
    "OfficialScienceWorldEpisodeFactory",
    "OfficialScienceWorldStepResult",
    "OfficialScienceWorldTask",
    "OfficialScienceWorldTextEnv",
    "OfficialScienceWorldTextEnvFactory",
]
