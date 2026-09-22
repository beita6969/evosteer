"""Answer-separated adapters for benchmark-native direct interaction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from skillev.evaluation.direct_baseline import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev_private.benchmarks.alfworld_official import OfficialALFWorldTextEnv
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldTextEnv
from skillev_private.benchmarks.webshop_official import OfficialWebShopTextEnv


@dataclass(slots=True)
class DirectWebShopEnvironment:
    """Expose the official WebShop process through native commands."""

    environment: OfficialWebShopTextEnv
    session_id: str
    expected_instruction: str
    _last_observation: str | None = None
    _last_reward: float = 0.0
    _terminal: bool = False

    async def reset(self) -> NativePublicState:
        text = await asyncio.to_thread(self.environment.reset, self.session_id)
        if self.environment.instruction_text != self.expected_instruction:
            raise RuntimeError("WebShop goal differs from the frozen public task")
        actions = self.environment.get_available_actions()
        self._last_observation = text
        return NativePublicState(text, actions)

    async def step(self, action: str) -> NativeEnvironmentStep:
        result = await asyncio.to_thread(self.environment.step, action)
        self._last_reward = result.reward
        self._terminal = result.terminal
        actions = self.environment.get_available_actions()
        self._last_observation = result.observation_text
        return NativeEnvironmentStep(
            self._last_observation,
            result.terminal,
            result.reward,
            result.terminal and result.reward >= 1.0,
            None,
            available_actions=actions,
        )

    async def outcome(self) -> NativeEnvironmentOutcome:
        return NativeEnvironmentOutcome(self._last_reward, self._last_reward >= 1.0, self._terminal)

    async def close(self) -> None:
        await self.environment.close()


@dataclass(slots=True)
class DirectALFWorldEnvironment:
    """Expose one official ALFWorld game with its admissible commands."""

    environment: OfficialALFWorldTextEnv
    seed: int
    expected_instruction: str
    _commands: tuple[str, ...] = ()
    _last_observation: str | None = None
    _success: bool = False
    _terminal: bool = False

    async def reset(self) -> NativePublicState:
        result = await asyncio.to_thread(self.environment.reset, self.seed)
        if result.instruction_text != self.expected_instruction:
            raise RuntimeError("ALFWorld instruction differs from the frozen public task")
        self._commands = result.admissible_commands
        self._last_observation = result.observation_text
        return NativePublicState(result.observation_text, self._commands)

    async def step(self, action: str) -> NativeEnvironmentStep:
        result = await asyncio.to_thread(self.environment.step, action)
        self._commands = result.admissible_commands
        self._terminal = result.terminal
        self._success = result.success is True
        self._last_observation = result.observation_text
        return NativeEnvironmentStep(
            self._last_observation,
            result.terminal,
            float(self._success),
            self._success,
            None,
            available_actions=self._commands,
        )

    async def outcome(self) -> NativeEnvironmentOutcome:
        return NativeEnvironmentOutcome(float(self._success), self._success, self._terminal)

    async def close(self) -> None:
        await self.environment.close()


@dataclass(slots=True)
class DirectScienceWorldEnvironment:
    """Expose one official ScienceWorld variation with normalized native score."""

    environment: OfficialScienceWorldTextEnv
    task_name: str
    variation_index: int
    seed: int
    max_steps: int
    _score: float = 0.0
    _terminal: bool = False
    _raw_score: float | None = None
    _steps: int = 0
    _public_state: dict[str, str] = field(default_factory=dict)
    _native_moves: int | None = None
    _private_diagnostics: dict[str, object] = field(default_factory=dict)

    async def reset(self) -> NativePublicState:
        observation = await asyncio.to_thread(
            self.environment.reset,
            task_name=self.task_name,
            variation_index=self.variation_index,
            seed=self.seed,
            max_steps=self.max_steps,
        )
        self._score, self._terminal, self._raw_score, self._steps = 0.0, False, None, 0
        self._private_diagnostics = {}
        self._public_state = getattr(self.environment, "public_state", {})
        self._raw_score = getattr(self.environment, "reset_score", None)
        self._native_moves = getattr(self.environment, "native_moves", None)
        if self._raw_score is not None:
            self._score = max(0.0, self._raw_score) / 100
        return NativePublicState(observation)

    async def step(self, action: str) -> NativeEnvironmentStep:
        result = await asyncio.to_thread(self.environment.step, action)
        self._score = result.score / 100.0
        self._raw_score = result.raw_score
        self._public_state, self._native_moves = result.public_state, result.native_moves
        self._private_diagnostics = result.private_diagnostics
        self._steps += 1
        self._terminal = result.terminal
        return NativeEnvironmentStep(
            result.observation_text,
            result.terminal,
            self._score,
            self._score >= 1.0,
            None,
        )

    async def outcome(self) -> NativeEnvironmentOutcome:
        return NativeEnvironmentOutcome(
            self._score,
            self._score >= 1.0,
            self._terminal,
            terminated_by_horizon=not self._terminal,
        )

    async def close(self) -> None:
        await self.environment.close()


__all__ = [
    "DirectALFWorldEnvironment",
    "DirectScienceWorldEnvironment",
    "DirectWebShopEnvironment",
]
