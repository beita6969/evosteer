"""Dependency-light rollout session bundles with explicit skill ownership."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from skillev.runtime import FullRetrievedSkillContext, RolloutEnvironmentSession

from .environment import TerminalEvaluator


@dataclass(frozen=True, slots=True)
class UnskilledRolloutSessionBundle:
    environment: RolloutEnvironmentSession
    evaluator: TerminalEvaluator
    cleanup: Callable[[], Awaitable[None]] | None = None


@dataclass(frozen=True, slots=True)
class RolloutSessionBundle:
    environment: RolloutEnvironmentSession
    evaluator: TerminalEvaluator
    retrieved_skills: tuple[FullRetrievedSkillContext, ...]
    cleanup: Callable[[], Awaitable[None]] | None = None


__all__ = ["RolloutSessionBundle", "UnskilledRolloutSessionBundle"]
