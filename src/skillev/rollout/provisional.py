"""Completed-step training channel, independent of traces and terminal labels."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from skillev.contracts import TrajectoryStep
from skillev.policy.interface import EncodedPolicyPrompt, ModelInputWindow
from skillev.rollout.types import PolicySnapshot


@dataclass(frozen=True, slots=True)
class ProvisionalStep:
    trajectory_id: str
    task_id: str
    policy: PolicySnapshot
    library_version: str
    query: str
    initial_text: str
    previous_steps: tuple[TrajectoryStep, ...]
    step: TrajectoryStep
    input_window: ModelInputWindow | None = None
    # Actual admitted action-phase input, including the declared input window.
    # This is private execution data, not an additional model-visible field.
    forward_input: EncodedPolicyPrompt | None = None


ProvisionalStepSink: TypeAlias = Callable[[ProvisionalStep], Awaitable[None]]
