"""Typed worker result for bounded Protocol 11 smoke attempts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProtocolV11AttemptWorkerResult:
    completed_trajectories: int
    infrastructure_failures: int
    optimizer_steps: int
    commits: int

    def __post_init__(self) -> None:
        if (
            min(
                self.completed_trajectories,
                self.infrastructure_failures,
                self.optimizer_steps,
                self.commits,
            )
            < 0
        ):
            raise ValueError("Protocol 11 worker counts cannot be negative")
        if self.commits > self.optimizer_steps:
            raise ValueError("Protocol 11 cannot commit more steps than it optimized")


__all__ = ["ProtocolV11AttemptWorkerResult"]
