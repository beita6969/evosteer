"""Typed acceptance records for bounded no-update and one-step debug runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoUpdateControlReceipt:
    condition_id: str
    trajectory_count: int
    reward_count: int
    optimizer_step: int
    active_skill_count: int
    retrieved_skill_count: int
    checkpoint_written: bool
    infrastructure_failures: int

    def validate(self) -> None:
        if self.condition_id != "structured-no-skill":
            raise ValueError("no-update control requires structured-no-skill")
        if self.trajectory_count != 16 or self.reward_count != 16:
            raise ValueError("no-update control requires one complete batch")
        if self.optimizer_step != 0 or self.checkpoint_written:
            raise ValueError("no-update control mutated training state")
        if self.active_skill_count or self.retrieved_skill_count:
            raise ValueError("no-update control received skills")
        if self.infrastructure_failures:
            raise ValueError("no-update control had infrastructure failures")


@dataclass(frozen=True, slots=True)
class OneStepDebugReceipt:
    trajectory_count: int
    reward_count: int
    optimizer_steps: int
    commits: int
    adapter_revisions: int
    checkpoints: int
    resume_cursor: int
    infrastructure_failures: int

    def validate(self) -> None:
        if self.trajectory_count != 16 or self.reward_count != 16:
            raise ValueError("one-step debug requires one complete batch")
        if (
            self.optimizer_steps,
            self.commits,
            self.adapter_revisions,
            self.checkpoints,
            self.resume_cursor,
        ) != (1, 1, 1, 1, 1):
            raise ValueError("one-step transaction did not publish exactly once")
        if self.infrastructure_failures:
            raise ValueError("one-step debug had infrastructure failures")


__all__ = ["NoUpdateControlReceipt", "OneStepDebugReceipt"]
