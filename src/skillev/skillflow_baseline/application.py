"""Thin Protocol 10 application boundary around the official-derived trainer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.runtime.emitter import RuntimeEventEmitter
from skillev.runtime.event_log import EventType

from .config import ExactSkillFlowProtocolConfig
from .rollout import ExactSkillFlowTaskAdapter


class ExactSkillFlowTrainer(Protocol):
    max_steps: int
    _current_step: int

    def setup(self, train_data: list[dict[str, Any]], val_data: list[dict[str, Any]]) -> None: ...

    def train(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExactSkillFlowAttemptSummary:
    initial_optimizer_step: int
    final_optimizer_step: int


@dataclass(slots=True)
class ExactSkillFlowApplication:
    """Method-neutral run surface for the canonical Protocol 10 coordinator."""

    trainer: ExactSkillFlowTrainer
    ordered_tasks: tuple[ExactSkillFlowTaskAdapter, ...]
    config: ExactSkillFlowProtocolConfig
    upstream_revision: str
    parity_contract: str
    after_setup: Callable[[ExactSkillFlowTrainer], None] | None = None
    emitter: RuntimeEventEmitter | None = None

    def __post_init__(self) -> None:
        if len(self.ordered_tasks) != 4608:
            raise ValueError("exact SkillFlow requires the frozen 4,608 episodes")
        if tuple(item.sequence_position for item in self.ordered_tasks) != tuple(range(4608)):
            raise ValueError("exact SkillFlow task positions are not contiguous")
        if self.trainer.max_steps != 288:
            raise ValueError("exact SkillFlow trainer must execute 288 steps")

    async def run(
        self,
        plan: ExactAttemptRunPlan,
        *,
        maximum_steps_this_attempt: int | None = None,
    ) -> ExactSkillFlowAttemptSummary:
        if plan.total_training_steps != 288:
            raise ValueError("exact SkillFlow run plan differs from Protocol 10")
        self.trainer.setup(
            [item.to_upstream() for item in self.ordered_tasks],
            [],
        )
        if self.after_setup is not None:
            self.after_setup(self.trainer)
        initial = self.trainer._current_step
        if maximum_steps_this_attempt is not None:
            if type(maximum_steps_this_attempt) is not int or maximum_steps_this_attempt < 1:
                raise ValueError("bounded run steps must be a positive integer")
            self.trainer.max_steps = min(288, initial + maximum_steps_this_attempt)
        if self.emitter is not None and initial == 0:
            self.emitter.emit(
                EventType.EXACT_SKILLFLOW_INITIALIZED,
                {
                    "parity_contract": self.parity_contract,
                    "upstream_revision": self.upstream_revision,
                },
            )
        self.trainer.train()
        return ExactSkillFlowAttemptSummary(initial, self.trainer._current_step)


def build_exact_skillflow_protocol_v10_application(
    *,
    trainer: ExactSkillFlowTrainer,
    ordered_tasks: tuple[ExactSkillFlowTaskAdapter, ...],
    config: ExactSkillFlowProtocolConfig,
    after_setup: Callable[[ExactSkillFlowTrainer], None] | None = None,
    emitter: RuntimeEventEmitter | None = None,
) -> ExactSkillFlowApplication:
    """Build only the upstream-derived application; Bayesian types are rejected."""

    if not type(trainer).__name__.endswith("GFlowNetTrainer"):
        raise TypeError("exact SkillFlow requires the upstream-derived trainer")
    return ExactSkillFlowApplication(
        trainer=trainer,
        ordered_tasks=ordered_tasks,
        config=config,
        upstream_revision=config.upstream_revision,
        parity_contract=config.parity_contract,
        after_setup=after_setup,
        emitter=emitter,
    )


__all__ = [
    "ExactSkillFlowApplication",
    "ExactSkillFlowAttemptSummary",
    "build_exact_skillflow_protocol_v10_application",
]
