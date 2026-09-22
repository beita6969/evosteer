"""Whole-step CUDA OOM recovery from steady to expanded gradient workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .gradient_worker import (
    AggregatedGradientResult,
    CostBalancedGradientWorkerPool,
    GradientWorkerOOMError,
    SealedGradientBatch,
)


class OOMRecoveryPhase(StrEnum):
    RUN_STEADY = "run_steady"
    GPU3_OOM_DETECTED = "gpu3_oom_detected"
    ABORT_CURRENT_STEP = "abort_current_step"
    DISCARD_UNCOMMITTED_GRADS = "discard_uncommitted_grads"
    PERSIST_OOM_EVENT_AND_BATCH = "persist_oom_event_and_batch"
    RESTORE_LAST_COMPLETED_STEP = "restore_last_completed_step"
    START_EXPANDED_PROFILE = "start_expanded_profile"
    RETRY_IDENTICAL_BATCH = "retry_identical_batch"
    VERIFY_EXACTLY_ONCE = "verify_exactly_once"
    READY_TO_COMMIT = "ready_to_commit"


class OOMRecoveryHooks(Protocol):
    def abort_current_step(self, batch: SealedGradientBatch) -> None: ...

    def discard_uncommitted_gradients(self) -> None: ...

    def persist_oom_event_and_batch(
        self,
        error: GradientWorkerOOMError,
        batch: SealedGradientBatch,
    ) -> None: ...

    def restore_last_completed_step(self) -> None: ...

    def start_expanded_profile(self) -> CostBalancedGradientWorkerPool: ...


@dataclass(slots=True)
class OOMRecoveryCoordinator:
    """Retry exactly one sealed batch after a typed gradient-worker CUDA OOM."""

    hooks: OOMRecoveryHooks
    phase: OOMRecoveryPhase = field(default=OOMRecoveryPhase.RUN_STEADY, init=False)
    transitions: list[OOMRecoveryPhase] = field(default_factory=list, init=False)

    def execute(
        self,
        *,
        batch: SealedGradientBatch,
        steady_pool: CostBalancedGradientWorkerPool,
    ) -> AggregatedGradientResult:
        self._advance(OOMRecoveryPhase.RUN_STEADY)
        try:
            result = steady_pool.compute(batch, attempt=1)
        except GradientWorkerOOMError as error:
            if error.batch_id != batch.batch_id:
                raise ValueError("OOM refers to a different sealed batch") from error
            self._advance(OOMRecoveryPhase.GPU3_OOM_DETECTED)
            self._advance(OOMRecoveryPhase.ABORT_CURRENT_STEP)
            self.hooks.abort_current_step(batch)
            self._advance(OOMRecoveryPhase.DISCARD_UNCOMMITTED_GRADS)
            self.hooks.discard_uncommitted_gradients()
            self._advance(OOMRecoveryPhase.PERSIST_OOM_EVENT_AND_BATCH)
            self.hooks.persist_oom_event_and_batch(error, batch)
            self._advance(OOMRecoveryPhase.RESTORE_LAST_COMPLETED_STEP)
            self.hooks.restore_last_completed_step()
            self._advance(OOMRecoveryPhase.START_EXPANDED_PROFILE)
            expanded_pool = self.hooks.start_expanded_profile()
            if len(expanded_pool.workers) < 2:
                raise RuntimeError(
                    "expanded profile requires primary and standby workers"
                ) from error
            self._advance(OOMRecoveryPhase.RETRY_IDENTICAL_BATCH)
            result = expanded_pool.compute(batch, attempt=2)
        self._advance(OOMRecoveryPhase.VERIFY_EXACTLY_ONCE)
        self._verify(batch, result)
        self._advance(OOMRecoveryPhase.READY_TO_COMMIT)
        return result

    def _advance(self, phase: OOMRecoveryPhase) -> None:
        self.phase = phase
        self.transitions.append(phase)

    @staticmethod
    def _verify(batch: SealedGradientBatch, result: AggregatedGradientResult) -> None:
        if result.batch_id != batch.batch_id or result.optimizer_step != batch.optimizer_step:
            raise ValueError("gradient result does not belong to the sealed batch")
        expected = tuple(item.item_id for item in batch.items)
        if result.processed_item_ids != expected:
            raise ValueError("gradient result did not preserve exact sealed-batch order")


ExpandedPoolFactory = Callable[[], CostBalancedGradientWorkerPool]


__all__ = [
    "ExpandedPoolFactory",
    "OOMRecoveryCoordinator",
    "OOMRecoveryHooks",
    "OOMRecoveryPhase",
]
