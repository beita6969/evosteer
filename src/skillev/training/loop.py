"""The complete fixed-population ``collect → train`` TTB loop."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from skillev.contracts import JsonValue, TrainingStepCommit, TrainingStepReportValue
from skillev.policy import PolicyBackbone
from skillev.rollout import (
    CanonicalInitialContextAssembler,
    DecodingSnapshot,
    RolloutGenerator,
)
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    BudgetLedger,
    BudgetVector,
    EventEnvelope,
    EventType,
    RuntimeEventEmitter,
    RuntimeSnapshot,
    RuntimeSnapshotIdentity,
)
from skillev.runtime.attempt_run_plan import RemainingAttemptCapacity

from .checkpoint import TrainingCheckpointStore
from .config import TrainerConfig
from .deadline import TrainingDeadline
from .inflight import InFlightBatchStore
from .planning import (
    AppliedTrainingState,
    BatchReadyTrainingState,
    CollectedTrainingBatch,
    IdleTrainingState,
    TrainingRuntimeState,
    TrainingStepExecutionContext,
)
from .projections import ProjectionTransition, TrainingProjectionPipeline, TrainingStepSource
from .rollout_workflow import (
    RolloutBatchPerformanceReport,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
)
from .runtime_components import (
    RolloutBatchCollector,
    RolloutSessionFactory,
    SkillLibraryView,
    TaskProvider,
    TTBOptimizerKernel,
)
from .stability import ReferencePolicyScorer
from .step_math import ComponentGradientNorms, PreparedTTBStep, TTBStepPreparer
from .step_timing import StepTiming
from .streaming_step import GradientStepStream


class TrainingLoop:
    """Alternate one complete on-policy batch with exactly one AdamW update."""

    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        generator: RolloutGenerator,
        task_provider: TaskProvider,
        session_factory: RolloutSessionFactory,
        context_assembler: CanonicalInitialContextAssembler,
        library: SkillLibraryView,
        config: TrainerConfig,
        ledger: BudgetLedger,
        emitter: RuntimeEventEmitter,
        clock: Callable[[], str],
        projections: TrainingProjectionPipeline,
        checkpoint_store: TrainingCheckpointStore,
        sampling_schedule_hash: str,
        ordered_task_sequence_hash: str,
        gradient_preparer: TTBStepPreparer | None = None,
        reference_scorer: ReferencePolicyScorer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
    ) -> None:
        self._collector = RolloutBatchCollector(
            generator=generator,
            task_provider=task_provider,
            session_factory=session_factory,
            context_assembler=context_assembler,
            library=library,
            config=config,
            ledger=ledger,
            emitter=emitter,
            clock=clock,
            sampling_schedule_hash=sampling_schedule_hash,
            ordered_task_sequence_hash=ordered_task_sequence_hash,
            condition_id=config.rollout.condition_id,
            initial_context_profile=InitialContextProfile.TRAINED_SKILLEV,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
        )
        self._kernel = TTBOptimizerKernel(
            backbone=backbone,
            generator=generator,
            config=config,
            clock=clock,
            checkpoint_store=checkpoint_store,
            gradient_preparer=gradient_preparer,
            reference_scorer=reference_scorer,
        )
        self._projections = projections
        self._emitter = emitter
        self._state: TrainingRuntimeState = IdleTrainingState()
        self._pending: PendingTrainingStep | None = None
        self._streamed_prepared: PreparedTTBStep | None = None
        self._timings: tuple[StepTiming, ...] = ()
        self._process_instance = f"{os.getpid()}:{time.monotonic_ns()}"
        self._execution_stage = ("idle", time.perf_counter())
        self._collecting = False
        self._live_gradient_stream: GradientStepStream | None = None
        self.execution_deadline: TrainingDeadline | None = None
        self._timing_start = self._rollout_end = self._gradient_start = self._gradient_end = 0.0
        self._overlap_completed = 0
        self._rank_metrics: tuple[dict[str, JsonValue], ...] = ()
        self._last_finalized_report: TrainingStepReportValue | None = None

    @property
    def optimizer_step(self) -> int:
        return self._kernel.optimizer_step

    def enable_update_observation(self) -> None:
        """Enable read-only parameter/Adam evidence for subsequent optimizer steps."""
        self._kernel.enable_update_observation()

    def configure_inflight(self, directory: Path, *, condition: dict[str, JsonValue]) -> None:
        if self._collecting or self._pending is not None:
            raise RuntimeError("in-flight persistence must be configured before collection")
        self._collector.inflight_store = InFlightBatchStore(directory, condition=condition)

    @property
    def rollout_progress(self) -> dict[str, object]:
        return {
            "clock": "host-monotonic",
            "process_instance_id": self._process_instance,
            "trajectories": [row.snapshot() for row in self._collector._live_rollouts.values()],
            "resources": self._collector._workflow_resources.active_snapshot(),
        }

    @property
    def state(self) -> TrainingRuntimeState:
        return self._state

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._kernel.optimizer

    @property
    def backbone(self) -> PolicyBackbone:
        return self._kernel.backbone

    @property
    def ledger(self) -> BudgetLedger:
        return self._collector.ledger

    @property
    def decoding_snapshot(self) -> DecodingSnapshot:
        return self._collector.decoding_snapshot

    @property
    def policy_snapshot_id(self) -> str:
        return self._kernel.policy_snapshot().snapshot_id

    @property
    def config(self) -> TrainerConfig:
        return self._kernel.config

    @property
    def scientific_base_seed(self) -> int:
        return self._kernel.config.rollout.base_seed

    @property
    def task_provider(self) -> TaskProvider:
        return self._collector.task_provider

    @property
    def library(self) -> SkillLibraryView:
        return self._collector.library

    @property
    def workflow_binding(self) -> RolloutWorkflowBinding:
        return self._collector.workflow_binding

    @property
    def workflow_resources(self) -> RolloutWorkflowResources:
        return self._collector.workflow_resources

    @property
    def gradient_preparer(self) -> TTBStepPreparer | None:
        return self._kernel.gradient_preparer

    @property
    def last_rollout_performance(self) -> RolloutBatchPerformanceReport | None:
        return self._collector.last_performance_report

    @property
    def last_finalized_report(self) -> TrainingStepReportValue | None:
        """Most recent fully durable step, for answer-free live telemetry."""

        return self._last_finalized_report

    @property
    def finalized_timings(self) -> tuple[StepTiming, ...]:
        return self._timings

    @property
    def execution_progress(self) -> dict[str, object]:
        """Controller phase exists even without a live CUDA gradient stream."""
        stage, started = self._execution_stage
        return {
            "process_instance_id": self._process_instance,
            "transaction_stage": stage,
            "transaction_stage_started": started,
            "transaction_stage_age_seconds": time.perf_counter() - started,
        }

    @property
    def gradient_progress(self) -> dict[str, object] | None:
        stream = self._live_gradient_stream
        if stream is None:
            return None
        return {
            **stream.progress_snapshot(),
            **self.execution_progress,
        }

    def set_execution_stage(self, stage: str) -> None:
        self._execution_stage = stage, time.perf_counter()

    async def collect_batch(self) -> CollectedTrainingBatch:
        if not isinstance(self._state, IdleTrainingState) or self._collecting:
            raise RuntimeError("collect_batch requires idle training state")
        if self.execution_deadline is not None:
            self.execution_deadline.require_next_step(time.monotonic())
        self.set_execution_stage("collecting")
        self._collecting = True
        self._timing_start = time.perf_counter()
        stream = self._kernel.create_gradient_stream()
        self._live_gradient_stream = stream
        try:
            timeout_seconds = (
                None
                if self.execution_deadline is None
                else self.execution_deadline.active_step_seconds
            )
            async with asyncio.timeout(timeout_seconds):
                batch = await self._collector.collect(
                    optimizer_step=self.optimizer_step + 1, gradient_stream=stream
                )
                self._rollout_end = time.perf_counter()
                if stream is not None:
                    self.set_execution_stage("sealing-gradient-batch")
                    self._streamed_prepared = await stream.seal(batch)
                    assert stream.gradient_started is not None
                    assert stream.gradient_finished is not None
                    self._gradient_start, self._gradient_end = (
                        stream.gradient_started,
                        stream.gradient_finished,
                    )
                    self._overlap_completed = stream.completed_before_last_rollout
                    self._rank_metrics = stream.rank_metrics
            self._state = BatchReadyTrainingState(batch)
            return batch
        finally:
            if not isinstance(self._state, BatchReadyTrainingState):
                if stream is not None:
                    await stream.discard_uncommitted()
                self._streamed_prepared = None
            self._collecting = False

    def validate_attempt_budget(
        self,
        *,
        capacity: RemainingAttemptCapacity,
        phi_per_cycle_maximum: BudgetVector,
    ) -> None:
        self._collector.validate_attempt_budget(
            capacity=capacity,
            phi_per_cycle_maximum=phi_per_cycle_maximum,
        )

    def apply_step(self, context: TrainingStepExecutionContext) -> TrainingStepReportValue:
        """Apply optimizer math while deferring the externally visible commit.

        The evolution driver owns the wider transaction: projection install,
        detector/evolution work, adapter publication, and the durable full
        checkpoint must all succeed before :meth:`finalize_applied_step` emits
        the sole committed-step event.  A failure after this method is fatal to
        the process; resume restores the last complete checkpoint rather than
        continuing with this in-memory transition.
        """

        state = self._state
        if not isinstance(state, BatchReadyTrainingState):
            raise RuntimeError("apply_step requires a ready batch")
        if not isinstance(context, TrainingStepExecutionContext):
            raise TypeError("train_step requires TrainingStepExecutionContext")
        batch = state.batch
        self.set_execution_stage("projection-preview-and-optimizer")

        if self._streamed_prepared is None:
            self._gradient_start = time.perf_counter()
            prepared = self._kernel.prepare(batch)
            self._gradient_end = time.perf_counter()
        else:
            prepared = self._streamed_prepared
            if (
                prepared.batch is not batch
                or prepared.snapshot_before.snapshot_id != self.policy_snapshot_id
            ):
                raise ValueError("streamed gradients differ from the current batch or policy")
            self._streamed_prepared = None
        source = TrainingStepSource(
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            artifacts=batch.artifacts,
            stats=prepared.stats,
            edge_records=prepared.edges,
        )
        projection = self._projections.preview(source)
        report = self._kernel.apply(prepared)
        snapshot_after = self._kernel.policy_snapshot()
        commit = TrainingStepCommit(
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            policy_snapshot_before=prepared.snapshot_before.snapshot_id,
            policy_snapshot_after=snapshot_after.snapshot_id,
            library_version=batch.library_version,
            records=tuple(artifact.record for artifact in batch.artifacts),
            edge_records=prepared.edges,
            stats=prepared.stats,
            posterior_batch=projection.posterior_batch,
            report=report,
            run_cursor_after=context.run_cursor_after,
        )
        self._pending = PendingTrainingStep(
            commit=commit,
            projection=projection,
            report=report,
        )
        self._state = AppliedTrainingState(
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
        )
        return report

    def install_applied_projection(self) -> None:
        state = self._state
        pending = self._pending
        if not isinstance(state, AppliedTrainingState) or pending is None:
            raise RuntimeError("projection install requires an applied optimizer step")
        if state.projection_installed:
            raise RuntimeError("pending projection is already installed")
        self.set_execution_stage("projection-install")
        self._projections.commit(pending.projection)
        self._state = AppliedTrainingState(
            batch_id=state.batch_id,
            optimizer_step=state.optimizer_step,
            projection_installed=True,
        )

    def finalize_applied_step(self) -> TrainingStepReportValue:
        """Publish the one commit event after the outer transaction is durable."""

        state = self._state
        pending = self._pending
        if (
            not isinstance(state, AppliedTrainingState)
            or not state.projection_installed
            or pending is None
        ):
            raise RuntimeError("finalize requires an installed pending projection")
        self._emitter.emit(EventType.TRAINING_STEP_COMMITTED, pending.commit.to_value())
        self._record_timing(pending)
        self._last_finalized_report = pending.report
        self._pending = None
        self._state = IdleTrainingState()
        return pending.report

    def _record_timing(self, pending: PendingTrainingStep) -> None:
        self._timings = (
            *self._timings,
            StepTiming(
                pending.commit.batch_id,
                pending.report.optimizer_step,
                self._timing_start,
                self._rollout_end,
                self._gradient_start,
                self._gradient_end,
                time.perf_counter(),
                self.last_rollout_performance,
                self._overlap_completed,
                self._rank_metrics,
                process_instance_id=self._process_instance,
                process_step_index=len(self._timings) + 1,
                service_instance_id=os.environ.get("SKILLEV_SERVING_INSTANCE"),
                rollout_detail=cast(dict[str, JsonValue], self.rollout_progress),
                gradient_detail=cast(dict[str, JsonValue], self.gradient_progress),
            ),
        )

    @property
    def pending_commit(self) -> TrainingStepCommit:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no training step is pending")
        return pending.commit

    def finalize_applied_step_event(self, event: EventEnvelope) -> TrainingStepReportValue:
        """Publish a transaction-prepared step event and clear the pending state."""

        state = self._state
        pending = self._pending
        if (
            not isinstance(state, AppliedTrainingState)
            or not state.projection_installed
            or pending is None
        ):
            raise RuntimeError("finalize requires an installed pending projection")
        if event.event_type is not EventType.TRAINING_STEP_COMMITTED:
            raise ValueError("prepared event is not a training-step commit")
        if event.payload != pending.commit.to_value():
            raise ValueError("prepared event differs from the pending training step")
        self._emitter.publish_prepared(event)
        self._record_timing(pending)
        self._last_finalized_report = pending.report
        self._pending = None
        self._state = IdleTrainingState()
        return pending.report

    def train_step(self, context: TrainingStepExecutionContext) -> TrainingStepReportValue:
        """Compatibility helper for callers without a wider step transaction."""

        self.apply_step(context)
        self.install_applied_projection()
        return self.finalize_applied_step()

    async def run_one(self, context: TrainingStepExecutionContext) -> TrainingStepReportValue:
        """Collect and commit exactly one cursor-bound optimizer step."""

        await self.collect_batch()
        return self.train_step(context)

    def reset_partition(self, seed: int) -> str:
        if self._collecting:
            raise RuntimeError("cannot reset parameters during collection or streamed backward")
        state = self._state
        if not (
            isinstance(state, IdleTrainingState)
            or (isinstance(state, AppliedTrainingState) and state.projection_installed)
        ):
            raise RuntimeError(
                "partition reset requires idle state or an installed outer transaction"
            )
        return self._kernel.reset_partition(seed)

    @property
    def checkpoint_due(self) -> bool:
        return self._kernel.checkpoint_due

    def restore_policy_optimizer_exact(
        self,
        directory: str | Path,
        *,
        expected_identity: RuntimeSnapshotIdentity,
    ) -> RuntimeSnapshot:
        if not isinstance(self._state, IdleTrainingState):
            raise RuntimeError("restore requires idle training state")
        return self._kernel.restore_policy_optimizer_exact(
            directory,
            expected_identity=expected_identity,
        )


__all__ = [
    "ComponentGradientNorms",
    "RolloutSessionFactory",
    "SkillLibraryView",
    "TaskProvider",
    "TrainingLoop",
    "TrainingStepExecutionContext",
]


@dataclass(frozen=True, slots=True)
class PendingTrainingStep:
    commit: TrainingStepCommit
    projection: ProjectionTransition
    report: TrainingStepReportValue
