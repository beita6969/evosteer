"""Independent no-Bayesian training source over shared runtime components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skillev.contracts import (
    TrainingStepReportValue,
)
from skillev.evolution.detector import DetectorRuntimeState
from skillev.experiments.arm_events import ArmEventType, LiveArmEventLog
from skillev.policy import PolicyBackbone
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot, RolloutGenerator
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    AttemptRunProgress,
    BudgetLedger,
    BudgetVector,
    EventEnvelope,
    EventType,
    FlowOnlyRuntimeExecutionState,
    OrderedTaskCursorState,
    RuntimeEventEmitter,
    RuntimeSnapshot,
    RuntimeSnapshotIdentity,
    SkillLibraryState,
)
from skillev.runtime.attempt_run_plan import RemainingAttemptCapacity
from skillev.training import (
    AppliedTrainingState,
    BatchReadyTrainingState,
    CollectedTrainingBatch,
    IdleTrainingState,
    RolloutSessionFactory,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
    SkillLibraryView,
    TaskProvider,
    TrainerConfig,
    TrainingCheckpointSnapshot,
    TrainingCheckpointStore,
    TrainingRuntimeState,
    TrainingStepExecutionContext,
    TTBStepPreparer,
)
from skillev.training.runtime_components import RolloutBatchCollector, TTBOptimizerKernel

from .no_bayesian_calibration import (
    FlowOnlyProjectionPipeline,
    FlowOnlyProjectionTransition,
    FlowOnlyTrainingSource,
)
from .no_bayesian_contracts import FlowOnlyTrainingStepCommit

if TYPE_CHECKING:
    import torch


class FlowOnlyTrainingLoop:
    """No-Bayesian source and projection over shared collector/kernel objects."""

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
        projections: FlowOnlyProjectionPipeline,
        checkpoint_store: TrainingCheckpointStore,
        arm_event_log: LiveArmEventLog | None,
        sampling_schedule_hash: str,
        ordered_task_sequence_hash: str,
        gradient_preparer: TTBStepPreparer | None = None,
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
            condition_id="trained-skillev-no-bayesian@1",
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
        )
        self._flow_projections = projections
        self._arm_event_log = arm_event_log
        self._emitter = emitter
        self._state: TrainingRuntimeState = IdleTrainingState()
        self._pending: PendingFlowOnlyTrainingStep | None = None

    @property
    def optimizer_step(self) -> int:
        return self._kernel.optimizer_step

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
    def generator(self) -> RolloutGenerator:
        return self._collector.generator

    @property
    def ledger(self) -> BudgetLedger:
        return self._collector.ledger

    @property
    def decoding_snapshot(self) -> DecodingSnapshot:
        return self._collector.decoding_snapshot

    @property
    def config(self) -> TrainerConfig:
        return self._kernel.config

    @property
    def scientific_base_seed(self) -> int:
        return self._kernel.config.rollout.base_seed

    @property
    def checkpoint_due(self) -> bool:
        return self._kernel.checkpoint_due

    @property
    def policy_snapshot_id(self) -> str:
        return self._kernel.policy_snapshot().snapshot_id

    @property
    def workflow_resources(self) -> RolloutWorkflowResources:
        return self._collector.workflow_resources

    @property
    def gradient_preparer(self) -> TTBStepPreparer | None:
        return self._kernel.gradient_preparer

    async def collect_batch(self) -> CollectedTrainingBatch:
        if not isinstance(self._state, IdleTrainingState):
            raise RuntimeError("collect_batch requires idle flow-only state")
        batch = await self._collector.collect(optimizer_step=self.optimizer_step + 1)
        self._state = BatchReadyTrainingState(batch)
        return batch

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
        state = self._state
        if not isinstance(state, BatchReadyTrainingState):
            raise RuntimeError("apply_step requires a ready flow-only batch")
        if not isinstance(context, TrainingStepExecutionContext):
            raise TypeError("train_step requires TrainingStepExecutionContext")
        batch = state.batch
        prepared = self._kernel.prepare(batch)
        source = FlowOnlyTrainingSource(
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            records=tuple(artifact.record for artifact in batch.artifacts),
            stats=prepared.stats,
            edge_records=prepared.edges,
        )
        projection = self._flow_projections.preview(source)
        report = self._kernel.apply(prepared)
        commit = FlowOnlyTrainingStepCommit(
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            policy_snapshot_before=prepared.snapshot_before.snapshot_id,
            policy_snapshot_after=self._kernel.policy_snapshot().snapshot_id,
            library_version=batch.library_version,
            records=tuple(artifact.record for artifact in batch.artifacts),
            edge_records=prepared.edges,
            stats=prepared.stats,
            report=report,
            run_cursor_after=context.run_cursor_after,
        )
        self._pending = PendingFlowOnlyTrainingStep(
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
            raise RuntimeError("projection install requires an applied flow-only step")
        if state.projection_installed:
            raise RuntimeError("flow-only projection is already installed")
        self._flow_projections.commit(pending.projection)
        self._state = AppliedTrainingState(
            batch_id=state.batch_id,
            optimizer_step=state.optimizer_step,
            projection_installed=True,
        )

    @property
    def pending_commit(self) -> FlowOnlyTrainingStepCommit:
        if self._pending is None:
            raise RuntimeError("no flow-only training step is pending")
        return self._pending.commit

    def prepare_applied_step_event(self) -> EventEnvelope:
        return self._emitter.prepare(
            EventType.FLOW_ONLY_TRAINING_STEP_COMMITTED,
            self.pending_commit.to_value(),
        )

    def finalize_applied_step_event(self, event: EventEnvelope) -> TrainingStepReportValue:
        state = self._state
        pending = self._pending
        if (
            not isinstance(state, AppliedTrainingState)
            or not state.projection_installed
            or pending is None
        ):
            raise RuntimeError("finalize requires an installed flow-only projection")
        if (
            event.event_type is not EventType.FLOW_ONLY_TRAINING_STEP_COMMITTED
            or event.payload != pending.commit.to_value()
        ):
            raise ValueError("prepared event differs from the pending flow-only step")
        self._emitter.publish_prepared(event)
        if self._arm_event_log is not None:
            self._arm_event_log.append(
                ArmEventType.FLOW_ONLY_TRAINING_STEP_COMMITTED,
                pending.commit.to_value(),
            )
        self._pending = None
        self._state = IdleTrainingState()
        return pending.report

    def finalize_applied_step(self) -> TrainingStepReportValue:
        return self.finalize_applied_step_event(self.prepare_applied_step_event())

    def train_step(self, context: TrainingStepExecutionContext) -> TrainingStepReportValue:
        self.apply_step(context)
        self.install_applied_projection()
        return self.finalize_applied_step()

    async def run_one(self, context: TrainingStepExecutionContext) -> TrainingStepReportValue:
        await self.collect_batch()
        return self.train_step(context)

    def reset_partition(self, seed: int) -> str:
        if not (
            isinstance(self._state, IdleTrainingState)
            or (isinstance(self._state, AppliedTrainingState) and self._state.projection_installed)
        ):
            raise RuntimeError("partition reset requires a committed flow-only projection")
        return self._kernel.reset_partition(seed)

    def snapshot(
        self,
        *,
        detector_state: DetectorRuntimeState,
        run_progress: AttemptRunProgress,
        snapshot_identity: RuntimeSnapshotIdentity,
    ) -> TrainingCheckpointSnapshot:
        task_cursor = self._collector.task_provider.runtime_state
        library_state = self._collector.library.state
        if not isinstance(task_cursor, OrderedTaskCursorState):
            raise TypeError("task provider runtime state must be OrderedTaskCursorState")
        if not isinstance(library_state, SkillLibraryState):
            raise TypeError("flow-only library state must be SkillLibraryState")
        return TrainingCheckpointSnapshot(
            optimizer_step=self.optimizer_step,
            experiment_id=self.config.execution.experiment_id,
            identity=snapshot_identity,
            backbone=self.backbone,
            optimizer=self.optimizer,
            execution_state=FlowOnlyRuntimeExecutionState(
                task_cursor=task_cursor,
                run_cursor=run_progress.state,
                library=library_state,
                projections=self._flow_projections.runtime_state(),
                detector=detector_state,
            ),
        )

    def restore_policy_optimizer_exact(
        self,
        directory: str | Path,
        *,
        expected_identity: RuntimeSnapshotIdentity,
    ) -> RuntimeSnapshot:
        if not isinstance(self._state, IdleTrainingState):
            raise RuntimeError("restore requires idle flow-only state")
        return self._kernel.restore_policy_optimizer_exact(
            directory,
            expected_identity=expected_identity,
        )


__all__ = ["FlowOnlyTrainingLoop", "FlowOnlyTrainingStepCommit"]


@dataclass(frozen=True, slots=True)
class PendingFlowOnlyTrainingStep:
    commit: FlowOnlyTrainingStepCommit
    projection: FlowOnlyProjectionTransition
    report: TrainingStepReportValue
