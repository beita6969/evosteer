"""Outer training -> detect -> complete Phi -> reset -> continue driver."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from skillev.contracts import (
    AuthoringUsageValue,
    EvolutionCycleCommitted,
    EvolutionMutationValue,
    EvolutionNoOpCommitted,
    EvolutionPhaseOpened,
    JsonValue,
    PhaseCheckpointPublished,
    PhaseTransitionEvent,
    PosteriorCellState,
    TrainingStepCommit,
    TrainingStepReportValue,
    stable_hash,
)
from skillev.diagnostics import BatchDiagnostics, DiagnosticsState
from skillev.evolution.detector import AwaitingDetectorSegment
from skillev.evolution.evidence import AuthoringEdgeEvidence
from skillev.runtime import (
    AdapterGeneration,
    BudgetLedger,
    BudgetVector,
    EventEnvelope,
    EventType,
    FullAttemptSummary,
    RuntimeEventEmitter,
    RuntimeSnapshotStore,
    SkillLibrary,
    StepAdapterPublisher,
    StepTransactionJournal,
    StepTransactionRecord,
    StepTransactionState,
    describe_phase_checkpoint_artifact,
)
from skillev.runtime.attempt_failures import (
    EventAppendFailedError,
    EvolutionMutationFailedError,
    LibraryApplyFailedError,
    PartitionResetFailedError,
)
from skillev.runtime.attempt_run_plan import (
    AttemptRunCursorState,
    AttemptRunProgress,
    ExactAttemptRunPlan,
    RemainingAttemptCapacity,
    RunSlotKind,
)

from .authoring import (
    AuthoringFailedError,
    SkillAuthor,
    SkillAuthoringAuthority,
    authoring_reservation_id,
)
from .cold_start import preview_method_phase
from .config import EvolutionConfig
from .decision import (
    FullEvolutionDecision,
    InsufficientEvolutionBudgetError,
    VerifiedNoOpEvolutionDecision,
    required_phi_budget,
)
from .detector import (
    DetectorObservation,
    DetectorRuntimeState,
    NoPhaseTransition,
    PhaseTransitionDetected,
)
from .evidence import (
    PosteriorEvidenceView,
    TrajectoryEvidenceView,
    WindowFlowView,
    assemble_posterior_evidence_view,
)
from .execution import (
    EvolutionMutation,
    authoring_request_for_proposal,
    build_evolution_mutation,
)
from .projection_snapshot import PhaseProjectionSnapshot

if TYPE_CHECKING:
    from skillev.training.checkpoint import TrainingCheckpointSnapshot
    from skillev.training.planning import TrainingStepExecutionContext
    from skillev.training.projections import FullProjectionRuntimeState


class EvolutionTrainingLoop(Protocol):
    @property
    def optimizer_step(self) -> int: ...

    @property
    def checkpoint_due(self) -> bool: ...

    @property
    def policy_snapshot_id(self) -> str: ...

    @property
    def ledger(self) -> BudgetLedger: ...

    @property
    def scientific_base_seed(self) -> int: ...

    async def collect_batch(self) -> object: ...

    def apply_step(
        self,
        context: TrainingStepExecutionContext,
    ) -> TrainingStepReportValue: ...

    def set_execution_stage(self, stage: str) -> None:
        """Update only the read-only execution sidecar."""
        ...

    def install_applied_projection(self) -> None: ...

    def finalize_applied_step(self) -> TrainingStepReportValue: ...

    @property
    def pending_commit(self) -> TrainingStepCommit: ...

    def finalize_applied_step_event(self, event: EventEnvelope) -> TrainingStepReportValue: ...

    def reset_partition(self, seed: int) -> str: ...

    def validate_attempt_budget(
        self,
        *,
        capacity: RemainingAttemptCapacity,
        phi_per_cycle_maximum: BudgetVector,
    ) -> None: ...


class EvolutionSnapshotFactory(Protocol):
    def snapshot(self) -> TrainingCheckpointSnapshot: ...


class EvolutionProjectionView(Protocol):
    def freeze_for_phase(self, phase: PhaseTransitionEvent) -> PhaseProjectionSnapshot: ...

    @property
    def latest_diagnostic(self) -> BatchDiagnostics: ...

    @property
    def diagnostics_state(self) -> DiagnosticsState: ...

    @property
    def calibration_cells(self) -> tuple[PosteriorCellState, ...]: ...

    @property
    def event_ids_by_skill(self) -> Mapping[str, tuple[str, ...]]: ...

    @property
    def event_ids_by_cell(self) -> Mapping[str, tuple[str, ...]]: ...

    def diagnostics_for_batch_ids(
        self,
        batch_ids: tuple[str, ...],
    ) -> tuple[BatchDiagnostics, ...]: ...

    def authoring_evidence_for_diagnostics(
        self,
        diagnostics: tuple[BatchDiagnostics, ...],
    ) -> Mapping[str, AuthoringEdgeEvidence]: ...

    def preview_reset(
        self,
        *,
        old_version: str,
        new_version: str,
    ) -> FullProjectionRuntimeState: ...

    def commit_reset(self, state: FullProjectionRuntimeState) -> None: ...

    def cold_start_supported(self, config: EvolutionConfig) -> bool: ...


class EvolutionPhaseDetector(Protocol):
    @property
    def state(self) -> DetectorRuntimeState: ...

    def preview_observation(self, diagnostic: BatchDiagnostics) -> DetectorObservation: ...

    def commit_observation(self, observation: DetectorObservation) -> None: ...

    def close_no_op(self, phase: PhaseTransitionEvent, reason: str) -> None: ...

    def preview_reset(
        self,
        *,
        old_version: str,
        new_version: str,
    ) -> AwaitingDetectorSegment: ...

    def commit_reset(self, state: AwaitingDetectorSegment) -> None: ...


class EvolutionDecisionPolicy(Protocol):
    def decide_from_views(
        self,
        *,
        window_flow: WindowFlowView,
        posterior: PosteriorEvidenceView,
        trajectories: TrajectoryEvidenceView,
        config: EvolutionConfig,
    ) -> FullEvolutionDecision: ...


@dataclass(frozen=True, slots=True)
class PhiBudgetAuthority:
    cap: BudgetVector

    def require_available(self, required: BudgetVector) -> None:
        if not required.fits_within(self.cap):
            raise InsufficientEvolutionBudgetError(
                "complete Phi decision exceeds the attempt envelope"
            )

    @property
    def available(self) -> BudgetVector:
        return self.cap


EvolutionRunSummary = FullAttemptSummary


@dataclass(frozen=True, slots=True)
class CommittedStepDurability:
    checkpoint_name: str
    adapter_revision: str

    def __post_init__(self) -> None:
        if Path(self.checkpoint_name).name != self.checkpoint_name:
            raise ValueError("committed checkpoint name must be one path component")
        if not self.adapter_revision.strip():
            raise ValueError("committed adapter revision must be non-empty")


@dataclass(frozen=True, slots=True)
class PreparedStepDurability:
    checkpoint_name: str
    prepared_adapter: object | None

    def __post_init__(self) -> None:
        if Path(self.checkpoint_name).name != self.checkpoint_name:
            raise ValueError("prepared checkpoint name must be one path component")


class RunPlanCycleLimitExceededError(RuntimeError):
    """A verified phase exceeded the predeclared exact-cycle capacity."""


def phase_detection_enabled(slot_kind: RunSlotKind) -> bool:
    """Return whether the pre-registered slot admits a new phase."""

    if not isinstance(slot_kind, RunSlotKind):
        raise TypeError("slot_kind must be RunSlotKind")
    return slot_kind is RunSlotKind.PHASE_SEARCH


class EvolutionLoop:
    """Resolve every detected phase completely inside the current child attempt."""

    def __init__(
        self,
        *,
        training_loop: EvolutionTrainingLoop,
        projections: EvolutionProjectionView,
        detector: EvolutionPhaseDetector,
        library: SkillLibrary,
        author: SkillAuthor,
        policy: EvolutionDecisionPolicy,
        authority: SkillAuthoringAuthority,
        phi_budget: PhiBudgetAuthority,
        config: EvolutionConfig,
        emitter: RuntimeEventEmitter,
        snapshot_store: RuntimeSnapshotStore,
        snapshot_factory: EvolutionSnapshotFactory,
        run_progress: AttemptRunProgress,
        step_adapter_publisher: StepAdapterPublisher | None = None,
        phase_checkpoint_cycle_ordinals: tuple[int, ...] = (),
        step_transaction_journal: StepTransactionJournal | None = None,
    ) -> None:
        self._training_loop = training_loop
        self._projections = projections
        self._detector = detector
        self._library = library
        self._author = author
        self._policy = policy
        self._authority = authority
        self._phi_budget = phi_budget
        self._config = config
        self._emitter = emitter
        self._snapshot_store = snapshot_store
        self._snapshot_factory = snapshot_factory
        self._run_progress = run_progress
        self._step_adapter_publisher = step_adapter_publisher
        self._step_transaction_journal = step_transaction_journal
        self._failed = False
        self._current_adapter_published = False
        if (
            not isinstance(phase_checkpoint_cycle_ordinals, tuple)
            or any(type(item) is not int or item < 1 for item in phase_checkpoint_cycle_ordinals)
            or tuple(sorted(set(phase_checkpoint_cycle_ordinals)))
            != phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("phase checkpoint cycles must be sorted unique positives")
        self._phase_checkpoint_cycle_ordinals = phase_checkpoint_cycle_ordinals
        self._final_snapshot_directory: Path | None = None

    def save_initial_snapshot(self) -> Path:
        """Give even an interrupted first optimizer step a complete rollback point."""
        if self._training_loop.optimizer_step != 0:
            raise ValueError("initial snapshot requires an untrained application")
        return Path(
            self._snapshot_store.save(
                self._snapshot_factory.snapshot(), name="initial-step-00000000"
            )
        )

    def save_condition_boundary(self, name: str) -> Path:
        """Persist a fully restored method under an explicitly changed condition."""
        if self._step_transaction_journal is None or self._step_transaction_journal.pending():
            raise RuntimeError("condition boundary requires settled step transactions")
        return Path(self._snapshot_store.save(self._snapshot_factory.snapshot(), name=name))

    @property
    def final_training_snapshot_directory(self) -> Path:
        """Private checkpoint emitted unconditionally at successful run completion."""

        if self._final_snapshot_directory is None:
            raise RuntimeError("final training snapshot is unavailable before run completion")
        return self._final_snapshot_directory

    @property
    def step_adapter_publisher(self) -> StepAdapterPublisher | None:
        return self._step_adapter_publisher

    @property
    def author(self) -> SkillAuthor:
        return self._author

    @property
    def step_transaction_journal(self) -> StepTransactionJournal | None:
        return self._step_transaction_journal

    async def run(
        self,
        plan: ExactAttemptRunPlan,
        *,
        maximum_steps_this_attempt: int | None = None,
        stop_requested: Callable[[], bool | Awaitable[bool]] | None = None,
    ) -> EvolutionRunSummary:
        if self._failed:
            raise RuntimeError("failed training transaction requires a fresh application restore")
        completed = False
        try:
            from skillev.training.stopping import TrainingPausedError

            try:
                summary = await self._run(
                    plan,
                    maximum_steps_this_attempt=maximum_steps_this_attempt,
                    stop_requested=stop_requested,
                )
            except TrainingPausedError:
                completed = True
                raise
            completed = True
            return summary
        finally:
            self._failed = not completed

    async def _run(
        self,
        plan: ExactAttemptRunPlan,
        *,
        maximum_steps_this_attempt: int | None = None,
        stop_requested: Callable[[], bool | Awaitable[bool]] | None = None,
    ) -> EvolutionRunSummary:
        if not isinstance(plan, ExactAttemptRunPlan):
            raise TypeError("EvolutionLoop.run requires ExactAttemptRunPlan")
        if plan.content_hash != self._run_progress.plan.content_hash:
            raise ValueError("run plan differs from application construction")
        initial_cursor = self._run_progress.state
        initial_cursor.require_plan(plan)
        initial_optimizer_step = self._training_loop.optimizer_step
        if maximum_steps_this_attempt is not None and (
            type(maximum_steps_this_attempt) is not int or maximum_steps_this_attempt < 1
        ):
            raise ValueError("bounded run steps must be a positive integer")
        target_steps = plan.total_training_steps
        if maximum_steps_this_attempt is not None:
            target_steps = min(
                target_steps,
                initial_cursor.completed_training_steps + maximum_steps_this_attempt,
            )
        remaining_steps = target_steps - initial_cursor.completed_training_steps
        if remaining_steps < 0:
            raise ValueError("run cursor is beyond run plan")
        self._training_loop.validate_attempt_budget(
            capacity=RemainingAttemptCapacity(
                training_steps=remaining_steps,
                possible_cycles=plan.maximum_cycles - initial_cursor.committed_cycles,
            ),
            phi_per_cycle_maximum=self._phi_budget.available,
        )
        if (
            remaining_steps
            and self._step_adapter_publisher is not None
            and not self._current_adapter_published
        ):
            self._publish_current_adapter()
        reports: list[TrainingStepReportValue] = []
        cycles_before = initial_cursor.committed_cycles
        actions_before = initial_cursor.committed_actions

        while self._run_progress.state.completed_training_steps < target_steps:
            requested = False if stop_requested is None else stop_requested()
            # Fixed-panel inference can await its own isolated budget here.
            # No next batch or optimizer work starts while quality is unresolved.
            should_stop = await requested if isawaitable(requested) else requested
            if should_stop:
                from skillev.training.stopping import TrainingPausedError

                # Only reached before collection or after ALL durability and
                # publication phases. No provisional work is a checkpoint.
                self._training_loop.ledger.assert_fully_settled()
                step = self._training_loop.optimizer_step
                checkpoint = Path(
                    self._snapshot_store.save(
                        self._snapshot_factory.snapshot(),
                        name=f"paused-step-{step:08d}-{uuid4().hex[:12]}",
                    )
                )
                self._training_loop.set_execution_stage("paused-after-checkpoint")
                raise TrainingPausedError(step, checkpoint)
            position = self._run_progress.state.completed_training_steps + 1
            slot_kind = plan.slot_kind(position)
            from skillev.training.planning import TrainingStepExecutionContext

            next_step_cursor = self._run_progress.preview_training_step()
            batch = await self._training_loop.collect_batch()
            transaction = self._begin_step_transaction(batch)
            applied_report = self._training_loop.apply_step(
                TrainingStepExecutionContext(next_step_cursor.to_source_value())
            )
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.OPTIMIZER_APPLIED,
                policy_snapshot_after=self._training_loop.policy_snapshot_id,
            )
            self._training_loop.install_applied_projection()
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.PROJECTION_INSTALLED,
            )
            self._run_progress.commit_training_step_state(next_step_cursor)

            from .detector import NoPhaseTransitionReason
            from .reporting import phase_coverage_report

            self._training_loop.set_execution_stage("phase-detection-and-evolution")
            diagnostic = self._projections.latest_diagnostic
            if (
                diagnostic.optimizer_step != position
                or diagnostic.batch_id != applied_report.batch_id
                or diagnostic.library_version != self._library.current_version
            ):
                raise ValueError("phase detector did not receive this step's installed diagnostic")
            detection_payload: dict[str, JsonValue] = {
                "optimizer_step": position,
                "batch_id": diagnostic.batch_id,
                "library_version": diagnostic.library_version,
                "reason": NoPhaseTransitionReason.SLOT_DISABLED.value,
            }
            batch_exemplars = tuple(
                self._projections.authoring_evidence_for_diagnostics((diagnostic,)).values()
            )
            detection_payload.update(
                phase_coverage_report(
                    diagnostic,
                    self._detector.state,
                    batch_exemplars,
                    self._config,
                )
            )
            evolution_events: tuple[tuple[EventType, object], ...] = (
                (EventType.PHASE_DETECTION_RECORDED, detection_payload),
            )
            phase_resolved = False
            if phase_detection_enabled(slot_kind):
                observation, supported = preview_method_phase(
                    self._detector, self._projections, diagnostic, self._config
                )
                detection_payload["cold_start_supported"] = supported
                self._detector.commit_observation(observation)
                detection_payload.update(
                    phase_coverage_report(
                        diagnostic,
                        observation.next_state,
                        batch_exemplars,
                        self._config,
                    )
                )
                match observation:
                    case NoPhaseTransition(reason=reason):
                        evolution_events = (
                            (
                                EventType.PHASE_DETECTION_RECORDED,
                                {**detection_payload, "reason": reason.value},
                            ),
                        )
                    case PhaseTransitionDetected(event=phase, triggering_window=window):
                        if self._run_progress.state.committed_cycles >= plan.maximum_cycles:
                            raise RunPlanCycleLimitExceededError(
                                "detector produced more cycles than the frozen run plan"
                            )
                        _, evolution_events = self._execute_phase_now(phase, window)
                        evolution_events = (
                            (
                                EventType.PHASE_DETECTION_RECORDED,
                                {
                                    **detection_payload,
                                    "reason": "phase-detected",
                                    "trigger_rule": phase.trigger_rule.value,
                                },
                            ),
                            *evolution_events,
                        )
                        phase_resolved = True
                    case _:
                        from typing import assert_never

                        assert_never(observation)
            if not phase_resolved and self._step_transaction_journal is not None:
                from .authoring_journal import require_no_saved_phase

                require_no_saved_phase(self._authoring_directory(position))
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.EVOLUTION_RESOLVED,
            )
            self._training_loop.set_execution_stage("checkpoint-publication")
            prepared_durability = self._prepare_step_durability()
            prepared_events = (
                ()
                if transaction is None
                else self._emitter.prepare_many(
                    (
                        (
                            EventType.TRAINING_STEP_COMMITTED,
                            self._training_loop.pending_commit.to_value(),
                        ),
                        *evolution_events,
                    )
                )
            )
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.CHECKPOINT_PUBLISHED,
                checkpoint_name=prepared_durability.checkpoint_name,
                source_events=prepared_events,
            )
            self._training_loop.set_execution_stage("adapter-publication")
            durability = self._commit_step_adapter(prepared_durability)
            self._training_loop.set_execution_stage("source-event-publication")
            if transaction is None:
                reports.append(self._training_loop.finalize_applied_step())
                self._publish_evolution_events(evolution_events)
            else:
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.ADAPTER_COMMITTED,
                    adapter_revision=durability.adapter_revision,
                )
                reports.append(self._training_loop.finalize_applied_step_event(prepared_events[0]))
                self._publish_prepared_evolution_events(prepared_events[1:])
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.SOURCE_EVENTS_PUBLISHED,
                )
                self._advance_step_transaction(transaction, StepTransactionState.COMMITTED)
            self._training_loop.set_execution_stage("committed")

        final_cursor = self._run_progress.state
        if final_cursor.completed_training_steps != target_steps:
            raise RuntimeError("successful run ended before its requested target")
        self._final_snapshot_directory = Path(
            self._snapshot_store.save(
                self._snapshot_factory.snapshot(),
                name=f"final-step-{self._training_loop.optimizer_step:08d}",
            )
        )

        return EvolutionRunSummary(
            reports=tuple(reports),
            planned_training_steps_this_attempt=remaining_steps,
            completed_training_steps_this_attempt=len(reports),
            actions_committed_this_attempt=final_cursor.committed_actions - actions_before,
            cycles_committed_this_attempt=final_cursor.committed_cycles - cycles_before,
            cycles_committed_in_run=final_cursor.committed_cycles,
            initial_optimizer_step=initial_optimizer_step,
            final_optimizer_step=self._training_loop.optimizer_step,
            final_library_version=self._library.current_version,
            final_policy_snapshot_id=self._training_loop.policy_snapshot_id,
        )

    def _prepare_step_durability(self) -> PreparedStepDurability:
        prepared: object | None = None
        if self._step_adapter_publisher is not None:
            prepared = self._step_adapter_publisher.prepare(
                optimizer_step=self._training_loop.optimizer_step,
                policy_snapshot_id=self._training_loop.policy_snapshot_id,
            )
        with ExitStack() as rollback:
            if self._step_adapter_publisher is not None and prepared is not None:
                rollback.callback(self._step_adapter_publisher.rollback, prepared)
            snapshot = self._snapshot_factory.snapshot()
            checkpoint = Path(
                self._snapshot_store.save(
                    snapshot,
                    name=f"step-{self._training_loop.optimizer_step:08d}",
                )
            )
            # Ordinary step snapshots are a three-deep crash-recovery journal.
            # A cadence snapshot is a separately named durable training product,
            # so retention cannot discard the requested every-N-step series.
            if self._training_loop.checkpoint_due:
                self._snapshot_store.save(
                    snapshot,
                    name=f"cadence-step-{self._training_loop.optimizer_step:08d}",
                )
            self._snapshot_store.retain_recent(keep_recent=3)
            rollback.pop_all()
        return PreparedStepDurability(checkpoint.name, prepared)

    def _commit_step_adapter(
        self,
        prepared: PreparedStepDurability,
    ) -> CommittedStepDurability:
        adapter_revision = "no-external-adapter"
        if self._step_adapter_publisher is not None:
            if prepared.prepared_adapter is None:
                raise RuntimeError("step adapter preparation is absent")
            with ExitStack() as rollback:
                rollback.callback(
                    self._step_adapter_publisher.rollback,
                    prepared.prepared_adapter,
                )
                generation = self._step_adapter_publisher.commit(prepared.prepared_adapter)
                rollback.pop_all()
            self._current_adapter_published = True
            if self._step_transaction_journal is not None:
                if not isinstance(generation, AdapterGeneration):
                    raise TypeError("formal adapter publisher returned no generation")
                adapter_revision = generation.adapter_revision
            elif isinstance(generation, AdapterGeneration):
                adapter_revision = generation.adapter_revision
        return CommittedStepDurability(prepared.checkpoint_name, adapter_revision)

    def publish_restored_adapter(self) -> AdapterGeneration | None:
        """Publish a restored checkpoint; local-only applications have no server."""

        if self._step_adapter_publisher is None:
            return None
        generation = self._publish_current_adapter()
        if not isinstance(generation, AdapterGeneration):
            raise TypeError("formal restored adapter publisher returned no generation")
        return generation

    def _publish_current_adapter(self) -> object | None:
        """Publish the restored policy before the next rollout may begin."""

        if self._step_adapter_publisher is None:  # pragma: no cover - guarded caller
            return None
        generation = self._step_adapter_publisher.restore(
            optimizer_step=self._training_loop.optimizer_step,
            policy_snapshot_id=self._training_loop.policy_snapshot_id,
        )
        self._current_adapter_published = True
        return generation

    def _authoring_directory(self, optimizer_step: int) -> Path:
        if self._step_transaction_journal is None:
            raise RuntimeError("durable authoring requires a step journal")
        return self._step_transaction_journal.directory / f"authoring-step-{optimizer_step:08d}"

    def _execute_phase_now(
        self,
        phase: PhaseTransitionEvent,
        triggering_window: tuple[BatchDiagnostics, ...],
    ) -> tuple[int, tuple[tuple[EventType, object], ...]]:
        expected_batch_ids = (
            *phase.previous_window.member_batch_ids,
            *phase.current_window.member_batch_ids,
        )
        if tuple(item.batch_id for item in triggering_window) != expected_batch_ids:
            raise ValueError("detector window differs from phase evidence")

        if phase.library_version != self._library.current_version:
            raise ValueError("phase belongs to another active library")
        frozen = self._projections.freeze_for_phase(phase)
        decision = self._policy.decide_from_views(
            window_flow=WindowFlowView(
                phase_event=phase,
                diagnostics=frozen.diagnostics,
                active_skill_ids=self._library.active_skill_ids,
                applicability_by_skill={
                    skill_id: self._library.document(skill_id).applicability
                    for skill_id in self._library.active_skill_ids
                },
                task_family_universe=self._authority.allowed_task_families,
            ),
            posterior=assemble_posterior_evidence_view(
                active_skill_ids=self._library.active_skill_ids,
                cells=frozen.calibration_cells,
                event_ids_by_cell=frozen.event_ids_by_cell,
            ),
            trajectories=TrajectoryEvidenceView(
                authoring_by_edge_id=frozen.authoring_by_edge_id,
                source_by_trajectory=frozen.source_by_trajectory,
            ),
            config=self._config,
        )
        author = self._author
        if self._step_transaction_journal is not None:
            from .authoring_journal import EvolutionAuthoringJournal

            author = EvolutionAuthoringJournal(
                self._authoring_directory(self._training_loop.optimizer_step),
                decision={
                    "phase": phase.to_value(),
                    "decision": decision.to_value(),
                    "policy_snapshot_id": self._training_loop.policy_snapshot_id,
                    "evolution_config": self._config.to_value(),
                },
            ).bind(self._author, self._training_loop.ledger)
        if isinstance(decision, VerifiedNoOpEvolutionDecision):
            self._detector.close_no_op(phase, decision.reason)
            current_cursor = self._run_progress.state.to_source_value()
            return 0, (
                (
                    EventType.EVOLUTION_PHASE_OPENED,
                    EvolutionPhaseOpened(
                        phase_event=phase,
                        run_cursor_at_phase=current_cursor,
                    ).to_value(),
                ),
                (
                    EventType.EVOLUTION_NO_OP_COMMITTED,
                    EvolutionNoOpCommitted(
                        phase_event_id=phase.event_id,
                        optimizer_step=self._training_loop.optimizer_step,
                        decision_reason=decision.reason,
                        library_version=self._library.current_version,
                        run_cursor_after=current_cursor,
                    ).to_value(),
                ),
            )
        required = required_phi_budget(decision, self._config)
        self._phi_budget.require_available(required)
        cycle_ordinal = self._run_progress.state.committed_cycles + 1
        base_seed = self._training_loop.scientific_base_seed
        requests_by_proposal = tuple(
            authoring_request_for_proposal(
                proposal,
                proposal_index=index,
                phase_event=phase,
                library=self._library,
                authority=self._authority,
                base_seed=base_seed,
                cycle_ordinal=cycle_ordinal,
            )
            for index, proposal in enumerate(decision.proposals)
        )
        authoring_requests = tuple(
            request for request in requests_by_proposal if request is not None
        )
        settled_before = self._training_loop.ledger.settled
        try:
            mutation = build_evolution_mutation(
                decision,
                author=author,
                library=self._library,
                phase_event=phase,
                config=self._config,
                authority=self._authority,
                base_seed=base_seed,
                cycle_ordinal=cycle_ordinal,
            )
        except AuthoringFailedError:
            raise
        except (TypeError, ValueError) as error:
            raise EvolutionMutationFailedError(
                "complete Phi mutation construction failed"
            ) from error
        authoring_usage = self._training_loop.ledger.settled.subtract(settled_before)
        try:
            next_library_state = self._library.preview(mutation)
            next_detector = self._detector.preview_reset(
                old_version=mutation.library_version_before,
                new_version=mutation.library_version_after,
            )
            next_projection = self._projections.preview_reset(
                old_version=mutation.library_version_before,
                new_version=mutation.library_version_after,
            )
        except (TypeError, ValueError) as error:
            raise EvolutionMutationFailedError("complete Phi mutation preview failed") from error
        reset_seed = partition_reset_seed(
            base_seed=base_seed,
            cycle_ordinal=cycle_ordinal,
        )

        next_run_cursor = self._run_progress.preview_cycle(action_count=len(mutation.actions))

        try:
            z_version_after_reset = self._training_loop.reset_partition(reset_seed)
        except (RuntimeError, ValueError) as error:
            raise PartitionResetFailedError("deterministic partition reset failed") from error
        try:
            self._library.apply(next_library_state)
        except (RuntimeError, ValueError) as error:
            raise LibraryApplyFailedError("prepared skill library application failed") from error
        cycle = EvolutionCycleCommitted(
            phase_event_id=phase.event_id,
            optimizer_step=self._training_loop.optimizer_step,
            mutation=_mutation_value(mutation),
            decision_content_hash=decision.content_hash,
            proposal_content_hashes=decision.proposal_content_hashes,
            authoring_reservation_ids=tuple(
                authoring_reservation_id(request) for request in authoring_requests
            ),
            authoring_usage=AuthoringUsageValue(
                input_tokens=authoring_usage.input_tokens,
                output_tokens=authoring_usage.output_tokens,
                model_calls=authoring_usage.model_calls,
            ),
            z_reset_seed=reset_seed,
            z_version_after_reset=z_version_after_reset,
            library_version_before=mutation.library_version_before,
            library_version_after=mutation.library_version_after,
            run_cursor_after=next_run_cursor.to_source_value(),
        )
        phase_opened = EvolutionPhaseOpened(
            phase_event=phase,
            run_cursor_at_phase=self._run_progress.state.to_source_value(),
        ).to_value()
        self._detector.commit_reset(next_detector)
        self._projections.commit_reset(next_projection)
        self._run_progress.commit_cycle(next_run_cursor)
        events: tuple[tuple[EventType, object], ...] = (
            (EventType.EVOLUTION_PHASE_OPENED, phase_opened),
            (EventType.EVOLUTION_CYCLE_COMMITTED, cycle.to_value()),
        )
        if next_run_cursor.committed_cycles in self._phase_checkpoint_cycle_ordinals:
            events = (*events, self._publish_phase_checkpoint(phase, next_run_cursor))
        return len(mutation.actions), events

    def _publish_phase_checkpoint(
        self,
        phase: PhaseTransitionEvent,
        run_cursor_after: AttemptRunCursorState,
    ) -> tuple[EventType, object]:
        """Persist the exact post-cycle state for a read-only anchor child.

        This is deliberately after the committed library/Z/projection/cursor
        state is installed.  It does not run an evaluator and feeds no result
        back into the training path.
        """

        directory = Path(
            self._snapshot_store.save(
                self._snapshot_factory.snapshot(),
                name=(
                    f"phase-{run_cursor_after.committed_cycles:08d}-"
                    f"step-{self._training_loop.optimizer_step:08d}"
                ),
            )
        )
        artifact = describe_phase_checkpoint_artifact(
            directory,
            phase_event_id=phase.event_id,
            policy_snapshot_id=self._training_loop.policy_snapshot_id,
            library_version=self._library.current_version,
            optimizer_step=self._training_loop.optimizer_step,
            run_cursor_after=run_cursor_after.to_source_value(),
        )
        return (
            EventType.PHASE_CHECKPOINT_PUBLISHED,
            PhaseCheckpointPublished(
                phase_event_id=phase.event_id,
                artifact=artifact,
            ).to_value(),
        )

    def _begin_step_transaction(self, batch: object) -> StepTransactionRecord | None:
        if self._step_transaction_journal is None:
            return None
        from skillev.training.planning import CollectedTrainingBatch

        if not isinstance(batch, CollectedTrainingBatch):
            raise TypeError("formal transaction requires a collected training batch")
        return self._step_transaction_journal.begin(
            optimizer_step=batch.optimizer_step,
            batch_id=batch.batch_id,
            policy_snapshot_before=batch.policy_snapshot_id,
        )

    def _publish_evolution_events(
        self,
        events: tuple[tuple[EventType, object], ...],
    ) -> None:
        try:
            for event_type, payload in events:
                self._emitter.emit(event_type, payload)
        except EventAppendFailedError:
            raise
        except OSError as error:
            raise EventAppendFailedError("full-method source event append failed") from error

    def _publish_prepared_evolution_events(
        self,
        events: tuple[EventEnvelope, ...],
    ) -> None:
        try:
            for event in events:
                self._emitter.publish_prepared(event)
        except EventAppendFailedError:
            raise
        except OSError as error:
            raise EventAppendFailedError("full-method source event append failed") from error

    def _advance_step_transaction(
        self,
        record: StepTransactionRecord | None,
        state: StepTransactionState,
        *,
        policy_snapshot_after: str | None = None,
        checkpoint_name: str | None = None,
        adapter_revision: str | None = None,
        source_events: tuple[EventEnvelope, ...] | None = None,
    ) -> StepTransactionRecord | None:
        if record is None:
            return None
        if self._step_transaction_journal is None:  # pragma: no cover - paired state
            raise RuntimeError("step transaction record has no journal")
        return self._step_transaction_journal.advance(
            record,
            state,
            policy_snapshot_after=policy_snapshot_after,
            checkpoint_name=checkpoint_name,
            adapter_revision=adapter_revision,
            source_events=source_events,
        )


def partition_reset_seed(*, base_seed: int, cycle_ordinal: int) -> int:
    digest = stable_hash(
        {
            "base_seed": base_seed,
            "cycle_ordinal": cycle_ordinal,
            "operation": "partition-reset",
        }
    ).removeprefix("sha256:")
    return int(digest[:16], 16)


def _mutation_value(mutation: EvolutionMutation) -> EvolutionMutationValue:
    return EvolutionMutationValue(
        library_version_before=mutation.library_version_before,
        library_version_after=mutation.library_version_after,
        new_documents=tuple(document.to_value() for document in mutation.new_documents),
        active_skill_ids_after=mutation.active_skill_ids_after,
        actions=mutation.actions,
    )


__all__ = [
    "EvolutionDecisionPolicy",
    "EvolutionLoop",
    "EvolutionPhaseDetector",
    "EvolutionProjectionView",
    "EvolutionRunSummary",
    "EvolutionSnapshotFactory",
    "EvolutionTrainingLoop",
    "PhiBudgetAuthority",
    "RunPlanCycleLimitExceededError",
    "partition_reset_seed",
    "phase_detection_enabled",
]
