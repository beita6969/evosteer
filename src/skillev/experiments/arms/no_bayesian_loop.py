"""Independent same-call driver for the no-Bayesian experiment arm."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skillev.contracts import (
    AuthoringUsageValue,
    PhaseTransitionEvent,
    RunCursorValue,
    TrainingStepReportValue,
)
from skillev.diagnostics import BatchDiagnostics
from skillev.evolution import (
    NoPhaseTransition,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
    PhiBudgetAuthority,
    SkillAuthor,
    SkillAuthoringAuthority,
    TrajectoryEvidenceView,
    WindowFlowView,
    authoring_reservation_id,
    partition_reset_seed,
    phase_detection_enabled,
)
from skillev.evolution.authoring import AuthoringFailedError
from skillev.evolution.config import EvolutionConfig
from skillev.experiments.arm_events import ArmEventType, LiveArmEventLog
from skillev.runtime import (
    AdapterGeneration,
    EventEnvelope,
    EventType,
    FlowOnlyAttemptSummary,
    RuntimeEventEmitter,
    RuntimeSnapshotIdentity,
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
)
from skillev.training import TrainingStepExecutionContext

from .no_bayesian_calibration import (
    FlowOnlyEvolutionPolicy,
    FlowOnlyProjectionPipeline,
)
from .no_bayesian_contracts import (
    FlowOnlyCycleResult,
    FlowOnlyPhaseCheckpointPublished,
    FlowOnlyPhaseOpened,
)
from .no_bayesian_execution import (
    build_flow_only_mutation,
    flow_only_authoring_request_for_proposal,
    required_flow_only_phi_budget,
)
from .no_bayesian_training import FlowOnlyTrainingLoop


class FlowOnlyResultSink(Protocol):
    def append_phase(self, event: PhaseTransitionEvent, cursor: RunCursorValue) -> None: ...

    def append_cycle(self, result: FlowOnlyCycleResult) -> None: ...

    def append_phase_checkpoint(self, result: FlowOnlyPhaseCheckpointPublished) -> None: ...


@dataclass(frozen=True, slots=True)
class ArmFlowOnlyResultSink:
    log: LiveArmEventLog

    def append_phase(self, event: PhaseTransitionEvent, cursor: RunCursorValue) -> None:
        self.log.append(
            ArmEventType.FLOW_ONLY_PHASE_OPENED,
            FlowOnlyPhaseOpened(event, cursor).to_value(),
        )

    def append_cycle(self, result: FlowOnlyCycleResult) -> None:
        self.log.append(ArmEventType.FLOW_ONLY_CYCLE_COMMITTED, result.to_value())

    def append_phase_checkpoint(self, result: FlowOnlyPhaseCheckpointPublished) -> None:
        self.log.append(ArmEventType.FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED, result.to_value())


FlowOnlyRunSummary = FlowOnlyAttemptSummary


class FlowOnlyEvolutionLoop:
    """Train one step and resolve every verified flow-only phase in that call."""

    def __init__(
        self,
        *,
        training_loop: FlowOnlyTrainingLoop,
        projections: FlowOnlyProjectionPipeline,
        detector: PhaseTransitionDetector,
        library: SkillLibrary,
        author: SkillAuthor,
        policy: FlowOnlyEvolutionPolicy,
        authority: SkillAuthoringAuthority,
        result_sink: FlowOnlyResultSink | None,
        phi_budget: PhiBudgetAuthority,
        snapshot_store: RuntimeSnapshotStore,
        config: EvolutionConfig,
        run_progress: AttemptRunProgress,
        snapshot_identity: RuntimeSnapshotIdentity,
        emitter: RuntimeEventEmitter,
        phase_checkpoint_cycle_ordinals: tuple[int, ...] = (),
        step_adapter_publisher: StepAdapterPublisher | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
    ) -> None:
        self._training_loop = training_loop
        self._projections = projections
        self._detector = detector
        self._library = library
        self._author = author
        self._policy = policy
        self._authority = authority
        self._result_sink = result_sink
        self._phi_budget = phi_budget
        self._snapshot_store = snapshot_store
        self._config = config
        self._run_progress = run_progress
        self._snapshot_identity = snapshot_identity
        self._emitter = emitter
        if (
            not isinstance(phase_checkpoint_cycle_ordinals, tuple)
            or any(type(item) is not int or item < 1 for item in phase_checkpoint_cycle_ordinals)
            or tuple(sorted(set(phase_checkpoint_cycle_ordinals)))
            != phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("flow-only phase checkpoint cycles must be sorted unique positives")
        self._phase_checkpoint_cycle_ordinals = phase_checkpoint_cycle_ordinals
        self._step_adapter_publisher = step_adapter_publisher
        self._step_transaction_journal = step_transaction_journal
        self._current_adapter_published = False
        self._final_snapshot_directory: Path | None = None

    @property
    def step_adapter_publisher(self) -> StepAdapterPublisher | None:
        return self._step_adapter_publisher

    @property
    def author(self) -> SkillAuthor:
        return self._author

    @property
    def step_transaction_journal(self) -> StepTransactionJournal | None:
        return self._step_transaction_journal

    @property
    def final_training_snapshot_directory(self) -> Path:
        if self._final_snapshot_directory is None:
            raise RuntimeError("final flow-only training snapshot is unavailable before completion")
        return self._final_snapshot_directory

    async def run(
        self,
        plan: ExactAttemptRunPlan,
        *,
        maximum_steps_this_attempt: int | None = None,
    ) -> FlowOnlyRunSummary:
        if not isinstance(plan, ExactAttemptRunPlan):
            raise TypeError("FlowOnlyEvolutionLoop.run requires ExactAttemptRunPlan")
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
        self._training_loop.validate_attempt_budget(
            capacity=RemainingAttemptCapacity(
                training_steps=remaining_steps,
                possible_cycles=plan.maximum_cycles - initial_cursor.committed_cycles,
            ),
            phi_per_cycle_maximum=self._phi_budget.available,
        )
        if remaining_steps and not self._current_adapter_published:
            self.publish_restored_adapter()
        reports: list[TrainingStepReportValue] = []
        cycles_before = initial_cursor.committed_cycles
        actions_before = initial_cursor.committed_actions
        while self._run_progress.state.completed_training_steps < target_steps:
            position = self._run_progress.state.completed_training_steps + 1
            slot_kind = plan.slot_kind(position)
            next_step_cursor = self._run_progress.preview_training_step()
            batch = await self._training_loop.collect_batch()
            transaction = self._begin_step_transaction(batch)
            report = self._training_loop.apply_step(
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
            reports.append(report)
            self._run_progress.commit_training_step_state(next_step_cursor)
            evolution_events: tuple[tuple[EventType, object], ...] = ()
            if phase_detection_enabled(slot_kind):
                observation = self._detector.observe(self._projections.latest_diagnostic)
                match observation:
                    case NoPhaseTransition():
                        pass
                    case PhaseTransitionDetected(event=phase, triggering_window=window):
                        if self._run_progress.state.committed_cycles >= plan.maximum_cycles:
                            raise RuntimeError(
                                "detector produced more cycles than the frozen run plan"
                            )
                        _, evolution_events = self._execute_phase_now(phase, window)
                    case _ as unreachable:
                        from typing import assert_never

                        assert_never(unreachable)
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.EVOLUTION_RESOLVED,
            )
            if transaction is None:
                self._training_loop.finalize_applied_step()
                self._publish_evolution_events(evolution_events)
                self._save_checkpoint_if_due()
                self._publish_step_adapter()
            else:
                prepared_events = self._emitter.prepare_many(
                    (
                        (
                            EventType.FLOW_ONLY_TRAINING_STEP_COMMITTED,
                            self._training_loop.pending_commit.to_value(),
                        ),
                        *evolution_events,
                    )
                )
                prepared_adapter = self._prepare_step_adapter()
                checkpoint = Path(
                    self._snapshot_store.save(
                        self._training_loop.snapshot(
                            detector_state=self._detector.state,
                            run_progress=self._run_progress,
                            snapshot_identity=self._snapshot_identity,
                        ),
                        name=f"flow-only-step-{self._training_loop.optimizer_step:08d}",
                    )
                )
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.CHECKPOINT_PUBLISHED,
                    checkpoint_name=checkpoint.name,
                    source_events=prepared_events,
                )
                generation = self._commit_prepared_step_adapter(prepared_adapter)
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.ADAPTER_COMMITTED,
                    adapter_revision=generation.adapter_revision,
                )
                self._training_loop.finalize_applied_step_event(prepared_events[0])
                self._publish_prepared_evolution_events(prepared_events[1:])
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.SOURCE_EVENTS_PUBLISHED,
                )
                self._advance_step_transaction(
                    transaction,
                    StepTransactionState.COMMITTED,
                )
        final_cursor = self._run_progress.state
        if final_cursor.completed_training_steps != target_steps:
            raise RuntimeError("successful flow-only run ended before its requested target")
        self._final_snapshot_directory = Path(
            self._snapshot_store.save(
                self._training_loop.snapshot(
                    detector_state=self._detector.state,
                    run_progress=self._run_progress,
                    snapshot_identity=self._snapshot_identity,
                ),
                name=f"flow-only-final-step-{self._training_loop.optimizer_step:08d}",
            )
        )
        return FlowOnlyRunSummary(
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

    def publish_restored_adapter(self) -> AdapterGeneration | None:
        """Publish the current forward policy before any formal rollout."""

        if self._step_adapter_publisher is None:
            return None
        generation = self._step_adapter_publisher.restore(
            optimizer_step=self._training_loop.optimizer_step,
            policy_snapshot_id=self._training_loop.policy_snapshot_id,
        )
        if not isinstance(generation, AdapterGeneration):
            raise TypeError("flow-only adapter publisher returned no generation")
        self._current_adapter_published = True
        return generation

    def _publish_step_adapter(self) -> None:
        if self._step_adapter_publisher is not None:
            self.publish_restored_adapter()

    def _prepare_step_adapter(self) -> object:
        if self._step_adapter_publisher is None:
            raise RuntimeError("formal flow-only step has no adapter publisher")
        return self._step_adapter_publisher.prepare(
            optimizer_step=self._training_loop.optimizer_step,
            policy_snapshot_id=self._training_loop.policy_snapshot_id,
        )

    def _commit_prepared_step_adapter(self, prepared: object) -> AdapterGeneration:
        if self._step_adapter_publisher is None:
            raise RuntimeError("formal flow-only step has no adapter publisher")
        with ExitStack() as rollback:
            rollback.callback(self._step_adapter_publisher.rollback, prepared)
            generation = self._step_adapter_publisher.commit(prepared)
            rollback.pop_all()
        if not isinstance(generation, AdapterGeneration):
            raise TypeError("flow-only adapter publisher returned no generation")
        self._current_adapter_published = True
        return generation

    def _begin_step_transaction(self, batch: object) -> StepTransactionRecord | None:
        if self._step_transaction_journal is None:
            return None
        from skillev.training import CollectedTrainingBatch

        if not isinstance(batch, CollectedTrainingBatch):
            raise TypeError("flow-only transaction requires a collected batch")
        return self._step_transaction_journal.begin(
            optimizer_step=batch.optimizer_step,
            batch_id=batch.batch_id,
            policy_snapshot_before=batch.policy_snapshot_id,
        )

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
        if self._step_transaction_journal is None:
            raise RuntimeError("flow-only transaction record has no journal")
        return self._step_transaction_journal.advance(
            record,
            state,
            policy_snapshot_after=policy_snapshot_after,
            checkpoint_name=checkpoint_name,
            adapter_revision=adapter_revision,
            source_events=source_events,
        )

    def _save_checkpoint_if_due(self) -> None:
        if self._training_loop.checkpoint_due:
            self._snapshot_store.save(
                self._training_loop.snapshot(
                    detector_state=self._detector.state,
                    run_progress=self._run_progress,
                    snapshot_identity=self._snapshot_identity,
                ),
                name=f"flow-only-step-{self._training_loop.optimizer_step:08d}",
            )

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
            raise ValueError("flow-only detector window differs from phase evidence")
        diagnostics = self._projections.diagnostics_for_batch_ids(expected_batch_ids)
        decision = self._policy.decide_from_views(
            window_flow=WindowFlowView(
                phase_event=phase,
                diagnostics=diagnostics,
                active_skill_ids=self._library.active_skill_ids,
                applicability_by_skill={
                    skill_id: self._library.document(skill_id).applicability
                    for skill_id in self._library.active_skill_ids
                },
                task_family_universe=self._authority.allowed_task_families,
            ),
            trajectories=TrajectoryEvidenceView(
                authoring_by_edge_id=self._projections.authoring_evidence_for_batch_ids(
                    expected_batch_ids
                )
            ),
            config=self._config,
        )
        required = required_flow_only_phi_budget(
            decision,
            max_authoring_completion_tokens=(self._config.max_authoring_completion_tokens),
            max_authoring_prompt_tokens=self._config.max_authoring_prompt_tokens,
        )
        self._phi_budget.require_available(required)
        cycle_ordinal = self._run_progress.state.committed_cycles + 1
        base_seed = self._training_loop.scientific_base_seed
        authoring_requests = tuple(
            flow_only_authoring_request_for_proposal(
                proposal,
                phase_event=phase,
                library=self._library,
                authority=self._authority,
                proposal_index=index,
                base_seed=base_seed,
                cycle_ordinal=cycle_ordinal,
            )
            for index, proposal in enumerate(decision.proposals)
        )
        settled_before = self._training_loop.ledger.settled
        try:
            mutation = build_flow_only_mutation(
                decision,
                author=self._author,
                authority=self._authority,
                library=self._library,
                phase_event=phase,
                optimizer_step=self._training_loop.optimizer_step,
                max_skill_instruction_tokens_per_draft=(
                    self._config.max_skill_instruction_tokens_per_draft
                ),
                base_seed=base_seed,
                cycle_ordinal=cycle_ordinal,
            )
        except AuthoringFailedError:
            raise
        except (TypeError, ValueError) as error:
            raise EvolutionMutationFailedError(
                "flow-only Phi mutation construction failed"
            ) from error
        authoring_usage = self._training_loop.ledger.settled.subtract(settled_before)
        try:
            next_library = self._library.preview_contents(
                library_version_before=mutation.library_version_before,
                library_version_after=mutation.library_version_after,
                new_documents=mutation.new_documents,
                active_skill_ids_after=mutation.active_skill_ids_after,
            )
            next_detector = self._detector.preview_reset(
                old_version=mutation.library_version_before,
                new_version=mutation.library_version_after,
            )
            next_projection = self._projections.preview_reset(
                old_version=mutation.library_version_before,
                new_version=mutation.library_version_after,
            )
        except (TypeError, ValueError) as error:
            raise EvolutionMutationFailedError("flow-only Phi mutation preview failed") from error

        reset_seed = partition_reset_seed(
            base_seed=base_seed,
            cycle_ordinal=cycle_ordinal,
        )
        next_run_cursor = self._run_progress.preview_cycle(action_count=len(mutation.actions))
        try:
            z_version = self._training_loop.reset_partition(reset_seed)
        except (RuntimeError, ValueError) as error:
            raise PartitionResetFailedError("flow-only partition reset failed") from error
        try:
            self._library.apply(next_library)
        except (RuntimeError, ValueError) as error:
            raise LibraryApplyFailedError(
                "flow-only prepared skill library application failed"
            ) from error
        result = FlowOnlyCycleResult(
            phase_event_id=mutation.phase_event_id,
            optimizer_step=mutation.optimizer_step,
            library_version_before=mutation.library_version_before,
            library_version_after=mutation.library_version_after,
            new_documents=tuple(item.to_value() for item in mutation.new_documents),
            active_skill_ids_after=mutation.active_skill_ids_after,
            actions=mutation.actions,
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
            z_version_after_reset=z_version,
            run_cursor_after=next_run_cursor.to_source_value(),
        )
        phase_opened = FlowOnlyPhaseOpened(
            phase,
            self._run_progress.state.to_source_value(),
        )
        self._detector.commit_reset(next_detector)
        self._projections.commit_reset(next_projection)
        self._run_progress.commit_cycle(next_run_cursor)
        events: tuple[tuple[EventType, object], ...] = (
            (EventType.FLOW_ONLY_PHASE_OPENED, phase_opened.to_value()),
            (EventType.FLOW_ONLY_CYCLE_COMMITTED, result.to_value()),
        )
        if next_run_cursor.committed_cycles in self._phase_checkpoint_cycle_ordinals:
            events = (*events, self._prepare_phase_checkpoint(phase, next_run_cursor))
        return len(result.actions), events

    def _prepare_phase_checkpoint(
        self,
        phase: PhaseTransitionEvent,
        run_cursor_after: AttemptRunCursorState,
    ) -> tuple[EventType, object]:
        """Persist a flow-only phase checkpoint for transactional publication.

        The saved runtime state is the native ``FlowOnlyRuntimeExecutionState``;
        no full-method posterior is created or substituted for this arm.
        """

        directory = Path(
            self._snapshot_store.save(
                self._training_loop.snapshot(
                    detector_state=self._detector.state,
                    run_progress=self._run_progress,
                    snapshot_identity=self._snapshot_identity,
                ),
                name=(
                    f"flow-only-phase-{run_cursor_after.committed_cycles:08d}-"
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
            EventType.FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED,
            FlowOnlyPhaseCheckpointPublished(
                phase_event_id=phase.event_id, artifact=artifact
            ).to_value(),
        )

    def _publish_evolution_events(
        self,
        events: tuple[tuple[EventType, object], ...],
    ) -> None:
        for event_type, payload in events:
            self._emitter.emit(event_type, payload)
            self._append_legacy_evolution_event(event_type, payload)

    def _publish_prepared_evolution_events(
        self,
        events: tuple[EventEnvelope, ...],
    ) -> None:
        for event in events:
            self._emitter.publish_prepared(event)
            self._append_legacy_evolution_event(event.event_type, event.payload)

    def _append_legacy_evolution_event(
        self,
        event_type: EventType,
        payload: object,
    ) -> None:
        if self._result_sink is None:
            return
        try:
            if event_type is EventType.FLOW_ONLY_PHASE_OPENED:
                opened = FlowOnlyPhaseOpened.from_value(payload)
                self._result_sink.append_phase(
                    opened.phase_event,
                    opened.run_cursor_at_phase,
                )
            elif event_type is EventType.FLOW_ONLY_CYCLE_COMMITTED:
                self._result_sink.append_cycle(FlowOnlyCycleResult.from_value(payload))
            elif event_type is EventType.FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED:
                self._result_sink.append_phase_checkpoint(
                    FlowOnlyPhaseCheckpointPublished.from_value(payload)
                )
            else:
                raise ValueError("unsupported flow-only evolution event")
        except EventAppendFailedError:
            raise
        except OSError as error:
            raise EventAppendFailedError("flow-only source event append failed") from error


__all__ = [
    "ArmFlowOnlyResultSink",
    "FlowOnlyEvolutionLoop",
    "FlowOnlyResultSink",
    "FlowOnlyRunSummary",
]
