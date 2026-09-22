"""Closed event reducer for the independent no-Bayesian source stream."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias

from skillev.contracts import RunCursorValue
from skillev.evolution import partition_reset_seed
from skillev.experiments.arm_events import (
    ArmEventEnvelope,
    ArmEventType,
    read_arm_event_history,
)
from skillev.experiments.arms.no_bayesian_contracts import (
    FlowOnlyCycleResult,
    FlowOnlyLibraryInitialized,
    FlowOnlyPhaseCheckpointPublished,
    FlowOnlyPhaseOpened,
    FlowOnlyTrainingStepCommit,
)
from skillev.experiments.protocol import AblationArm
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptSourceLogKind,
    PublishedSuccessfulAttemptBundle,
    RunSlotKind,
    SkillDocument,
    SkillLibraryState,
)

from .source_reducer import AuditEvidenceMismatchError, AuditSourceOrderError

if TYPE_CHECKING:
    from skillev.experiments.attempt_identity import PublishedAttemptIdentity


@dataclass(frozen=True, slots=True)
class _Await:
    pass


@dataclass(frozen=True, slots=True)
class _Segment:
    library_state: SkillLibraryState
    cursor_before: RunCursorValue
    training_steps: tuple[FlowOnlyTrainingStepCommit, ...]
    required_z_reset_version: str | None = None


@dataclass(frozen=True, slots=True)
class _Phase:
    library_state: SkillLibraryState
    cursor_before: RunCursorValue
    training_steps: tuple[FlowOnlyTrainingStepCommit, ...]
    phase: FlowOnlyPhaseOpened


FlowOnlyAuditState: TypeAlias = _Await | _Segment | _Phase


@dataclass(frozen=True, slots=True)
class CommittedFlowOnlySegment:
    library_state: SkillLibraryState
    cursor_before: RunCursorValue
    cursor_after: RunCursorValue
    training_steps: tuple[FlowOnlyTrainingStepCommit, ...]
    phase: FlowOnlyPhaseOpened | None
    cycle: FlowOnlyCycleResult | None
    phase_checkpoint: FlowOnlyPhaseCheckpointPublished | None

    @property
    def library_version(self) -> str:
        return self.library_state.current_version


FlowOnlySourceEvent: TypeAlias = (
    FlowOnlyLibraryInitialized
    | FlowOnlyTrainingStepCommit
    | FlowOnlyPhaseOpened
    | FlowOnlyCycleResult
    | FlowOnlyPhaseCheckpointPublished
)


class FlowOnlyAuditSourceReducer:
    def __init__(self, identity: PublishedAttemptIdentity) -> None:
        if identity.builder_kind is not AttemptBuilderKind.NO_BAYESIAN:
            raise TypeError("flow-only reducer requires no-Bayesian identity")
        self._identity = identity
        self._plan = identity.run_plan
        self._state: FlowOnlyAuditState = _Await()
        self._segments: list[CommittedFlowOnlySegment] = []
        self._cursor: RunCursorValue | None = None
        self._last_optimizer_step: int | None = None

    def consume_published_bundle(
        self, bundle: PublishedSuccessfulAttemptBundle
    ) -> tuple[CommittedFlowOnlySegment, ...]:
        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        if exact.builder_kind is not AttemptBuilderKind.NO_BAYESIAN:
            raise TypeError("flow-only reducer rejects non-flow-only bundle")
        descriptor = exact.source_log(AttemptSourceLogKind.FLOW_ONLY)
        for envelope in read_arm_event_history(exact.source_log_path(descriptor)):
            if envelope.arm is not AblationArm.SKILLFLOW_DISABLED:
                raise AuditSourceOrderError("arm event has wrong arm identity")
            if envelope.run_id != exact.run_id or envelope.attempt_id != exact.attempt_id:
                raise AuditSourceOrderError("arm event identity differs from publication")
            self._consume(flow_only_source_from_envelope(envelope))
        self._finish()
        return tuple(self._segments)

    def _consume(self, event: FlowOnlySourceEvent) -> None:
        match self._state, event:
            case _Await(), FlowOnlyLibraryInitialized() as initialized:
                if initialized.method_identity_hash != self._identity.method_identity_hash:
                    raise AuditEvidenceMismatchError("flow-only library identity differs")
                if initialized.run_cursor != self._identity.initial_run_cursor.to_source_value():
                    raise AuditSourceOrderError("flow-only library cursor differs")
                if initialized.initial_optimizer_step != self._identity.initial_optimizer_step:
                    raise AuditSourceOrderError("flow-only optimizer origin differs")
                state = _library_from_initialized(initialized)
                if state.current_version != self._identity.initial_library_version:
                    raise AuditEvidenceMismatchError("flow-only seed library differs")
                if state.state_hash != self._identity.initial_skill_library_state_hash:
                    raise AuditEvidenceMismatchError("flow-only seed library state differs")
                self._cursor = initialized.run_cursor
                self._last_optimizer_step = initialized.initial_optimizer_step
                self._state = _Segment(state, initialized.run_cursor, ())
            case _Segment() as state, FlowOnlyTrainingStepCommit() as step:
                if self._cursor is None or self._last_optimizer_step is None:
                    raise AuditSourceOrderError("flow-only training precedes initialization")
                expected = RunCursorValue(
                    run_plan_hash=self._cursor.run_plan_hash,
                    completed_training_steps=self._cursor.completed_training_steps + 1,
                    committed_cycles=self._cursor.committed_cycles,
                    committed_actions=self._cursor.committed_actions,
                )
                if (
                    step.run_cursor_after != expected
                    or step.optimizer_step != self._last_optimizer_step + 1
                ):
                    raise AuditSourceOrderError(
                        "flow-only training cursor or optimizer step differs"
                    )
                if step.library_version != state.library_state.current_version:
                    raise AuditSourceOrderError("flow-only training uses another library")
                if state.required_z_reset_version is not None and (
                    step.report.z_version.rpartition("@")[0]
                    != state.required_z_reset_version.rpartition("@")[0]
                ):
                    raise AuditEvidenceMismatchError(
                        "flow-only continuation does not descend from reset Z"
                    )
                if expected.completed_training_steps > self._plan.total_training_steps:
                    raise AuditSourceOrderError("flow-only training exceeds run plan")
                self._cursor, self._last_optimizer_step = expected, step.optimizer_step
                self._state = _Segment(
                    state.library_state, state.cursor_before, (*state.training_steps, step)
                )
            case _Segment() as state, FlowOnlyPhaseOpened() as opened:
                if self._cursor is None or not state.training_steps:
                    raise AuditSourceOrderError("flow-only phase has no preceding training")
                if opened.run_cursor_at_phase != self._cursor:
                    raise AuditSourceOrderError("flow-only phase cursor differs")
                if (
                    self._plan.slot_kind(self._cursor.completed_training_steps)
                    is not RunSlotKind.PHASE_SEARCH
                ):
                    raise AuditSourceOrderError("flow-only phase opened in closure slot")
                if opened.phase_event.library_version != state.library_state.current_version:
                    raise AuditSourceOrderError("flow-only phase uses another library")
                self._state = _Phase(
                    state.library_state, state.cursor_before, state.training_steps, opened
                )
            case _Phase() as state, FlowOnlyCycleResult() as cycle:
                if self._cursor is None or self._last_optimizer_step is None:
                    raise AuditSourceOrderError("flow-only cycle precedes initialization")
                if (
                    cycle.phase_event_id != state.phase.phase_event.event_id
                    or cycle.optimizer_step != self._last_optimizer_step
                ):
                    raise AuditSourceOrderError("flow-only cycle differs from phase")
                if cycle.library_version_before != state.library_state.current_version:
                    raise AuditSourceOrderError("flow-only cycle starts from another library")
                expected = RunCursorValue(
                    run_plan_hash=self._cursor.run_plan_hash,
                    completed_training_steps=self._cursor.completed_training_steps,
                    committed_cycles=self._cursor.committed_cycles + 1,
                    committed_actions=self._cursor.committed_actions + len(cycle.actions),
                )
                if (
                    cycle.run_cursor_after != expected
                    or expected.committed_cycles > self._plan.maximum_cycles
                ):
                    raise AuditSourceOrderError("flow-only cycle cursor differs")
                if cycle.z_reset_seed != partition_reset_seed(
                    base_seed=self._identity.application_config.trainer.rollout.base_seed,
                    cycle_ordinal=expected.committed_cycles,
                ):
                    raise AuditEvidenceMismatchError("flow-only reset seed differs")
                expected_reset_version = f"z-reset-{cycle.z_reset_seed:016x}@0"
                if cycle.z_version_after_reset != expected_reset_version:
                    raise AuditEvidenceMismatchError("flow-only reset Z version differs")
                rebuilt = apply_flow_only_mutation(state.library_state, cycle)
                self._segments.append(
                    CommittedFlowOnlySegment(
                        state.library_state,
                        state.cursor_before,
                        expected,
                        state.training_steps,
                        state.phase,
                        cycle,
                        None,
                    )
                )
                self._cursor = expected
                self._state = _Segment(
                    rebuilt,
                    expected,
                    (),
                    required_z_reset_version=expected_reset_version,
                )
            case _Segment() as state, FlowOnlyPhaseCheckpointPublished() as published:
                self._consume_phase_checkpoint(state, published)
            case _:
                raise AuditSourceOrderError("flow-only source event is invalid in current state")

    def _consume_phase_checkpoint(
        self,
        state: _Segment,
        published: FlowOnlyPhaseCheckpointPublished,
    ) -> None:
        """Validate one declared post-cycle anchor without a posterior surrogate."""

        if self._cursor is None or state.training_steps or not self._segments:
            raise AuditSourceOrderError("flow-only checkpoint is not immediately post-cycle")
        previous = self._segments[-1]
        phase, cycle = previous.phase, previous.cycle
        if phase is None or cycle is None or previous.phase_checkpoint is not None:
            raise AuditSourceOrderError("flow-only checkpoint has no unique preceding phase")
        if (
            published.phase_event_id != phase.phase_event.event_id
            or cycle.run_cursor_after != self._cursor
            or cycle.run_cursor_after != state.cursor_before
            or cycle.library_version_after != state.library_state.current_version
        ):
            raise AuditEvidenceMismatchError("flow-only checkpoint differs from its phase cycle")
        artifact = published.artifact
        if (
            artifact.optimizer_step != cycle.optimizer_step
            or artifact.library_version != cycle.library_version_after
            or artifact.run_cursor_after != cycle.run_cursor_after
        ):
            raise AuditEvidenceMismatchError("flow-only checkpoint artifact differs from cycle")
        if (
            artifact.run_cursor_after.committed_cycles
            not in self._identity.phase_checkpoint_cycle_ordinals
        ):
            raise AuditSourceOrderError("flow-only checkpoint cycle is not preregistered")
        self._segments[-1] = replace(previous, phase_checkpoint=published)

    def _finish(self) -> None:
        if (
            self._cursor is None
            or self._cursor.completed_training_steps != self._plan.total_training_steps
        ):
            raise AuditSourceOrderError("flow-only published success did not complete run plan")
        match self._state:
            case _Segment() as state:
                if state.required_z_reset_version is not None and not state.training_steps:
                    raise AuditSourceOrderError("flow-only cycle has no continuation training step")
                self._segments.append(
                    CommittedFlowOnlySegment(
                        state.library_state,
                        state.cursor_before,
                        self._cursor,
                        state.training_steps,
                        None,
                        None,
                        None,
                    )
                )
            case _Phase():
                raise AuditSourceOrderError("flow-only attempt ends with unclosed phase")
            case _Await():
                raise AuditSourceOrderError("flow-only attempt has no library initialization")


def flow_only_source_from_envelope(envelope: ArmEventEnvelope) -> FlowOnlySourceEvent:
    match envelope.event_type:
        case ArmEventType.FLOW_ONLY_LIBRARY_INITIALIZED:
            return FlowOnlyLibraryInitialized.from_value(envelope.payload)
        case ArmEventType.FLOW_ONLY_TRAINING_STEP_COMMITTED:
            return FlowOnlyTrainingStepCommit.from_value(envelope.payload)
        case ArmEventType.FLOW_ONLY_PHASE_OPENED:
            return FlowOnlyPhaseOpened.from_value(envelope.payload)
        case ArmEventType.FLOW_ONLY_CYCLE_COMMITTED:
            return FlowOnlyCycleResult.from_value(envelope.payload)
        case ArmEventType.FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED:
            return FlowOnlyPhaseCheckpointPublished.from_value(envelope.payload)
        case _:
            raise AuditSourceOrderError("unsupported flow-only arm event")


def _library_from_initialized(initialized: FlowOnlyLibraryInitialized) -> SkillLibraryState:
    documents = tuple(SkillDocument.from_value(item) for item in initialized.documents)
    mapping = {item.manifest.skill_id: item for item in documents}
    if len(mapping) != len(documents):
        raise AuditEvidenceMismatchError("flow-only initialization repeats document")
    try:
        return SkillLibraryState(mapping, initialized.active_skill_ids, initialized.library_version)
    except ValueError as error:
        raise AuditEvidenceMismatchError("flow-only initialization library hash differs") from error


def apply_flow_only_mutation(
    state: SkillLibraryState, cycle: FlowOnlyCycleResult
) -> SkillLibraryState:
    documents = dict(state.documents)
    created = tuple(SkillDocument.from_value(item) for item in cycle.new_documents)
    produced = {skill for action in cycle.actions for skill in action.produced_skill_ids}
    if produced != {item.manifest.skill_id for item in created}:
        raise AuditEvidenceMismatchError("flow-only action products differ from documents")
    active = set(state.active_skill_ids)
    for action in cycle.actions:
        if action.target_skill_ids and action.target_skill_ids[0] not in active:
            raise AuditEvidenceMismatchError("flow-only action targets inactive skill")
        active.difference_update(action.target_skill_ids)
        active.update(action.produced_skill_ids)
    for document in created:
        if document.manifest.skill_id in documents:
            raise AuditEvidenceMismatchError("flow-only action reuses skill ID")
        documents[document.manifest.skill_id] = document
    expected_active = tuple(sorted(active))
    if cycle.active_skill_ids_after != expected_active:
        raise AuditEvidenceMismatchError("flow-only mutation active skill set differs")
    try:
        return SkillLibraryState(documents, expected_active, cycle.library_version_after)
    except ValueError as error:
        raise AuditEvidenceMismatchError("flow-only mutation library hash differs") from error


__all__ = [
    "CommittedFlowOnlySegment",
    "FlowOnlyAuditSourceReducer",
    "apply_flow_only_mutation",
    "flow_only_source_from_envelope",
]
