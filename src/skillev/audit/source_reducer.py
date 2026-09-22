"""Closed full-shaped source reducer driven by a published run identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from skillev.contracts import (
    EvolutionCycleCommitted,
    EvolutionNoOpCommitted,
    EvolutionPhaseOpened,
    LibraryInitialized,
    RunCursorValue,
    TrainingStepCommit,
)
from skillev.evolution import partition_reset_seed
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptSourceLogKind,
    EventEnvelope,
    EventType,
    PublishedSuccessfulAttemptBundle,
    RunSlotKind,
    SkillDocument,
    SkillLibraryState,
)

if TYPE_CHECKING:
    from skillev.experiments.attempt_identity import PublishedAttemptIdentity


class AuditSourceOrderError(ValueError):
    """The published source sequence is not an exact run-plan state machine."""


class AuditEvidenceMismatchError(ValueError):
    """A published source cannot be reproduced from its preceding sources."""


@dataclass(frozen=True, slots=True)
class AwaitLibraryInitialization:
    kind: str = "await-library"


@dataclass(frozen=True, slots=True)
class InLibrarySegment:
    library_state: SkillLibraryState
    cursor_before: RunCursorValue
    training_steps: tuple[TrainingStepCommit, ...]
    kind: str = "in-library-segment"


@dataclass(frozen=True, slots=True)
class PhaseOpened:
    library_state: SkillLibraryState
    cursor_before: RunCursorValue
    training_steps: tuple[TrainingStepCommit, ...]
    phase: EvolutionPhaseOpened
    kind: str = "phase-opened"


AuditReducerState: TypeAlias = AwaitLibraryInitialization | InLibrarySegment | PhaseOpened


@dataclass(frozen=True, slots=True)
class CommittedLibrarySegment:
    library_state: SkillLibraryState
    cursor_before: RunCursorValue
    cursor_after: RunCursorValue
    training_steps: tuple[TrainingStepCommit, ...]
    phase: EvolutionPhaseOpened | None
    cycle: EvolutionCycleCommitted | None
    no_op: EvolutionNoOpCommitted | None = None

    @property
    def library_version(self) -> str:
        return self.library_state.current_version


ScientificSourceEvent: TypeAlias = (
    LibraryInitialized
    | TrainingStepCommit
    | EvolutionPhaseOpened
    | EvolutionCycleCommitted
    | EvolutionNoOpCommitted
)


class AuditSourceReducer:
    """Reduce only hash-covered full-method sources under one public identity."""

    def __init__(self, identity: PublishedAttemptIdentity) -> None:
        if identity.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
            raise TypeError("full-shaped reducer rejects the no-Bayesian identity")
        self._identity = identity
        self._plan = identity.run_plan
        self._state: AuditReducerState = AwaitLibraryInitialization()
        self._segments: list[CommittedLibrarySegment] = []
        self._last_optimizer_step: int | None = None
        self._cursor: RunCursorValue | None = None

    def consume_published_bundle(
        self,
        bundle: PublishedSuccessfulAttemptBundle,
    ) -> tuple[CommittedLibrarySegment, ...]:
        if not isinstance(bundle, PublishedSuccessfulAttemptBundle):
            raise TypeError("audit requires a published successful attempt bundle")
        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        if exact.builder_kind is not self._identity.builder_kind:
            raise TypeError("bundle builder differs from the audit identity")
        descriptor = exact.source_log(AttemptSourceLogKind.FULL_METHOD)
        for line in exact.source_log_path(descriptor).read_text(encoding="utf-8").splitlines():
            envelope = EventEnvelope.from_value(json.loads(line))
            if envelope.run_id != exact.run_id or envelope.attempt_id != exact.attempt_id:
                raise AuditSourceOrderError("event identity differs from its published attempt")
            source = _scientific_source(envelope)
            if source is not None:
                self._consume(source)
        self._finish()
        return tuple(self._segments)

    def _consume(self, event: ScientificSourceEvent) -> None:
        match self._state, event:
            case AwaitLibraryInitialization(), LibraryInitialized() as initialized:
                self._consume_library_initialization(initialized)
            case InLibrarySegment() as state, TrainingStepCommit() as step:
                self._consume_training_step(state, step)
            case InLibrarySegment() as state, EvolutionPhaseOpened() as opened:
                self._consume_phase(state, opened)
            case PhaseOpened() as state, EvolutionCycleCommitted() as cycle:
                self._consume_cycle(state, cycle)
            case PhaseOpened() as state, EvolutionNoOpCommitted() as no_op:
                self._consume_no_op(state, no_op)
            case _:
                raise AuditSourceOrderError(
                    f"source {type(event).__name__} is invalid in "
                    f"state {type(self._state).__name__}"
                )

    def _consume_library_initialization(self, initialized: LibraryInitialized) -> None:
        if initialized.method_identity_hash != self._identity.method_identity_hash:
            raise AuditEvidenceMismatchError("library source method identity differs")
        expected_cursor = self._identity.initial_run_cursor.to_source_value()
        if initialized.run_cursor != expected_cursor:
            raise AuditSourceOrderError("library source starts from another run cursor")
        if initialized.initial_optimizer_step != self._identity.initial_optimizer_step:
            raise AuditSourceOrderError("library source optimizer origin differs")
        state = _initial_library_state(initialized)
        if state.current_version != self._identity.initial_library_version:
            raise AuditEvidenceMismatchError("library source differs from identity seed library")
        if state.state_hash != self._identity.initial_skill_library_state_hash:
            raise AuditEvidenceMismatchError("library source differs from identity seed state")
        self._cursor = initialized.run_cursor
        self._last_optimizer_step = initialized.initial_optimizer_step
        self._state = InLibrarySegment(
            library_state=state,
            cursor_before=initialized.run_cursor,
            training_steps=(),
        )

    def _consume_training_step(
        self,
        state: InLibrarySegment,
        step: TrainingStepCommit,
    ) -> None:
        if self._cursor is None or self._last_optimizer_step is None:
            raise AuditSourceOrderError("training source appeared before initialization")
        expected_cursor = _after_training_step(self._cursor)
        if step.run_cursor_after != expected_cursor:
            raise AuditSourceOrderError("training source run cursor is not the next step")
        if step.optimizer_step != self._last_optimizer_step + 1:
            raise AuditSourceOrderError("optimizer steps are not consecutive")
        if step.library_version != state.library_state.current_version:
            raise AuditSourceOrderError("training step uses another library")
        position = step.run_cursor_after.completed_training_steps
        if position > self._plan.total_training_steps:
            raise AuditSourceOrderError("training source exceeds the exact run plan")
        self._cursor = step.run_cursor_after
        self._last_optimizer_step = step.optimizer_step
        self._state = InLibrarySegment(
            library_state=state.library_state,
            cursor_before=state.cursor_before,
            training_steps=(*state.training_steps, step),
        )

    def _consume_phase(
        self,
        state: InLibrarySegment,
        opened: EvolutionPhaseOpened,
    ) -> None:
        if self._cursor is None:
            raise AuditSourceOrderError("phase source appeared before initialization")
        if not state.training_steps:
            raise AuditSourceOrderError("phase opened before any training source")
        if opened.run_cursor_at_phase != self._cursor:
            raise AuditSourceOrderError("phase source cursor differs from latest training step")
        position = self._cursor.completed_training_steps
        if self._plan.slot_kind(position) is not RunSlotKind.PHASE_SEARCH:
            raise AuditSourceOrderError("phase opened in a closure training slot")
        if opened.phase_event.library_version != state.library_state.current_version:
            raise AuditSourceOrderError("phase uses another library")
        self._state = PhaseOpened(
            library_state=state.library_state,
            cursor_before=state.cursor_before,
            training_steps=state.training_steps,
            phase=opened,
        )

    def _consume_cycle(
        self,
        state: PhaseOpened,
        cycle: EvolutionCycleCommitted,
    ) -> None:
        if self._cursor is None or self._last_optimizer_step is None:
            raise AuditSourceOrderError("cycle source appeared before initialization")
        if cycle.phase_event_id != state.phase.phase_event.event_id:
            raise AuditSourceOrderError("cycle references another phase")
        if cycle.optimizer_step != self._last_optimizer_step:
            raise AuditSourceOrderError("cycle optimizer step differs from its segment")
        if cycle.library_version_before != state.library_state.current_version:
            raise AuditSourceOrderError("cycle starts from another library")
        expected_cursor = _after_cycle(self._cursor, action_count=len(cycle.mutation.actions))
        if cycle.run_cursor_after != expected_cursor:
            raise AuditSourceOrderError("cycle source cursor does not advance exact counts")
        if expected_cursor.committed_cycles > self._plan.maximum_cycles:
            raise AuditSourceOrderError("cycle source exceeds run-plan cycle bound")
        if cycle.z_reset_seed != partition_reset_seed(
            base_seed=self._identity.application_config.trainer.rollout.base_seed,
            cycle_ordinal=expected_cursor.committed_cycles,
        ):
            raise AuditEvidenceMismatchError("cycle Z reset seed differs")
        rebuilt = apply_mutation_pure(state.library_state, cycle)
        self._segments.append(
            CommittedLibrarySegment(
                library_state=state.library_state,
                cursor_before=state.cursor_before,
                cursor_after=cycle.run_cursor_after,
                training_steps=state.training_steps,
                phase=state.phase,
                cycle=cycle,
                no_op=None,
            )
        )
        self._cursor = cycle.run_cursor_after
        self._state = InLibrarySegment(
            library_state=rebuilt,
            cursor_before=cycle.run_cursor_after,
            training_steps=(),
        )

    def _consume_no_op(
        self,
        state: PhaseOpened,
        no_op: EvolutionNoOpCommitted,
    ) -> None:
        if self._cursor is None or self._last_optimizer_step is None:
            raise AuditSourceOrderError("no-op source appeared before initialization")
        if no_op.phase_event_id != state.phase.phase_event.event_id:
            raise AuditSourceOrderError("no-op source references another phase")
        if no_op.optimizer_step != self._last_optimizer_step:
            raise AuditSourceOrderError("no-op optimizer step differs from its segment")
        if no_op.library_version != state.library_state.current_version:
            raise AuditSourceOrderError("no-op source uses another library")
        if no_op.run_cursor_after != self._cursor:
            raise AuditSourceOrderError("no-op source must preserve the run cursor")
        self._segments.append(
            CommittedLibrarySegment(
                library_state=state.library_state,
                cursor_before=state.cursor_before,
                cursor_after=self._cursor,
                training_steps=state.training_steps,
                phase=state.phase,
                cycle=None,
                no_op=no_op,
            )
        )
        self._state = InLibrarySegment(
            library_state=state.library_state,
            cursor_before=self._cursor,
            training_steps=(),
        )

    def _finish(self) -> None:
        if self._cursor is None:
            raise AuditSourceOrderError("published attempt has no run cursor")
        if self._cursor.completed_training_steps != self._plan.total_training_steps:
            raise AuditSourceOrderError("published success did not complete run plan")
        match self._state:
            case AwaitLibraryInitialization():
                raise AuditSourceOrderError("published attempt has no library initialization")
            case PhaseOpened():
                raise AuditSourceOrderError("published attempt ends with an unclosed phase")
            case InLibrarySegment() as state:
                self._segments.append(
                    CommittedLibrarySegment(
                        library_state=state.library_state,
                        cursor_before=state.cursor_before,
                        cursor_after=self._cursor,
                        training_steps=state.training_steps,
                        phase=None,
                        cycle=None,
                        no_op=None,
                    )
                )


def _after_training_step(cursor: RunCursorValue) -> RunCursorValue:
    return RunCursorValue(
        run_plan_hash=cursor.run_plan_hash,
        completed_training_steps=cursor.completed_training_steps + 1,
        committed_cycles=cursor.committed_cycles,
        committed_actions=cursor.committed_actions,
    )


def _after_cycle(cursor: RunCursorValue, *, action_count: int) -> RunCursorValue:
    if type(action_count) is not int or action_count < 1:
        raise AuditSourceOrderError("cycle source requires at least one action")
    return RunCursorValue(
        run_plan_hash=cursor.run_plan_hash,
        completed_training_steps=cursor.completed_training_steps,
        committed_cycles=cursor.committed_cycles + 1,
        committed_actions=cursor.committed_actions + action_count,
    )


def _scientific_source(envelope: EventEnvelope) -> ScientificSourceEvent | None:
    match envelope.event_type:
        case EventType.LIBRARY_INITIALIZED:
            return LibraryInitialized.from_value(envelope.payload)
        case EventType.TRAINING_STEP_COMMITTED:
            return TrainingStepCommit.from_value(envelope.payload)
        case EventType.EVOLUTION_PHASE_OPENED:
            return EvolutionPhaseOpened.from_value(envelope.payload)
        case EventType.EVOLUTION_CYCLE_COMMITTED:
            return EvolutionCycleCommitted.from_value(envelope.payload)
        case EventType.EVOLUTION_NO_OP_COMMITTED:
            return EvolutionNoOpCommitted.from_value(envelope.payload)
        case _:
            return None


def _initial_library_state(initialized: LibraryInitialized) -> SkillLibraryState:
    documents = tuple(SkillDocument.from_value(value) for value in initialized.documents)
    by_id = {document.manifest.skill_id: document for document in documents}
    if len(by_id) != len(documents):
        raise AuditEvidenceMismatchError("library initialization repeats a skill ID")
    try:
        return SkillLibraryState(
            documents=by_id,
            active_skill_ids=initialized.active_skill_ids,
            current_version=initialized.library_version,
        )
    except ValueError as error:
        raise AuditEvidenceMismatchError("library initialization hash differs") from error


def apply_mutation_pure(
    state: SkillLibraryState,
    cycle: EvolutionCycleCommitted,
) -> SkillLibraryState:
    mutation = cycle.mutation
    if mutation.library_version_before != state.current_version:
        raise AuditEvidenceMismatchError("mutation starts from another library")
    documents = dict(state.documents)
    new_documents = tuple(SkillDocument.from_value(value) for value in mutation.new_documents)
    new_document_ids = tuple(document.manifest.skill_id for document in new_documents)
    if len(set(new_document_ids)) != len(new_document_ids):
        raise AuditEvidenceMismatchError("cycle repeats a new skill document")
    _verify_action_plan(state=state, cycle=cycle, new_document_ids=new_document_ids)
    for document in new_documents:
        skill_id = document.manifest.skill_id
        if skill_id in documents:
            raise AuditEvidenceMismatchError("cycle reuses a skill ID")
        documents[skill_id] = document
    try:
        rebuilt = SkillLibraryState(
            documents=documents,
            active_skill_ids=mutation.active_skill_ids_after,
            current_version=mutation.library_version_after,
        )
    except ValueError as error:
        raise AuditEvidenceMismatchError("cycle library mutation hash differs") from error
    if rebuilt.current_version != cycle.library_version_after:
        raise AuditEvidenceMismatchError("cycle after-version differs from rebuilt library")
    return rebuilt


def _verify_action_plan(
    *,
    state: SkillLibraryState,
    cycle: EvolutionCycleCommitted,
    new_document_ids: tuple[str, ...],
) -> None:
    mutation = cycle.mutation
    active = set(state.active_skill_ids)
    known = set(state.documents)
    produced: set[str] = set()
    action_ids: set[str] = set()
    targeted: set[str] = set()
    for action in mutation.actions:
        if action.action_id in action_ids:
            raise AuditEvidenceMismatchError("cycle repeats an evolution action")
        action_ids.add(action.action_id)
        if action.phase_event_id != cycle.phase_event_id:
            raise AuditEvidenceMismatchError("evolution action references another phase")
        if action.lineage_ref != cycle.phase_event_id:
            raise AuditEvidenceMismatchError("evolution action lineage differs from its phase")
        if action.library_version_before != mutation.library_version_before:
            raise AuditEvidenceMismatchError("evolution action before-version differs")
        if action.library_version_after != mutation.library_version_after:
            raise AuditEvidenceMismatchError("evolution action after-version differs")
        for target_skill_id in action.target_skill_ids:
            if target_skill_id in targeted:
                raise AuditEvidenceMismatchError("cycle targets one skill more than once")
            if target_skill_id not in active:
                raise AuditEvidenceMismatchError("evolution action targets an inactive skill")
            targeted.add(target_skill_id)
            active.remove(target_skill_id)
        for produced_skill_id in action.produced_skill_ids:
            if produced_skill_id in known or produced_skill_id in produced:
                raise AuditEvidenceMismatchError("evolution action reuses a skill ID")
            produced.add(produced_skill_id)
            active.add(produced_skill_id)
    if produced != set(new_document_ids):
        raise AuditEvidenceMismatchError("evolution action products differ from new documents")
    if tuple(sorted(active)) != mutation.active_skill_ids_after:
        raise AuditEvidenceMismatchError("evolution actions differ from the active set")


__all__ = [
    "AuditEvidenceMismatchError",
    "AuditSourceOrderError",
    "AuditSourceReducer",
    "CommittedLibrarySegment",
    "apply_mutation_pure",
]
