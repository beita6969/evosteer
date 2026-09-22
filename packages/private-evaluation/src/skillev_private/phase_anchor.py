"""Private admission for immutable, phase-bound progress-anchor states.

An anchor is selected by its predeclared phase event ID, never by scanning a
checkpoint directory for a latest entry.  The source event binds the private
checkpoint bytes to the completed phase; evaluation consumes that state through
the normal read-only frozen runner and cannot feed a result into training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from skillev.audit.published_identity import read_published_attempt_identity
from skillev.contracts import (
    JsonValue,
    PhaseCheckpointArtifact,
    PhaseCheckpointPublished,
    PosteriorCellState,
    normalize_json,
    stable_hash,
)
from skillev.contracts.identity import validate_sha256
from skillev.contracts.ttb_source_events import EvolutionCycleCommitted, EvolutionPhaseOpened
from skillev.experiments import (
    AblationArm,
    AttemptPurpose,
    FormalRunLedger,
    FrozenTaskSequenceIdentity,
    SchedulePurpose,
    TrainingEvolutionCounts,
)
from skillev.experiments.arm_events import ArmEventType, read_arm_event_history
from skillev.experiments.arms.no_bayesian_contracts import (
    FlowOnlyCycleResult,
    FlowOnlyPhaseCheckpointPublished,
    FlowOnlyPhaseOpened,
)
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptSourceLogKind,
    AttemptSucceeded,
    FlowOnlyRuntimeExecutionState,
    FullRuntimeExecutionState,
    PublishedSuccessfulAttemptBundle,
    RuntimeSnapshot,
    SkillLibraryState,
    read_attempt_outcome,
)
from skillev.runtime.attempt_publication import sha256_file, sha256_tree
from skillev.runtime.event_log import EventType
from skillev.runtime.event_log_reader import read_event_history
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence

PHASE_ANCHOR_INFERENCE_STATE_FORMAT = "skillev-phase-anchor-inference-state@3"
PHASE_ANCHOR_EVALUATION_INPUT_FORMAT = "skillev-private-phase-anchor-input@2"
_PHASE_ANCHOR_ADMISSION = object()


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


@runtime_checkable
class PhaseCheckpointArtifactResolver(Protocol):
    """Resolve one source-bound phase checkpoint without publishing its path."""

    def resolve_phase_checkpoint_artifact(
        self,
        *,
        bundle: PublishedSuccessfulAttemptBundle,
        artifact: PhaseCheckpointArtifact,
    ) -> Path: ...


@dataclass(frozen=True, slots=True, init=False)
class PhaseAnchorInferenceState:
    """The immutable post-cycle state allowed to run an IID progress anchor."""

    policy_snapshot_directory: Path
    policy_snapshot_hash: str
    phase_checkpoint_artifact: PhaseCheckpointArtifact
    library: SkillLibraryState
    calibration_cells: tuple[PosteriorCellState, ...]
    method_identity_hash: str
    source_training_attempt_id: str
    source_formal_run_group_id: str
    source_training_identity_hash: str
    source_exact_input_sha256: str
    source_training_outcome_sha256: str
    source_phase_event_id: str
    evolution_counts: TrainingEvolutionCounts
    format: str = PHASE_ANCHOR_INFERENCE_STATE_FORMAT

    def __init__(
        self,
        *,
        policy_snapshot_directory: Path,
        policy_snapshot_hash: str,
        phase_checkpoint_artifact: PhaseCheckpointArtifact,
        library: SkillLibraryState,
        calibration_cells: tuple[PosteriorCellState, ...],
        method_identity_hash: str,
        source_training_attempt_id: str,
        source_formal_run_group_id: str,
        source_training_identity_hash: str,
        source_exact_input_sha256: str,
        source_training_outcome_sha256: str,
        source_phase_event_id: str,
        evolution_counts: TrainingEvolutionCounts,
        state_format: str = PHASE_ANCHOR_INFERENCE_STATE_FORMAT,
        _admission: object,
    ) -> None:
        if _admission is not _PHASE_ANCHOR_ADMISSION:
            raise TypeError("phase anchor state must come from a published phase checkpoint")
        object.__setattr__(self, "policy_snapshot_directory", policy_snapshot_directory)
        object.__setattr__(self, "policy_snapshot_hash", policy_snapshot_hash)
        object.__setattr__(self, "phase_checkpoint_artifact", phase_checkpoint_artifact)
        object.__setattr__(self, "library", library)
        object.__setattr__(self, "calibration_cells", calibration_cells)
        object.__setattr__(self, "method_identity_hash", method_identity_hash)
        object.__setattr__(self, "source_training_attempt_id", source_training_attempt_id)
        object.__setattr__(self, "source_formal_run_group_id", source_formal_run_group_id)
        object.__setattr__(self, "source_training_identity_hash", source_training_identity_hash)
        object.__setattr__(self, "source_exact_input_sha256", source_exact_input_sha256)
        object.__setattr__(self, "source_training_outcome_sha256", source_training_outcome_sha256)
        object.__setattr__(self, "source_phase_event_id", source_phase_event_id)
        object.__setattr__(self, "evolution_counts", evolution_counts)
        object.__setattr__(self, "format", state_format)
        self._validate()

    def _validate(self) -> None:
        if (
            not isinstance(self.policy_snapshot_directory, Path)
            or not self.policy_snapshot_directory.is_absolute()
        ):
            raise ValueError("phase anchor checkpoint directory must be absolute")
        validate_sha256(self.policy_snapshot_hash)
        if not isinstance(self.phase_checkpoint_artifact, PhaseCheckpointArtifact):
            raise TypeError("phase anchor requires a PhaseCheckpointArtifact")
        if self.policy_snapshot_hash != self.phase_checkpoint_artifact.artifact_sha256:
            raise ValueError("phase anchor checkpoint differs from artifact identity")
        if not isinstance(self.library, SkillLibraryState):
            raise TypeError("phase anchor requires SkillLibraryState")
        if self.library.current_version != self.phase_checkpoint_artifact.library_version:
            raise ValueError("phase anchor library differs from checkpoint artifact")
        if not isinstance(self.calibration_cells, tuple) or any(
            not isinstance(item, PosteriorCellState) for item in self.calibration_cells
        ):
            raise TypeError("phase anchor calibration_cells are invalid")
        if len({item.z.cell_key(item.skill_id) for item in self.calibration_cells}) != len(
            self.calibration_cells
        ):
            raise ValueError("phase anchor repeats posterior cells")
        _text(self.method_identity_hash, field="method_identity_hash")
        _text(self.source_training_attempt_id, field="source_training_attempt_id")
        validate_sha256(self.source_formal_run_group_id)
        validate_sha256(self.source_training_identity_hash)
        validate_sha256(self.source_exact_input_sha256)
        validate_sha256(self.source_training_outcome_sha256)
        _text(self.source_phase_event_id, field="source_phase_event_id")
        if self.source_phase_event_id != self.phase_checkpoint_artifact.phase_event_id:
            raise ValueError("phase anchor state differs from checkpoint phase")
        if not isinstance(self.evolution_counts, TrainingEvolutionCounts):
            raise TypeError("phase anchor evolution counts are invalid")
        cursor = self.phase_checkpoint_artifact.run_cursor_after
        if (
            self.evolution_counts.training_step_count != cursor.completed_training_steps
            or self.evolution_counts.cycle_count != cursor.committed_cycles
            or self.evolution_counts.action_count != cursor.committed_actions
        ):
            raise ValueError("phase anchor evolution counts differ from checkpoint cursor")
        if self.format != PHASE_ANCHOR_INFERENCE_STATE_FORMAT:
            raise ValueError("unsupported phase anchor inference state format")

    @classmethod
    def from_published_phase(
        cls,
        bundle: PublishedSuccessfulAttemptBundle,
        *,
        phase_event_id: str,
        artifact_resolver: PhaseCheckpointArtifactResolver,
        formal_run_ledger: FormalRunLedger,
    ) -> PhaseAnchorInferenceState:
        """Admit one explicit source-bound phase checkpoint.

        The event log is searched only for the supplied phase event ID.  There
        is no fallback to a newer checkpoint or a directory-order convention.
        """

        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        outcome = read_attempt_outcome(exact.outcome_path)
        if not isinstance(outcome, AttemptSucceeded):
            raise ValueError("phase anchor requires a successful training outcome")
        identity = read_published_attempt_identity(exact)
        if identity.purpose is not AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
            raise ValueError("phase anchor requires a formal benchmark training attempt")
        if not isinstance(formal_run_ledger, FormalRunLedger):
            raise TypeError("phase anchor requires FormalRunLedger")
        formal_run_ledger.require_bundle_terminal_success(exact)
        if type(phase_event_id) is not str or not phase_event_id:
            raise ValueError("phase anchor requires an explicit phase event ID")
        if not isinstance(artifact_resolver, PhaseCheckpointArtifactResolver):
            raise TypeError("phase anchor requires a checkpoint artifact resolver")

        if exact.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
            artifact, phase_count = _source_flow_only_phase_artifact(
                exact,
                phase_event_id=phase_event_id,
            )
        else:
            artifact, phase_count = _source_phase_artifact(exact, phase_event_id=phase_event_id)
        if identity.formal_execution is None:  # narrowed by formal identity admission
            raise ValueError("phase anchor source has no formal execution freeze")
        identity.formal_execution.requires_progress_anchor_cycle(
            artifact.run_cursor_after.committed_cycles
        )
        directory = artifact_resolver.resolve_phase_checkpoint_artifact(
            bundle=exact,
            artifact=artifact,
        ).resolve()
        if sha256_tree(directory) != artifact.artifact_sha256:
            raise ValueError("resolved phase checkpoint bytes differ from source artifact")
        runtime_state_path = directory / "runtime_state.json"
        if sha256_file(runtime_state_path) != artifact.runtime_state_sha256:
            raise ValueError("resolved phase runtime state differs from source artifact")
        metadata = RuntimeSnapshot.from_value(
            normalize_json(json.loads(runtime_state_path.read_text(encoding="utf-8")))
        )
        execution_state = metadata.execution_state
        if (
            metadata.identity.method_identity_hash != identity.method_identity_hash
            or metadata.optimizer_step != artifact.optimizer_step
            or execution_state.library.current_version != artifact.library_version
            or execution_state.run_cursor.to_source_value() != artifact.run_cursor_after
        ):
            raise ValueError("resolved phase checkpoint differs from its source descriptor")
        if exact.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
            if not isinstance(execution_state, FlowOnlyRuntimeExecutionState):
                raise ValueError("phase anchor runtime kind differs from its source arm")
            cells: tuple[PosteriorCellState, ...] = ()
        else:
            if not isinstance(execution_state, FullRuntimeExecutionState):
                raise ValueError("phase anchor runtime kind differs from its source arm")
            cells = execution_state.projections.calibration_cells
        return cls(
            policy_snapshot_directory=directory,
            policy_snapshot_hash=artifact.artifact_sha256,
            phase_checkpoint_artifact=artifact,
            library=execution_state.library,
            calibration_cells=cells,
            method_identity_hash=identity.method_identity_hash,
            source_training_attempt_id=exact.attempt_id,
            source_formal_run_group_id=formal_run_ledger.manifest.run_group_id,
            source_training_identity_hash=identity.content_hash,
            source_exact_input_sha256=identity.exact_input_sha256,
            source_training_outcome_sha256=exact.outcome_sha256,
            source_phase_event_id=phase_event_id,
            evolution_counts=TrainingEvolutionCounts(
                training_step_count=artifact.run_cursor_after.completed_training_steps,
                phase_count=phase_count,
                cycle_count=artifact.run_cursor_after.committed_cycles,
                action_count=artifact.run_cursor_after.committed_actions,
            ),
            _admission=_PHASE_ANCHOR_ADMISSION,
        )

    def require_published_phase(
        self,
        bundle: PublishedSuccessfulAttemptBundle,
        *,
        artifact_resolver: PhaseCheckpointArtifactResolver,
        formal_run_ledger: FormalRunLedger,
    ) -> None:
        """Re-admit the immutable checkpoint before a private worker executes."""

        expected = type(self).from_published_phase(
            bundle,
            phase_event_id=self.source_phase_event_id,
            artifact_resolver=artifact_resolver,
            formal_run_ledger=formal_run_ledger,
        )
        if self != expected:
            raise ValueError("phase anchor state differs from the published phase source")

    @property
    def policy_snapshot_id(self) -> str:
        return self.phase_checkpoint_artifact.policy_snapshot_id

    @property
    def optimizer_step(self) -> int:
        return self.phase_checkpoint_artifact.optimizer_step

    @property
    def content_hash(self) -> str:
        return stable_hash(
            {
                "calibration_cells": [item.to_value() for item in self.calibration_cells],
                "format": self.format,
                "evolution_counts": self.evolution_counts.to_value(),
                "library": self.library.to_value(),
                "method_identity_hash": self.method_identity_hash,
                "phase_checkpoint_artifact": self.phase_checkpoint_artifact.to_value(),
                "policy_snapshot_hash": self.policy_snapshot_hash,
                "source_exact_input_sha256": self.source_exact_input_sha256,
                "source_formal_run_group_id": self.source_formal_run_group_id,
                "source_phase_event_id": self.source_phase_event_id,
                "source_training_attempt_id": self.source_training_attempt_id,
                "source_training_identity_hash": self.source_training_identity_hash,
                "source_training_outcome_sha256": self.source_training_outcome_sha256,
            }
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "calibration_cells": [item.to_value() for item in self.calibration_cells],
            "format": self.format,
            "evolution_counts": self.evolution_counts.to_value(),
            "library": self.library.to_value(),
            "method_identity_hash": self.method_identity_hash,
            "phase_checkpoint_artifact": self.phase_checkpoint_artifact.to_value(),
            "policy_snapshot_directory": self.policy_snapshot_directory.as_posix(),
            "policy_snapshot_hash": self.policy_snapshot_hash,
            "source_exact_input_sha256": self.source_exact_input_sha256,
            "source_formal_run_group_id": self.source_formal_run_group_id,
            "source_phase_event_id": self.source_phase_event_id,
            "source_training_attempt_id": self.source_training_attempt_id,
            "source_training_identity_hash": self.source_training_identity_hash,
            "source_training_outcome_sha256": self.source_training_outcome_sha256,
        }

    @classmethod
    def _from_value(cls, value: object) -> PhaseAnchorInferenceState:
        data = normalize_json(value)
        fields = {
            "calibration_cells",
            "format",
            "evolution_counts",
            "library",
            "method_identity_hash",
            "phase_checkpoint_artifact",
            "policy_snapshot_directory",
            "policy_snapshot_hash",
            "source_exact_input_sha256",
            "source_formal_run_group_id",
            "source_phase_event_id",
            "source_training_attempt_id",
            "source_training_identity_hash",
            "source_training_outcome_sha256",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("phase anchor inference state has incompatible fields")
        cells = data["calibration_cells"]
        if not isinstance(cells, list):
            raise ValueError("phase anchor calibration cells must be an array")
        if any(
            type(data[name]) is not str
            for name in fields
            - {
                "calibration_cells",
                "evolution_counts",
                "library",
                "phase_checkpoint_artifact",
            }
        ):
            raise ValueError("phase anchor inference state text fields are invalid")
        return cls(
            policy_snapshot_directory=Path(data["policy_snapshot_directory"]),
            policy_snapshot_hash=data["policy_snapshot_hash"],
            phase_checkpoint_artifact=PhaseCheckpointArtifact.from_value(
                data["phase_checkpoint_artifact"]
            ),
            library=SkillLibraryState.from_value(data["library"]),
            calibration_cells=tuple(PosteriorCellState.from_value(item) for item in cells),
            method_identity_hash=data["method_identity_hash"],
            source_training_attempt_id=data["source_training_attempt_id"],
            source_formal_run_group_id=data["source_formal_run_group_id"],
            source_training_identity_hash=data["source_training_identity_hash"],
            source_exact_input_sha256=data["source_exact_input_sha256"],
            source_training_outcome_sha256=data["source_training_outcome_sha256"],
            source_phase_event_id=data["source_phase_event_id"],
            evolution_counts=TrainingEvolutionCounts.from_value(data["evolution_counts"]),
            state_format=data["format"],
            _admission=_PHASE_ANCHOR_ADMISSION,
        )


@dataclass(frozen=True, slots=True)
class PhaseAnchorEvaluationInput:
    """One AIME-style no-update sequence fixed to an explicit phase checkpoint."""

    state: PhaseAnchorInferenceState
    task_sequence: PrivateFrozenTaskSequence
    format: str = PHASE_ANCHOR_EVALUATION_INPUT_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.state, PhaseAnchorInferenceState):
            raise TypeError("phase anchor input requires PhaseAnchorInferenceState")
        if not isinstance(self.task_sequence, PrivateFrozenTaskSequence):
            raise TypeError("phase anchor input requires PrivateFrozenTaskSequence")
        if self.task_sequence.identity.purpose is not SchedulePurpose.IID_PROGRESS:
            raise ValueError("phase anchors require an IID progress schedule")
        if self.format != PHASE_ANCHOR_EVALUATION_INPUT_FORMAT:
            raise ValueError("unsupported phase anchor evaluation input format")

    @property
    def task_sequence_identity(self) -> FrozenTaskSequenceIdentity:
        return self.task_sequence.identity

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "state": self.state.to_value(),
            "task_sequence": self.task_sequence.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PhaseAnchorEvaluationInput:
        data = normalize_json(value)
        if not isinstance(data, dict) or set(data) != {"format", "state", "task_sequence"}:
            raise ValueError("phase anchor evaluation input has incompatible fields")
        if type(data["format"]) is not str:
            raise ValueError("phase anchor evaluation input format must be text")
        return cls(
            state=PhaseAnchorInferenceState._from_value(data["state"]),
            task_sequence=PrivateFrozenTaskSequence.from_value(data["task_sequence"]),
            format=data["format"],
        )


def _source_phase_artifact(
    bundle: PublishedSuccessfulAttemptBundle,
    *,
    phase_event_id: str,
) -> tuple[PhaseCheckpointArtifact, int]:
    events = read_event_history(bundle.event_log_path)
    phases: dict[str, tuple[int, EvolutionPhaseOpened]] = {}
    cycles: dict[str, tuple[int, EvolutionCycleCommitted]] = {}
    publications: list[tuple[int, PhaseCheckpointPublished]] = []
    for index, event in enumerate(events):
        if event.event_type is EventType.EVOLUTION_PHASE_OPENED:
            opened = EvolutionPhaseOpened.from_value(event.payload)
            if opened.phase_event.event_id in phases:
                raise ValueError("published source repeats a phase event ID")
            phases[opened.phase_event.event_id] = (index, opened)
        elif event.event_type is EventType.EVOLUTION_CYCLE_COMMITTED:
            cycle = EvolutionCycleCommitted.from_value(event.payload)
            if cycle.phase_event_id in cycles:
                raise ValueError("published source repeats a committed phase ID")
            cycles[cycle.phase_event_id] = (index, cycle)
        elif event.event_type is EventType.PHASE_CHECKPOINT_PUBLISHED:
            publications.append((index, PhaseCheckpointPublished.from_value(event.payload)))
    selected = tuple(item for item in publications if item[1].phase_event_id == phase_event_id)
    if len(selected) != 1:
        raise ValueError("published source has no unique phase checkpoint for the requested phase")
    publication_index, publication = selected[0]
    if phase_event_id not in phases or phase_event_id not in cycles:
        raise ValueError("phase checkpoint source lacks its phase/cycle evidence")
    phase_index, phase = phases[phase_event_id]
    cycle_index, cycle = cycles[phase_event_id]
    artifact = publication.artifact
    if not phase_index < cycle_index < publication_index:
        raise ValueError("phase checkpoint source ordering is invalid")
    if (
        artifact.optimizer_step != cycle.optimizer_step
        or artifact.library_version != cycle.library_version_after
        or artifact.run_cursor_after != cycle.run_cursor_after
        or phase.phase_event.library_version != cycle.library_version_before
        or phase.run_cursor_at_phase.completed_training_steps != cycle.optimizer_step
    ):
        raise ValueError("phase checkpoint descriptor differs from committed phase state")
    phase_count = sum(index <= publication_index for index, _ in phases.values())
    if phase_count != artifact.run_cursor_after.committed_cycles:
        raise ValueError("phase checkpoint source phase count differs from committed cycles")
    return artifact, phase_count


def _source_flow_only_phase_artifact(
    bundle: PublishedSuccessfulAttemptBundle,
    *,
    phase_event_id: str,
) -> tuple[PhaseCheckpointArtifact, int]:
    """Reduce the native no-Bayesian source stream without posterior stand-ins."""

    if bundle.builder_kind is not AttemptBuilderKind.NO_BAYESIAN:
        raise ValueError("flow-only phase source requires the no-Bayesian arm")
    source_path = bundle.source_log_path(bundle.source_log(AttemptSourceLogKind.FLOW_ONLY))
    events = read_arm_event_history(source_path)
    phases: dict[str, tuple[int, FlowOnlyPhaseOpened]] = {}
    cycles: dict[str, tuple[int, FlowOnlyCycleResult]] = {}
    publications: list[tuple[int, FlowOnlyPhaseCheckpointPublished]] = []
    for index, event in enumerate(events):
        if (
            event.arm is not AblationArm.SKILLFLOW_DISABLED
            or event.run_id != bundle.run_id
            or event.attempt_id != bundle.attempt_id
        ):
            raise ValueError("flow-only phase source differs from its published attempt")
        if event.event_type is ArmEventType.FLOW_ONLY_PHASE_OPENED:
            opened = FlowOnlyPhaseOpened.from_value(event.payload)
            if opened.phase_event.event_id in phases:
                raise ValueError("flow-only source repeats a phase event ID")
            phases[opened.phase_event.event_id] = (index, opened)
        elif event.event_type is ArmEventType.FLOW_ONLY_CYCLE_COMMITTED:
            cycle = FlowOnlyCycleResult.from_value(event.payload)
            if cycle.phase_event_id in cycles:
                raise ValueError("flow-only source repeats a committed phase ID")
            cycles[cycle.phase_event_id] = (index, cycle)
        elif event.event_type is ArmEventType.FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED:
            publications.append((index, FlowOnlyPhaseCheckpointPublished.from_value(event.payload)))
    selected = tuple(item for item in publications if item[1].phase_event_id == phase_event_id)
    if len(selected) != 1:
        raise ValueError("flow-only source has no unique checkpoint for the requested phase")
    publication_index, publication = selected[0]
    if phase_event_id not in phases or phase_event_id not in cycles:
        raise ValueError("flow-only checkpoint source lacks its phase/cycle evidence")
    phase_index, phase = phases[phase_event_id]
    cycle_index, cycle = cycles[phase_event_id]
    artifact = publication.artifact
    if not phase_index < cycle_index < publication_index:
        raise ValueError("flow-only checkpoint source ordering is invalid")
    if (
        artifact.optimizer_step != cycle.optimizer_step
        or artifact.library_version != cycle.library_version_after
        or artifact.run_cursor_after != cycle.run_cursor_after
        or phase.run_cursor_at_phase.completed_training_steps != cycle.optimizer_step
    ):
        raise ValueError("flow-only checkpoint differs from committed phase state")
    phase_count = sum(index <= publication_index for index, _ in phases.values())
    if phase_count != artifact.run_cursor_after.committed_cycles:
        raise ValueError("flow-only checkpoint phase count differs from committed cycles")
    return artifact, phase_count


__all__ = [
    "PHASE_ANCHOR_EVALUATION_INPUT_FORMAT",
    "PHASE_ANCHOR_INFERENCE_STATE_FORMAT",
    "PhaseAnchorEvaluationInput",
    "PhaseAnchorInferenceState",
    "PhaseCheckpointArtifactResolver",
]
