"""Shared frozen-inference state contracts with no experiment-worker imports.

The private experiment package exports its concrete worker graph directly.  A
frozen runner must nevertheless consume the state contract without importing
that package root, otherwise its eager exports create a circular worker import.
This module is the dependency-light home for that shared state contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skillev.audit import AuditedFinalTrainingState
from skillev.audit.published_identity import read_published_attempt_identity
from skillev.contracts import (
    JsonValue,
    PosteriorCellState,
    normalize_json,
    stable_hash,
    validate_sha256,
)
from skillev.experiments import (
    AttemptPurpose,
    FormalRunLedger,
    FrozenTaskSequenceIdentity,
    SchedulePurpose,
    TrainingEvolutionCounts,
)
from skillev.runtime import (
    AttemptSucceeded,
    FinalTrainingArtifact,
    PublishedSuccessfulAttemptBundle,
    RuntimeSnapshot,
    SkillLibraryState,
    read_attempt_outcome,
)
from skillev.runtime.attempt_publication import sha256_file, sha256_tree
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence

FROZEN_INFERENCE_STATE_FORMAT = "skillev-frozen-inference-state@5"
FROZEN_EVALUATION_INPUT_FORMAT = "skillev-private-frozen-evaluation-input@4"
_FROZEN_STATE_ADMISSION = object()


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


class FinalTrainingArtifactResolver(Protocol):
    """Resolve a hash-bound private checkpoint without exposing its path publicly."""

    def resolve_final_training_artifact(
        self,
        *,
        bundle: PublishedSuccessfulAttemptBundle,
        artifact: FinalTrainingArtifact,
    ) -> Path: ...


class AuditedTrainingAttempt(Protocol):
    @property
    def attempt_id(self) -> str: ...

    @property
    def public_identity_content_hash(self) -> str: ...

    @property
    def formal_artifacts_verified(self) -> bool: ...

    @property
    def final_training_state(self) -> AuditedFinalTrainingState: ...


@dataclass(frozen=True, slots=True, init=False)
class FrozenInferenceState:
    """The immutable policy/library/posterior state consumed by an evaluation."""

    policy_snapshot_directory: Path
    policy_snapshot_hash: str
    final_training_artifact: FinalTrainingArtifact
    library: SkillLibraryState
    calibration_cells: tuple[PosteriorCellState, ...]
    method_identity_hash: str
    source_training_attempt_id: str
    source_formal_run_group_id: str
    source_training_identity_hash: str
    source_exact_input_sha256: str
    source_training_outcome_sha256: str
    evolution_counts: TrainingEvolutionCounts
    format: str = FROZEN_INFERENCE_STATE_FORMAT

    def __init__(
        self,
        *,
        policy_snapshot_directory: Path,
        policy_snapshot_hash: str,
        final_training_artifact: FinalTrainingArtifact,
        library: SkillLibraryState,
        calibration_cells: tuple[PosteriorCellState, ...],
        method_identity_hash: str,
        source_training_attempt_id: str,
        source_formal_run_group_id: str,
        source_training_identity_hash: str,
        source_exact_input_sha256: str,
        source_training_outcome_sha256: str,
        evolution_counts: TrainingEvolutionCounts,
        state_format: str = FROZEN_INFERENCE_STATE_FORMAT,
        _admission: object,
    ) -> None:
        if _admission is not _FROZEN_STATE_ADMISSION:
            raise TypeError("frozen inference state must come from a published training audit")
        object.__setattr__(self, "policy_snapshot_directory", policy_snapshot_directory)
        object.__setattr__(self, "policy_snapshot_hash", policy_snapshot_hash)
        object.__setattr__(self, "final_training_artifact", final_training_artifact)
        object.__setattr__(self, "library", library)
        object.__setattr__(self, "calibration_cells", calibration_cells)
        object.__setattr__(self, "method_identity_hash", method_identity_hash)
        object.__setattr__(self, "source_training_attempt_id", source_training_attempt_id)
        object.__setattr__(self, "source_formal_run_group_id", source_formal_run_group_id)
        object.__setattr__(self, "source_training_identity_hash", source_training_identity_hash)
        object.__setattr__(self, "source_exact_input_sha256", source_exact_input_sha256)
        object.__setattr__(self, "source_training_outcome_sha256", source_training_outcome_sha256)
        object.__setattr__(self, "evolution_counts", evolution_counts)
        object.__setattr__(self, "format", state_format)
        self._validate()

    def _validate(self) -> None:
        if (
            not isinstance(self.policy_snapshot_directory, Path)
            or not self.policy_snapshot_directory.is_absolute()
        ):
            raise ValueError("frozen policy snapshot directory must be absolute")
        _text(self.policy_snapshot_hash, field="policy_snapshot_hash")
        if not isinstance(self.final_training_artifact, FinalTrainingArtifact):
            raise TypeError("frozen inference state requires final training artifact")
        if self.policy_snapshot_hash != self.final_training_artifact.artifact_sha256:
            raise ValueError("frozen policy snapshot differs from final artifact identity")
        if not isinstance(self.library, SkillLibraryState):
            raise TypeError("frozen inference state requires SkillLibraryState")
        if not isinstance(self.calibration_cells, tuple) or any(
            not isinstance(item, PosteriorCellState) for item in self.calibration_cells
        ):
            raise TypeError("frozen inference state calibration_cells are invalid")
        if len({item.z.cell_key(item.skill_id) for item in self.calibration_cells}) != len(
            self.calibration_cells
        ):
            raise ValueError("frozen inference state repeats posterior cells")
        _text(self.method_identity_hash, field="method_identity_hash")
        _text(self.source_training_attempt_id, field="source_training_attempt_id")
        validate_sha256(self.source_formal_run_group_id)
        validate_sha256(self.source_training_identity_hash)
        validate_sha256(self.source_exact_input_sha256)
        validate_sha256(self.source_training_outcome_sha256)
        if not isinstance(self.evolution_counts, TrainingEvolutionCounts):
            raise TypeError("frozen inference state evolution counts are invalid")
        if self.format != FROZEN_INFERENCE_STATE_FORMAT:
            raise ValueError("unsupported frozen inference state format")

    @classmethod
    def from_published_training(
        cls,
        bundle: PublishedSuccessfulAttemptBundle,
        audit_result: AuditedTrainingAttempt,
        artifact_resolver: FinalTrainingArtifactResolver,
        formal_run_ledger: FormalRunLedger,
    ) -> FrozenInferenceState:
        """Build the only accepted frozen state from audited formal training."""

        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        outcome = read_attempt_outcome(exact.outcome_path)
        if not isinstance(outcome, AttemptSucceeded):
            raise ValueError("frozen evaluation requires a successful training outcome")
        identity = read_published_attempt_identity(exact)
        if identity.purpose is not AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
            raise ValueError("frozen evaluation requires a formal benchmark training attempt")
        if not isinstance(formal_run_ledger, FormalRunLedger):
            raise TypeError("frozen evaluation requires FormalRunLedger")
        formal_run_ledger.require_bundle_terminal_success(exact)
        if audit_result.formal_artifacts_verified is not True:
            raise ValueError("frozen evaluation requires an audit with measured formal artifacts")
        if (
            audit_result.attempt_id != exact.attempt_id
            or audit_result.public_identity_content_hash != exact.public_identity_content_hash
        ):
            raise ValueError("audited training result belongs to another published attempt")
        if outcome.final_training_artifact != exact.final_training_artifact:
            raise ValueError("published final training artifact differs from outcome")
        audited = audit_result.final_training_state
        artifact = exact.final_training_artifact
        if (
            artifact.policy_snapshot_id != audited.final_policy_snapshot_id
            or artifact.library_version != audited.library.current_version
            or artifact.optimizer_step != audited.final_optimizer_step
        ):
            raise ValueError("audited source state differs from final checkpoint descriptor")
        directory = artifact_resolver.resolve_final_training_artifact(
            bundle=exact,
            artifact=artifact,
        ).resolve()
        if sha256_tree(directory) != artifact.artifact_sha256:
            raise ValueError("resolved final checkpoint bytes differ from published artifact")
        runtime_state_path = directory / "runtime_state.json"
        if sha256_file(runtime_state_path) != artifact.runtime_state_sha256:
            raise ValueError("resolved runtime state differs from published artifact")
        metadata = RuntimeSnapshot.from_value(
            normalize_json(json.loads(runtime_state_path.read_text(encoding="utf-8")))
        )
        if (
            metadata.identity.method_identity_hash != identity.method_identity_hash
            or metadata.optimizer_step != artifact.optimizer_step
            or metadata.execution_state.library != audited.library
        ):
            raise ValueError("resolved final checkpoint differs from audited training state")
        return cls(
            policy_snapshot_directory=directory,
            policy_snapshot_hash=artifact.artifact_sha256,
            final_training_artifact=artifact,
            library=audited.library,
            calibration_cells=audited.calibration_cells,
            method_identity_hash=identity.method_identity_hash,
            source_training_attempt_id=exact.attempt_id,
            source_formal_run_group_id=formal_run_ledger.manifest.run_group_id,
            source_training_identity_hash=identity.content_hash,
            source_exact_input_sha256=identity.exact_input_sha256,
            source_training_outcome_sha256=exact.outcome_sha256,
            evolution_counts=audited.evolution_counts,
            _admission=_FROZEN_STATE_ADMISSION,
        )

    def require_audited_published_training(
        self,
        bundle: PublishedSuccessfulAttemptBundle,
        audit_result: AuditedTrainingAttempt,
        formal_run_ledger: FormalRunLedger,
    ) -> None:
        """Require this state to equal one freshly audited published success."""

        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        outcome = read_attempt_outcome(exact.outcome_path)
        if not isinstance(outcome, AttemptSucceeded):
            raise ValueError("frozen evaluation requires a successful training outcome")
        identity = read_published_attempt_identity(exact)
        if identity.purpose is not AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
            raise ValueError("frozen evaluation requires a formal benchmark training attempt")
        if not isinstance(formal_run_ledger, FormalRunLedger):
            raise TypeError("frozen evaluation requires FormalRunLedger")
        formal_run_ledger.require_bundle_terminal_success(exact)
        if audit_result.formal_artifacts_verified is not True:
            raise ValueError("frozen evaluation requires an audit with measured formal artifacts")
        if (
            self.source_training_attempt_id != exact.attempt_id
            or self.source_formal_run_group_id != formal_run_ledger.manifest.run_group_id
            or self.source_training_identity_hash != identity.content_hash
            or self.source_exact_input_sha256 != exact.exact_input_sha256
            or self.source_training_outcome_sha256 != exact.outcome_sha256
            or self.final_training_artifact != exact.final_training_artifact
            or outcome.final_training_artifact != exact.final_training_artifact
        ):
            raise ValueError("frozen state differs from its published training success")
        audited = audit_result.final_training_state
        if (
            audit_result.attempt_id != exact.attempt_id
            or audit_result.public_identity_content_hash != identity.content_hash
            or audited.library != self.library
            or audited.calibration_cells != self.calibration_cells
            or audited.final_policy_snapshot_id != self.final_training_artifact.policy_snapshot_id
            or audited.final_optimizer_step != self.final_training_artifact.optimizer_step
            or audited.evolution_counts != self.evolution_counts
        ):
            raise ValueError("frozen state differs from the freshly audited training state")

    @property
    def library_hash(self) -> str:
        return stable_hash(self.library.to_value())

    @property
    def policy_snapshot_id(self) -> str:
        """The one policy snapshot admitted for this evaluation state."""

        return self.final_training_artifact.policy_snapshot_id

    @property
    def posterior_hash(self) -> str:
        return stable_hash([item.to_value() for item in self.calibration_cells])

    @property
    def content_hash(self) -> str:
        return stable_hash(
            {
                "calibration_cells": [item.to_value() for item in self.calibration_cells],
                "final_training_artifact": self.final_training_artifact.to_value(),
                "evolution_counts": self.evolution_counts.to_value(),
                "format": self.format,
                "library": self.library.to_value(),
                "method_identity_hash": self.method_identity_hash,
                "policy_snapshot_hash": self.policy_snapshot_hash,
                "source_exact_input_sha256": self.source_exact_input_sha256,
                "source_formal_run_group_id": self.source_formal_run_group_id,
                "source_training_attempt_id": self.source_training_attempt_id,
                "source_training_identity_hash": self.source_training_identity_hash,
                "source_training_outcome_sha256": self.source_training_outcome_sha256,
            }
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "calibration_cells": [item.to_value() for item in self.calibration_cells],
            "format": self.format,
            "final_training_artifact": self.final_training_artifact.to_value(),
            "evolution_counts": self.evolution_counts.to_value(),
            "library": self.library.to_value(),
            "method_identity_hash": self.method_identity_hash,
            "policy_snapshot_directory": self.policy_snapshot_directory.as_posix(),
            "policy_snapshot_hash": self.policy_snapshot_hash,
            "source_exact_input_sha256": self.source_exact_input_sha256,
            "source_formal_run_group_id": self.source_formal_run_group_id,
            "source_training_attempt_id": self.source_training_attempt_id,
            "source_training_identity_hash": self.source_training_identity_hash,
            "source_training_outcome_sha256": self.source_training_outcome_sha256,
        }

    @classmethod
    def _from_value(cls, value: object) -> FrozenInferenceState:
        data = normalize_json(value)
        fields = {
            "calibration_cells",
            "format",
            "final_training_artifact",
            "evolution_counts",
            "library",
            "method_identity_hash",
            "policy_snapshot_directory",
            "policy_snapshot_hash",
            "source_exact_input_sha256",
            "source_formal_run_group_id",
            "source_training_attempt_id",
            "source_training_identity_hash",
            "source_training_outcome_sha256",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("frozen inference state has incompatible fields")
        raw_cells = data["calibration_cells"]
        if not isinstance(raw_cells, list):
            raise ValueError("frozen inference state cells must be an array")
        if any(
            type(data[name]) is not str
            for name in fields
            - {"calibration_cells", "evolution_counts", "final_training_artifact", "library"}
        ):
            raise ValueError("frozen inference state text fields are invalid")
        return cls(
            policy_snapshot_directory=Path(data["policy_snapshot_directory"]),
            policy_snapshot_hash=data["policy_snapshot_hash"],
            final_training_artifact=FinalTrainingArtifact.from_value(
                data["final_training_artifact"]
            ),
            library=SkillLibraryState.from_value(data["library"]),
            calibration_cells=tuple(PosteriorCellState.from_value(item) for item in raw_cells),
            method_identity_hash=data["method_identity_hash"],
            source_training_attempt_id=data["source_training_attempt_id"],
            source_formal_run_group_id=data["source_formal_run_group_id"],
            source_training_identity_hash=data["source_training_identity_hash"],
            source_exact_input_sha256=data["source_exact_input_sha256"],
            source_training_outcome_sha256=data["source_training_outcome_sha256"],
            evolution_counts=TrainingEvolutionCounts.from_value(data["evolution_counts"]),
            state_format=data["format"],
            _admission=_FROZEN_STATE_ADMISSION,
        )


@dataclass(frozen=True, slots=True)
class FrozenEvaluationInput:
    """One no-update evaluation schedule pinned to a frozen inference state."""

    state: FrozenInferenceState
    task_sequence: PrivateFrozenTaskSequence
    format: str = FROZEN_EVALUATION_INPUT_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.state, FrozenInferenceState):
            raise TypeError("frozen evaluation input requires FrozenInferenceState")
        if not isinstance(self.task_sequence, PrivateFrozenTaskSequence):
            raise TypeError("frozen evaluation input requires PrivateFrozenTaskSequence")
        if self.task_sequence.identity.purpose not in {
            SchedulePurpose.IID_PROGRESS,
            SchedulePurpose.IID_EVALUATION,
            SchedulePurpose.OOD_EVALUATION,
        }:
            raise ValueError("frozen evaluation cannot consume the optimizer training schedule")
        if self.format != FROZEN_EVALUATION_INPUT_FORMAT:
            raise ValueError("unsupported frozen evaluation input format")

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
    def from_value(cls, value: object) -> FrozenEvaluationInput:
        data = normalize_json(value)
        if not isinstance(data, dict) or set(data) != {"format", "state", "task_sequence"}:
            raise ValueError("frozen evaluation input has incompatible fields")
        if type(data["format"]) is not str:
            raise ValueError("frozen evaluation input format must be text")
        return cls(
            state=FrozenInferenceState._from_value(data["state"]),
            task_sequence=PrivateFrozenTaskSequence.from_value(data["task_sequence"]),
            format=data["format"],
        )


__all__ = [
    "FROZEN_EVALUATION_INPUT_FORMAT",
    "FROZEN_INFERENCE_STATE_FORMAT",
    "FinalTrainingArtifactResolver",
    "FrozenEvaluationInput",
    "FrozenInferenceState",
]
