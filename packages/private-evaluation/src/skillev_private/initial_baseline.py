"""Private admission for the exact pre-training evaluation state.

The IID/OOD baseline is not a shortened training attempt and never borrows a
later checkpoint.  It is the exact formal initial LoRA/Z partition, the
formal seed library, and the Beta prior (therefore no observed posterior
cells).  The public identity carries hashes only; the checkpoint location
remains private.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillev.audit.published_identity import read_published_attempt_identity
from skillev.contracts import JsonValue, PosteriorCellState, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256
from skillev.experiments import (
    AttemptPurpose,
    FormalRunLedger,
    FrozenTaskSequenceIdentity,
    PublishedAttemptIdentity,
    SchedulePurpose,
    TrainingEvolutionCounts,
)
from skillev.policy import PrivateInitialCheckpointBinding, TrainableStateIdentity
from skillev.rollout import PolicySnapshot
from skillev.runtime import PublishedSuccessfulAttemptBundle, SkillDocument, SkillLibraryState
from skillev.runtime.attempt_publication import sha256_tree
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence

INITIAL_BASELINE_INFERENCE_STATE_FORMAT = "skillev-initial-baseline-inference-state@2"
INITIAL_BASELINE_EVALUATION_INPUT_FORMAT = "skillev-private-initial-baseline-input@2"
_LOCAL_POLICY_BACKEND_ID = "hf-local"
_INITIAL_BASELINE_ADMISSION = object()


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _initial_policy_snapshot(
    *,
    checkpoint: PrivateInitialCheckpointBinding,
    source_identity: PublishedAttemptIdentity,
) -> PolicySnapshot:
    """Read the light checkpoint metadata required to pin initial rollout bytes."""

    path = checkpoint.path.resolve() / "policy_state.json"
    data = normalize_json(json.loads(path.read_text(encoding="utf-8")))
    expected = {
        "backbone_id",
        "backward_version",
        "format",
        "forward_version",
        "optimizer_step",
        "trainable_state",
        "z_version",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("initial policy checkpoint state has incompatible fields")
    if type(data["optimizer_step"]) is not int or data["optimizer_step"] != 0:
        raise ValueError("initial policy checkpoint must have optimizer step zero")
    if any(
        type(data[field]) is not str
        for field in ("backbone_id", "backward_version", "format", "forward_version", "z_version")
    ):
        raise TypeError("initial policy checkpoint state text fields are invalid")
    checkpoint_state = TrainableStateIdentity.from_value(data["trainable_state"])
    if checkpoint_state != checkpoint.trainable_state:
        raise ValueError("initial policy checkpoint metadata differs from its binding")
    if data["backbone_id"] != source_identity.backbone_deployment_hash:
        raise ValueError("initial policy checkpoint targets another formal deployment")
    return PolicySnapshot.create(
        backbone_id=data["backbone_id"],
        forward_adapter_version=data["forward_version"],
        tokenizer_id=source_identity.tokenizer_identity.tokenizer_id,
        backend_id=_LOCAL_POLICY_BACKEND_ID,
        initial_trainable_state_hash=checkpoint.trainable_state.content_hash,
    )


@dataclass(frozen=True, slots=True, init=False)
class InitialBaselineInferenceState:
    """The only formal frozen state allowed before any optimizer step."""

    policy_snapshot_directory: Path
    policy_snapshot_hash: str
    policy_snapshot: PolicySnapshot
    initial_trainable_state: TrainableStateIdentity
    library: SkillLibraryState
    calibration_cells: tuple[PosteriorCellState, ...]
    method_identity_hash: str
    source_attempt_id: str
    source_formal_run_group_id: str
    source_identity_hash: str
    source_exact_input_sha256: str
    format: str = INITIAL_BASELINE_INFERENCE_STATE_FORMAT

    def __init__(
        self,
        *,
        policy_snapshot_directory: Path,
        policy_snapshot_hash: str,
        policy_snapshot: PolicySnapshot,
        initial_trainable_state: TrainableStateIdentity,
        library: SkillLibraryState,
        calibration_cells: tuple[PosteriorCellState, ...],
        method_identity_hash: str,
        source_attempt_id: str,
        source_formal_run_group_id: str,
        source_identity_hash: str,
        source_exact_input_sha256: str,
        state_format: str = INITIAL_BASELINE_INFERENCE_STATE_FORMAT,
        _admission: object,
    ) -> None:
        if _admission is not _INITIAL_BASELINE_ADMISSION:
            raise TypeError("initial baseline state must come from a formal source admission")
        object.__setattr__(self, "policy_snapshot_directory", policy_snapshot_directory)
        object.__setattr__(self, "policy_snapshot_hash", policy_snapshot_hash)
        object.__setattr__(self, "policy_snapshot", policy_snapshot)
        object.__setattr__(self, "initial_trainable_state", initial_trainable_state)
        object.__setattr__(self, "library", library)
        object.__setattr__(self, "calibration_cells", calibration_cells)
        object.__setattr__(self, "method_identity_hash", method_identity_hash)
        object.__setattr__(self, "source_attempt_id", source_attempt_id)
        object.__setattr__(self, "source_formal_run_group_id", source_formal_run_group_id)
        object.__setattr__(self, "source_identity_hash", source_identity_hash)
        object.__setattr__(self, "source_exact_input_sha256", source_exact_input_sha256)
        object.__setattr__(self, "format", state_format)
        self._validate()

    def _validate(self) -> None:
        if (
            not isinstance(self.policy_snapshot_directory, Path)
            or not self.policy_snapshot_directory.is_absolute()
        ):
            raise ValueError("initial baseline checkpoint directory must be absolute")
        validate_sha256(self.policy_snapshot_hash)
        if not isinstance(self.policy_snapshot, PolicySnapshot):
            raise TypeError("initial baseline requires PolicySnapshot")
        if not isinstance(self.initial_trainable_state, TrainableStateIdentity):
            raise TypeError("initial baseline requires initial trainable state")
        if (
            self.policy_snapshot.backbone_id
            != self.initial_trainable_state.backbone_deployment_hash
        ):
            raise ValueError("initial baseline snapshot targets another deployment")
        if (
            self.policy_snapshot.initial_trainable_state_hash
            != self.initial_trainable_state.content_hash
        ):
            raise ValueError("initial baseline snapshot differs from initial trainable state")
        if not isinstance(self.library, SkillLibraryState):
            raise TypeError("initial baseline requires SkillLibraryState")
        if self.calibration_cells != ():
            raise ValueError("initial baseline must contain the unobserved Beta prior only")
        _text(self.method_identity_hash, field="method_identity_hash")
        _text(self.source_attempt_id, field="source_attempt_id")
        validate_sha256(self.source_formal_run_group_id)
        validate_sha256(self.source_identity_hash)
        validate_sha256(self.source_exact_input_sha256)
        if self.format != INITIAL_BASELINE_INFERENCE_STATE_FORMAT:
            raise ValueError("unsupported initial baseline inference state format")

    @classmethod
    def from_formal_input(
        cls,
        *,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        seed_documents: tuple[SkillDocument, ...],
        source_identity: PublishedAttemptIdentity,
        source_bundle: PublishedSuccessfulAttemptBundle,
        formal_run_ledger: FormalRunLedger,
    ) -> InitialBaselineInferenceState:
        """Admit the exact initial state from one formal private input."""

        if not isinstance(initial_checkpoint, PrivateInitialCheckpointBinding):
            raise TypeError("initial baseline requires a private initial checkpoint")
        if not isinstance(source_identity, PublishedAttemptIdentity):
            raise TypeError("initial baseline requires a published formal identity")
        if source_identity.purpose is not AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
            raise ValueError("initial baseline requires a formal benchmark training identity")
        if not isinstance(source_bundle, PublishedSuccessfulAttemptBundle):
            raise TypeError("initial baseline requires a published formal source bundle")
        if not isinstance(formal_run_ledger, FormalRunLedger):
            raise TypeError("initial baseline requires the formal run ledger")
        exact_bundle = PublishedSuccessfulAttemptBundle.open_exact(source_bundle.directory)
        formal_run_ledger.require_bundle_terminal_success(exact_bundle)
        if read_published_attempt_identity(exact_bundle) != source_identity:
            raise ValueError("initial baseline bundle differs from formal source identity")
        if (
            formal_run_ledger.manifest.protocol_hash != source_identity.protocol_hash
            or formal_run_ledger.manifest.protocol_freeze_id != source_identity.protocol_freeze_id
        ):
            raise ValueError("initial baseline manifest differs from the formal source")
        matching_slots = tuple(
            slot
            for slot in formal_run_ledger.manifest.slots
            if slot.builder_kind is source_identity.builder_kind
        )
        if len(matching_slots) != 1:
            raise ValueError("formal source has no unique run-manifest slot")
        source_slot = matching_slots[0]
        if (
            source_slot.exact_input_sha256 != source_identity.exact_input_sha256
            or source_slot.public_identity_content_hash != source_identity.content_hash
        ):
            raise ValueError("formal run-manifest slot differs from the formal source")
        if not isinstance(seed_documents, tuple) or any(
            not isinstance(item, SkillDocument) for item in seed_documents
        ):
            raise TypeError("initial baseline seed library is invalid")
        if source_identity.initial_trainable_state != initial_checkpoint.trainable_state:
            raise ValueError("initial baseline checkpoint differs from formal source identity")
        directory = initial_checkpoint.path.resolve()
        policy_snapshot = _initial_policy_snapshot(
            checkpoint=initial_checkpoint,
            source_identity=source_identity,
        )
        library = SkillLibraryState.from_seed_documents(seed_documents)
        if library.current_version != source_identity.initial_library_version:
            raise ValueError("initial baseline library differs from formal source identity")
        if library.state_hash != source_identity.initial_skill_library_state_hash:
            raise ValueError("initial baseline library state differs from formal source identity")
        return cls(
            policy_snapshot_directory=directory,
            policy_snapshot_hash=sha256_tree(directory),
            policy_snapshot=policy_snapshot,
            initial_trainable_state=initial_checkpoint.trainable_state,
            library=library,
            calibration_cells=(),
            method_identity_hash=source_identity.method_identity_hash,
            source_attempt_id=source_slot.attempt_id,
            source_formal_run_group_id=formal_run_ledger.manifest.run_group_id,
            source_identity_hash=source_identity.content_hash,
            source_exact_input_sha256=source_identity.exact_input_sha256,
            _admission=_INITIAL_BASELINE_ADMISSION,
        )

    def require_formal_initial(
        self,
        *,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        seed_documents: tuple[SkillDocument, ...],
        source_identity: PublishedAttemptIdentity,
        source_bundle: PublishedSuccessfulAttemptBundle,
        formal_run_ledger: FormalRunLedger,
    ) -> None:
        """Reject a baseline input that is not the exact formal initial state."""

        expected = type(self).from_formal_input(
            initial_checkpoint=initial_checkpoint,
            seed_documents=seed_documents,
            source_identity=source_identity,
            source_bundle=source_bundle,
            formal_run_ledger=formal_run_ledger,
        )
        if self != expected:
            raise ValueError("initial baseline state differs from the formal source input")

    @property
    def policy_snapshot_id(self) -> str:
        return self.policy_snapshot.snapshot_id

    @property
    def optimizer_step(self) -> int:
        return 0

    @property
    def evolution_counts(self) -> TrainingEvolutionCounts:
        """The formal baseline is evaluated before any training or evolution."""

        return TrainingEvolutionCounts(
            training_step_count=0,
            phase_count=0,
            cycle_count=0,
            action_count=0,
        )

    @property
    def source_training_attempt_id(self) -> str:
        """Use the source arm's preregistered slot in public aggregates."""

        return self.source_attempt_id

    @property
    def content_hash(self) -> str:
        return stable_hash(
            {
                "calibration_cells": [],
                "format": self.format,
                "initial_trainable_state": self.initial_trainable_state.to_value(),
                "library": self.library.to_value(),
                "method_identity_hash": self.method_identity_hash,
                "policy_snapshot": self.policy_snapshot.to_value(),
                "policy_snapshot_hash": self.policy_snapshot_hash,
                "source_attempt_id": self.source_attempt_id,
                "source_formal_run_group_id": self.source_formal_run_group_id,
                "source_exact_input_sha256": self.source_exact_input_sha256,
                "source_identity_hash": self.source_identity_hash,
            }
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "calibration_cells": [],
            "format": self.format,
            "initial_trainable_state": self.initial_trainable_state.to_value(),
            "library": self.library.to_value(),
            "method_identity_hash": self.method_identity_hash,
            "policy_snapshot": self.policy_snapshot.to_value(),
            "policy_snapshot_directory": self.policy_snapshot_directory.as_posix(),
            "policy_snapshot_hash": self.policy_snapshot_hash,
            "source_attempt_id": self.source_attempt_id,
            "source_formal_run_group_id": self.source_formal_run_group_id,
            "source_exact_input_sha256": self.source_exact_input_sha256,
            "source_identity_hash": self.source_identity_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> InitialBaselineInferenceState:
        data = normalize_json(value)
        fields = {
            "calibration_cells",
            "format",
            "initial_trainable_state",
            "library",
            "method_identity_hash",
            "policy_snapshot",
            "policy_snapshot_directory",
            "policy_snapshot_hash",
            "source_attempt_id",
            "source_formal_run_group_id",
            "source_exact_input_sha256",
            "source_identity_hash",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("initial baseline inference state has incompatible fields")
        if data["calibration_cells"] != []:
            raise ValueError("initial baseline cannot deserialize observed posterior cells")
        for field in (
            "format",
            "method_identity_hash",
            "policy_snapshot_directory",
            "policy_snapshot_hash",
            "source_attempt_id",
            "source_formal_run_group_id",
            "source_exact_input_sha256",
            "source_identity_hash",
        ):
            if type(data[field]) is not str:
                raise TypeError("initial baseline inference state text fields are invalid")
        return cls(
            policy_snapshot_directory=Path(data["policy_snapshot_directory"]),
            policy_snapshot_hash=data["policy_snapshot_hash"],
            policy_snapshot=PolicySnapshot.from_value(data["policy_snapshot"]),
            initial_trainable_state=TrainableStateIdentity.from_value(
                data["initial_trainable_state"]
            ),
            library=SkillLibraryState.from_value(data["library"]),
            calibration_cells=(),
            method_identity_hash=data["method_identity_hash"],
            source_attempt_id=data["source_attempt_id"],
            source_formal_run_group_id=data["source_formal_run_group_id"],
            source_identity_hash=data["source_identity_hash"],
            source_exact_input_sha256=data["source_exact_input_sha256"],
            state_format=data["format"],
            _admission=_INITIAL_BASELINE_ADMISSION,
        )


@dataclass(frozen=True, slots=True)
class InitialBaselineEvaluationInput:
    """One no-update schedule pinned to the formal pre-training state."""

    state: InitialBaselineInferenceState
    task_sequence: PrivateFrozenTaskSequence
    format: str = INITIAL_BASELINE_EVALUATION_INPUT_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.state, InitialBaselineInferenceState):
            raise TypeError("initial baseline input requires InitialBaselineInferenceState")
        if not isinstance(self.task_sequence, PrivateFrozenTaskSequence):
            raise TypeError("initial baseline input requires PrivateFrozenTaskSequence")
        if self.task_sequence.identity.purpose not in {
            SchedulePurpose.IID_EVALUATION,
            SchedulePurpose.OOD_EVALUATION,
        }:
            raise ValueError("initial baseline cannot consume training or progress schedules")
        if self.format != INITIAL_BASELINE_EVALUATION_INPUT_FORMAT:
            raise ValueError("unsupported initial baseline evaluation input format")

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
    def from_value(cls, value: object) -> InitialBaselineEvaluationInput:
        data = normalize_json(value)
        if not isinstance(data, dict) or set(data) != {"format", "state", "task_sequence"}:
            raise ValueError("initial baseline input has incompatible fields")
        if type(data["format"]) is not str:
            raise TypeError("initial baseline input format must be text")
        return cls(
            state=InitialBaselineInferenceState.from_value(data["state"]),
            task_sequence=PrivateFrozenTaskSequence.from_value(data["task_sequence"]),
            format=data["format"],
        )


__all__ = [
    "INITIAL_BASELINE_EVALUATION_INPUT_FORMAT",
    "INITIAL_BASELINE_INFERENCE_STATE_FORMAT",
    "InitialBaselineEvaluationInput",
    "InitialBaselineInferenceState",
]
