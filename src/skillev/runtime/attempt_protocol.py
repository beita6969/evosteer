"""Closed IPC contracts for one non-retryable, publication-complete attempt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, TypeAlias, TypeVar

from skillev.contracts import JsonValue, TrainingStepReportValue, canonical_json, normalize_json

ATTEMPT_IPC_FORMAT = "skillev-attempt-ipc@5"
FINAL_TRAINING_ARTIFACT_FORMAT = "skillev-final-training-artifact@1"
FORMAL_PUBLICATION_ADMISSION_FORMAT = "skillev-formal-publication-admission@1"


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if type(normalized) is not dict or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _path_text(value: object, *, field: str) -> Path:
    return Path(_text(value, field=field))


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    prefix, separator, digest = text.partition(":")
    if prefix != "sha256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"{field} must be a sha256 identity")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a sha256 identity") from error
    return text


class AttemptBuilderKind(StrEnum):
    """The preregistered application graphs admitted by the worker."""

    FULL = "full"
    NO_BAYESIAN = "no-bayesian"
    UNIT_FLOW = "unit-flow"
    CAPPED_FLOW = "capped-flow"
    CLIPPED_IMPORTANCE = "clipped-importance"
    POSTERIOR_MEAN = "posterior-mean"
    RESIDUAL_ONLY_PHASE = "residual-only-phase"


@dataclass(frozen=True, slots=True)
class FormalPublicationAdmission:
    """Terminal-backed provenance required before a formal bundle is published.

    The formal supervisor creates this only after the manifest-selected child
    has written a durable terminal.  It deliberately contains no paths or
    secret claim token, so the published bundle can be independently joined
    to the formal ledger without exposing private launch material.
    """

    run_group_id: str
    manifest_content_hash: str
    slot_builder_kind: AttemptBuilderKind
    slot_attempt_id: str
    slot_exact_input_sha256: str
    slot_public_identity_content_hash: str
    claim_token_sha256: str
    terminal_outcome_sha256: str
    format: str = FORMAL_PUBLICATION_ADMISSION_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "run_group_id",
            "manifest_content_hash",
            "slot_exact_input_sha256",
            "slot_public_identity_content_hash",
            "claim_token_sha256",
            "terminal_outcome_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        if not isinstance(self.slot_builder_kind, AttemptBuilderKind):
            raise TypeError("formal publication admission requires a closed builder kind")
        _text(self.slot_attempt_id, field="formal publication slot attempt ID")
        if self.format != FORMAL_PUBLICATION_ADMISSION_FORMAT:
            raise ValueError("unsupported formal publication admission format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "claim_token_sha256": self.claim_token_sha256,
            "format": self.format,
            "manifest_content_hash": self.manifest_content_hash,
            "run_group_id": self.run_group_id,
            "slot_attempt_id": self.slot_attempt_id,
            "slot_builder_kind": self.slot_builder_kind.value,
            "slot_exact_input_sha256": self.slot_exact_input_sha256,
            "slot_public_identity_content_hash": self.slot_public_identity_content_hash,
            "terminal_outcome_sha256": self.terminal_outcome_sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalPublicationAdmission:
        data = _object(
            value,
            fields={
                "claim_token_sha256",
                "format",
                "manifest_content_hash",
                "run_group_id",
                "slot_attempt_id",
                "slot_builder_kind",
                "slot_exact_input_sha256",
                "slot_public_identity_content_hash",
                "terminal_outcome_sha256",
            },
            label="FormalPublicationAdmission",
        )
        if any(type(data[field]) is not str for field in data):
            raise TypeError("formal publication admission fields must be text")
        return cls(
            run_group_id=data["run_group_id"],
            manifest_content_hash=data["manifest_content_hash"],
            slot_builder_kind=AttemptBuilderKind(data["slot_builder_kind"]),
            slot_attempt_id=data["slot_attempt_id"],
            slot_exact_input_sha256=data["slot_exact_input_sha256"],
            slot_public_identity_content_hash=data["slot_public_identity_content_hash"],
            claim_token_sha256=data["claim_token_sha256"],
            terminal_outcome_sha256=data["terminal_outcome_sha256"],
            format=data["format"],
        )


class AttemptSourceLogKind(StrEnum):
    """The semantic role of one hash-covered published source file."""

    FULL_METHOD = "full-method-source"
    FLOW_ONLY = "flow-only-source"
    OPERATIONAL = "operational-events"


@dataclass(frozen=True, slots=True)
class AttemptSourceLogDigest:
    name: str
    kind: AttemptSourceLogKind
    sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or Path(self.name).name != self.name:
            raise ValueError("source log name must be one non-empty path component")
        if not isinstance(self.kind, AttemptSourceLogKind):
            raise TypeError("source log kind must be AttemptSourceLogKind")
        _sha256(self.sha256, field="source log sha256")

    def to_value(self) -> dict[str, JsonValue]:
        return {"kind": self.kind.value, "name": self.name, "sha256": self.sha256}

    @classmethod
    def from_value(cls, value: object) -> AttemptSourceLogDigest:
        data = _object(value, fields={"kind", "name", "sha256"}, label="source log digest")
        return cls(
            name=_text(data["name"], field="source log name"),
            kind=AttemptSourceLogKind(_text(data["kind"], field="source log kind")),
            sha256=_sha256(data["sha256"], field="source log sha256"),
        )


def expected_source_log_layout(
    kind: AttemptBuilderKind,
) -> tuple[tuple[str, AttemptSourceLogKind], ...]:
    if kind is AttemptBuilderKind.NO_BAYESIAN:
        return (
            ("arm-events.jsonl", AttemptSourceLogKind.FLOW_ONLY),
            ("events.jsonl", AttemptSourceLogKind.OPERATIONAL),
        )
    return (("events.jsonl", AttemptSourceLogKind.FULL_METHOD),)


def validate_source_log_layout(
    builder_kind: AttemptBuilderKind,
    logs: tuple[AttemptSourceLogDigest, ...],
) -> None:
    expected = expected_source_log_layout(builder_kind)
    actual = tuple((item.name, item.kind) for item in logs)
    if actual != expected:
        raise ValueError("source log layout differs from builder kind")
    if len({item.name for item in logs}) != len(logs):
        raise ValueError("source log layout repeats a filename")


class AttemptFailureStage(StrEnum):
    BUILD = "build"
    EXECUTION = "execution"
    TERMINAL_EVALUATION = "terminal-evaluation"
    TTB_SCORING = "ttb-scoring"
    BUDGET_FINALIZATION = "budget-finalization"
    SUMMARY = "summary"
    PUBLICATION = "publication"
    INTERNAL = "internal"


class AttemptFailureCode(StrEnum):
    INVALID_EXACT_INPUT = "invalid-exact-input"
    BUDGET_EXCEEDED = "budget-exceeded"
    ROLLOUT_INFRASTRUCTURE_FAILED = "rollout-infrastructure-failed"
    TERMINAL_EVALUATOR_FAILED = "terminal-evaluator-failed"
    TRAINING_MEMORY_EXHAUSTED = "training-memory-exhausted"
    AUTHORING_FAILED = "authoring-failed"
    MUTATION_FAILED = "mutation-failed"
    LIBRARY_APPLY_FAILED = "library-apply-failed"
    PARTITION_RESET_FAILED = "partition-reset-failed"
    EVENT_APPEND_FAILED = "event-append-failed"
    INTERNAL_ATTEMPT_FAILURE = "internal-attempt-failure"


_FAILURE_PUBLIC_TEXT: dict[AttemptFailureCode, str] = {
    AttemptFailureCode.INVALID_EXACT_INPUT: "attempt input is invalid",
    AttemptFailureCode.BUDGET_EXCEEDED: "attempt budget is insufficient",
    AttemptFailureCode.ROLLOUT_INFRASTRUCTURE_FAILED: "rollout infrastructure failed",
    AttemptFailureCode.TERMINAL_EVALUATOR_FAILED: "terminal evaluator failed",
    AttemptFailureCode.TRAINING_MEMORY_EXHAUSTED: "training scoring exhausted device memory",
    AttemptFailureCode.AUTHORING_FAILED: "skill authoring failed",
    AttemptFailureCode.MUTATION_FAILED: "skill mutation failed",
    AttemptFailureCode.LIBRARY_APPLY_FAILED: "skill library update failed",
    AttemptFailureCode.PARTITION_RESET_FAILED: "partition reset failed",
    AttemptFailureCode.EVENT_APPEND_FAILED: "attempt event append failed",
    AttemptFailureCode.INTERNAL_ATTEMPT_FAILURE: "internal attempt failure",
}


_SUMMARY_FIELDS = {
    "actions_committed_this_attempt",
    "completed_training_steps_this_attempt",
    "cycles_committed_in_run",
    "cycles_committed_this_attempt",
    "final_library_version",
    "final_optimizer_step",
    "final_policy_snapshot_id",
    "initial_optimizer_step",
    "kind",
    "planned_training_steps_this_attempt",
    "reports",
}


@dataclass(frozen=True, slots=True)
class FullAttemptSummary:
    reports: tuple[TrainingStepReportValue, ...]
    planned_training_steps_this_attempt: int
    completed_training_steps_this_attempt: int
    actions_committed_this_attempt: int
    cycles_committed_this_attempt: int
    cycles_committed_in_run: int
    initial_optimizer_step: int
    final_optimizer_step: int
    final_library_version: str
    final_policy_snapshot_id: str
    kind: ClassVar[str] = "full"

    def __post_init__(self) -> None:
        _validate_summary(self)

    def to_value(self) -> dict[str, JsonValue]:
        return _summary_to_value(self)

    @classmethod
    def from_value(cls, value: object) -> FullAttemptSummary:
        return _summary_from_value(cls, value)


@dataclass(frozen=True, slots=True)
class FlowOnlyAttemptSummary:
    reports: tuple[TrainingStepReportValue, ...]
    planned_training_steps_this_attempt: int
    completed_training_steps_this_attempt: int
    actions_committed_this_attempt: int
    cycles_committed_this_attempt: int
    cycles_committed_in_run: int
    initial_optimizer_step: int
    final_optimizer_step: int
    final_library_version: str
    final_policy_snapshot_id: str
    kind: ClassVar[str] = "flow-only"

    def __post_init__(self) -> None:
        _validate_summary(self)

    def to_value(self) -> dict[str, JsonValue]:
        return _summary_to_value(self)

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyAttemptSummary:
        return _summary_from_value(cls, value)


AttemptRunSummary: TypeAlias = FullAttemptSummary | FlowOnlyAttemptSummary
SummaryT = TypeVar("SummaryT", FullAttemptSummary, FlowOnlyAttemptSummary)


def _validate_summary(summary: AttemptRunSummary) -> None:
    for field in (
        "planned_training_steps_this_attempt",
        "completed_training_steps_this_attempt",
        "actions_committed_this_attempt",
        "cycles_committed_this_attempt",
        "cycles_committed_in_run",
        "initial_optimizer_step",
        "final_optimizer_step",
    ):
        _non_negative_int(getattr(summary, field), field=field)
    if summary.completed_training_steps_this_attempt != len(summary.reports):
        raise ValueError("completed step count differs from reports")
    if summary.completed_training_steps_this_attempt != summary.planned_training_steps_this_attempt:
        raise ValueError("successful attempt did not complete its exact run plan")
    if (
        summary.final_optimizer_step - summary.initial_optimizer_step
        != summary.completed_training_steps_this_attempt
    ):
        raise ValueError("summary optimizer-step delta differs from completed steps")
    if summary.cycles_committed_this_attempt > summary.cycles_committed_in_run:
        raise ValueError("attempt cycles cannot exceed run-total cycles")
    if summary.actions_committed_this_attempt < summary.cycles_committed_this_attempt:
        raise ValueError("each committed cycle requires at least one action")
    _text(summary.final_library_version, field="final_library_version")
    _text(summary.final_policy_snapshot_id, field="final_policy_snapshot_id")


def _summary_to_value(summary: AttemptRunSummary) -> dict[str, JsonValue]:
    return {
        "actions_committed_this_attempt": summary.actions_committed_this_attempt,
        "completed_training_steps_this_attempt": summary.completed_training_steps_this_attempt,
        "cycles_committed_in_run": summary.cycles_committed_in_run,
        "cycles_committed_this_attempt": summary.cycles_committed_this_attempt,
        "final_library_version": summary.final_library_version,
        "final_optimizer_step": summary.final_optimizer_step,
        "final_policy_snapshot_id": summary.final_policy_snapshot_id,
        "initial_optimizer_step": summary.initial_optimizer_step,
        "kind": summary.kind,
        "planned_training_steps_this_attempt": summary.planned_training_steps_this_attempt,
        "reports": [report.to_value() for report in summary.reports],
    }


def _summary_from_value(
    summary_type: type[SummaryT],
    value: object,
) -> SummaryT:
    data = _object(value, fields=_SUMMARY_FIELDS, label=summary_type.__name__)
    if data["kind"] != summary_type.kind:
        raise ValueError(f"{summary_type.__name__} has the wrong kind")
    reports = data["reports"]
    if type(reports) is not list:
        raise TypeError("summary reports must be an array")
    return summary_type(
        reports=tuple(TrainingStepReportValue.from_value(item) for item in reports),
        planned_training_steps_this_attempt=_non_negative_int(
            data["planned_training_steps_this_attempt"],
            field="planned_training_steps_this_attempt",
        ),
        completed_training_steps_this_attempt=_non_negative_int(
            data["completed_training_steps_this_attempt"],
            field="completed_training_steps_this_attempt",
        ),
        actions_committed_this_attempt=_non_negative_int(
            data["actions_committed_this_attempt"], field="actions_committed_this_attempt"
        ),
        cycles_committed_this_attempt=_non_negative_int(
            data["cycles_committed_this_attempt"], field="cycles_committed_this_attempt"
        ),
        cycles_committed_in_run=_non_negative_int(
            data["cycles_committed_in_run"], field="cycles_committed_in_run"
        ),
        initial_optimizer_step=_non_negative_int(
            data["initial_optimizer_step"], field="initial_optimizer_step"
        ),
        final_optimizer_step=_non_negative_int(
            data["final_optimizer_step"], field="final_optimizer_step"
        ),
        final_library_version=_text(data["final_library_version"], field="final_library_version"),
        final_policy_snapshot_id=_text(
            data["final_policy_snapshot_id"], field="final_policy_snapshot_id"
        ),
    )


def attempt_summary_from_value(value: object) -> AttemptRunSummary:
    if type(value) is not dict:
        raise TypeError("attempt summary must be an object")
    kind = value.get("kind")
    if kind == FullAttemptSummary.kind:
        return FullAttemptSummary.from_value(value)
    if kind == FlowOnlyAttemptSummary.kind:
        return FlowOnlyAttemptSummary.from_value(value)
    raise ValueError("unsupported attempt summary kind")


@dataclass(frozen=True, slots=True)
class AttemptRequest:
    run_id: str
    attempt_id: str
    builder_kind: AttemptBuilderKind
    exact_input_path: Path
    exact_input_sha256: str
    private_bundle_directory: Path
    request_type: ClassVar[str] = "attempt-request"
    format: str = ATTEMPT_IPC_FORMAT

    def __post_init__(self) -> None:
        _text(self.run_id, field="run_id")
        _text(self.attempt_id, field="attempt_id")
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("builder_kind must be AttemptBuilderKind")
        _sha256(self.exact_input_sha256, field="exact_input_sha256")
        if self.format != ATTEMPT_IPC_FORMAT:
            raise ValueError("unsupported attempt IPC format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "exact_input_path": str(self.exact_input_path),
            "exact_input_sha256": self.exact_input_sha256,
            "format": self.format,
            "private_bundle_directory": str(self.private_bundle_directory),
            "request_type": self.request_type,
            "run_id": self.run_id,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> AttemptRequest:
        data = _object(
            value,
            fields={
                "attempt_id",
                "builder_kind",
                "exact_input_path",
                "exact_input_sha256",
                "format",
                "private_bundle_directory",
                "request_type",
                "run_id",
            },
            label="AttemptRequest",
        )
        if data["request_type"] != cls.request_type:
            raise ValueError("IPC object is not an attempt request")
        return cls(
            run_id=_text(data["run_id"], field="run_id"),
            attempt_id=_text(data["attempt_id"], field="attempt_id"),
            builder_kind=AttemptBuilderKind(_text(data["builder_kind"], field="builder_kind")),
            exact_input_path=_path_text(data["exact_input_path"], field="exact_input_path"),
            exact_input_sha256=_sha256(data["exact_input_sha256"], field="exact_input_sha256"),
            private_bundle_directory=_path_text(
                data["private_bundle_directory"], field="private_bundle_directory"
            ),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FinalTrainingArtifact:
    """Path-free identity of the mandatory final private training snapshot."""

    artifact_sha256: str
    runtime_state_sha256: str
    policy_snapshot_id: str
    library_version: str
    optimizer_step: int
    format: str = FINAL_TRAINING_ARTIFACT_FORMAT

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, field="final artifact sha256")
        _sha256(self.runtime_state_sha256, field="final runtime state sha256")
        _text(self.policy_snapshot_id, field="final policy snapshot ID")
        _text(self.library_version, field="final library version")
        _non_negative_int(self.optimizer_step, field="final optimizer step")
        if self.format != FINAL_TRAINING_ARTIFACT_FORMAT:
            raise ValueError("unsupported final training artifact format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "format": self.format,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_id": self.policy_snapshot_id,
            "runtime_state_sha256": self.runtime_state_sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> FinalTrainingArtifact:
        data = _object(
            value,
            fields={
                "artifact_sha256",
                "format",
                "library_version",
                "optimizer_step",
                "policy_snapshot_id",
                "runtime_state_sha256",
            },
            label="FinalTrainingArtifact",
        )
        return cls(
            artifact_sha256=_sha256(data["artifact_sha256"], field="final artifact sha256"),
            runtime_state_sha256=_sha256(
                data["runtime_state_sha256"], field="final runtime state sha256"
            ),
            policy_snapshot_id=_text(data["policy_snapshot_id"], field="final policy snapshot ID"),
            library_version=_text(data["library_version"], field="final library version"),
            optimizer_step=_non_negative_int(data["optimizer_step"], field="final optimizer step"),
            format=_text(data["format"], field="final training artifact format"),
        )


@dataclass(frozen=True, slots=True)
class AttemptSucceeded:
    attempt_id: str
    builder_kind: AttemptBuilderKind
    exact_input_sha256: str
    public_identity_sha256: str
    public_identity_content_hash: str
    source_logs: tuple[AttemptSourceLogDigest, ...]
    summary: AttemptRunSummary
    final_training_artifact: FinalTrainingArtifact
    formal_run_group_id: str | None = None
    outcome_type: ClassVar[str] = "succeeded"
    format: str = ATTEMPT_IPC_FORMAT

    def __post_init__(self) -> None:
        _text(self.attempt_id, field="attempt_id")
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("builder_kind must be AttemptBuilderKind")
        _sha256(self.exact_input_sha256, field="exact_input_sha256")
        _sha256(self.public_identity_sha256, field="public_identity_sha256")
        _sha256(self.public_identity_content_hash, field="public_identity_content_hash")
        if self.formal_run_group_id is not None:
            _sha256(self.formal_run_group_id, field="formal_run_group_id")
        if not isinstance(self.summary, FullAttemptSummary | FlowOnlyAttemptSummary):
            raise TypeError("summary must be an AttemptRunSummary")
        if not isinstance(self.final_training_artifact, FinalTrainingArtifact):
            raise TypeError("successful attempt requires its final training artifact")
        validate_source_log_layout(self.builder_kind, self.source_logs)
        expected_summary_kind = (
            FlowOnlyAttemptSummary.kind
            if self.builder_kind is AttemptBuilderKind.NO_BAYESIAN
            else FullAttemptSummary.kind
        )
        if self.summary.kind != expected_summary_kind:
            raise ValueError("builder kind and attempt summary kind differ")
        if (
            self.final_training_artifact.policy_snapshot_id != self.summary.final_policy_snapshot_id
            or self.final_training_artifact.library_version != self.summary.final_library_version
            or self.final_training_artifact.optimizer_step != self.summary.final_optimizer_step
        ):
            raise ValueError("final training artifact differs from successful summary")
        if self.format != ATTEMPT_IPC_FORMAT:
            raise ValueError("unsupported attempt IPC format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "exact_input_sha256": self.exact_input_sha256,
            "final_training_artifact": self.final_training_artifact.to_value(),
            "format": self.format,
            "formal_run_group_id": self.formal_run_group_id,
            "outcome_type": self.outcome_type,
            "public_identity_content_hash": self.public_identity_content_hash,
            "public_identity_sha256": self.public_identity_sha256,
            "source_logs": [item.to_value() for item in self.source_logs],
            "summary": self.summary.to_value(),
        }


@dataclass(frozen=True, slots=True)
class AttemptFailed:
    attempt_id: str
    builder_kind: AttemptBuilderKind
    exact_input_sha256: str
    code: AttemptFailureCode
    stage: AttemptFailureStage
    exception_type: str
    outcome_type: ClassVar[str] = "failed"
    format: str = ATTEMPT_IPC_FORMAT

    def __post_init__(self) -> None:
        _text(self.attempt_id, field="attempt_id")
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("builder_kind must be AttemptBuilderKind")
        _sha256(self.exact_input_sha256, field="exact_input_sha256")
        if not isinstance(self.code, AttemptFailureCode):
            raise TypeError("code must be AttemptFailureCode")
        if not isinstance(self.stage, AttemptFailureStage):
            raise TypeError("stage must be AttemptFailureStage")
        _text(self.exception_type, field="exception_type")
        if self.format != ATTEMPT_IPC_FORMAT:
            raise ValueError("unsupported attempt IPC format")

    @property
    def public_message(self) -> str:
        return _FAILURE_PUBLIC_TEXT[self.code]

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "code": self.code.value,
            "exact_input_sha256": self.exact_input_sha256,
            "exception_type": self.exception_type,
            "format": self.format,
            "outcome_type": self.outcome_type,
            "public_message": self.public_message,
            "stage": self.stage.value,
        }


AttemptOutcome: TypeAlias = AttemptSucceeded | AttemptFailed


def attempt_outcome_from_value(value: object) -> AttemptOutcome:
    if type(value) is not dict:
        raise TypeError("attempt outcome must be an object")
    outcome_type = value.get("outcome_type")
    if outcome_type == AttemptSucceeded.outcome_type:
        data = _object(
            value,
            fields={
                "attempt_id",
                "builder_kind",
                "exact_input_sha256",
                "final_training_artifact",
                "format",
                "formal_run_group_id",
                "outcome_type",
                "public_identity_content_hash",
                "public_identity_sha256",
                "source_logs",
                "summary",
            },
            label="AttemptSucceeded",
        )
        raw_logs = data["source_logs"]
        if type(raw_logs) is not list:
            raise TypeError("source_logs must be an array")
        if data["formal_run_group_id"] is not None and type(data["formal_run_group_id"]) is not str:
            raise TypeError("formal_run_group_id must be text or null")
        return AttemptSucceeded(
            attempt_id=_text(data["attempt_id"], field="attempt_id"),
            builder_kind=AttemptBuilderKind(_text(data["builder_kind"], field="builder_kind")),
            exact_input_sha256=_sha256(data["exact_input_sha256"], field="exact_input_sha256"),
            public_identity_sha256=_sha256(
                data["public_identity_sha256"], field="public_identity_sha256"
            ),
            public_identity_content_hash=_sha256(
                data["public_identity_content_hash"], field="public_identity_content_hash"
            ),
            source_logs=tuple(AttemptSourceLogDigest.from_value(item) for item in raw_logs),
            summary=attempt_summary_from_value(data["summary"]),
            final_training_artifact=FinalTrainingArtifact.from_value(
                data["final_training_artifact"]
            ),
            formal_run_group_id=data["formal_run_group_id"],
            format=_text(data["format"], field="format"),
        )
    if outcome_type == AttemptFailed.outcome_type:
        data = _object(
            value,
            fields={
                "attempt_id",
                "builder_kind",
                "code",
                "exact_input_sha256",
                "exception_type",
                "format",
                "outcome_type",
                "public_message",
                "stage",
            },
            label="AttemptFailed",
        )
        code = AttemptFailureCode(_text(data["code"], field="code"))
        if data["public_message"] != _FAILURE_PUBLIC_TEXT[code]:
            raise ValueError("public failure text differs from its code")
        return AttemptFailed(
            attempt_id=_text(data["attempt_id"], field="attempt_id"),
            builder_kind=AttemptBuilderKind(_text(data["builder_kind"], field="builder_kind")),
            exact_input_sha256=_sha256(data["exact_input_sha256"], field="exact_input_sha256"),
            code=code,
            stage=AttemptFailureStage(_text(data["stage"], field="stage")),
            exception_type=_text(data["exception_type"], field="exception_type"),
            format=_text(data["format"], field="format"),
        )
    raise ValueError("unsupported attempt outcome type")


def read_attempt_outcome(path: Path) -> AttemptOutcome:
    return attempt_outcome_from_value(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "ATTEMPT_IPC_FORMAT",
    "FINAL_TRAINING_ARTIFACT_FORMAT",
    "FORMAL_PUBLICATION_ADMISSION_FORMAT",
    "AttemptBuilderKind",
    "AttemptFailed",
    "AttemptFailureCode",
    "AttemptFailureStage",
    "AttemptOutcome",
    "AttemptRequest",
    "AttemptRunSummary",
    "AttemptSourceLogDigest",
    "AttemptSourceLogKind",
    "AttemptSucceeded",
    "FinalTrainingArtifact",
    "FlowOnlyAttemptSummary",
    "FormalPublicationAdmission",
    "FullAttemptSummary",
    "attempt_outcome_from_value",
    "attempt_summary_from_value",
    "expected_source_log_layout",
    "read_attempt_outcome",
    "validate_source_log_layout",
]
