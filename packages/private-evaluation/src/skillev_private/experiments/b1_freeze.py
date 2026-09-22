"""Result-blind B1 composition over the already frozen B0 catalog.

This module only freezes inputs.  It does not claim a training/evaluation
slot, create a ledger, load a model, or execute a benchmark.  Private task
IDs remain in the private schedule files and exact attempt inputs; the summary
contains only their hashes and predeclared slot identities.

An evaluation's checkpoint bytes do not exist at B1 time.  B1 therefore
freezes an :class:`B1EvaluationSlotPlan` for every arm/kind/phase ordinal.
The plan deterministically binds to the existing state-specific
``FormalEvaluationSlot`` only after the corresponding frozen state exists.
This avoids inventing a checkpoint hash before training while fixing every
result-blind coordinate in advance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillev.contracts import (
    SCIENTIFIC_SAMPLING_ALGORITHM,
    JsonValue,
    canonical_json,
    normalize_json,
    scientific_sampling_schedule_hash,
    stable_hash,
    validate_sha256,
)
from skillev.experiments import (
    SCHEDULE_PURPOSES,
    ExperimentProtocol,
    FormalRunManifest,
    FormalRunSlot,
    FrozenTaskSequenceIdentity,
    ProtocolFreeze,
    SchedulePurpose,
)
from skillev.experiments.formal_run_manifest import FORMAL_RUN_MANIFEST_FORMAT
from skillev.runtime import AttemptBuilderKind
from skillev.runtime.attempt_publication import sha256_bytes
from skillev_private.benchmarks.catalog import PrivateBenchmarkCatalog
from skillev_private.benchmarks.curriculum import (
    PrivateFrozenTaskSequence,
    build_result_blind_sequence,
)

from .benchmark_attempt_input import PrivateBenchmarkAttemptInput
from .formal_evaluation_manifest import (
    FORMAL_EVALUATION_HARDWARE_POLICY_HASH,
    FormalEvaluationKind,
    FormalEvaluationSlot,
)

B1_EVALUATION_SLOT_PLAN_FORMAT = "skillev-b1-evaluation-slot-plan@3"
B1_EVALUATION_PLAN_FORMAT = "skillev-b1-evaluation-plan@3"
B1_FREEZE_SUMMARY_FORMAT = "skillev-b1-result-blind-freeze@3"
B1_RESULT_COUNTS_FORMAT = "skillev-b1-zero-result-counts@1"


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _non_negative(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _canonical_bytes(value: dict[str, JsonValue]) -> bytes:
    return canonical_json(value).encode("utf-8") + b"\n"


def _write_once(path: Path, value: dict[str, JsonValue]) -> str:
    encoded = _canonical_bytes(value)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(encoded)


def _expected_schedule_purpose(kind: FormalEvaluationKind) -> SchedulePurpose:
    if kind in (
        FormalEvaluationKind.INITIAL_IID,
        FormalEvaluationKind.PROGRESS_IID,
        FormalEvaluationKind.FINAL_IID,
    ):
        return SchedulePurpose.IID_EVALUATION
    if kind in (FormalEvaluationKind.INITIAL_OOD, FormalEvaluationKind.FINAL_OOD):
        return SchedulePurpose.OOD_EVALUATION
    raise ValueError("unsupported formal evaluation kind")


@dataclass(frozen=True, slots=True)
class B1ResultCounts:
    """Mechanical proof that B1 contains no formal outcome."""

    result_ingestion_count: int = 0
    training_terminal_count: int = 0
    evaluation_terminal_count: int = 0
    primary_aggregate_count: int = 0
    format: str = B1_RESULT_COUNTS_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "result_ingestion_count",
            "training_terminal_count",
            "evaluation_terminal_count",
            "primary_aggregate_count",
        ):
            if _non_negative(getattr(self, field), field=field) != 0:
                raise ValueError("B1 freeze must precede every formal result")
        if self.format != B1_RESULT_COUNTS_FORMAT:
            raise ValueError("unsupported B1 result-count format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evaluation_terminal_count": self.evaluation_terminal_count,
            "format": self.format,
            "primary_aggregate_count": self.primary_aggregate_count,
            "result_ingestion_count": self.result_ingestion_count,
            "training_terminal_count": self.training_terminal_count,
        }

    @classmethod
    def from_value(cls, value: object) -> B1ResultCounts:
        data = _object(
            value,
            fields={
                "evaluation_terminal_count",
                "format",
                "primary_aggregate_count",
                "result_ingestion_count",
                "training_terminal_count",
            },
            label="B1 result counts",
        )
        return cls(
            result_ingestion_count=_non_negative(
                data["result_ingestion_count"], field="result_ingestion_count"
            ),
            training_terminal_count=_non_negative(
                data["training_terminal_count"], field="training_terminal_count"
            ),
            evaluation_terminal_count=_non_negative(
                data["evaluation_terminal_count"], field="evaluation_terminal_count"
            ),
            primary_aggregate_count=_non_negative(
                data["primary_aggregate_count"], field="primary_aggregate_count"
            ),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class B1EvaluationSlotPlan:
    """One state-independent evaluation coordinate fixed at B1."""

    source_formal_run_group_id: str
    source_builder_kind: AttemptBuilderKind
    source_training_attempt_id: str
    source_training_identity_hash: str
    kind: FormalEvaluationKind
    anchor_ordinal: int
    task_sequence_identity: FrozenTaskSequenceIdentity
    sampling_schedule_hash: str
    implementation_build_hash: str
    execution_hardware_policy_hash: str
    slot_plan_id: str
    format: str = B1_EVALUATION_SLOT_PLAN_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "source_formal_run_group_id",
            "source_training_identity_hash",
            "sampling_schedule_hash",
            "implementation_build_hash",
            "execution_hardware_policy_hash",
            "slot_plan_id",
        ):
            validate_sha256(getattr(self, field))
        if not isinstance(self.source_builder_kind, AttemptBuilderKind):
            raise TypeError("B1 evaluation plan requires an attempt builder kind")
        _text(self.source_training_attempt_id, field="source_training_attempt_id")
        if not isinstance(self.kind, FormalEvaluationKind):
            raise TypeError("B1 evaluation plan requires FormalEvaluationKind")
        if type(self.anchor_ordinal) is not int or self.anchor_ordinal < 0:
            raise ValueError("B1 evaluation anchor ordinal must be non-negative")
        if self.kind is FormalEvaluationKind.PROGRESS_IID:
            if self.anchor_ordinal < 1:
                raise ValueError("progress evaluation requires a positive cycle ordinal")
        elif self.anchor_ordinal != 0:
            raise ValueError("non-progress evaluation cannot have a phase ordinal")
        if not isinstance(self.task_sequence_identity, FrozenTaskSequenceIdentity):
            raise TypeError("B1 evaluation plan requires a frozen task sequence")
        if self.task_sequence_identity.purpose is not _expected_schedule_purpose(self.kind):
            raise ValueError("B1 evaluation kind uses another task schedule")
        if self.execution_hardware_policy_hash != FORMAL_EVALUATION_HARDWARE_POLICY_HASH:
            raise ValueError("B1 evaluation hardware policy differs from formal evaluation")
        if self.format != B1_EVALUATION_SLOT_PLAN_FORMAT:
            raise ValueError("unsupported B1 evaluation slot-plan format")
        if self.slot_plan_id != stable_hash(self._identity_value()):
            raise ValueError("B1 evaluation slot-plan ID differs from its coordinate")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "anchor_ordinal": self.anchor_ordinal,
            "execution_hardware_policy_hash": self.execution_hardware_policy_hash,
            "format": self.format,
            "implementation_build_hash": self.implementation_build_hash,
            "kind": self.kind.value,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "source_builder_kind": self.source_builder_kind.value,
            "source_formal_run_group_id": self.source_formal_run_group_id,
            "source_training_attempt_id": self.source_training_attempt_id,
            "source_training_identity_hash": self.source_training_identity_hash,
            "task_sequence_identity": self.task_sequence_identity.to_value(),
        }

    @classmethod
    def create(cls, **values: object) -> B1EvaluationSlotPlan:
        wire = {
            **values,
            "format": B1_EVALUATION_SLOT_PLAN_FORMAT,
        }
        normalized = {
            key: (
                item.value
                if isinstance(item, AttemptBuilderKind | FormalEvaluationKind)
                else item.to_value()
                if isinstance(item, FrozenTaskSequenceIdentity)
                else item
            )
            for key, item in wire.items()
        }
        return cls(**values, slot_plan_id=stable_hash(normalized))  # type: ignore[arg-type]

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._identity_value(), "slot_plan_id": self.slot_plan_id}

    @classmethod
    def from_value(cls, value: object) -> B1EvaluationSlotPlan:
        fields = {
            "anchor_ordinal",
            "execution_hardware_policy_hash",
            "format",
            "implementation_build_hash",
            "kind",
            "sampling_schedule_hash",
            "slot_plan_id",
            "source_builder_kind",
            "source_formal_run_group_id",
            "source_training_attempt_id",
            "source_training_identity_hash",
            "task_sequence_identity",
        }
        data = _object(value, fields=fields, label="B1 evaluation slot plan")
        if type(data["anchor_ordinal"]) is not int:
            raise TypeError("B1 evaluation anchor ordinal must be an integer")
        text_fields = fields - {"anchor_ordinal", "task_sequence_identity"}
        for field in text_fields:
            _text(data[field], field=field)
        return cls(
            source_formal_run_group_id=data["source_formal_run_group_id"],
            source_builder_kind=AttemptBuilderKind(data["source_builder_kind"]),
            source_training_attempt_id=data["source_training_attempt_id"],
            source_training_identity_hash=data["source_training_identity_hash"],
            kind=FormalEvaluationKind(data["kind"]),
            anchor_ordinal=data["anchor_ordinal"],
            task_sequence_identity=FrozenTaskSequenceIdentity.from_value(
                data["task_sequence_identity"]
            ),
            sampling_schedule_hash=data["sampling_schedule_hash"],
            implementation_build_hash=data["implementation_build_hash"],
            execution_hardware_policy_hash=data["execution_hardware_policy_hash"],
            slot_plan_id=data["slot_plan_id"],
            format=data["format"],
        )

    def bind_state(
        self,
        *,
        frozen_state_hash: str,
        policy_snapshot_id: str,
        library_version: str,
    ) -> FormalEvaluationSlot:
        """Bind the preregistered coordinate to one admitted future state."""

        validate_sha256(frozen_state_hash)
        validate_sha256(library_version)
        _text(policy_snapshot_id, field="policy_snapshot_id")
        return FormalEvaluationSlot.create(
            source_formal_run_group_id=self.source_formal_run_group_id,
            source_builder_kind=self.source_builder_kind,
            source_training_attempt_id=self.source_training_attempt_id,
            source_training_identity_hash=self.source_training_identity_hash,
            kind=self.kind,
            anchor_ordinal=self.anchor_ordinal,
            frozen_state_hash=frozen_state_hash,
            policy_snapshot_id=policy_snapshot_id,
            library_version=library_version,
            task_sequence_identity=self.task_sequence_identity,
            sampling_schedule_hash=self.sampling_schedule_hash,
            implementation_build_hash=self.implementation_build_hash,
            execution_hardware_policy_hash=self.execution_hardware_policy_hash,
        )


@dataclass(frozen=True, slots=True)
class B1EvaluationPlan:
    protocol_hash: str
    protocol_freeze_id: str
    source_formal_run_group_id: str
    slots: tuple[B1EvaluationSlotPlan, ...]
    plan_id: str
    format: str = B1_EVALUATION_PLAN_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "protocol_hash",
            "protocol_freeze_id",
            "source_formal_run_group_id",
            "plan_id",
        ):
            validate_sha256(getattr(self, field))
        if len(self.slots) < 1 or any(
            not isinstance(item, B1EvaluationSlotPlan) for item in self.slots
        ):
            raise ValueError("B1 evaluation plan requires slot plans")
        if len({item.slot_plan_id for item in self.slots}) != len(self.slots):
            raise ValueError("B1 evaluation plan repeats a slot")
        if any(
            item.source_formal_run_group_id != self.source_formal_run_group_id
            for item in self.slots
        ):
            raise ValueError("B1 evaluation plans belong to another training group")
        if self.format != B1_EVALUATION_PLAN_FORMAT:
            raise ValueError("unsupported B1 evaluation-plan format")
        if self.plan_id != stable_hash(self._identity_value()):
            raise ValueError("B1 evaluation plan ID differs from its slots")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "slots": [item.to_value() for item in self.slots],
            "source_formal_run_group_id": self.source_formal_run_group_id,
        }

    @classmethod
    def create(
        cls,
        *,
        protocol_hash: str,
        protocol_freeze_id: str,
        source_formal_run_group_id: str,
        slots: tuple[B1EvaluationSlotPlan, ...],
    ) -> B1EvaluationPlan:
        identity = {
            "format": B1_EVALUATION_PLAN_FORMAT,
            "protocol_freeze_id": protocol_freeze_id,
            "protocol_hash": protocol_hash,
            "slots": [item.to_value() for item in slots],
            "source_formal_run_group_id": source_formal_run_group_id,
        }
        return cls(
            protocol_hash=protocol_hash,
            protocol_freeze_id=protocol_freeze_id,
            source_formal_run_group_id=source_formal_run_group_id,
            slots=slots,
            plan_id=stable_hash(identity),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._identity_value(), "plan_id": self.plan_id}

    @classmethod
    def from_value(cls, value: object) -> B1EvaluationPlan:
        data = _object(
            value,
            fields={
                "format",
                "plan_id",
                "protocol_freeze_id",
                "protocol_hash",
                "slots",
                "source_formal_run_group_id",
            },
            label="B1 evaluation plan",
        )
        raw_slots = data["slots"]
        if not isinstance(raw_slots, list):
            raise TypeError("B1 evaluation slots must be an array")
        return cls(
            protocol_hash=_text(data["protocol_hash"], field="protocol_hash"),
            protocol_freeze_id=_text(data["protocol_freeze_id"], field="protocol_freeze_id"),
            source_formal_run_group_id=_text(
                data["source_formal_run_group_id"], field="source_formal_run_group_id"
            ),
            slots=tuple(B1EvaluationSlotPlan.from_value(item) for item in raw_slots),
            plan_id=_text(data["plan_id"], field="plan_id"),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class B1FreezeSummary:
    b0_report_hash: str
    f2_f3_proof_hash: str
    b1_run_admission_hash: str
    exact_admission_report_hash: str
    source_commit: str
    formal_execution_hash: str
    base_model_artifact_hash: str
    tokenizer_artifact_hash: str
    implementation_build_hash: str
    initial_trainable_state_hash: str
    initial_library_version: str
    initial_skill_library_state_hash: str
    metric_reporting_protocol_hash: str
    scientific_sampling_schedule_hash: str
    protocol_hash: str
    protocol_freeze_id: str
    formal_run_group_id: str
    formal_run_manifest_hash: str
    schedule_content_hashes: tuple[str, ...]
    schedule_file_hashes: tuple[str, ...]
    exact_input_hashes: tuple[str, ...]
    training_slot_ids: tuple[str, ...]
    evaluation_plan_id: str
    evaluation_slot_plan_ids: tuple[str, ...]
    result_counts: B1ResultCounts
    summary_id: str
    format: str = B1_FREEZE_SUMMARY_FORMAT

    def __post_init__(self) -> None:
        hash_fields = (
            "b0_report_hash",
            "f2_f3_proof_hash",
            "b1_run_admission_hash",
            "exact_admission_report_hash",
            "formal_execution_hash",
            "base_model_artifact_hash",
            "tokenizer_artifact_hash",
            "implementation_build_hash",
            "initial_trainable_state_hash",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "metric_reporting_protocol_hash",
            "scientific_sampling_schedule_hash",
            "protocol_hash",
            "protocol_freeze_id",
            "formal_run_group_id",
            "formal_run_manifest_hash",
            "evaluation_plan_id",
            "summary_id",
        )
        for field in hash_fields:
            validate_sha256(getattr(self, field))
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("B1 source commit must be a full lowercase Git commit")
        for values in (
            self.schedule_content_hashes,
            self.schedule_file_hashes,
            self.exact_input_hashes,
            self.evaluation_slot_plan_ids,
        ):
            if not values:
                raise ValueError("B1 freeze summary contains an empty identity set")
            for item in values:
                validate_sha256(item)
        if len(self.schedule_content_hashes) != len(SCHEDULE_PURPOSES):
            raise ValueError("B1 freeze summary requires every protocol schedule")
        if len(self.schedule_file_hashes) != len(SCHEDULE_PURPOSES):
            raise ValueError("B1 freeze summary requires every private schedule file")
        if len(self.exact_input_hashes) != len(AttemptBuilderKind):
            raise ValueError("B1 freeze summary requires seven exact inputs")
        if len(self.training_slot_ids) != len(AttemptBuilderKind):
            raise ValueError("B1 freeze summary requires seven training slots")
        if any(type(item) is not str or not item for item in self.training_slot_ids):
            raise ValueError("B1 training slot IDs must be non-empty text")
        expected_evaluation_slots = len(AttemptBuilderKind) * 6
        if len(self.evaluation_slot_plan_ids) != expected_evaluation_slots:
            raise ValueError("B1 freeze summary lacks an evaluation coordinate")
        if not isinstance(self.result_counts, B1ResultCounts):
            raise TypeError("B1 freeze summary requires zero result counts")
        if self.format != B1_FREEZE_SUMMARY_FORMAT:
            raise ValueError("unsupported B1 freeze summary format")
        if self.summary_id != stable_hash(self._identity_value()):
            raise ValueError("B1 summary ID differs from its frozen identities")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "b0_report_hash": self.b0_report_hash,
            "b1_run_admission_hash": self.b1_run_admission_hash,
            "base_model_artifact_hash": self.base_model_artifact_hash,
            "evaluation_plan_id": self.evaluation_plan_id,
            "evaluation_slot_plan_ids": list(self.evaluation_slot_plan_ids),
            "exact_input_hashes": list(self.exact_input_hashes),
            "exact_admission_report_hash": self.exact_admission_report_hash,
            "f2_f3_proof_hash": self.f2_f3_proof_hash,
            "formal_execution_hash": self.formal_execution_hash,
            "formal_run_group_id": self.formal_run_group_id,
            "formal_run_manifest_hash": self.formal_run_manifest_hash,
            "format": self.format,
            "implementation_build_hash": self.implementation_build_hash,
            "initial_library_version": self.initial_library_version,
            "initial_skill_library_state_hash": self.initial_skill_library_state_hash,
            "initial_trainable_state_hash": self.initial_trainable_state_hash,
            "metric_reporting_protocol_hash": self.metric_reporting_protocol_hash,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "result_counts": self.result_counts.to_value(),
            "schedule_content_hashes": list(self.schedule_content_hashes),
            "schedule_file_hashes": list(self.schedule_file_hashes),
            "scientific_sampling_schedule_hash": self.scientific_sampling_schedule_hash,
            "source_commit": self.source_commit,
            "tokenizer_artifact_hash": self.tokenizer_artifact_hash,
            "training_slot_ids": list(self.training_slot_ids),
        }

    @classmethod
    def create(cls, **values: object) -> B1FreezeSummary:
        identity = {**values, "format": B1_FREEZE_SUMMARY_FORMAT}
        normalized = {
            key: item.to_value() if isinstance(item, B1ResultCounts) else item
            for key, item in identity.items()
        }
        return cls(**values, summary_id=stable_hash(normalized))  # type: ignore[arg-type]

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._identity_value(), "summary_id": self.summary_id}

    @classmethod
    def from_value(cls, value: object) -> B1FreezeSummary:
        fields = {
            "b0_report_hash",
            "b1_run_admission_hash",
            "base_model_artifact_hash",
            "evaluation_plan_id",
            "evaluation_slot_plan_ids",
            "exact_input_hashes",
            "exact_admission_report_hash",
            "f2_f3_proof_hash",
            "formal_execution_hash",
            "formal_run_group_id",
            "formal_run_manifest_hash",
            "format",
            "implementation_build_hash",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "initial_trainable_state_hash",
            "metric_reporting_protocol_hash",
            "protocol_freeze_id",
            "protocol_hash",
            "result_counts",
            "schedule_content_hashes",
            "schedule_file_hashes",
            "scientific_sampling_schedule_hash",
            "source_commit",
            "summary_id",
            "tokenizer_artifact_hash",
            "training_slot_ids",
        }
        data = _object(value, fields=fields, label="B1 freeze summary")
        tuple_fields = {
            "evaluation_slot_plan_ids",
            "exact_input_hashes",
            "schedule_content_hashes",
            "schedule_file_hashes",
            "training_slot_ids",
        }
        for field in tuple_fields:
            raw = data[field]
            if not isinstance(raw, list) or any(type(item) is not str for item in raw):
                raise TypeError(f"{field} must be a text array")
        text_fields = fields - tuple_fields - {"result_counts"}
        for field in text_fields:
            _text(data[field], field=field)
        return cls(
            b0_report_hash=data["b0_report_hash"],
            f2_f3_proof_hash=data["f2_f3_proof_hash"],
            b1_run_admission_hash=data["b1_run_admission_hash"],
            exact_admission_report_hash=data["exact_admission_report_hash"],
            source_commit=data["source_commit"],
            formal_execution_hash=data["formal_execution_hash"],
            base_model_artifact_hash=data["base_model_artifact_hash"],
            tokenizer_artifact_hash=data["tokenizer_artifact_hash"],
            implementation_build_hash=data["implementation_build_hash"],
            initial_trainable_state_hash=data["initial_trainable_state_hash"],
            initial_library_version=data["initial_library_version"],
            initial_skill_library_state_hash=data["initial_skill_library_state_hash"],
            metric_reporting_protocol_hash=data["metric_reporting_protocol_hash"],
            scientific_sampling_schedule_hash=data["scientific_sampling_schedule_hash"],
            protocol_hash=data["protocol_hash"],
            protocol_freeze_id=data["protocol_freeze_id"],
            formal_run_group_id=data["formal_run_group_id"],
            formal_run_manifest_hash=data["formal_run_manifest_hash"],
            schedule_content_hashes=tuple(data["schedule_content_hashes"]),
            schedule_file_hashes=tuple(data["schedule_file_hashes"]),
            exact_input_hashes=tuple(data["exact_input_hashes"]),
            training_slot_ids=tuple(data["training_slot_ids"]),
            evaluation_plan_id=data["evaluation_plan_id"],
            evaluation_slot_plan_ids=tuple(data["evaluation_slot_plan_ids"]),
            result_counts=B1ResultCounts.from_value(data["result_counts"]),
            summary_id=data["summary_id"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class B1FreezeArtifacts:
    schedules: tuple[PrivateFrozenTaskSequence, ...]
    protocol: ExperimentProtocol
    protocol_freeze: ProtocolFreeze
    exact_inputs: tuple[PrivateBenchmarkAttemptInput, ...]
    exact_input_hashes: tuple[str, ...]
    run_manifest: FormalRunManifest
    evaluation_plan: B1EvaluationPlan
    summary: B1FreezeSummary


def build_b1_schedules(
    catalog: PrivateBenchmarkCatalog,
    *,
    training_task_count: int,
) -> tuple[PrivateFrozenTaskSequence, ...]:
    """Build the three exact, result-blind schedules in protocol order."""

    if type(training_task_count) is not int or training_task_count < 1:
        raise ValueError("B1 training task count must be positive")
    return tuple(
        build_result_blind_sequence(
            catalog,
            purpose=purpose,
            task_count=(training_task_count if purpose is SchedulePurpose.IID_TRAINING else None),
        )
        for purpose in SCHEDULE_PURPOSES
    )


def _evaluation_slot_plans(
    *,
    protocol: ExperimentProtocol,
    manifest: FormalRunManifest,
) -> tuple[B1EvaluationSlotPlan, ...]:
    schedules = {item.purpose: item for item in protocol.schedule_identities}
    sampling_hash = scientific_sampling_schedule_hash(base_seed=protocol.seed)
    if protocol.formal_execution.sampling_schedule_algorithm != SCIENTIFIC_SAMPLING_ALGORITHM:
        raise ValueError("B1 protocol uses another scientific sampling schedule")
    kinds_and_ordinals = (
        (FormalEvaluationKind.INITIAL_IID, 0),
        (FormalEvaluationKind.INITIAL_OOD, 0),
        *tuple(
            (FormalEvaluationKind.PROGRESS_IID, ordinal)
            for ordinal in protocol.formal_execution.progress_anchor_cycle_ordinals
        ),
        (FormalEvaluationKind.FINAL_IID, 0),
        (FormalEvaluationKind.FINAL_OOD, 0),
    )
    plans: list[B1EvaluationSlotPlan] = []
    for slot in manifest.slots:
        for kind, ordinal in kinds_and_ordinals:
            plans.append(
                B1EvaluationSlotPlan.create(
                    source_formal_run_group_id=manifest.run_group_id,
                    source_builder_kind=slot.builder_kind,
                    source_training_attempt_id=slot.attempt_id,
                    source_training_identity_hash=slot.public_identity_content_hash,
                    kind=kind,
                    anchor_ordinal=ordinal,
                    task_sequence_identity=schedules[_expected_schedule_purpose(kind)],
                    sampling_schedule_hash=sampling_hash,
                    implementation_build_hash=(
                        protocol.formal_execution.implementation_build.content_hash
                    ),
                    execution_hardware_policy_hash=(FORMAL_EVALUATION_HARDWARE_POLICY_HASH),
                )
            )
    return tuple(plans)


def build_b1_freeze_artifacts(
    *,
    schedules: tuple[PrivateFrozenTaskSequence, ...],
    protocol: ExperimentProtocol,
    protocol_freeze: ProtocolFreeze,
    exact_inputs: tuple[PrivateBenchmarkAttemptInput, ...],
    b0_report_hash: str,
    f2_f3_proof_hash: str,
) -> B1FreezeArtifacts:
    """Compose all B1 identities without executing or claiming any slot."""

    validate_sha256(b0_report_hash)
    validate_sha256(f2_f3_proof_hash)
    if tuple(item.identity.purpose for item in schedules) != SCHEDULE_PURPOSES:
        raise ValueError("B1 schedules must contain the protocol purposes in order")
    identities = tuple(item.identity for item in schedules)
    if protocol.schedule_identities != identities:
        raise ValueError("B1 private schedules differ from the protocol identities")
    if protocol_freeze.protocol_hash != protocol.content_hash:
        raise ValueError("B1 protocol freeze differs from the protocol")
    if protocol_freeze != ProtocolFreeze.create(protocol):
        raise ValueError("B1 protocol freeze is not the canonical zero-result freeze")
    if tuple(item.builder_kind for item in exact_inputs) != tuple(AttemptBuilderKind):
        raise ValueError("B1 exact inputs must contain the seven arms in builder order")
    training_sequence = schedules[0]
    if any(
        item.protocol != protocol
        or item.protocol_freeze != protocol_freeze
        or item.training_sequence != training_sequence
        for item in exact_inputs
    ):
        raise ValueError("B1 exact input differs from the shared protocol or training schedule")
    if any(item.rollout_workflow != exact_inputs[0].rollout_workflow for item in exact_inputs[1:]):
        raise ValueError("B1 exact arms differ in rollout workflow execution binding")
    run_admission = protocol.formal_execution.b1_run_admission
    if run_admission.f2_f3_proof_hash != f2_f3_proof_hash:
        raise ValueError("B1 run admission differs from the named F2/F3 proof")
    for exact in exact_inputs:
        run_admission.require_exact_input(exact)

    exact_bytes = tuple(_canonical_bytes(item.to_value()) for item in exact_inputs)
    exact_hashes = tuple(sha256_bytes(item) for item in exact_bytes)
    public_identities = tuple(
        item.public_identity(exact_input_sha256=exact_hash)
        for item, exact_hash in zip(exact_inputs, exact_hashes, strict=True)
    )
    cross_arm = tuple(
        item.cross_arm_execution_identity(exact_input_sha256=exact_hash)
        for item, exact_hash in zip(exact_inputs, exact_hashes, strict=True)
    )
    if any(item != cross_arm[0] for item in cross_arm[1:]):
        raise ValueError("B1 exact arms differ outside their declared method axis")
    slots = tuple(
        FormalRunSlot(
            builder_kind=item.builder_kind,
            attempt_id=(
                f"b1-{item.builder_kind.value}-"
                f"{protocol_freeze.freeze_id.removeprefix('sha256:')[:16]}"
            ),
            exact_input_sha256=exact_hash,
            public_identity_content_hash=identity.content_hash,
        )
        for item, exact_hash, identity in zip(
            exact_inputs, exact_hashes, public_identities, strict=True
        )
    )
    manifest = FormalRunManifest.create(protocol_freeze=protocol_freeze, slots=slots)
    evaluation_slots = _evaluation_slot_plans(protocol=protocol, manifest=manifest)
    evaluation_plan = B1EvaluationPlan.create(
        protocol_hash=protocol.content_hash,
        protocol_freeze_id=protocol_freeze.freeze_id,
        source_formal_run_group_id=manifest.run_group_id,
        slots=evaluation_slots,
    )
    schedule_bytes = tuple(_canonical_bytes(item.to_value()) for item in schedules)
    result_counts = B1ResultCounts()
    summary = B1FreezeSummary.create(
        b0_report_hash=b0_report_hash,
        f2_f3_proof_hash=f2_f3_proof_hash,
        b1_run_admission_hash=run_admission.content_hash,
        exact_admission_report_hash=run_admission.exact_admission_report_hash,
        source_commit=protocol.formal_execution.implementation_build.source_commit,
        formal_execution_hash=protocol.formal_execution.content_hash,
        base_model_artifact_hash=protocol.formal_execution.base_model_artifact.content_hash,
        tokenizer_artifact_hash=protocol.formal_execution.tokenizer_artifact.content_hash,
        implementation_build_hash=protocol.formal_execution.implementation_build.content_hash,
        initial_trainable_state_hash=(
            protocol.formal_execution.initial_trainable_state.content_hash
        ),
        initial_library_version=protocol.formal_execution.initial_library_version,
        initial_skill_library_state_hash=(
            protocol.formal_execution.initial_skill_library_state_hash
        ),
        metric_reporting_protocol_hash=stable_hash(protocol.evaluation_reporting.to_value()),
        scientific_sampling_schedule_hash=protocol.formal_execution.sampling_schedule_hash,
        protocol_hash=protocol.content_hash,
        protocol_freeze_id=protocol_freeze.freeze_id,
        formal_run_group_id=manifest.run_group_id,
        formal_run_manifest_hash=manifest.content_hash,
        schedule_content_hashes=tuple(item.identity.content_hash for item in schedules),
        schedule_file_hashes=tuple(sha256_bytes(item) for item in schedule_bytes),
        exact_input_hashes=exact_hashes,
        training_slot_ids=tuple(item.attempt_id for item in manifest.slots),
        evaluation_plan_id=evaluation_plan.plan_id,
        evaluation_slot_plan_ids=tuple(item.slot_plan_id for item in evaluation_plan.slots),
        result_counts=result_counts,
    )
    return B1FreezeArtifacts(
        schedules=schedules,
        protocol=protocol,
        protocol_freeze=protocol_freeze,
        exact_inputs=exact_inputs,
        exact_input_hashes=exact_hashes,
        run_manifest=manifest,
        evaluation_plan=evaluation_plan,
        summary=summary,
    )


def write_b1_freeze_artifacts(
    artifacts: B1FreezeArtifacts,
    output_directory: Path,
) -> None:
    """Publish one fresh private B1 directory without creating a run ledger."""

    if not isinstance(artifacts, B1FreezeArtifacts):
        raise TypeError("B1 writer requires B1FreezeArtifacts")
    if not isinstance(output_directory, Path) or not output_directory.is_absolute():
        raise ValueError("B1 output directory must be an absolute Path")
    output_directory.mkdir(mode=0o700)
    schedule_directory = output_directory / "schedules"
    exact_input_directory = output_directory / "exact-inputs"
    schedule_directory.mkdir()
    exact_input_directory.mkdir()

    observed_schedule_hashes: list[str] = []
    for sequence in artifacts.schedules:
        path = schedule_directory / f"{sequence.identity.purpose.value}.json"
        sequence.write_once(path)
        observed_schedule_hashes.append(sha256_bytes(path.read_bytes()))
    if tuple(observed_schedule_hashes) != artifacts.summary.schedule_file_hashes:
        raise RuntimeError("written B1 schedule bytes differ from the frozen summary")

    _write_once(output_directory / "experiment-protocol.json", artifacts.protocol.to_value())
    _write_once(
        output_directory / "experiment-protocol-freeze.json",
        artifacts.protocol_freeze.to_value(),
    )
    observed_exact_hashes: list[str] = []
    for exact in artifacts.exact_inputs:
        path = exact_input_directory / f"{exact.builder_kind.value}.json"
        observed_exact_hashes.append(_write_once(path, exact.to_value()))
    if tuple(observed_exact_hashes) != artifacts.exact_input_hashes:
        raise RuntimeError("written B1 exact-input bytes differ from the formal slots")

    if artifacts.run_manifest.format != FORMAL_RUN_MANIFEST_FORMAT:
        raise ValueError("B1 formal run manifest has an unsupported format")
    _write_once(output_directory / "formal-run-manifest.json", artifacts.run_manifest.to_value())
    _write_once(
        output_directory / "formal-evaluation-plan.json",
        artifacts.evaluation_plan.to_value(),
    )
    _write_once(
        output_directory / "zero-result-counts.json",
        artifacts.summary.result_counts.to_value(),
    )
    _write_once(output_directory / "b1-freeze-summary.json", artifacts.summary.to_value())


__all__ = [
    "B1_EVALUATION_PLAN_FORMAT",
    "B1_EVALUATION_SLOT_PLAN_FORMAT",
    "B1_FREEZE_SUMMARY_FORMAT",
    "B1_RESULT_COUNTS_FORMAT",
    "B1EvaluationPlan",
    "B1EvaluationSlotPlan",
    "B1FreezeArtifacts",
    "B1FreezeSummary",
    "B1ResultCounts",
    "build_b1_freeze_artifacts",
    "build_b1_schedules",
    "write_b1_freeze_artifacts",
]
