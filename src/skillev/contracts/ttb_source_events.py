"""Protocol-v3 scientific source-event payloads.

The live method has one source event per training step, one initial-library
event, one phase-open event, and one committed evolution-cycle event.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from .canonical import JsonValue, normalize_json, stable_hash
from .identity import validate_sha256
from .ttb_calibration import PosteriorBatchUpdate
from .ttb_common import require_non_empty_text
from .ttb_evolution import (
    EvolutionActionRecord,
    EvolutionActionType,
    evolution_action_from_value,
)
from .ttb_training import EdgeLogprobRecord, TTBBatchStats
from .ttb_trajectory import TrajectoryRecord
from .ttb_transition import PhaseTransitionEvent

TRAINING_STEP_REPORT_FORMAT = "skillev-training-step-report@3"
TRAINING_STEP_COMMIT_FORMAT = "skillev-training-step-commit@4"
LIBRARY_INITIALIZED_FORMAT = "skillev-library-initialized@4"
EVOLUTION_PHASE_OPEN_FORMAT = "skillev-evolution-phase-open@5"
EVOLUTION_CYCLE_COMMIT_FORMAT = "skillev-evolution-cycle-commit@7"
EVOLUTION_NO_OP_COMMIT_FORMAT = "skillev-evolution-no-op-commit@1"
PHASE_CHECKPOINT_ARTIFACT_FORMAT = "skillev-phase-checkpoint-artifact@1"
PHASE_CHECKPOINT_PUBLISHED_FORMAT = "skillev-phase-checkpoint-published@1"
RUN_CURSOR_VALUE_FORMAT = "skillev-source-run-cursor@1"


def _object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _text(value: JsonValue, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    require_non_empty_text(value, field=field)
    return value


def _integer(value: JsonValue, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: JsonValue, *, field: str, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    if upper is not None and number > upper:
        raise ValueError(f"{field} must not exceed {upper}")
    return number


def _timestamp(value: JsonValue, *, field: str) -> tuple[str, datetime]:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return text, parsed


def _array(value: JsonValue, *, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _json_objects(
    values: tuple[Mapping[str, JsonValue], ...],
    *,
    field: str,
) -> tuple[Mapping[str, JsonValue], ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple")
    normalized: list[Mapping[str, JsonValue]] = []
    for value in values:
        item = normalize_json(value)
        if not isinstance(item, dict):
            raise ValueError(f"{field} must contain JSON objects")
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class RunCursorValue:
    """Dependency-light source representation of the frozen run cursor."""

    run_plan_hash: str
    completed_training_steps: int
    committed_cycles: int
    committed_actions: int
    format: str = RUN_CURSOR_VALUE_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.run_plan_hash)
        for field in (
            "completed_training_steps",
            "committed_cycles",
            "committed_actions",
        ):
            _integer(getattr(self, field), field=field, minimum=0)
        if self.committed_actions < self.committed_cycles:
            raise ValueError("run cursor has fewer actions than cycles")
        if self.format != RUN_CURSOR_VALUE_FORMAT:
            raise ValueError("unsupported source run cursor format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "committed_actions": self.committed_actions,
            "committed_cycles": self.committed_cycles,
            "completed_training_steps": self.completed_training_steps,
            "format": self.format,
            "run_plan_hash": self.run_plan_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> RunCursorValue:
        data = _object(
            value,
            label="RunCursorValue",
            fields=frozenset(
                {
                    "committed_actions",
                    "committed_cycles",
                    "completed_training_steps",
                    "format",
                    "run_plan_hash",
                }
            ),
        )
        return cls(
            run_plan_hash=_text(data["run_plan_hash"], field="run_plan_hash"),
            completed_training_steps=_integer(
                data["completed_training_steps"],
                field="completed_training_steps",
                minimum=0,
            ),
            committed_cycles=_integer(
                data["committed_cycles"], field="committed_cycles", minimum=0
            ),
            committed_actions=_integer(
                data["committed_actions"], field="committed_actions", minimum=0
            ),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class TrainingStepReportValue:
    optimizer_step: int
    batch_id: str
    torch_batch_loss: float
    audited_batch_loss: float
    mean_reward: float
    grad_norm_forward: float
    grad_norm_backward: float
    grad_norm_z: float
    forward_adapter_version: str
    backward_adapter_version: str
    z_version: str
    started_at: str
    completed_at: str
    format: str = TRAINING_STEP_REPORT_FORMAT
    optimization_diagnostics: JsonValue = None
    optimizer_transition: JsonValue = None

    def __post_init__(self) -> None:
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        require_non_empty_text(self.batch_id, field="batch_id")
        for field, value in (
            ("torch_batch_loss", self.torch_batch_loss),
            ("audited_batch_loss", self.audited_batch_loss),
            ("grad_norm_forward", self.grad_norm_forward),
            ("grad_norm_backward", self.grad_norm_backward),
            ("grad_norm_z", self.grad_norm_z),
        ):
            object.__setattr__(self, field, _number(value, field=field))
        object.__setattr__(
            self,
            "mean_reward",
            _number(self.mean_reward, field="mean_reward", upper=1.0),
        )
        for field, version in (
            ("forward_adapter_version", self.forward_adapter_version),
            ("backward_adapter_version", self.backward_adapter_version),
            ("z_version", self.z_version),
        ):
            require_non_empty_text(version, field=field)
        if self.format not in {
            TRAINING_STEP_REPORT_FORMAT,
            "skillev-training-step-report@4",
            "skillev-training-step-report@5",
        }:
            raise ValueError("unsupported training step report format")
        if self.optimization_diagnostics is not None:
            if self.format not in {
                "skillev-training-step-report@4",
                "skillev-training-step-report@5",
            } or not isinstance(self.optimization_diagnostics, dict):
                raise ValueError("optimization diagnostics require the candidate report format")
            normalize_json(self.optimization_diagnostics)
        if self.optimizer_transition is not None:
            if self.format != "skillev-training-step-report@5" or not isinstance(
                self.optimizer_transition, dict
            ):
                raise ValueError("optimizer transition requires report@5")
            normalize_json(self.optimizer_transition)
        started_at, started = _timestamp(self.started_at, field="started_at")
        completed_at, completed = _timestamp(self.completed_at, field="completed_at")
        if completed < started:
            raise ValueError("completed_at cannot precede started_at")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **(
                {"optimization_diagnostics": self.optimization_diagnostics}
                if self.format
                in {"skillev-training-step-report@4", "skillev-training-step-report@5"}
                else {}
            ),
            **(
                {"optimizer_transition": self.optimizer_transition}
                if self.format == "skillev-training-step-report@5"
                else {}
            ),
            "audited_batch_loss": self.audited_batch_loss,
            "backward_adapter_version": self.backward_adapter_version,
            "batch_id": self.batch_id,
            "completed_at": self.completed_at,
            "format": self.format,
            "forward_adapter_version": self.forward_adapter_version,
            "grad_norm_backward": self.grad_norm_backward,
            "grad_norm_forward": self.grad_norm_forward,
            "grad_norm_z": self.grad_norm_z,
            "mean_reward": self.mean_reward,
            "optimizer_step": self.optimizer_step,
            "started_at": self.started_at,
            "torch_batch_loss": self.torch_batch_loss,
            "z_version": self.z_version,
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingStepReportValue:
        data = _object(
            value,
            label="TrainingStepReportValue",
            fields=frozenset(
                {
                    "audited_batch_loss",
                    "backward_adapter_version",
                    "batch_id",
                    "completed_at",
                    "format",
                    "forward_adapter_version",
                    "grad_norm_backward",
                    "grad_norm_forward",
                    "grad_norm_z",
                    "mean_reward",
                    "optimizer_step",
                    "started_at",
                    "torch_batch_loss",
                    "z_version",
                }
                | (
                    {"optimization_diagnostics"}
                    if isinstance(value, dict)
                    and value.get("format")
                    in {"skillev-training-step-report@4", "skillev-training-step-report@5"}
                    else set()
                )
                | (
                    {"optimizer_transition"}
                    if isinstance(value, dict)
                    and value.get("format") == "skillev-training-step-report@5"
                    else set()
                )
            ),
        )
        return cls(
            optimizer_transition=data.get("optimizer_transition"),
            optimization_diagnostics=data.get("optimization_diagnostics"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            batch_id=_text(data["batch_id"], field="batch_id"),
            torch_batch_loss=_number(data["torch_batch_loss"], field="torch_batch_loss"),
            audited_batch_loss=_number(
                data["audited_batch_loss"],
                field="audited_batch_loss",
            ),
            mean_reward=_number(data["mean_reward"], field="mean_reward", upper=1.0),
            grad_norm_forward=_number(
                data["grad_norm_forward"],
                field="grad_norm_forward",
            ),
            grad_norm_backward=_number(
                data["grad_norm_backward"],
                field="grad_norm_backward",
            ),
            grad_norm_z=_number(data["grad_norm_z"], field="grad_norm_z"),
            forward_adapter_version=_text(
                data["forward_adapter_version"],
                field="forward_adapter_version",
            ),
            backward_adapter_version=_text(
                data["backward_adapter_version"],
                field="backward_adapter_version",
            ),
            z_version=_text(data["z_version"], field="z_version"),
            started_at=_text(data["started_at"], field="started_at"),
            completed_at=_text(data["completed_at"], field="completed_at"),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class TrainingStepCommit:
    batch_id: str
    optimizer_step: int
    policy_snapshot_before: str
    policy_snapshot_after: str
    library_version: str
    records: tuple[TrajectoryRecord, ...]
    edge_records: tuple[EdgeLogprobRecord, ...]
    stats: TTBBatchStats
    posterior_batch: PosteriorBatchUpdate
    report: TrainingStepReportValue
    run_cursor_after: RunCursorValue
    format: str = TRAINING_STEP_COMMIT_FORMAT

    def __post_init__(self) -> None:
        require_non_empty_text(self.batch_id, field="batch_id")
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        for field, value in (
            ("policy_snapshot_before", self.policy_snapshot_before),
            ("policy_snapshot_after", self.policy_snapshot_after),
            ("library_version", self.library_version),
        ):
            require_non_empty_text(value, field=field)
        if self.format != TRAINING_STEP_COMMIT_FORMAT:
            raise ValueError("unsupported training step commit format")
        if not isinstance(self.records, tuple) or any(
            not isinstance(record, TrajectoryRecord) for record in self.records
        ):
            raise ValueError("records must contain TrajectoryRecord values")
        if not self.records:
            raise ValueError("training step commit requires records")
        if not isinstance(self.edge_records, tuple) or any(
            not isinstance(record, EdgeLogprobRecord) for record in self.edge_records
        ):
            raise ValueError("edge_records must contain EdgeLogprobRecord values")
        if not isinstance(self.stats, TTBBatchStats):
            raise ValueError("stats must be TTBBatchStats")
        if not isinstance(self.posterior_batch, PosteriorBatchUpdate):
            raise ValueError("posterior_batch must be PosteriorBatchUpdate")
        if not isinstance(self.report, TrainingStepReportValue):
            raise ValueError("report must be TrainingStepReportValue")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("training commit requires run_cursor_after")
        if self.run_cursor_after.completed_training_steps < 1:
            raise ValueError("training commit cursor must include its completed step")
        if self.stats.batch_id != self.batch_id:
            raise ValueError("training step stats batch_id differs")
        if self.stats.optimizer_step != self.optimizer_step:
            raise ValueError("training step stats optimizer_step differs")
        record_ids = tuple(record.trajectory_id for record in self.records)
        residual_ids = tuple(residual.trajectory_id for residual in self.stats.residuals)
        if record_ids != residual_ids:
            raise ValueError("training records and residuals have different order")
        if self.posterior_batch.batch_id != self.batch_id:
            raise ValueError("posterior batch targets another training batch")
        if self.report.batch_id != self.batch_id:
            raise ValueError("training report targets another batch")
        if self.report.optimizer_step != self.optimizer_step:
            raise ValueError("training report targets another optimizer step")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "edge_records": [record.to_value() for record in self.edge_records],
            "format": self.format,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_after": self.policy_snapshot_after,
            "policy_snapshot_before": self.policy_snapshot_before,
            "posterior_batch": self.posterior_batch.to_value(),
            "records": [record.to_value() for record in self.records],
            "report": self.report.to_value(),
            "run_cursor_after": self.run_cursor_after.to_value(),
            "stats": self.stats.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingStepCommit:
        data = _object(
            value,
            label="TrainingStepCommit",
            fields=frozenset(
                {
                    "batch_id",
                    "edge_records",
                    "format",
                    "library_version",
                    "optimizer_step",
                    "policy_snapshot_after",
                    "policy_snapshot_before",
                    "posterior_batch",
                    "records",
                    "report",
                    "run_cursor_after",
                    "stats",
                }
            ),
        )
        return cls(
            batch_id=_text(data["batch_id"], field="batch_id"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            policy_snapshot_before=_text(
                data["policy_snapshot_before"],
                field="policy_snapshot_before",
            ),
            policy_snapshot_after=_text(
                data["policy_snapshot_after"],
                field="policy_snapshot_after",
            ),
            library_version=_text(data["library_version"], field="library_version"),
            records=tuple(
                TrajectoryRecord.from_value(record)
                for record in _array(data["records"], field="records")
            ),
            edge_records=tuple(
                EdgeLogprobRecord.from_value(record)
                for record in _array(data["edge_records"], field="edge_records")
            ),
            stats=TTBBatchStats.from_value(data["stats"]),
            posterior_batch=PosteriorBatchUpdate.from_value(data["posterior_batch"]),
            report=TrainingStepReportValue.from_value(data["report"]),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class LibraryInitialized:
    documents: tuple[Mapping[str, JsonValue], ...]
    active_skill_ids: tuple[str, ...]
    library_version: str
    initial_optimizer_step: int
    method_identity_hash: str
    run_cursor: RunCursorValue
    format: str = LIBRARY_INITIALIZED_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "documents",
            _json_objects(self.documents, field="documents"),
        )
        if not isinstance(self.active_skill_ids, tuple) or any(
            not isinstance(skill_id, str) or not skill_id.strip()
            for skill_id in self.active_skill_ids
        ):
            raise ValueError("active_skill_ids must contain non-empty text")
        if tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids:
            raise ValueError("active_skill_ids must be sorted and unique")
        require_non_empty_text(self.library_version, field="library_version")
        _integer(self.initial_optimizer_step, field="initial_optimizer_step", minimum=0)
        validate_sha256(self.method_identity_hash)
        if not isinstance(self.run_cursor, RunCursorValue):
            raise TypeError("library initialization requires RunCursorValue")
        if self.format != LIBRARY_INITIALIZED_FORMAT:
            raise ValueError("unsupported library-initialized format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "active_skill_ids": list(self.active_skill_ids),
            "documents": [normalize_json(document) for document in self.documents],
            "format": self.format,
            "initial_optimizer_step": self.initial_optimizer_step,
            "library_version": self.library_version,
            "method_identity_hash": self.method_identity_hash,
            "run_cursor": self.run_cursor.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> LibraryInitialized:
        data = _object(
            value,
            label="LibraryInitialized",
            fields=frozenset(
                {
                    "active_skill_ids",
                    "documents",
                    "format",
                    "initial_optimizer_step",
                    "library_version",
                    "method_identity_hash",
                    "run_cursor",
                }
            ),
        )
        documents = _array(data["documents"], field="documents")
        if any(not isinstance(document, dict) for document in documents):
            raise ValueError("documents must contain JSON objects")
        return cls(
            documents=tuple(cast(dict[str, JsonValue], document) for document in documents),
            active_skill_ids=tuple(
                _text(item, field="active_skill_ids")
                for item in _array(data["active_skill_ids"], field="active_skill_ids")
            ),
            library_version=_text(data["library_version"], field="library_version"),
            initial_optimizer_step=_integer(
                data["initial_optimizer_step"], field="initial_optimizer_step", minimum=0
            ),
            method_identity_hash=_text(data["method_identity_hash"], field="method_identity_hash"),
            run_cursor=RunCursorValue.from_value(data["run_cursor"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionPhaseOpened:
    phase_event: PhaseTransitionEvent
    run_cursor_at_phase: RunCursorValue
    format: str = EVOLUTION_PHASE_OPEN_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.phase_event, PhaseTransitionEvent):
            raise ValueError("phase_event must be PhaseTransitionEvent")
        if not isinstance(self.run_cursor_at_phase, RunCursorValue):
            raise TypeError("phase source requires run cursor")
        if self.run_cursor_at_phase.completed_training_steps < 1:
            raise ValueError("phase cannot open before a training step")
        if self.format != EVOLUTION_PHASE_OPEN_FORMAT:
            raise ValueError("unsupported phase-open format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "phase_event": self.phase_event.to_value(),
            "run_cursor_at_phase": self.run_cursor_at_phase.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionPhaseOpened:
        data = _object(
            value,
            label="EvolutionPhaseOpened",
            fields=frozenset({"format", "phase_event", "run_cursor_at_phase"}),
        )
        return cls(
            phase_event=PhaseTransitionEvent.from_value(data["phase_event"]),
            run_cursor_at_phase=RunCursorValue.from_value(data["run_cursor_at_phase"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionNoOpCommitted:
    """A verified phase whose frozen Phi thresholds admit no mutation."""

    phase_event_id: str
    optimizer_step: int
    decision_reason: str
    library_version: str
    run_cursor_after: RunCursorValue
    format: str = EVOLUTION_NO_OP_COMMIT_FORMAT

    def __post_init__(self) -> None:
        require_non_empty_text(self.phase_event_id, field="phase_event_id")
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        require_non_empty_text(self.decision_reason, field="decision_reason")
        require_non_empty_text(self.library_version, field="library_version")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("no-op evolution source requires run cursor")
        if self.format != EVOLUTION_NO_OP_COMMIT_FORMAT:
            raise ValueError("unsupported evolution no-op format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "decision_reason": self.decision_reason,
            "format": self.format,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "phase_event_id": self.phase_event_id,
            "run_cursor_after": self.run_cursor_after.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionNoOpCommitted:
        data = _object(
            value,
            label="EvolutionNoOpCommitted",
            fields=frozenset(
                {
                    "decision_reason",
                    "format",
                    "library_version",
                    "optimizer_step",
                    "phase_event_id",
                    "run_cursor_after",
                }
            ),
        )
        return cls(
            phase_event_id=_text(data["phase_event_id"], field="phase_event_id"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            decision_reason=_text(data["decision_reason"], field="decision_reason"),
            library_version=_text(data["library_version"], field="library_version"),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionMutationValue:
    library_version_before: str
    library_version_after: str
    new_documents: tuple[Mapping[str, JsonValue], ...]
    active_skill_ids_after: tuple[str, ...]
    actions: tuple[EvolutionActionRecord, ...]

    def __post_init__(self) -> None:
        require_non_empty_text(
            self.library_version_before,
            field="library_version_before",
        )
        require_non_empty_text(
            self.library_version_after,
            field="library_version_after",
        )
        if self.library_version_before == self.library_version_after:
            raise ValueError("evolution mutation must change the library version")
        object.__setattr__(
            self,
            "new_documents",
            _json_objects(self.new_documents, field="new_documents"),
        )
        if tuple(sorted(set(self.active_skill_ids_after))) != self.active_skill_ids_after:
            raise ValueError("active_skill_ids_after must be sorted and unique")
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ValueError("evolution mutation requires actions")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "actions": [action.to_value() for action in self.actions],
            "active_skill_ids_after": list(self.active_skill_ids_after),
            "library_version_after": self.library_version_after,
            "library_version_before": self.library_version_before,
            "new_documents": [normalize_json(document) for document in self.new_documents],
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionMutationValue:
        data = _object(
            value,
            label="EvolutionMutationValue",
            fields=frozenset(
                {
                    "actions",
                    "active_skill_ids_after",
                    "library_version_after",
                    "library_version_before",
                    "new_documents",
                }
            ),
        )
        raw_documents = _array(data["new_documents"], field="new_documents")
        if any(not isinstance(document, dict) for document in raw_documents):
            raise ValueError("new_documents must contain JSON objects")
        return cls(
            library_version_before=_text(
                data["library_version_before"],
                field="library_version_before",
            ),
            library_version_after=_text(
                data["library_version_after"],
                field="library_version_after",
            ),
            new_documents=tuple(cast(dict[str, JsonValue], item) for item in raw_documents),
            active_skill_ids_after=tuple(
                _text(item, field="active_skill_ids_after")
                for item in _array(
                    data["active_skill_ids_after"],
                    field="active_skill_ids_after",
                )
            ),
            actions=tuple(
                evolution_action_from_value(item)
                for item in _array(data["actions"], field="actions")
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionCycleCommitted:
    phase_event_id: str
    optimizer_step: int
    mutation: EvolutionMutationValue
    decision_content_hash: str
    proposal_content_hashes: tuple[str, ...]
    authoring_reservation_ids: tuple[str, ...]
    authoring_usage: AuthoringUsageValue
    z_reset_seed: int
    z_version_after_reset: str
    library_version_before: str
    library_version_after: str
    run_cursor_after: RunCursorValue
    format: str = EVOLUTION_CYCLE_COMMIT_FORMAT

    def __post_init__(self) -> None:
        require_non_empty_text(self.phase_event_id, field="phase_event_id")
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        if not isinstance(self.mutation, EvolutionMutationValue):
            raise ValueError("mutation must be EvolutionMutationValue")
        validate_sha256(self.decision_content_hash)
        if not isinstance(self.proposal_content_hashes, tuple) or not self.proposal_content_hashes:
            raise ValueError("cycle requires proposal content hashes")
        for value in self.proposal_content_hashes:
            validate_sha256(value)
        action_hashes = tuple(action.proposal_content_hash for action in self.mutation.actions)
        if action_hashes != self.proposal_content_hashes:
            raise ValueError("cycle proposals differ from mutation action provenance")
        if any(type(item) is not str or not item for item in self.authoring_reservation_ids):
            raise ValueError("authoring reservation IDs must be non-empty text")
        if len(set(self.authoring_reservation_ids)) != len(self.authoring_reservation_ids):
            raise ValueError("cycle repeats an authoring reservation")
        if not isinstance(self.authoring_usage, AuthoringUsageValue):
            raise ValueError("authoring_usage must be AuthoringUsageValue")
        expected_calls = sum(
            action.action_type is not EvolutionActionType.PRUNE for action in self.mutation.actions
        )
        if self.authoring_usage.model_calls != expected_calls:
            raise ValueError("authoring usage differs from mutation model calls")
        if len(self.authoring_reservation_ids) != expected_calls:
            raise ValueError("authoring reservations differ from mutation model calls")
        if type(self.z_reset_seed) is not int or not 0 <= self.z_reset_seed < 2**64:
            raise ValueError("z_reset_seed must be uint64")
        require_non_empty_text(self.z_version_after_reset, field="z_version_after_reset")
        if self.library_version_before != self.mutation.library_version_before:
            raise ValueError("cycle before-version differs from mutation")
        if self.library_version_after != self.mutation.library_version_after:
            raise ValueError("cycle after-version differs from mutation")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("cycle requires run_cursor_after")
        if self.run_cursor_after.committed_cycles < 1:
            raise ValueError("cycle cursor must contain a committed cycle")
        if self.format != EVOLUTION_CYCLE_COMMIT_FORMAT:
            raise ValueError("unsupported evolution-cycle format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_reservation_ids": list(self.authoring_reservation_ids),
            "authoring_usage": self.authoring_usage.to_value(),
            "decision_content_hash": self.decision_content_hash,
            "format": self.format,
            "library_version_after": self.library_version_after,
            "library_version_before": self.library_version_before,
            "mutation": self.mutation.to_value(),
            "optimizer_step": self.optimizer_step,
            "phase_event_id": self.phase_event_id,
            "proposal_content_hashes": list(self.proposal_content_hashes),
            "run_cursor_after": self.run_cursor_after.to_value(),
            "z_reset_seed": self.z_reset_seed,
            "z_version_after_reset": self.z_version_after_reset,
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionCycleCommitted:
        data = _object(
            value,
            label="EvolutionCycleCommitted",
            fields=frozenset(
                {
                    "authoring_reservation_ids",
                    "authoring_usage",
                    "decision_content_hash",
                    "format",
                    "library_version_after",
                    "library_version_before",
                    "mutation",
                    "optimizer_step",
                    "phase_event_id",
                    "proposal_content_hashes",
                    "run_cursor_after",
                    "z_reset_seed",
                    "z_version_after_reset",
                }
            ),
        )
        return cls(
            phase_event_id=_text(data["phase_event_id"], field="phase_event_id"),
            optimizer_step=_integer(
                data["optimizer_step"],
                field="optimizer_step",
                minimum=1,
            ),
            mutation=EvolutionMutationValue.from_value(data["mutation"]),
            decision_content_hash=_text(
                data["decision_content_hash"], field="decision_content_hash"
            ),
            proposal_content_hashes=tuple(
                _text(item, field="proposal_content_hashes")
                for item in _array(data["proposal_content_hashes"], field="proposal_content_hashes")
            ),
            authoring_reservation_ids=tuple(
                _text(item, field="authoring_reservation_ids")
                for item in _array(
                    data["authoring_reservation_ids"],
                    field="authoring_reservation_ids",
                )
            ),
            authoring_usage=AuthoringUsageValue.from_value(data["authoring_usage"]),
            z_reset_seed=_integer(data["z_reset_seed"], field="z_reset_seed", minimum=0),
            z_version_after_reset=_text(
                data["z_version_after_reset"], field="z_version_after_reset"
            ),
            library_version_before=_text(
                data["library_version_before"],
                field="library_version_before",
            ),
            library_version_after=_text(
                data["library_version_after"],
                field="library_version_after",
            ),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class PhaseCheckpointArtifact:
    """Path-free identity of one immutable post-cycle runtime snapshot.

    A phase anchor may consume this descriptor only by its explicit
    ``phase_event_id``.  The checkpoint remains private; this record binds its
    contents and its post-cycle execution state into the published source log.
    """

    artifact_sha256: str
    runtime_state_sha256: str
    policy_snapshot_id: str
    library_version: str
    optimizer_step: int
    phase_event_id: str
    run_cursor_after: RunCursorValue
    format: str = PHASE_CHECKPOINT_ARTIFACT_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.artifact_sha256)
        validate_sha256(self.runtime_state_sha256)
        require_non_empty_text(self.policy_snapshot_id, field="policy_snapshot_id")
        require_non_empty_text(self.library_version, field="library_version")
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        require_non_empty_text(self.phase_event_id, field="phase_event_id")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("phase checkpoint requires run_cursor_after")
        if self.run_cursor_after.committed_cycles < 1:
            raise ValueError("phase checkpoint must follow a committed cycle")
        if self.format != PHASE_CHECKPOINT_ARTIFACT_FORMAT:
            raise ValueError("unsupported phase checkpoint artifact format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "format": self.format,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "phase_event_id": self.phase_event_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "run_cursor_after": self.run_cursor_after.to_value(),
            "runtime_state_sha256": self.runtime_state_sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> PhaseCheckpointArtifact:
        data = _object(
            value,
            label="PhaseCheckpointArtifact",
            fields=frozenset(
                {
                    "artifact_sha256",
                    "format",
                    "library_version",
                    "optimizer_step",
                    "phase_event_id",
                    "policy_snapshot_id",
                    "run_cursor_after",
                    "runtime_state_sha256",
                }
            ),
        )
        return cls(
            artifact_sha256=_text(data["artifact_sha256"], field="artifact_sha256"),
            runtime_state_sha256=_text(data["runtime_state_sha256"], field="runtime_state_sha256"),
            policy_snapshot_id=_text(data["policy_snapshot_id"], field="policy_snapshot_id"),
            library_version=_text(data["library_version"], field="library_version"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            phase_event_id=_text(data["phase_event_id"], field="phase_event_id"),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class PhaseCheckpointPublished:
    """Source event connecting a committed phase to its exact checkpoint."""

    phase_event_id: str
    artifact: PhaseCheckpointArtifact
    format: str = PHASE_CHECKPOINT_PUBLISHED_FORMAT

    def __post_init__(self) -> None:
        require_non_empty_text(self.phase_event_id, field="phase_event_id")
        if not isinstance(self.artifact, PhaseCheckpointArtifact):
            raise TypeError("phase checkpoint publication requires an artifact")
        if self.artifact.phase_event_id != self.phase_event_id:
            raise ValueError("phase checkpoint artifact belongs to another phase")
        if self.format != PHASE_CHECKPOINT_PUBLISHED_FORMAT:
            raise ValueError("unsupported phase checkpoint publication format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifact": self.artifact.to_value(),
            "format": self.format,
            "phase_event_id": self.phase_event_id,
        }

    @classmethod
    def from_value(cls, value: object) -> PhaseCheckpointPublished:
        data = _object(
            value,
            label="PhaseCheckpointPublished",
            fields=frozenset({"artifact", "format", "phase_event_id"}),
        )
        return cls(
            phase_event_id=_text(data["phase_event_id"], field="phase_event_id"),
            artifact=PhaseCheckpointArtifact.from_value(data["artifact"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class AuthoringUsageValue:
    """Exact aggregate model usage between Phi preview and cycle publication."""

    input_tokens: int
    output_tokens: int
    model_calls: int

    def __post_init__(self) -> None:
        for field in ("input_tokens", "output_tokens", "model_calls"):
            _integer(getattr(self, field), field=field, minimum=0)
        if self.model_calls == 0 and (self.input_tokens != 0 or self.output_tokens != 0):
            raise ValueError("zero authoring calls cannot consume tokens")
        if self.model_calls > 0 and self.input_tokens == 0:
            raise ValueError("authoring calls require prompt tokens")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "input_tokens": self.input_tokens,
            "model_calls": self.model_calls,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_value(cls, value: object) -> AuthoringUsageValue:
        data = _object(
            value,
            label="AuthoringUsageValue",
            fields=frozenset({"input_tokens", "model_calls", "output_tokens"}),
        )
        return cls(
            input_tokens=_integer(data["input_tokens"], field="input_tokens", minimum=0),
            output_tokens=_integer(data["output_tokens"], field="output_tokens", minimum=0),
            model_calls=_integer(data["model_calls"], field="model_calls", minimum=0),
        )


__all__ = [
    "EVOLUTION_CYCLE_COMMIT_FORMAT",
    "EVOLUTION_NO_OP_COMMIT_FORMAT",
    "EVOLUTION_PHASE_OPEN_FORMAT",
    "LIBRARY_INITIALIZED_FORMAT",
    "PHASE_CHECKPOINT_ARTIFACT_FORMAT",
    "PHASE_CHECKPOINT_PUBLISHED_FORMAT",
    "RUN_CURSOR_VALUE_FORMAT",
    "TRAINING_STEP_COMMIT_FORMAT",
    "TRAINING_STEP_REPORT_FORMAT",
    "AuthoringUsageValue",
    "EvolutionCycleCommitted",
    "EvolutionMutationValue",
    "EvolutionNoOpCommitted",
    "EvolutionPhaseOpened",
    "LibraryInitialized",
    "PhaseCheckpointArtifact",
    "PhaseCheckpointPublished",
    "RunCursorValue",
    "TrainingStepCommit",
    "TrainingStepReportValue",
]
