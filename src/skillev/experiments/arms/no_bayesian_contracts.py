"""Closed source contracts for the no-Bayesian SkillFlow-style arm.

These records deliberately remain separate from the full-method contracts:
the arm has no posterior updates and must never masquerade as a full TTB
source stream.  They nevertheless carry the same complete run cursor,
decision provenance, and library-mutation information required for an exact
offline reconstruction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias, assert_never

from skillev.contracts import (
    AuthoringUsageValue,
    EdgeLogprobRecord,
    JsonValue,
    PhaseCheckpointArtifact,
    PhaseTransitionEvent,
    RunCursorValue,
    TrainingStepReportValue,
    TrajectoryRecord,
    TTBBatchStats,
    normalize_json,
    stable_hash,
)
from skillev.contracts.identity import validate_sha256
from skillev.evolution.evidence import AuthoringEdgeEvidence, is_generate_candidate_action
from skillev.runtime.skill_library import SkillLibraryState
from skillev.runtime.skills import SkillDocument

FLOW_ONLY_ARM_FORMAT = "skillev-no-bayesian-arm@3"
FLOW_ONLY_TRAINING_STEP_FORMAT = "skillev-flow-only-training-step@2"
FLOW_ONLY_LIBRARY_INITIALIZED_FORMAT = "skillev-flow-only-library-initialized@1"
FLOW_ONLY_PHASE_OPENED_FORMAT = "skillev-flow-only-phase-opened@1"
FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED_FORMAT = "skillev-flow-only-phase-checkpoint-published@1"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    validate_sha256(text)
    return text


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _quantile(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("flow_quantile must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("flow_quantile must lie in [0, 1]")
    return result


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _json_documents(
    values: tuple[Mapping[str, JsonValue], ...],
    *,
    field: str,
) -> tuple[Mapping[str, JsonValue], ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    output: list[Mapping[str, JsonValue]] = []
    for value in values:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise TypeError(f"{field} must contain JSON objects")
        output.append(normalized)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class FlowOnlyRetainEvidence:
    log_skill_marginal_flow: float
    flow_quantile: float

    def __post_init__(self) -> None:
        if isinstance(self.log_skill_marginal_flow, bool) or not isinstance(
            self.log_skill_marginal_flow, int | float
        ):
            raise ValueError("log_skill_marginal_flow must be numeric")
        flow = float(self.log_skill_marginal_flow)
        if not math.isfinite(flow):
            raise ValueError("log_skill_marginal_flow must be finite")
        object.__setattr__(self, "log_skill_marginal_flow", flow)
        object.__setattr__(self, "flow_quantile", _quantile(self.flow_quantile))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": "flow-only-retain",
            "flow_quantile": self.flow_quantile,
            "log_skill_marginal_flow": self.log_skill_marginal_flow,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyRetainEvidence:
        data = _object(
            value,
            fields={"evidence_type", "flow_quantile", "log_skill_marginal_flow"},
            label="FlowOnlyRetainEvidence",
        )
        if data["evidence_type"] != "flow-only-retain":
            raise ValueError("flow-only Retain evidence has another kind")
        return cls(
            log_skill_marginal_flow=data["log_skill_marginal_flow"],
            flow_quantile=data["flow_quantile"],
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyGenerateEvidence:
    importance_edge_ids: tuple[str, ...]
    minimum_absolute_log_importance: float
    importance_quantile: float
    importance_semantics: str

    def __post_init__(self) -> None:
        if not self.importance_edge_ids:
            raise ValueError("flow-only Generate requires uncovered edge IDs")
        if any(type(item) is not str or not item.strip() for item in self.importance_edge_ids):
            raise ValueError("importance edge IDs must be non-empty text")
        if len(set(self.importance_edge_ids)) != len(self.importance_edge_ids):
            raise ValueError("importance edge IDs must be unique")
        if (
            not math.isfinite(self.minimum_absolute_log_importance)
            or self.minimum_absolute_log_importance <= 0.0
        ):
            raise ValueError("flow-only Generate requires a positive absolute floor")
        if not 0.0 <= self.importance_quantile <= 1.0:
            raise ValueError("flow-only Generate quantile must lie in [0, 1]")
        if self.importance_semantics != "absolute-log-density-ratio@1":
            raise ValueError("flow-only Generate uses unsupported semantics")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": "flow-only-generate",
            "importance_edge_ids": list(self.importance_edge_ids),
            "importance_quantile": self.importance_quantile,
            "importance_semantics": self.importance_semantics,
            "minimum_absolute_log_importance": self.minimum_absolute_log_importance,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyGenerateEvidence:
        data = _object(
            value,
            fields={
                "evidence_type",
                "importance_edge_ids",
                "importance_quantile",
                "importance_semantics",
                "minimum_absolute_log_importance",
            },
            label="FlowOnlyGenerateEvidence",
        )
        values = data["importance_edge_ids"]
        if data["evidence_type"] != "flow-only-generate" or not isinstance(values, list):
            raise ValueError("flow-only Generate evidence is incompatible")
        if any(type(item) is not str for item in values):
            raise TypeError("flow-only Generate edge IDs must be text")
        return cls(
            importance_edge_ids=tuple(values),
            minimum_absolute_log_importance=_number(
                data["minimum_absolute_log_importance"],
                field="minimum_absolute_log_importance",
            ),
            importance_quantile=_quantile(data["importance_quantile"]),
            importance_semantics=_text(data["importance_semantics"], field="importance_semantics"),
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyRetainProposal:
    target_skill_id: str
    evidence: FlowOnlyRetainEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.target_skill_id, field="target_skill_id")
        _text(self.rationale_text, field="rationale_text")
        if not isinstance(self.evidence, FlowOnlyRetainEvidence):
            raise TypeError("FlowOnlyRetainProposal requires FlowOnlyRetainEvidence")
        if not isinstance(self.edge_exemplars, tuple) or not self.edge_exemplars:
            raise ValueError("flow-only Retain requires edge exemplars")


@dataclass(frozen=True, slots=True)
class FlowOnlyGenerateProposal:
    evidence: FlowOnlyGenerateEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.rationale_text, field="rationale_text")
        if not isinstance(self.evidence, FlowOnlyGenerateEvidence):
            raise TypeError("FlowOnlyGenerateProposal requires FlowOnlyGenerateEvidence")
        if tuple(item.edge_id for item in self.edge_exemplars) != self.evidence.importance_edge_ids:
            raise ValueError("Generate exemplars must exactly match arm evidence")
        for exemplar in self.edge_exemplars:
            if exemplar.absolute_log_importance < self.evidence.minimum_absolute_log_importance:
                raise ValueError("Generate exemplar is below the absolute importance floor")
            if exemplar.log_importance_quantile < self.evidence.importance_quantile:
                raise ValueError("Generate exemplar is outside the selected upper tail")
            if exemplar.invoked_skill_ids:
                raise ValueError("Generate exemplar already has skill coverage")
            if not is_generate_candidate_action(exemplar.action_kind):
                raise ValueError("Generate exemplar action kind is ineligible")


FlowOnlyProposal: TypeAlias = FlowOnlyRetainProposal | FlowOnlyGenerateProposal


def flow_only_proposal_to_value(proposal: FlowOnlyProposal) -> dict[str, JsonValue]:
    match proposal:
        case FlowOnlyRetainProposal():
            return {
                "edge_exemplars": [item.to_value() for item in proposal.edge_exemplars],
                "evidence": proposal.evidence.to_value(),
                "proposal_type": "retain-compress",
                "rationale_text": proposal.rationale_text,
                "target_skill_id": proposal.target_skill_id,
            }
        case FlowOnlyGenerateProposal():
            return {
                "edge_exemplars": [item.to_value() for item in proposal.edge_exemplars],
                "evidence": proposal.evidence.to_value(),
                "proposal_type": "generate",
                "rationale_text": proposal.rationale_text,
                "target_skill_id": None,
            }
        case _ as unreachable:
            assert_never(unreachable)


def flow_only_proposal_content_hash(proposal: FlowOnlyProposal) -> str:
    return stable_hash(flow_only_proposal_to_value(proposal))


@dataclass(frozen=True, slots=True)
class FlowOnlyDecision:
    phase_event_id: str
    proposals: tuple[FlowOnlyProposal, ...]

    def __post_init__(self) -> None:
        _text(self.phase_event_id, field="phase_event_id")
        if not self.proposals:
            raise ValueError("a verified flow-only phase requires a non-empty decision")

    @property
    def proposal_content_hashes(self) -> tuple[str, ...]:
        return tuple(flow_only_proposal_content_hash(item) for item in self.proposals)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "phase_event_id": self.phase_event_id,
            "proposals": [flow_only_proposal_to_value(item) for item in self.proposals],
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class FlowOnlyActionResult:
    action_id: str
    proposal_content_hash: str
    action_kind: str
    target_skill_ids: tuple[str, ...]
    produced_skill_ids: tuple[str, ...]
    evidence: FlowOnlyRetainEvidence | FlowOnlyGenerateEvidence
    rationale_text: str

    def __post_init__(self) -> None:
        _text(self.action_id, field="action_id")
        _sha256(self.proposal_content_hash, field="proposal_content_hash")
        _text(self.rationale_text, field="rationale_text")
        if self.action_kind not in {"retain-compress", "generate"}:
            raise ValueError("unsupported flow-only action kind")
        if self.action_kind == "retain-compress":
            if len(self.target_skill_ids) != 1 or len(self.produced_skill_ids) != 1:
                raise ValueError("flow-only Retain requires 1-to-1 identity")
            if not isinstance(self.evidence, FlowOnlyRetainEvidence):
                raise TypeError("flow-only Retain result has wrong evidence")
        else:
            if self.target_skill_ids or len(self.produced_skill_ids) != 1:
                raise ValueError("flow-only Generate requires 0-to-1 identity")
            if not isinstance(self.evidence, FlowOnlyGenerateEvidence):
                raise TypeError("flow-only Generate result has wrong evidence")
        for field, values in (
            ("target_skill_ids", self.target_skill_ids),
            ("produced_skill_ids", self.produced_skill_ids),
        ):
            if any(type(item) is not str or not item for item in values):
                raise ValueError(f"{field} must contain non-empty text")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "evidence": self.evidence.to_value(),
            "produced_skill_ids": list(self.produced_skill_ids),
            "proposal_content_hash": self.proposal_content_hash,
            "rationale_text": self.rationale_text,
            "target_skill_ids": list(self.target_skill_ids),
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyActionResult:
        data = _object(
            value,
            fields={
                "action_id",
                "action_kind",
                "evidence",
                "produced_skill_ids",
                "proposal_content_hash",
                "rationale_text",
                "target_skill_ids",
            },
            label="FlowOnlyActionResult",
        )
        targets = data["target_skill_ids"]
        products = data["produced_skill_ids"]
        if not isinstance(targets, list) or not isinstance(products, list):
            raise TypeError("flow-only action skill IDs must be arrays")
        if any(type(item) is not str for item in (*targets, *products)):
            raise TypeError("flow-only action skill IDs must be text")
        kind = _text(data["action_kind"], field="action_kind")
        evidence = (
            FlowOnlyRetainEvidence.from_value(data["evidence"])
            if kind == "retain-compress"
            else FlowOnlyGenerateEvidence.from_value(data["evidence"])
        )
        return cls(
            action_id=_text(data["action_id"], field="action_id"),
            proposal_content_hash=_sha256(
                data["proposal_content_hash"], field="proposal_content_hash"
            ),
            action_kind=kind,
            target_skill_ids=tuple(targets),
            produced_skill_ids=tuple(products),
            evidence=evidence,
            rationale_text=_text(data["rationale_text"], field="rationale_text"),
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyLibraryInitialized:
    documents: tuple[Mapping[str, JsonValue], ...]
    active_skill_ids: tuple[str, ...]
    library_version: str
    initial_optimizer_step: int
    method_identity_hash: str
    run_cursor: RunCursorValue
    format: str = FLOW_ONLY_LIBRARY_INITIALIZED_FORMAT

    def __post_init__(self) -> None:
        normalized = _json_documents(self.documents, field="documents")
        object.__setattr__(self, "documents", normalized)
        documents = tuple(SkillDocument.from_value(item) for item in normalized)
        state = SkillLibraryState(
            documents={item.manifest.skill_id: item for item in documents},
            active_skill_ids=self.active_skill_ids,
            current_version=self.library_version,
        )
        if len(state.documents) != len(documents):
            raise ValueError("flow-only library source repeats a document")
        _integer(self.initial_optimizer_step, field="initial_optimizer_step")
        _sha256(self.method_identity_hash, field="method_identity_hash")
        if not isinstance(self.run_cursor, RunCursorValue):
            raise TypeError("flow-only library source requires RunCursorValue")
        if self.format != FLOW_ONLY_LIBRARY_INITIALIZED_FORMAT:
            raise ValueError("unsupported flow-only library source format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "active_skill_ids": list(self.active_skill_ids),
            "documents": [dict(item) for item in self.documents],
            "format": self.format,
            "initial_optimizer_step": self.initial_optimizer_step,
            "library_version": self.library_version,
            "method_identity_hash": self.method_identity_hash,
            "run_cursor": self.run_cursor.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyLibraryInitialized:
        data = _object(
            value,
            fields={
                "active_skill_ids",
                "documents",
                "format",
                "initial_optimizer_step",
                "library_version",
                "method_identity_hash",
                "run_cursor",
            },
            label="FlowOnlyLibraryInitialized",
        )
        documents = data["documents"]
        active = data["active_skill_ids"]
        if not isinstance(documents, list) or not isinstance(active, list):
            raise TypeError("flow-only library source arrays are invalid")
        if any(not isinstance(item, dict) for item in documents) or any(
            type(item) is not str for item in active
        ):
            raise TypeError("flow-only library source contents are invalid")
        return cls(
            documents=tuple(documents),
            active_skill_ids=tuple(active),
            library_version=_text(data["library_version"], field="library_version"),
            initial_optimizer_step=_integer(
                data["initial_optimizer_step"], field="initial_optimizer_step"
            ),
            method_identity_hash=_sha256(
                data["method_identity_hash"], field="method_identity_hash"
            ),
            run_cursor=RunCursorValue.from_value(data["run_cursor"]),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyPhaseOpened:
    phase_event: PhaseTransitionEvent
    run_cursor_at_phase: RunCursorValue
    format: str = FLOW_ONLY_PHASE_OPENED_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.phase_event, PhaseTransitionEvent):
            raise TypeError("flow-only phase source requires PhaseTransitionEvent")
        if not isinstance(self.run_cursor_at_phase, RunCursorValue):
            raise TypeError("flow-only phase source requires RunCursorValue")
        if self.run_cursor_at_phase.completed_training_steps < 1:
            raise ValueError("flow-only phase cannot open before training")
        if self.format != FLOW_ONLY_PHASE_OPENED_FORMAT:
            raise ValueError("unsupported flow-only phase source format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "phase_event": self.phase_event.to_value(),
            "run_cursor_at_phase": self.run_cursor_at_phase.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyPhaseOpened:
        data = _object(
            value,
            fields={"format", "phase_event", "run_cursor_at_phase"},
            label="FlowOnlyPhaseOpened",
        )
        return cls(
            phase_event=PhaseTransitionEvent.from_value(data["phase_event"]),
            run_cursor_at_phase=RunCursorValue.from_value(data["run_cursor_at_phase"]),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyPhaseCheckpointPublished:
    """Flow-only source binding for one post-cycle progress-anchor snapshot.

    This is intentionally not the full-method ``PhaseCheckpointPublished``
    event: the no-Bayesian arm owns a separate source stream and carries no
    posterior payload.  The shared artifact only identifies immutable private
    checkpoint bytes and their post-cycle execution state.
    """

    phase_event_id: str
    artifact: PhaseCheckpointArtifact
    format: str = FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED_FORMAT

    def __post_init__(self) -> None:
        _text(self.phase_event_id, field="flow-only phase checkpoint phase_event_id")
        if not isinstance(self.artifact, PhaseCheckpointArtifact):
            raise TypeError("flow-only phase checkpoint requires PhaseCheckpointArtifact")
        if self.artifact.phase_event_id != self.phase_event_id:
            raise ValueError("flow-only phase checkpoint artifact belongs to another phase")
        if self.format != FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED_FORMAT:
            raise ValueError("unsupported flow-only phase checkpoint format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifact": self.artifact.to_value(),
            "format": self.format,
            "phase_event_id": self.phase_event_id,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyPhaseCheckpointPublished:
        data = _object(
            value,
            fields={"artifact", "format", "phase_event_id"},
            label="FlowOnlyPhaseCheckpointPublished",
        )
        return cls(
            phase_event_id=_text(data["phase_event_id"], field="phase_event_id"),
            artifact=PhaseCheckpointArtifact.from_value(data["artifact"]),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyTrainingStepCommit:
    batch_id: str
    optimizer_step: int
    policy_snapshot_before: str
    policy_snapshot_after: str
    library_version: str
    records: tuple[TrajectoryRecord, ...]
    edge_records: tuple[EdgeLogprobRecord, ...]
    stats: TTBBatchStats
    report: TrainingStepReportValue
    run_cursor_after: RunCursorValue
    format: str = FLOW_ONLY_TRAINING_STEP_FORMAT

    def __post_init__(self) -> None:
        _text(self.batch_id, field="batch_id")
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        for field in ("policy_snapshot_before", "policy_snapshot_after", "library_version"):
            _text(getattr(self, field), field=field)
        if not self.records or any(not isinstance(item, TrajectoryRecord) for item in self.records):
            raise ValueError("flow-only training source requires trajectory records")
        if any(not isinstance(item, EdgeLogprobRecord) for item in self.edge_records):
            raise TypeError("flow-only training edges are invalid")
        if not isinstance(self.stats, TTBBatchStats):
            raise TypeError("flow-only training stats are invalid")
        if not isinstance(self.report, TrainingStepReportValue):
            raise TypeError("flow-only training report is invalid")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("flow-only training source requires RunCursorValue")
        if self.run_cursor_after.completed_training_steps < 1:
            raise ValueError("flow-only training cursor must include its step")
        if self.format != FLOW_ONLY_TRAINING_STEP_FORMAT:
            raise ValueError("unsupported flow-only training source format")
        if self.stats.batch_id != self.batch_id or self.stats.optimizer_step != self.optimizer_step:
            raise ValueError("flow-only training stats identity differs")
        if tuple(record.trajectory_id for record in self.records) != tuple(
            residual.trajectory_id for residual in self.stats.residuals
        ):
            raise ValueError("flow-only records and residuals differ")
        if (
            self.report.batch_id != self.batch_id
            or self.report.optimizer_step != self.optimizer_step
        ):
            raise ValueError("flow-only training report identity differs")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "edge_records": [item.to_value() for item in self.edge_records],
            "format": self.format,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_after": self.policy_snapshot_after,
            "policy_snapshot_before": self.policy_snapshot_before,
            "records": [item.to_value() for item in self.records],
            "report": self.report.to_value(),
            "run_cursor_after": self.run_cursor_after.to_value(),
            "stats": self.stats.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyTrainingStepCommit:
        data = _object(
            value,
            fields={
                "batch_id",
                "edge_records",
                "format",
                "library_version",
                "optimizer_step",
                "policy_snapshot_after",
                "policy_snapshot_before",
                "records",
                "report",
                "run_cursor_after",
                "stats",
            },
            label="FlowOnlyTrainingStepCommit",
        )
        records = data["records"]
        edges = data["edge_records"]
        if not isinstance(records, list) or not isinstance(edges, list):
            raise TypeError("flow-only training record arrays are invalid")
        return cls(
            batch_id=_text(data["batch_id"], field="batch_id"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            policy_snapshot_before=_text(
                data["policy_snapshot_before"], field="policy_snapshot_before"
            ),
            policy_snapshot_after=_text(
                data["policy_snapshot_after"], field="policy_snapshot_after"
            ),
            library_version=_text(data["library_version"], field="library_version"),
            records=tuple(TrajectoryRecord.from_value(item) for item in records),
            edge_records=tuple(EdgeLogprobRecord.from_value(item) for item in edges),
            stats=TTBBatchStats.from_value(data["stats"]),
            report=TrainingStepReportValue.from_value(data["report"]),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyCycleResult:
    arm_id: ClassVar[str] = "no-bayesian-calibration"

    phase_event_id: str
    optimizer_step: int
    library_version_before: str
    library_version_after: str
    new_documents: tuple[Mapping[str, JsonValue], ...]
    active_skill_ids_after: tuple[str, ...]
    actions: tuple[FlowOnlyActionResult, ...]
    decision_content_hash: str
    proposal_content_hashes: tuple[str, ...]
    authoring_reservation_ids: tuple[str, ...]
    authoring_usage: AuthoringUsageValue
    z_reset_seed: int
    z_version_after_reset: str
    run_cursor_after: RunCursorValue
    format: str = FLOW_ONLY_ARM_FORMAT

    def __post_init__(self) -> None:
        _text(self.phase_event_id, field="phase_event_id")
        _integer(self.optimizer_step, field="optimizer_step")
        _text(self.library_version_before, field="library_version_before")
        _text(self.library_version_after, field="library_version_after")
        if self.library_version_before == self.library_version_after:
            raise ValueError("flow-only cycle must change the library")
        if not self.actions or any(
            not isinstance(item, FlowOnlyActionResult) for item in self.actions
        ):
            raise ValueError("flow-only cycle requires actions")
        documents_value = _json_documents(self.new_documents, field="new_documents")
        object.__setattr__(self, "new_documents", documents_value)
        documents = tuple(SkillDocument.from_value(item) for item in documents_value)
        document_ids = tuple(item.manifest.skill_id for item in documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("flow-only cycle repeats new documents")
        if any(type(item) is not str or not item for item in self.active_skill_ids_after):
            raise ValueError("flow-only active skills must be non-empty text")
        produced = tuple(skill for action in self.actions for skill in action.produced_skill_ids)
        if set(produced) != set(document_ids):
            raise ValueError("flow-only documents differ from action products")
        _sha256(self.decision_content_hash, field="decision_content_hash")
        if not self.proposal_content_hashes:
            raise ValueError("flow-only cycle requires proposal hashes")
        for item in self.proposal_content_hashes:
            _sha256(item, field="proposal_content_hash")
        if (
            tuple(item.proposal_content_hash for item in self.actions)
            != self.proposal_content_hashes
        ):
            raise ValueError("flow-only actions differ from proposal hashes")
        if any(type(item) is not str or not item for item in self.authoring_reservation_ids):
            raise ValueError("flow-only reservation IDs must be non-empty text")
        if len(set(self.authoring_reservation_ids)) != len(self.authoring_reservation_ids):
            raise ValueError("flow-only cycle repeats authoring reservations")
        if not isinstance(self.authoring_usage, AuthoringUsageValue):
            raise TypeError("flow-only cycle requires exact authoring usage")
        if self.authoring_usage.model_calls != len(self.authoring_reservation_ids):
            raise ValueError("flow-only usage differs from reservations")
        if type(self.z_reset_seed) is not int or not 0 <= self.z_reset_seed < 2**64:
            raise ValueError("flow-only z_reset_seed must be uint64")
        _text(self.z_version_after_reset, field="z_version_after_reset")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("flow-only cycle requires RunCursorValue")
        if self.run_cursor_after.committed_cycles < 1:
            raise ValueError("flow-only cycle cursor must include one cycle")
        if self.format != FLOW_ONLY_ARM_FORMAT:
            raise ValueError("flow-only cycle format is incompatible")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "actions": [item.to_value() for item in self.actions],
            "active_skill_ids_after": list(self.active_skill_ids_after),
            "arm_id": self.arm_id,
            "authoring_reservation_ids": list(self.authoring_reservation_ids),
            "authoring_usage": self.authoring_usage.to_value(),
            "decision_content_hash": self.decision_content_hash,
            "format": self.format,
            "library_version_after": self.library_version_after,
            "library_version_before": self.library_version_before,
            "new_documents": [dict(item) for item in self.new_documents],
            "optimizer_step": self.optimizer_step,
            "phase_event_id": self.phase_event_id,
            "proposal_content_hashes": list(self.proposal_content_hashes),
            "run_cursor_after": self.run_cursor_after.to_value(),
            "z_reset_seed": self.z_reset_seed,
            "z_version_after_reset": self.z_version_after_reset,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyCycleResult:
        data = _object(
            value,
            fields={
                "actions",
                "active_skill_ids_after",
                "arm_id",
                "authoring_reservation_ids",
                "authoring_usage",
                "decision_content_hash",
                "format",
                "library_version_after",
                "library_version_before",
                "new_documents",
                "optimizer_step",
                "phase_event_id",
                "proposal_content_hashes",
                "run_cursor_after",
                "z_reset_seed",
                "z_version_after_reset",
            },
            label="FlowOnlyCycleResult",
        )
        if data["arm_id"] != cls.arm_id:
            raise ValueError("flow-only cycle has another arm ID")
        actions = data["actions"]
        documents = data["new_documents"]
        active = data["active_skill_ids_after"]
        proposal_hashes = data["proposal_content_hashes"]
        reservations = data["authoring_reservation_ids"]
        if not all(
            isinstance(item, list)
            for item in (actions, documents, active, proposal_hashes, reservations)
        ):
            raise TypeError("flow-only cycle arrays are invalid")
        if any(type(item) is not str for item in (*active, *proposal_hashes, *reservations)):
            raise TypeError("flow-only cycle text arrays are invalid")
        if any(not isinstance(item, dict) for item in documents):
            raise TypeError("flow-only cycle documents must be objects")
        return cls(
            phase_event_id=_text(data["phase_event_id"], field="phase_event_id"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step"),
            library_version_before=_text(
                data["library_version_before"], field="library_version_before"
            ),
            library_version_after=_text(
                data["library_version_after"], field="library_version_after"
            ),
            new_documents=tuple(documents),
            active_skill_ids_after=tuple(active),
            actions=tuple(FlowOnlyActionResult.from_value(item) for item in actions),
            decision_content_hash=_sha256(
                data["decision_content_hash"], field="decision_content_hash"
            ),
            proposal_content_hashes=tuple(proposal_hashes),
            authoring_reservation_ids=tuple(reservations),
            authoring_usage=AuthoringUsageValue.from_value(data["authoring_usage"]),
            z_reset_seed=_integer(data["z_reset_seed"], field="z_reset_seed"),
            z_version_after_reset=_text(
                data["z_version_after_reset"], field="z_version_after_reset"
            ),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


__all__ = [
    "FLOW_ONLY_ARM_FORMAT",
    "FLOW_ONLY_LIBRARY_INITIALIZED_FORMAT",
    "FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED_FORMAT",
    "FLOW_ONLY_PHASE_OPENED_FORMAT",
    "FLOW_ONLY_TRAINING_STEP_FORMAT",
    "FlowOnlyActionResult",
    "FlowOnlyCycleResult",
    "FlowOnlyDecision",
    "FlowOnlyGenerateEvidence",
    "FlowOnlyGenerateProposal",
    "FlowOnlyLibraryInitialized",
    "FlowOnlyPhaseCheckpointPublished",
    "FlowOnlyPhaseOpened",
    "FlowOnlyProposal",
    "FlowOnlyRetainEvidence",
    "FlowOnlyRetainProposal",
    "FlowOnlyTrainingStepCommit",
    "flow_only_proposal_content_hash",
    "flow_only_proposal_to_value",
]
