"""Atomic typed handoff from committed training math to method projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from skillev.calibration import (
    CalibrationConfig,
    CalibrationEngine,
    CellQuery,
    ConfidenceMultiplier,
)
from skillev.contracts import (
    ContextFeature,
    EdgeLogprobRecord,
    JsonValue,
    PhaseTransitionEvent,
    PosteriorBatchUpdate,
    PosteriorCellState,
    TrajectoryRecord,
    TTBBatchStats,
)
from skillev.diagnostics import (
    BatchDiagnostics,
    CommittedDiagnostic,
    DiagnosticsConfig,
    DiagnosticsSource,
    DiagnosticsState,
    DiagnosticsTransition,
    LatestDiagnosticState,
    NoCommittedDiagnostic,
    OnlineFlowDiagnostics,
    diagnostics_library_version,
    diagnostics_state_from_value,
    reset_diagnostics_segment,
)
from skillev.evolution.config import EvolutionConfig
from skillev.evolution.evidence import (
    AuthoringEdgeEvidence,
    TrajectoryEvidenceView,
    authoring_evidence_for_diagnostic,
)
from skillev.evolution.projection_snapshot import PhaseProjectionSnapshot
from skillev.rollout import RolloutArtifact
from skillev.rollout.generator import RolloutTokenizerProtocol

from .evidence_context import TrajectoryEvidenceContext
from .posterior_state import (
    POSTERIOR_PROVENANCE_FORMAT,
    PosteriorEventProvenance,
    PosteriorEvidenceBatch,
    validate_posterior_provenance,
)


@dataclass(frozen=True, slots=True)
class TrainingStepSource:
    batch_id: str
    optimizer_step: int
    artifacts: tuple[RolloutArtifact, ...]
    stats: TTBBatchStats
    edge_records: tuple[EdgeLogprobRecord, ...]

    def __post_init__(self) -> None:
        if self.stats.batch_id != self.batch_id or self.stats.optimizer_step != self.optimizer_step:
            raise ValueError("projection source differs from its batch statistics")
        if not self.artifacts:
            raise ValueError("projection requires a complete nonempty batch")
        trajectory_ids = tuple(artifact.record.trajectory_id for artifact in self.artifacts)
        if trajectory_ids != tuple(item.trajectory_id for item in self.stats.residuals):
            raise ValueError("projection residuals differ from the sealed trajectory order")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("projection repeats a trajectory")
        snapshot = self.artifacts[0].manifest.policy_snapshot
        for artifact in self.artifacts:
            if (
                artifact.manifest.policy_snapshot != snapshot
                or artifact.manifest.library_version != self.stats.library_version
            ):
                raise ValueError("projection mixes rollout policies or skill libraries")
        if any(
            edge.context is None
            or edge.context.policy_snapshot_id != snapshot.snapshot_id
            or edge.forward_adapter_version != snapshot.forward_adapter_version
            for edge in self.edge_records
        ):
            raise ValueError("projection scores differ from the rollout policy snapshot")

    @property
    def policy_snapshot_id(self) -> str:
        return self.artifacts[0].manifest.policy_snapshot.snapshot_id

    @property
    def records(self) -> tuple[TrajectoryRecord, ...]:
        return tuple(artifact.record for artifact in self.artifacts)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifacts": [artifact.to_value() for artifact in self.artifacts],
            "batch_id": self.batch_id,
            "edge_records": [edge.to_value() for edge in self.edge_records],
            "optimizer_step": self.optimizer_step,
            "stats": self.stats.to_value(),
        }

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        tokenizer: RolloutTokenizerProtocol,
    ) -> TrainingStepSource:
        if not isinstance(value, dict) or set(value) != {
            "artifacts",
            "batch_id",
            "edge_records",
            "optimizer_step",
            "stats",
        }:
            raise ValueError("TrainingStepSource has incompatible fields")
        artifacts = value["artifacts"]
        edge_records = value["edge_records"]
        batch_id = value["batch_id"]
        optimizer_step = value["optimizer_step"]
        if not isinstance(artifacts, list) or not isinstance(edge_records, list):
            raise ValueError("TrainingStepSource arrays have wrong types")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("TrainingStepSource batch_id must be non-empty text")
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("TrainingStepSource optimizer_step must be positive")
        return cls(
            batch_id=batch_id,
            optimizer_step=optimizer_step,
            artifacts=tuple(
                RolloutArtifact.from_value(item, tokenizer=tokenizer) for item in artifacts
            ),
            stats=TTBBatchStats.from_value(value["stats"]),
            edge_records=tuple(EdgeLogprobRecord.from_value(item) for item in edge_records),
        )


@dataclass(frozen=True, slots=True)
class ProjectionBatchRuntimeState:
    """One bounded diagnostic and its answer-free authoring evidence."""

    diagnostic: BatchDiagnostics
    authoring_evidence: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        expected_edge_ids = tuple(
            f"{trajectory.trajectory_id}:{edge.step_index}"
            for trajectory in self.diagnostic.trajectories
            for edge in trajectory.edges
        )
        if tuple(item.edge_id for item in self.authoring_evidence) != expected_edge_ids:
            raise ValueError("answer-free evidence differs from diagnostic edge order")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_evidence": [item.to_value() for item in self.authoring_evidence],
            "diagnostic": self.diagnostic.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> ProjectionBatchRuntimeState:
        if not isinstance(value, dict) or set(value) != {
            "authoring_evidence",
            "diagnostic",
        }:
            raise ValueError("ProjectionBatchRuntimeState has incompatible fields")
        evidence = value["authoring_evidence"]
        if not isinstance(evidence, list):
            raise ValueError("projection authoring evidence must be an array")
        return cls(
            diagnostic=BatchDiagnostics.from_value(value["diagnostic"]),
            authoring_evidence=tuple(AuthoringEdgeEvidence.from_value(item) for item in evidence),
        )


def _latest_to_value(latest: LatestDiagnosticState) -> dict[str, JsonValue]:
    if isinstance(latest, NoCommittedDiagnostic):
        return {"kind": "none"}
    if isinstance(latest, CommittedDiagnostic):
        return {"diagnostic": latest.diagnostic.to_value(), "kind": "committed"}
    raise TypeError("unsupported latest diagnostic state")


def _latest_from_value(value: object) -> LatestDiagnosticState:
    if not isinstance(value, dict):
        raise TypeError("latest diagnostic state must be an object")
    kind = value.get("kind")
    if kind == "none" and set(value) == {"kind"}:
        return NoCommittedDiagnostic()
    if kind == "committed" and set(value) == {"diagnostic", "kind"}:
        return CommittedDiagnostic(BatchDiagnostics.from_value(value["diagnostic"]))
    raise ValueError("latest diagnostic state has incompatible fields")


@dataclass(frozen=True, slots=True)
class FullProjectionRuntimeState:
    diagnostics_state: DiagnosticsState
    latest_diagnostic: LatestDiagnosticState
    current_segment_batches: tuple[ProjectionBatchRuntimeState, ...]
    calibration_cells: tuple[PosteriorCellState, ...]
    posterior_provenance: PosteriorEventProvenance
    retained_batch_count: int
    revision: int = 0
    kind: str = "full"
    format: str = "skillev-full-projection-runtime-state@7"

    def __post_init__(self) -> None:
        if self.kind != "full" or self.format != "skillev-full-projection-runtime-state@7":
            raise ValueError("unsupported full projection runtime state")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("projection revision must be nonnegative")
        if type(self.retained_batch_count) is not int or self.retained_batch_count < 1:
            raise ValueError("retained_batch_count must be positive")
        if len(self.current_segment_batches) > self.retained_batch_count:
            raise ValueError("projection batch buffer exceeds its fixed bound")
        _validate_latest(
            latest=self.latest_diagnostic,
            batches=self.current_segment_batches,
        )
        if any(
            item.diagnostic.library_version != diagnostics_library_version(self.diagnostics_state)
            for item in self.current_segment_batches
        ):
            raise ValueError("projection batches cross library segments")
        validate_posterior_provenance(
            cells=self.calibration_cells,
            provenance=self.posterior_provenance,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "revision": self.revision,
            "calibration_cells": [cell.to_value() for cell in self.calibration_cells],
            "current_segment_batches": [item.to_value() for item in self.current_segment_batches],
            "diagnostics_state": self.diagnostics_state.to_value(),
            "format": self.format,
            "kind": self.kind,
            "latest_diagnostic": _latest_to_value(self.latest_diagnostic),
            "posterior_provenance": self.posterior_provenance.to_value(),
            "retained_batch_count": self.retained_batch_count,
        }

    @classmethod
    def from_value(cls, value: object) -> FullProjectionRuntimeState:
        expected = {
            "revision",
            "calibration_cells",
            "current_segment_batches",
            "diagnostics_state",
            "format",
            "kind",
            "latest_diagnostic",
            "posterior_provenance",
            "retained_batch_count",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("FullProjectionRuntimeState has incompatible fields")
        raw_batches = value["current_segment_batches"]
        raw_cells = value["calibration_cells"]
        if not isinstance(raw_batches, list) or not isinstance(raw_cells, list):
            raise TypeError("projection batches and cells must be arrays")
        retained = value["retained_batch_count"]
        if type(retained) is not int:
            raise TypeError("retained_batch_count must be an integer")
        return cls(
            revision=value["revision"],
            diagnostics_state=diagnostics_state_from_value(value["diagnostics_state"]),
            latest_diagnostic=_latest_from_value(value["latest_diagnostic"]),
            current_segment_batches=tuple(
                ProjectionBatchRuntimeState.from_value(item) for item in raw_batches
            ),
            calibration_cells=tuple(PosteriorCellState.from_value(item) for item in raw_cells),
            posterior_provenance=PosteriorEventProvenance.from_value(value["posterior_provenance"]),
            retained_batch_count=retained,
            kind=value["kind"],
            format=value["format"],
        )


@dataclass(frozen=True, slots=True)
class FlowOnlyProjectionRuntimeState:
    diagnostics_state: DiagnosticsState
    latest_diagnostic: LatestDiagnosticState
    current_segment_batches: tuple[ProjectionBatchRuntimeState, ...]
    retained_batch_count: int
    kind: str = "flow-only"
    format: str = "skillev-flow-only-projection-runtime-state@2"

    def __post_init__(self) -> None:
        if (
            self.kind != "flow-only"
            or self.format != "skillev-flow-only-projection-runtime-state@2"
        ):
            raise ValueError("unsupported flow-only projection runtime state")
        if type(self.retained_batch_count) is not int or self.retained_batch_count < 1:
            raise ValueError("retained_batch_count must be positive")
        if len(self.current_segment_batches) > self.retained_batch_count:
            raise ValueError("flow-only batch buffer exceeds its fixed bound")
        _validate_latest(latest=self.latest_diagnostic, batches=self.current_segment_batches)
        if any(
            item.diagnostic.library_version != diagnostics_library_version(self.diagnostics_state)
            for item in self.current_segment_batches
        ):
            raise ValueError("flow-only projection batches cross library segments")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "current_segment_batches": [item.to_value() for item in self.current_segment_batches],
            "diagnostics_state": self.diagnostics_state.to_value(),
            "format": self.format,
            "kind": self.kind,
            "latest_diagnostic": _latest_to_value(self.latest_diagnostic),
            "retained_batch_count": self.retained_batch_count,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowOnlyProjectionRuntimeState:
        expected = {
            "current_segment_batches",
            "diagnostics_state",
            "format",
            "kind",
            "latest_diagnostic",
            "retained_batch_count",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("FlowOnlyProjectionRuntimeState has incompatible fields")
        raw_batches = value["current_segment_batches"]
        retained = value["retained_batch_count"]
        if not isinstance(raw_batches, list) or type(retained) is not int:
            raise TypeError("flow-only projection state fields have incompatible types")
        return cls(
            diagnostics_state=diagnostics_state_from_value(value["diagnostics_state"]),
            latest_diagnostic=_latest_from_value(value["latest_diagnostic"]),
            current_segment_batches=tuple(
                ProjectionBatchRuntimeState.from_value(item) for item in raw_batches
            ),
            retained_batch_count=retained,
            kind=value["kind"],
            format=value["format"],
        )


ProjectionRuntimeState: TypeAlias = FullProjectionRuntimeState | FlowOnlyProjectionRuntimeState


def projection_state_from_value(value: object) -> ProjectionRuntimeState:
    if not isinstance(value, dict):
        raise TypeError("projection runtime state must be an object")
    kind = value.get("kind")
    if kind == "full":
        return FullProjectionRuntimeState.from_value(value)
    if kind == "flow-only":
        return FlowOnlyProjectionRuntimeState.from_value(value)
    raise ValueError("unsupported projection runtime state kind")


@dataclass(frozen=True, slots=True)
class ProjectionTransition:
    source: TrainingStepSource
    next_state: FullProjectionRuntimeState
    posterior_batch: PosteriorBatchUpdate
    base_revision: int

    def __post_init__(self) -> None:
        diagnostic = self.diagnostic
        if (
            self.next_state.revision != self.base_revision + 1
            or diagnostic.batch_id != self.source.batch_id
            or diagnostic.optimizer_step != self.source.optimizer_step
            or diagnostic.library_version != self.source.stats.library_version
            or self.posterior_batch.batch_id != self.source.batch_id
            or not self.next_state.posterior_provenance.batches
            or self.next_state.posterior_provenance.batches[-1].posterior != self.posterior_batch
        ):
            raise ValueError("projection transition mixes training-step identities")

    @property
    def diagnostic(self) -> BatchDiagnostics:
        latest = self.next_state.latest_diagnostic
        if not isinstance(latest, CommittedDiagnostic):
            raise RuntimeError("projection transition lacks a committed diagnostic")
        return latest.diagnostic


class TrainingProjectionPipeline(Protocol):
    def preview(self, source: TrainingStepSource) -> ProjectionTransition: ...

    def commit(self, transition: ProjectionTransition) -> None: ...

    @property
    def latest_diagnostic(self) -> BatchDiagnostics: ...

    def reset_library_segment(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None: ...

    @property
    def diagnostics_state(self) -> DiagnosticsState: ...

    @property
    def calibration_cells(self) -> tuple[PosteriorCellState, ...]: ...

    def runtime_state(self) -> FullProjectionRuntimeState: ...


class OnlineDiagnosticsProjection(Protocol):
    @property
    def config(self) -> DiagnosticsConfig: ...

    @property
    def state(self) -> DiagnosticsState: ...

    @property
    def latest_state(self) -> LatestDiagnosticState: ...

    def preview_from_state(
        self,
        source: DiagnosticsSource,
        state: DiagnosticsState,
    ) -> DiagnosticsTransition: ...


class MethodProjectionPipeline:
    """Full method projection published by one immutable state replacement."""

    def __init__(
        self,
        *,
        diagnostics: OnlineDiagnosticsProjection,
        calibration: CalibrationEngine,
        state: FullProjectionRuntimeState,
    ) -> None:
        self._diagnostics = diagnostics
        self._calibration = calibration
        self._state = state
        calibration.bind_committed_source(lambda: self._state.calibration_cells)

    @classmethod
    def fresh(
        cls,
        *,
        diagnostics_config: DiagnosticsConfig,
        calibration: CalibrationEngine,
        library_version: str,
    ) -> MethodProjectionPipeline:
        diagnostics = OnlineFlowDiagnostics.fresh(
            diagnostics_config,
            library_version=library_version,
        )
        return cls.from_fresh_components(
            diagnostics=diagnostics,
            calibration=calibration,
        )

    @classmethod
    def from_fresh_components(
        cls,
        *,
        diagnostics: OnlineDiagnosticsProjection,
        calibration: CalibrationEngine,
    ) -> MethodProjectionPipeline:
        retained = max(1, 2 * diagnostics.config.window_size)
        state = FullProjectionRuntimeState(
            diagnostics_state=diagnostics.state,
            latest_diagnostic=diagnostics.latest_state,
            current_segment_batches=(),
            calibration_cells=calibration.all_cells(),
            posterior_provenance=PosteriorEventProvenance.empty(),
            retained_batch_count=retained,
        )
        return cls(diagnostics=diagnostics, calibration=calibration, state=state)

    @classmethod
    def from_runtime_state(
        cls,
        *,
        diagnostics_config: DiagnosticsConfig,
        calibration_config: CalibrationConfig,
        state: FullProjectionRuntimeState,
    ) -> MethodProjectionPipeline:
        expected_retained = max(1, 2 * diagnostics_config.window_size)
        if state.retained_batch_count != expected_retained:
            raise ValueError("retained_batch_count differs from diagnostics config")
        state.posterior_provenance.require_reconstructed_cells(state.calibration_cells)
        diagnostics = OnlineFlowDiagnostics.from_runtime_state(
            diagnostics_config,
            state.diagnostics_state,
            state.latest_diagnostic,
        )
        calibration = CalibrationEngine.from_runtime_state(
            calibration_config,
            state.calibration_cells,
        )
        return cls(diagnostics=diagnostics, calibration=calibration, state=state)

    @property
    def latest_diagnostic(self) -> BatchDiagnostics:
        latest = self._state.latest_diagnostic
        if not isinstance(latest, CommittedDiagnostic):
            raise RuntimeError("no diagnostic has been committed in this segment")
        return latest.diagnostic

    @property
    def diagnostics_state(self) -> DiagnosticsState:
        return self._state.diagnostics_state

    @property
    def calibration_cells(self) -> tuple[PosteriorCellState, ...]:
        return self._state.calibration_cells

    def preview(self, source: TrainingStepSource) -> ProjectionTransition:
        state = self._state
        state.posterior_provenance.require_new_batch(
            source.batch_id,
            source.optimizer_step,
            tuple(record.trajectory_id for record in source.records),
        )
        diagnostic_transition = self._diagnostics.preview_from_state(
            source,
            state.diagnostics_state,
        )
        cells_before = {cell.z.cell_key(cell.skill_id): cell for cell in state.calibration_cells}
        calibration_transition = self._calibration.preview_from_cells(
            source=source,
            diagnostic=diagnostic_transition.diagnostic,
            cells_before=cells_before,
        )
        provenance = state.posterior_provenance.append_batch(
            PosteriorEvidenceBatch(
                optimizer_step=source.optimizer_step,
                policy_snapshot_id=source.policy_snapshot_id,
                library_version=source.stats.library_version,
                trajectory_ids=tuple(record.trajectory_id for record in source.records),
                posterior=calibration_transition.posterior_batch,
                trajectory_contexts=tuple(
                    TrajectoryEvidenceContext.from_artifact(item) for item in source.artifacts
                ),
            )
        )
        from skillev.evolution.trajectory_contrast import attach_source_contrasts

        batch_state = ProjectionBatchRuntimeState(
            diagnostic=diagnostic_transition.diagnostic,
            authoring_evidence=attach_source_contrasts(
                source.artifacts,
                authoring_evidence_for_diagnostic(
                    {record.trajectory_id: record for record in source.records},
                    diagnostic_transition.diagnostic,
                ),
            ),
        )
        batches = (*state.current_segment_batches, batch_state)[-state.retained_batch_count :]
        next_state = FullProjectionRuntimeState(
            revision=state.revision + 1,
            diagnostics_state=diagnostic_transition.next_state,
            latest_diagnostic=CommittedDiagnostic(diagnostic_transition.diagnostic),
            current_segment_batches=batches,
            calibration_cells=tuple(
                sorted(
                    calibration_transition.next_cells.values(),
                    key=lambda cell: (cell.z.content_hash, cell.skill_id),
                )
            ),
            posterior_provenance=provenance,
            retained_batch_count=state.retained_batch_count,
        )
        return ProjectionTransition(
            base_revision=state.revision,
            source=source,
            next_state=next_state,
            posterior_batch=calibration_transition.posterior_batch,
        )

    def commit(self, transition: ProjectionTransition) -> None:
        if (
            transition.base_revision != self._state.revision
            or transition.next_state.revision != self._state.revision + 1
        ):
            raise ValueError("projection transition was prepared from an old state")
        self._state = transition.next_state

    def query_configured(self, skill_id: str, z: ContextFeature) -> CellQuery:
        """Post-hoc query of the sole committed projection snapshot."""
        return self._calibration.query_configured(skill_id, z)

    def query_at_k(self, skill_id: str, z: ContextFeature, k: ConfidenceMultiplier) -> CellQuery:
        return self._calibration.query_at_k(skill_id, z, k)

    def reset_library_segment(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None:
        self.commit_reset(
            self.preview_reset(
                old_version=old_library_version,
                new_version=new_library_version,
            )
        )

    def preview_reset(
        self,
        *,
        old_version: str,
        new_version: str,
    ) -> FullProjectionRuntimeState:
        return FullProjectionRuntimeState(
            revision=self._state.revision + 1,
            diagnostics_state=reset_diagnostics_segment(
                self._state.diagnostics_state,
                old_library_version=old_version,
                new_library_version=new_version,
            ),
            latest_diagnostic=NoCommittedDiagnostic(),
            current_segment_batches=(),
            calibration_cells=self._state.calibration_cells,
            posterior_provenance=self._state.posterior_provenance,
            retained_batch_count=self._state.retained_batch_count,
        )

    def commit_reset(self, state: FullProjectionRuntimeState) -> None:
        if state.revision != self._state.revision + 1:
            raise ValueError("projection reset was prepared from an old state")
        self._state = state

    @property
    def diagnostics_history(self) -> tuple[BatchDiagnostics, ...]:
        return tuple(item.diagnostic for item in self._state.current_segment_batches)

    @property
    def event_ids_by_skill(self) -> Mapping[str, tuple[str, ...]]:
        return {
            skill_id: self._state.posterior_provenance.event_ids_for_skill(skill_id)
            for skill_id in sorted({cell.skill_id for cell in self._state.calibration_cells})
        }

    @property
    def event_ids_by_cell(self) -> Mapping[str, tuple[str, ...]]:
        return {
            cell.z.cell_key(cell.skill_id): self._state.posterior_provenance.event_ids_for_cell(
                cell.z.cell_key(cell.skill_id)
            )
            for cell in self._state.calibration_cells
        }

    @property
    def posterior_provenance(self) -> PosteriorEventProvenance:
        return self._state.posterior_provenance

    def diagnostics_for_batch_ids(
        self,
        batch_ids: tuple[str, ...],
    ) -> tuple[BatchDiagnostics, ...]:
        by_batch = {
            item.diagnostic.batch_id: item.diagnostic
            for item in self._state.current_segment_batches
        }
        if any(batch_id not in by_batch for batch_id in batch_ids):
            raise KeyError("phase references a diagnostic absent from bounded runtime state")
        return tuple(by_batch[batch_id] for batch_id in batch_ids)

    def authoring_evidence_for_diagnostics(
        self,
        diagnostics: tuple[BatchDiagnostics, ...],
    ) -> Mapping[str, AuthoringEdgeEvidence]:
        requested = tuple(item.batch_id for item in diagnostics)
        by_batch = {item.diagnostic.batch_id: item for item in self._state.current_segment_batches}
        if any(batch_id not in by_batch for batch_id in requested):
            raise KeyError("triggering window references a batch absent from bounded state")
        return {
            evidence.edge_id: evidence
            for batch_id in requested
            for evidence in by_batch[batch_id].authoring_evidence
        }

    def runtime_state(self) -> FullProjectionRuntimeState:
        return self._state

    def cold_start_supported(self, config: EvolutionConfig) -> bool:
        from skillev.diagnostics import ComparableResidualWindows
        from skillev.evolution.cold_start import cold_start_groups

        state = self._state
        latest = state.latest_diagnostic
        if config.cold_start is None or not isinstance(latest, CommittedDiagnostic):
            return False
        residual = latest.diagnostic.residual_window
        if not isinstance(residual, ComparableResidualWindows) or not residual.stagnant:
            return False
        selected = state.current_segment_batches[-2 * residual.window_size :]
        return bool(
            cold_start_groups(
                tuple(item.diagnostic for item in selected),
                TrajectoryEvidenceView(
                    {edge.edge_id: edge for item in selected for edge in item.authoring_evidence},
                    _source_coordinates(
                        state, tuple(item.diagnostic.batch_id for item in selected)
                    ),
                ),
                config,
            )
        )

    def freeze_for_phase(self, phase: PhaseTransitionEvent) -> PhaseProjectionSnapshot:
        """Read cells, provenance and window from one committed state reference."""
        state = self._state
        latest = state.latest_diagnostic
        if (
            not isinstance(latest, CommittedDiagnostic)
            or latest.diagnostic.optimizer_step != phase.triggered_at_step
            or latest.diagnostic.library_version != phase.library_version
        ):
            raise ValueError("phase evidence cutoff differs from the committed projection")
        batch_ids = (
            *phase.previous_window.member_batch_ids,
            *phase.current_window.member_batch_ids,
        )
        by_batch = {item.diagnostic.batch_id: item for item in state.current_segment_batches}
        selected = tuple(by_batch[batch_id] for batch_id in batch_ids)
        return PhaseProjectionSnapshot(
            source_by_trajectory=_source_coordinates(state, batch_ids),
            diagnostics=tuple(item.diagnostic for item in selected),
            calibration_cells=state.calibration_cells,
            event_ids_by_cell={
                cell.z.cell_key(cell.skill_id): state.posterior_provenance.event_ids_for_cell(
                    cell.z.cell_key(cell.skill_id)
                )
                for cell in state.calibration_cells
            },
            authoring_by_edge_id={
                evidence.edge_id: evidence
                for item in selected
                for evidence in item.authoring_evidence
            },
        )


def _source_coordinates(
    state: FullProjectionRuntimeState, batch_ids: tuple[str, ...]
) -> dict[str, tuple[str, str, str] | None]:
    selected = set(batch_ids)
    return {
        context.trajectory_id: context.source_key
        for batch in state.posterior_provenance.batches
        if batch.posterior.batch_id in selected
        for context in batch.trajectory_contexts
    }


def _validate_latest(
    *,
    latest: LatestDiagnosticState,
    batches: tuple[ProjectionBatchRuntimeState, ...],
) -> None:
    if isinstance(latest, NoCommittedDiagnostic):
        if batches:
            raise ValueError("no-committed diagnostic state cannot retain batches")
        return
    if isinstance(latest, CommittedDiagnostic):
        if not batches or latest.diagnostic != batches[-1].diagnostic:
            raise ValueError("latest diagnostic differs from the bounded batch tail")
        return
    raise TypeError("unsupported latest diagnostic state")


__all__ = [
    "POSTERIOR_PROVENANCE_FORMAT",
    "FlowOnlyProjectionRuntimeState",
    "FullProjectionRuntimeState",
    "MethodProjectionPipeline",
    "OnlineDiagnosticsProjection",
    "PosteriorEventProvenance",
    "ProjectionBatchRuntimeState",
    "ProjectionRuntimeState",
    "ProjectionTransition",
    "TrainingProjectionPipeline",
    "TrainingStepSource",
    "projection_state_from_value",
]
