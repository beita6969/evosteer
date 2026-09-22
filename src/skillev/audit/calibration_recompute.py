"""Dependency-light posterior replay kernels over committed source records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from skillev.calibration import (
    CalibrationConfig,
    calibration_updates_with_weights,
    flow_weights,
    prior_cell,
)
from skillev.contracts import (
    PosteriorBatchUpdate,
    PosteriorCellState,
    PosteriorUpdateEvent,
    TrainingStepCommit,
)
from skillev.diagnostics import BatchDiagnostics

from .source_reducer import AuditEvidenceMismatchError, CommittedLibrarySegment


class CalibrationAuditKernel(Protocol):
    """Frozen edge-weight rule used by one full-shaped arm audit."""

    def updates(
        self,
        *,
        commit: TrainingStepCommit,
        diagnostic: BatchDiagnostics,
        cells_before: Mapping[str, PosteriorCellState],
        config: CalibrationConfig,
    ) -> tuple[PosteriorUpdateEvent, ...]: ...


@dataclass(frozen=True, slots=True)
class FullCalibrationAuditKernel:
    def updates(
        self,
        *,
        commit: TrainingStepCommit,
        diagnostic: BatchDiagnostics,
        cells_before: Mapping[str, PosteriorCellState],
        config: CalibrationConfig,
    ) -> tuple[PosteriorUpdateEvent, ...]:
        return calibration_updates_with_weights(
            batch_id=commit.batch_id,
            flows=diagnostic.trajectories,
            records_by_id={record.trajectory_id: record for record in commit.records},
            cells_before=cells_before,
            config=config,
            weights=flow_weights(diagnostic.trajectories),
        )


@dataclass(frozen=True, slots=True)
class UnitFlowCalibrationAuditKernel:
    def updates(
        self,
        *,
        commit: TrainingStepCommit,
        diagnostic: BatchDiagnostics,
        cells_before: Mapping[str, PosteriorCellState],
        config: CalibrationConfig,
    ) -> tuple[PosteriorUpdateEvent, ...]:
        from skillev.experiments.arms.no_flow_weighting import unit_flow_weights_for

        return calibration_updates_with_weights(
            batch_id=commit.batch_id,
            flows=diagnostic.trajectories,
            records_by_id={record.trajectory_id: record for record in commit.records},
            cells_before=cells_before,
            config=config,
            weights=unit_flow_weights_for(diagnostic.trajectories),
        )


@dataclass(frozen=True, slots=True)
class CappedFlowCalibrationAuditKernel:
    cap: float

    def __post_init__(self) -> None:
        from skillev.experiments.protocol import CappedFlowWeightArmProtocol

        if self.cap != CappedFlowWeightArmProtocol.CAP:
            raise ValueError("capped-flow calibration kernel differs from preregistered cap")

    def updates(
        self,
        *,
        commit: TrainingStepCommit,
        diagnostic: BatchDiagnostics,
        cells_before: Mapping[str, PosteriorCellState],
        config: CalibrationConfig,
    ) -> tuple[PosteriorUpdateEvent, ...]:
        from skillev.experiments.arms.capped_flow_weight import capped_flow_weights_for

        return calibration_updates_with_weights(
            batch_id=commit.batch_id,
            flows=diagnostic.trajectories,
            records_by_id={record.trajectory_id: record for record in commit.records},
            cells_before=cells_before,
            config=config,
            weights=capped_flow_weights_for(diagnostic.trajectories, cap=self.cap),
        )


def recompute_posterior_step(
    commit: TrainingStepCommit,
    diagnostic: BatchDiagnostics,
    *,
    cells_before: Mapping[str, PosteriorCellState],
    config: CalibrationConfig,
    kernel: CalibrationAuditKernel,
) -> tuple[dict[str, PosteriorCellState], PosteriorBatchUpdate]:
    """Rebuild one source batch with a predeclared weighting kernel."""

    updates = kernel.updates(
        commit=commit,
        diagnostic=diagnostic,
        cells_before=cells_before,
        config=config,
    )
    recomputed = PosteriorBatchUpdate(batch_id=commit.batch_id, updates=updates)
    if recomputed.content_hash != commit.posterior_batch.content_hash:
        raise AuditEvidenceMismatchError("posterior batch does not recompute")
    next_cells = dict(cells_before)
    for update in updates:
        key = update.z.cell_key(update.skill_id)
        prior = next_cells.get(key)
        if prior is None:
            prior = prior_cell(skill_id=update.skill_id, z=update.z, config=config)
        next_cells[key] = prior.apply(update)
    return next_cells, recomputed


def recompute_posterior_batches(
    segments: tuple[CommittedLibrarySegment, ...],
    diagnostics: tuple[BatchDiagnostics, ...],
    *,
    config: CalibrationConfig,
    kernel: CalibrationAuditKernel,
) -> tuple[tuple[PosteriorCellState, ...], tuple[PosteriorBatchUpdate, ...]]:
    """Fold one already-recomputed diagnostic stream through a fixed kernel."""

    commits = tuple(commit for segment in segments for commit in segment.training_steps)
    if len(commits) != len(diagnostics):
        raise AuditEvidenceMismatchError(
            "diagnostics do not align with committed posterior batches"
        )
    cells: dict[str, PosteriorCellState] = {}
    batches: list[PosteriorBatchUpdate] = []
    for commit, diagnostic in zip(commits, diagnostics, strict=True):
        cells, batch = recompute_posterior_step(
            commit,
            diagnostic,
            cells_before=cells,
            config=config,
            kernel=kernel,
        )
        batches.append(batch)
    return (
        tuple(sorted(cells.values(), key=lambda cell: (cell.z.content_hash, cell.skill_id))),
        tuple(batches),
    )


__all__ = [
    "CalibrationAuditKernel",
    "CappedFlowCalibrationAuditKernel",
    "FullCalibrationAuditKernel",
    "UnitFlowCalibrationAuditKernel",
    "recompute_posterior_batches",
    "recompute_posterior_step",
]
