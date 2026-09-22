"""Transactional online Protocol-v3 posterior calibration."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from skillev.contracts import (
    ContextFeature,
    PosteriorBatchUpdate,
    PosteriorCellState,
)
from skillev.diagnostics import BatchDiagnostics, TrajectoryFlowDiagnostic

from .core import (
    CalibrationConfig,
    CellQuery,
    calibration_updates_with_weights,
    flow_weights,
    prior_cell,
    query_cell,
)

if TYPE_CHECKING:
    from skillev.contracts import EdgeLogprobRecord, TrajectoryRecord, TTBBatchStats


class CalibrationSource(Protocol):
    """The small committed-source view consumed by posterior calibration."""

    @property
    def batch_id(self) -> str: ...

    @property
    def records(self) -> tuple[TrajectoryRecord, ...]: ...

    @property
    def stats(self) -> TTBBatchStats: ...

    @property
    def edge_records(self) -> tuple[EdgeLogprobRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class ConfidenceMultiplier:
    """Explicit confidence parameter for offline sensitivity queries."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError("confidence multiplier must be numeric")
        normalized = float(self.value)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("confidence multiplier must be non-negative and finite")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True, slots=True)
class CalibrationTransition:
    next_cells: Mapping[str, PosteriorCellState]
    posterior_batch: PosteriorBatchUpdate
    cells_before: Mapping[str, PosteriorCellState]
    trajectory_ids: tuple[str, ...]
    base_revision: int


class CalibrationEngine:
    """Preview a whole posterior batch and publish it atomically in memory."""

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._cells: dict[str, PosteriorCellState] = {}
        self._cell_source: Callable[[], tuple[PosteriorCellState, ...]] | None = None
        self._revision = 0
        self._consumed_trajectories: set[str] = set()

    def bind_committed_source(self, source: Callable[[], tuple[PosteriorCellState, ...]]) -> None:
        """Transfer ownership to a projection; every query now reads its snapshot.

        A bound engine is only a calculator/view. It cannot publish an independent
        posterior, including through the standalone compatibility commit API.
        """

        if self._cell_source is not None:
            raise RuntimeError("calibration already belongs to a projection")
        self._cell_source = source
        self._cells.clear()

    def _current_cells(self) -> dict[str, PosteriorCellState]:
        if self._cell_source is not None:
            return {cell.z.cell_key(cell.skill_id): cell for cell in self._cell_source()}
        return self._cells

    @classmethod
    def from_runtime_state(
        cls,
        config: CalibrationConfig,
        cells: tuple[PosteriorCellState, ...],
    ) -> CalibrationEngine:
        engine = cls(config)
        exact: dict[str, PosteriorCellState] = {}
        for cell in cells:
            key = cell.z.cell_key(cell.skill_id)
            if key in exact:
                raise ValueError("calibration runtime state repeats a cell")
            if float(cell.alpha_0) != config.alpha_0:
                raise ValueError("cell alpha prior differs from runtime config")
            if float(cell.beta_0) != config.beta_0:
                raise ValueError("cell beta prior differs from runtime config")
            exact[key] = cell
        engine._cells = exact
        return engine

    @property
    def config(self) -> CalibrationConfig:
        return self._config

    def preview(
        self,
        *,
        source: CalibrationSource,
        diagnostic: BatchDiagnostics,
    ) -> CalibrationTransition:
        if self._consumed_trajectories.intersection(
            record.trajectory_id for record in source.records
        ):
            raise ValueError("calibration trajectory evidence has already been committed")
        return self.preview_from_cells(
            source=source,
            diagnostic=diagnostic,
            cells_before=self._current_cells(),
        )

    def preview_from_cells(
        self,
        *,
        source: CalibrationSource,
        diagnostic: BatchDiagnostics,
        cells_before: Mapping[str, PosteriorCellState],
    ) -> CalibrationTransition:
        if (
            diagnostic.batch_id != source.batch_id
            or diagnostic.library_version != source.stats.library_version
        ):
            raise ValueError("calibration diagnostic belongs to another batch or library")
        records = {record.trajectory_id: record for record in source.records}
        if len(records) != len(source.records):
            raise ValueError("training source contains duplicate trajectories")
        if tuple(flow.trajectory_id for flow in diagnostic.trajectories) != tuple(records):
            raise ValueError("calibration requires the complete ordered batch")
        updates = calibration_updates_with_weights(
            batch_id=source.batch_id,
            flows=diagnostic.trajectories,
            records_by_id=records,
            cells_before=cells_before,
            config=self._config,
            weights=self._flow_weights(diagnostic.trajectories),
        )
        next_cells = dict(cells_before)
        for update in updates:
            key = update.z.cell_key(update.skill_id)
            if key in next_cells:
                prior = next_cells[key]
            else:
                prior = prior_cell(
                    skill_id=update.skill_id,
                    z=update.z,
                    config=self._config,
                )
            next_cells[key] = prior.apply(update)
        return CalibrationTransition(
            cells_before=dict(cells_before),
            trajectory_ids=tuple(records),
            base_revision=self._revision,
            next_cells=next_cells,
            posterior_batch=PosteriorBatchUpdate(
                batch_id=source.batch_id,
                updates=updates,
            ),
        )

    def _flow_weights(
        self,
        flows: tuple[TrajectoryFlowDiagnostic, ...],
    ) -> Mapping[tuple[str, int], float]:
        return flow_weights(flows)

    def commit(self, transition: CalibrationTransition) -> None:
        if self._cell_source is not None:
            raise RuntimeError("only the owning projection may commit calibration")
        if transition.base_revision != self._revision or transition.cells_before != self._cells:
            raise ValueError("calibration transition was prepared from an old state")
        if self._consumed_trajectories.intersection(transition.trajectory_ids):
            raise ValueError("calibration trajectory evidence has already been committed")
        self._cells = dict(transition.next_cells)
        self._consumed_trajectories.update(transition.trajectory_ids)
        self._revision += 1

    def query_configured(
        self,
        skill_id: str,
        z: ContextFeature,
    ) -> CellQuery:
        return self._query(
            skill_id,
            z,
            ConfidenceMultiplier(self._config.default_k),
        )

    def query_at_k(
        self,
        skill_id: str,
        z: ContextFeature,
        k: ConfidenceMultiplier,
    ) -> CellQuery:
        if not isinstance(k, ConfidenceMultiplier):
            raise TypeError("query_at_k requires ConfidenceMultiplier")
        return self._query(skill_id, z, k)

    def _query(
        self,
        skill_id: str,
        z: ContextFeature,
        k: ConfidenceMultiplier,
    ) -> CellQuery:
        if not isinstance(z, ContextFeature):
            raise ValueError("calibration query z must be ContextFeature")
        key = z.cell_key(skill_id)
        cell = self._current_cells().get(key)
        if cell is None:
            cell = prior_cell(skill_id=skill_id, z=z, config=self._config)
        return query_cell(cell, k.value)

    def cells_for_skill(self, skill_id: str) -> tuple[PosteriorCellState, ...]:
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill_id must be non-empty text")
        return tuple(
            sorted(
                (cell for cell in self._current_cells().values() if cell.skill_id == skill_id),
                key=_cell_sort_key,
            )
        )

    def all_cells(self) -> tuple[PosteriorCellState, ...]:
        return tuple(sorted(self._current_cells().values(), key=_cell_sort_key))


def _cell_sort_key(cell: PosteriorCellState) -> tuple[str, str]:
    return cell.z.content_hash, cell.skill_id


__all__ = [
    "CalibrationEngine",
    "CalibrationSource",
    "CalibrationTransition",
    "ConfidenceMultiplier",
]
