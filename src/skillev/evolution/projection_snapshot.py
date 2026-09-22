"""One immutable projection read for a phase's decision and authoring inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from skillev.contracts import PosteriorCellState
from skillev.diagnostics import BatchDiagnostics

from .evidence import AuthoringEdgeEvidence


@dataclass(frozen=True, slots=True)
class PhaseProjectionSnapshot:
    diagnostics: tuple[BatchDiagnostics, ...]
    calibration_cells: tuple[PosteriorCellState, ...]
    event_ids_by_cell: Mapping[str, tuple[str, ...]]
    authoring_by_edge_id: Mapping[str, AuthoringEdgeEvidence]

    source_by_trajectory: Mapping[str, tuple[str, str, str] | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_by_trajectory", MappingProxyType(dict(self.source_by_trajectory))
        )
        object.__setattr__(
            self, "event_ids_by_cell", MappingProxyType(dict(self.event_ids_by_cell))
        )
        object.__setattr__(
            self, "authoring_by_edge_id", MappingProxyType(dict(self.authoring_by_edge_id))
        )
