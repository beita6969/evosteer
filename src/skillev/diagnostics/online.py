"""Transactional online projection for exact flow diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from .core import (
    BatchDiagnostics,
    DiagnosticsConfig,
    DiagnosticsState,
    FreshDiagnosticsSegment,
    assemble_batch_flow_input,
    observe_batch,
    reset_diagnostics_segment,
)

if TYPE_CHECKING:
    from skillev.contracts import EdgeLogprobRecord, TrajectoryRecord, TTBBatchStats


class DiagnosticsSource(Protocol):
    @property
    def records(self) -> tuple[TrajectoryRecord, ...]: ...

    @property
    def stats(self) -> TTBBatchStats: ...

    @property
    def edge_records(self) -> tuple[EdgeLogprobRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class DiagnosticsTransition:
    next_state: DiagnosticsState
    diagnostic: BatchDiagnostics


@dataclass(frozen=True, slots=True)
class NoCommittedDiagnostic:
    kind: str = "none"


@dataclass(frozen=True, slots=True)
class CommittedDiagnostic:
    diagnostic: BatchDiagnostics
    kind: str = "committed"


LatestDiagnosticState: TypeAlias = NoCommittedDiagnostic | CommittedDiagnostic


class OnlineFlowDiagnostics:
    """Preview without mutation and publish by one reference replacement."""

    def __init__(
        self,
        config: DiagnosticsConfig,
        *,
        state: DiagnosticsState,
        latest: LatestDiagnosticState,
    ) -> None:
        self._config = config
        self._state = state
        self._latest = latest

    @classmethod
    def fresh(
        cls,
        config: DiagnosticsConfig,
        *,
        library_version: str,
    ) -> OnlineFlowDiagnostics:
        return cls(
            config,
            state=FreshDiagnosticsSegment(expected_library_version=library_version),
            latest=NoCommittedDiagnostic(),
        )

    @classmethod
    def from_runtime_state(
        cls,
        config: DiagnosticsConfig,
        state: DiagnosticsState,
        latest: LatestDiagnosticState,
    ) -> OnlineFlowDiagnostics:
        return cls(config, state=state, latest=latest)

    @property
    def config(self) -> DiagnosticsConfig:
        return self._config

    @property
    def state(self) -> DiagnosticsState:
        return self._state

    @property
    def latest_state(self) -> LatestDiagnosticState:
        return self._latest

    @property
    def latest(self) -> BatchDiagnostics:
        match self._latest:
            case CommittedDiagnostic(diagnostic=diagnostic):
                return diagnostic
            case NoCommittedDiagnostic():
                raise RuntimeError("no diagnostic has been committed in this segment")
            case _:
                raise TypeError("unsupported latest-diagnostic state")

    def preview(self, source: DiagnosticsSource) -> DiagnosticsTransition:
        return self.preview_from_state(source, self._state)

    def preview_from_state(
        self,
        source: DiagnosticsSource,
        state: DiagnosticsState,
    ) -> DiagnosticsTransition:
        records = {record.trajectory_id: record for record in source.records}
        if len(records) != len(source.records):
            raise ValueError("training source contains duplicate trajectories")
        batch = assemble_batch_flow_input(source.stats, records, source.edge_records)
        next_state, diagnostic = observe_batch(state, batch, self._config)
        return DiagnosticsTransition(next_state=next_state, diagnostic=diagnostic)

    def commit(self, transition: DiagnosticsTransition) -> None:
        self._state = transition.next_state
        self._latest = CommittedDiagnostic(transition.diagnostic)

    def reset_library_segment(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None:
        self._state = reset_diagnostics_segment(
            self._state,
            old_library_version=old_library_version,
            new_library_version=new_library_version,
        )
        self._latest = NoCommittedDiagnostic()


__all__ = [
    "CommittedDiagnostic",
    "DiagnosticsSource",
    "DiagnosticsTransition",
    "LatestDiagnosticState",
    "NoCommittedDiagnostic",
    "OnlineFlowDiagnostics",
]
