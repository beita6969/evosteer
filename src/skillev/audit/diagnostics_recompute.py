"""Dependency-light diagnostic replay helpers for committed source segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import TrainingStepCommit
from skillev.diagnostics import (
    BatchDiagnostics,
    BatchFlowInput,
    DiagnosticsConfig,
    DiagnosticsState,
    FreshDiagnosticsSegment,
    assemble_batch_flow_input,
    observe_batch,
)

from .source_reducer import CommittedLibrarySegment


class DiagnosticsAuditKernel(Protocol):
    """The single pure diagnostics delta for a full-shaped arm audit."""

    def observe(
        self,
        state: DiagnosticsState,
        batch: BatchFlowInput,
        config: DiagnosticsConfig,
    ) -> tuple[DiagnosticsState, BatchDiagnostics]: ...


@dataclass(frozen=True, slots=True)
class FullDiagnosticsAuditKernel:
    def observe(
        self,
        state: DiagnosticsState,
        batch: BatchFlowInput,
        config: DiagnosticsConfig,
    ) -> tuple[DiagnosticsState, BatchDiagnostics]:
        return observe_batch(state, batch, config)


@dataclass(frozen=True, slots=True)
class ClippedImportanceDiagnosticsAuditKernel:
    clip: float

    def observe(
        self,
        state: DiagnosticsState,
        batch: BatchFlowInput,
        config: DiagnosticsConfig,
    ) -> tuple[DiagnosticsState, BatchDiagnostics]:
        from skillev.experiments.arms.clipped_importance import observe_clipped_batch

        return observe_clipped_batch(state, batch, config, clip=self.clip)


def diagnostic_input_for_commit(commit: TrainingStepCommit) -> BatchFlowInput:
    return assemble_batch_flow_input(
        commit.stats,
        {record.trajectory_id: record for record in commit.records},
        commit.edge_records,
    )


def recompute_diagnostics(
    segments: tuple[CommittedLibrarySegment, ...],
    *,
    config: DiagnosticsConfig,
    kernel: DiagnosticsAuditKernel,
) -> tuple[BatchDiagnostics, ...]:
    """Recompute all committed diagnostics with the arm's frozen pure kernel."""

    output: list[BatchDiagnostics] = []
    for segment in segments:
        state: DiagnosticsState = FreshDiagnosticsSegment(segment.library_version)
        for commit in segment.training_steps:
            state, diagnostic = kernel.observe(
                state,
                diagnostic_input_for_commit(commit),
                config,
            )
            output.append(diagnostic)
    return tuple(output)


__all__ = [
    "ClippedImportanceDiagnosticsAuditKernel",
    "DiagnosticsAuditKernel",
    "FullDiagnosticsAuditKernel",
    "diagnostic_input_for_commit",
    "recompute_diagnostics",
]
