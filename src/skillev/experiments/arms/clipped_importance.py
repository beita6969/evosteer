"""The clipped-importance arm's alternate trajectory estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass

from skillev.diagnostics import (
    ActiveDiagnosticsSegment,
    BatchDiagnostics,
    BatchFlowInput,
    CommittedDiagnostic,
    ComparableResidualWindows,
    DiagnosticsConfig,
    DiagnosticsSource,
    DiagnosticsState,
    DiagnosticsTransition,
    EdgeFlowDiagnostic,
    FreshDiagnosticsSegment,
    InsufficientResidualWindow,
    LatestDiagnosticState,
    LibrarySegmentMismatchError,
    NoCommittedDiagnostic,
    ResidualBatchEvidence,
    ResidualWindowEvidence,
    TrajectoryFlowDiagnostic,
    TrajectoryFlowInput,
    ZeroPreviousResidual,
    assemble_batch_flow_input,
    reset_diagnostics_segment,
    skill_marginal_flows,
)


@dataclass(frozen=True, slots=True)
class ClippedEdgeFlowDiagnostic:
    trajectory_id: str
    step_index: int
    raw_log_importance: float
    clipped_log_importance: float
    sample_log_state_weight: float
    invoked_skill_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClippedTrajectoryFlowDiagnostic:
    trajectory_id: str
    horizon: int
    edges: tuple[ClippedEdgeFlowDiagnostic, ...]
    terminal_sample_log_state_weight: float
    clip: float


def clipped_trajectory_flow(
    trajectory: TrajectoryFlowInput,
    clip: float,
) -> ClippedTrajectoryFlowDiagnostic:
    if (
        isinstance(clip, bool)
        or not isinstance(clip, int | float)
        or not math.isfinite(float(clip))
        or float(clip) <= 0.0
    ):
        raise ValueError("clip must be finite and positive")
    bound = float(clip)
    running = 0.0
    output: list[ClippedEdgeFlowDiagnostic] = []
    for edge, skills in zip(
        trajectory.edges,
        trajectory.invoked_skill_ids_by_step,
        strict=True,
    ):
        clipped = min(bound, max(-bound, edge.step_importance))
        running += clipped
        output.append(
            ClippedEdgeFlowDiagnostic(
                trajectory_id=trajectory.trajectory_id,
                step_index=edge.step_index,
                raw_log_importance=edge.step_importance,
                clipped_log_importance=clipped,
                sample_log_state_weight=running,
                invoked_skill_ids=skills,
            )
        )
    return ClippedTrajectoryFlowDiagnostic(
        trajectory_id=trajectory.trajectory_id,
        horizon=trajectory.horizon,
        edges=tuple(output),
        terminal_sample_log_state_weight=running,
        clip=bound,
    )


def _as_core_flow(
    trajectory: TrajectoryFlowInput,
    clip: float,
) -> TrajectoryFlowDiagnostic:
    clipped = clipped_trajectory_flow(trajectory, clip)
    return TrajectoryFlowDiagnostic(
        trajectory_id=clipped.trajectory_id,
        horizon=clipped.horizon,
        edges=tuple(
            EdgeFlowDiagnostic(
                trajectory_id=edge.trajectory_id,
                step_index=edge.step_index,
                log_importance=edge.raw_log_importance,
                sample_log_state_weight=edge.sample_log_state_weight,
                invoked_skill_ids=edge.invoked_skill_ids,
            )
            for edge in clipped.edges
        ),
        terminal_sample_log_state_weight=clipped.terminal_sample_log_state_weight,
    )


def _checked_clip(clip: float) -> float:
    if (
        isinstance(clip, bool)
        or not isinstance(clip, int | float)
        or not math.isfinite(float(clip))
        or float(clip) <= 0.0
    ):
        raise ValueError("clip must be finite and positive")
    return float(clip)


def observe_clipped_batch(
    state: DiagnosticsState,
    batch: BatchFlowInput,
    config: DiagnosticsConfig,
    *,
    clip: float,
) -> tuple[DiagnosticsState, BatchDiagnostics]:
    """Pure clipped-arm equivalent of :func:`observe_batch`.

    Both the live projection and offline arm audit use this function.  It
    accepts only the already joined canonical batch and never accesses model or
    rollout state.
    """

    bound = _checked_clip(clip)
    library_version = batch.stats.library_version
    match state:
        case FreshDiagnosticsSegment(expected_library_version=expected):
            if library_version != expected:
                raise LibrarySegmentMismatchError(
                    "clipped diagnostic batch differs from its explicit fresh segment"
                )
            prior_batches: tuple[ResidualBatchEvidence, ...] = ()
        case ActiveDiagnosticsSegment(
            library_version=expected,
            recent_squared_residual_batches=prior_batches,
        ):
            if library_version != expected:
                raise LibrarySegmentMismatchError(
                    "clipped diagnostic batch crosses an unreset library segment"
                )
        case _ as unmatched:
            from typing import assert_never

            assert_never(unmatched)

    by_id = {item.trajectory_id: item for item in batch.trajectories}
    trajectories = tuple(
        _as_core_flow(by_id[residual.trajectory_id], bound) for residual in batch.stats.residuals
    )
    current = ResidualBatchEvidence(tuple(residual.delta**2 for residual in batch.stats.residuals))
    all_batches = (*prior_batches, current)[-2 * config.window_size :]
    next_state: DiagnosticsState = ActiveDiagnosticsSegment(
        library_version=library_version,
        recent_squared_residual_batches=all_batches,
    )
    required = 2 * config.window_size
    if len(all_batches) < required:
        residual_window: ResidualWindowEvidence = InsufficientResidualWindow(
            observed_batch_count=len(all_batches),
            required_batch_count=required,
        )
    else:
        recent = all_batches[-2 * config.window_size :]
        previous_mean = _pooled_mean(
            tuple(item.delta_squared for item in recent[: config.window_size])
        )
        current_mean = _pooled_mean(
            tuple(item.delta_squared for item in recent[config.window_size :])
        )
        if previous_mean == 0.0:
            residual_window = ZeroPreviousResidual(
                library_version=library_version,
                window_size=config.window_size,
                previous_window_mean_delta_squared=previous_mean,
                current_window_mean_delta_squared=current_mean,
            )
        else:
            improvement = (previous_mean - current_mean) / previous_mean
            residual_window = ComparableResidualWindows(
                library_version=library_version,
                window_size=config.window_size,
                previous_window_mean_delta_squared=previous_mean,
                current_window_mean_delta_squared=current_mean,
                relative_improvement=improvement,
                rho=config.stagnation_rho,
                stagnant=improvement < config.stagnation_rho,
            )
    return (
        next_state,
        BatchDiagnostics(
            batch_id=batch.stats.batch_id,
            optimizer_step=batch.stats.optimizer_step,
            library_version=library_version,
            trajectories=trajectories,
            skill_flows=skill_marginal_flows(trajectories),
            residual_window=residual_window,
            config=config,
        ),
    )


class ClippedOnlineFlowDiagnostics:
    """Thin live wrapper around the clipped-arm pure diagnostics kernel."""

    def __init__(
        self,
        config: DiagnosticsConfig,
        *,
        clip: float,
        state: DiagnosticsState,
        latest: LatestDiagnosticState,
    ) -> None:
        self._config = config
        self._clip = _checked_clip(clip)
        self._state = state
        self._latest = latest

    @classmethod
    def fresh(
        cls,
        config: DiagnosticsConfig,
        *,
        clip: float,
        library_version: str,
    ) -> ClippedOnlineFlowDiagnostics:
        return cls(
            config,
            clip=clip,
            state=FreshDiagnosticsSegment(expected_library_version=library_version),
            latest=NoCommittedDiagnostic(),
        )

    @classmethod
    def from_runtime_state(
        cls,
        config: DiagnosticsConfig,
        *,
        clip: float,
        state: DiagnosticsState,
        latest: LatestDiagnosticState,
    ) -> ClippedOnlineFlowDiagnostics:
        return cls(config, clip=clip, state=state, latest=latest)

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
                raise RuntimeError("no diagnostic has been committed")
            case _ as unmatched:
                from typing import assert_never

                assert_never(unmatched)

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
        next_state, diagnostic = observe_clipped_batch(
            state,
            batch,
            self._config,
            clip=self._clip,
        )
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


def _pooled_mean(batches: tuple[tuple[float, ...], ...]) -> float:
    count = sum(len(batch) for batch in batches)
    return math.fsum(value for batch in batches for value in batch) / count


__all__ = [
    "ClippedEdgeFlowDiagnostic",
    "ClippedOnlineFlowDiagnostics",
    "ClippedTrajectoryFlowDiagnostic",
    "clipped_trajectory_flow",
    "observe_clipped_batch",
]
