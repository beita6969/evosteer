"""Pure full-shaped replay from committed source contracts only.

The seven full-shaped arms differ through small frozen kernels.  This module
keeps the audit input identical for all of them and never accepts a caller
provided diagnostics/calibration/decision configuration outside the published
attempt identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, assert_never

from skillev.calibration import CalibrationConfig, CalibrationEngine, CalibrationTransition
from skillev.contracts import (
    EdgeLogprobRecord,
    PosteriorCellState,
    PosteriorUpdateEvent,
    TrajectoryRecord,
    TTBBatchStats,
)
from skillev.diagnostics import (
    BatchDiagnostics,
    DiagnosticsConfig,
    DiagnosticsSource,
    DiagnosticsTransition,
    OnlineFlowDiagnostics,
)
from skillev.evolution import (
    AuthoringEdgeEvidence,
    DetectorObservation,
    EvolutionConfig,
    FullEvolutionDecision,
    FullEvolutionPolicy,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
    PosteriorEvidenceView,
    TrajectoryEvidenceView,
    VerifiedNoOpEvolutionDecision,
    WindowFlowView,
    assemble_posterior_evidence_view,
    authoring_evidence_for_diagnostic,
)
from skillev.experiments.attempt_identity import PublishedAttemptIdentity
from skillev.experiments.protocol import (
    CappedFlowWeightArmProtocol,
    ClippedImportanceArmProtocol,
)
from skillev.policy import AuthoringTokenizerProtocol
from skillev.runtime import AttemptBuilderKind, RunSlotKind

from .evolution_decision import verify_cycle_decision_and_documents
from .source_reducer import AuditEvidenceMismatchError, CommittedLibrarySegment


@dataclass(frozen=True, slots=True)
class CommittedDiagnosticsSource:
    """Dependency-light committed-source view; no rollout recreation in audit."""

    batch_id: str
    optimizer_step: int
    records: tuple[TrajectoryRecord, ...]
    stats: TTBBatchStats
    edge_records: tuple[EdgeLogprobRecord, ...]


class AuditDiagnostics(Protocol):
    def preview(self, source: DiagnosticsSource) -> DiagnosticsTransition: ...

    def commit(self, transition: DiagnosticsTransition) -> None: ...


class AuditCalibration(Protocol):
    def preview(
        self,
        *,
        source: CommittedDiagnosticsSource,
        diagnostic: BatchDiagnostics,
    ) -> CalibrationTransition: ...

    def commit(self, transition: CalibrationTransition) -> None: ...


class AuditPhaseDetector(Protocol):
    def observe(self, diagnostic: BatchDiagnostics) -> DetectorObservation: ...


class AuditEvolutionPolicy(Protocol):
    def decide_from_views(
        self,
        *,
        window_flow: WindowFlowView,
        posterior: PosteriorEvidenceView,
        trajectories: TrajectoryEvidenceView,
        config: EvolutionConfig,
    ) -> FullEvolutionDecision: ...


@dataclass(frozen=True, slots=True)
class RecomputedTrainingStep:
    batch_id: str
    diagnostic: BatchDiagnostics
    calibration_cells_after: tuple[PosteriorCellState, ...]
    posterior_provenance_after: AuditPosteriorProvenance
    authoring_by_edge_id: Mapping[str, AuthoringEdgeEvidence]


@dataclass(frozen=True, slots=True)
class RecomputedFullTimeline:
    steps: tuple[RecomputedTrainingStep, ...]
    decisions: tuple[FullEvolutionDecision, ...]
    authoring_reservation_ids_by_cycle: tuple[tuple[str, ...], ...]
    final_cells: tuple[PosteriorCellState, ...]
    final_provenance: AuditPosteriorProvenance


@dataclass(frozen=True, slots=True)
class AuditPosteriorProvenance:
    """Ordered posterior event provenance reconstructed from source records."""

    event_ids: tuple[str, ...] = ()
    cell_keys: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> AuditPosteriorProvenance:
        return cls()

    def append(self, update: PosteriorUpdateEvent) -> AuditPosteriorProvenance:
        if not isinstance(update, PosteriorUpdateEvent):
            raise TypeError("posterior provenance requires PosteriorUpdateEvent")
        event_id = update.event_id
        cell_key = update.z.cell_key(update.skill_id)
        if event_id in self.event_ids:
            raise AuditEvidenceMismatchError("posterior provenance repeats update event")
        return AuditPosteriorProvenance(
            event_ids=(*self.event_ids, event_id),
            cell_keys=(*self.cell_keys, cell_key),
        )

    def event_ids_for_cell(self, cell_key: str) -> tuple[str, ...]:
        return tuple(
            event_id
            for event_id, key in zip(self.event_ids, self.cell_keys, strict=True)
            if key == cell_key
        )


class FullShapedAuditKernel(Protocol):
    """The frozen arm deltas accepted by the common full-method audit."""

    def diagnostics(
        self,
        config: DiagnosticsConfig,
        *,
        library_version: str,
    ) -> AuditDiagnostics: ...

    def calibration(self, config: CalibrationConfig) -> AuditCalibration: ...

    def hydrate_calibration(
        self,
        config: CalibrationConfig,
        cells: tuple[PosteriorCellState, ...],
    ) -> AuditCalibration: ...

    def detector(
        self,
        *,
        diagnostics_config: DiagnosticsConfig,
        evolution_config: EvolutionConfig,
        library_version: str,
    ) -> AuditPhaseDetector: ...

    def policy(self) -> AuditEvolutionPolicy: ...


@dataclass(frozen=True, slots=True)
class LiteralFullAuditKernel:
    def diagnostics(
        self,
        config: DiagnosticsConfig,
        *,
        library_version: str,
    ) -> AuditDiagnostics:
        return OnlineFlowDiagnostics.fresh(config, library_version=library_version)

    def calibration(self, config: CalibrationConfig) -> CalibrationEngine:
        return CalibrationEngine(config)

    def hydrate_calibration(
        self,
        config: CalibrationConfig,
        cells: tuple[PosteriorCellState, ...],
    ) -> CalibrationEngine:
        return CalibrationEngine.from_runtime_state(config, cells)

    def detector(
        self,
        *,
        diagnostics_config: DiagnosticsConfig,
        evolution_config: EvolutionConfig,
        library_version: str,
    ) -> AuditPhaseDetector:
        return PhaseTransitionDetector.fresh(
            diagnostics_config=diagnostics_config,
            evolution_config=evolution_config,
            library_version=library_version,
        )

    def policy(self) -> AuditEvolutionPolicy:
        return FullEvolutionPolicy()


@dataclass(frozen=True, slots=True)
class UnitFlowAuditKernel(LiteralFullAuditKernel):
    def calibration(self, config: CalibrationConfig) -> CalibrationEngine:
        from skillev.experiments.arms.no_flow_weighting import UnitFlowCalibrationEngine

        return UnitFlowCalibrationEngine(config)

    def hydrate_calibration(
        self,
        config: CalibrationConfig,
        cells: tuple[PosteriorCellState, ...],
    ) -> CalibrationEngine:
        from skillev.experiments.arms.no_flow_weighting import UnitFlowCalibrationEngine

        return UnitFlowCalibrationEngine.from_runtime_state(config, cells)


@dataclass(frozen=True, slots=True)
class CappedFlowAuditKernel(LiteralFullAuditKernel):
    cap: float

    def __post_init__(self) -> None:
        from skillev.experiments.protocol import CappedFlowWeightArmProtocol

        if self.cap != CappedFlowWeightArmProtocol.CAP:
            raise ValueError("capped-flow audit kernel differs from preregistered cap")

    def calibration(self, config: CalibrationConfig) -> CalibrationEngine:
        from skillev.experiments.arms.capped_flow_weight import (
            CappedFlowCalibrationEngine,
            CappedFlowWeightConfig,
        )

        return CappedFlowCalibrationEngine(config, CappedFlowWeightConfig(self.cap))

    def hydrate_calibration(
        self,
        config: CalibrationConfig,
        cells: tuple[PosteriorCellState, ...],
    ) -> CalibrationEngine:
        from skillev.experiments.arms.capped_flow_weight import CappedFlowCalibrationEngine

        return CappedFlowCalibrationEngine.from_runtime_state(config, cells)


@dataclass(frozen=True, slots=True)
class ClippedImportanceAuditKernel(LiteralFullAuditKernel):
    clip: float

    def diagnostics(
        self,
        config: DiagnosticsConfig,
        *,
        library_version: str,
    ) -> AuditDiagnostics:
        from skillev.experiments.arms.clipped_importance import ClippedOnlineFlowDiagnostics

        return ClippedOnlineFlowDiagnostics.fresh(
            config,
            clip=self.clip,
            library_version=library_version,
        )


@dataclass(frozen=True, slots=True)
class PosteriorMeanAuditKernel(LiteralFullAuditKernel):
    def policy(self) -> AuditEvolutionPolicy:
        from skillev.experiments.arms.posterior_mean_decision import PosteriorMeanEvolutionPolicy

        return PosteriorMeanEvolutionPolicy()


@dataclass(frozen=True, slots=True)
class ResidualOnlyAuditKernel(LiteralFullAuditKernel):
    def detector(
        self,
        *,
        diagnostics_config: DiagnosticsConfig,
        evolution_config: EvolutionConfig,
        library_version: str,
    ) -> AuditPhaseDetector:
        del evolution_config
        from skillev.experiments.arms.residual_only_phase import ResidualOnlyPhaseDetector

        return ResidualOnlyPhaseDetector.fresh(
            diagnostics_config,
            library_version=library_version,
        )


def full_shaped_kernel_for_identity(
    identity: PublishedAttemptIdentity,
) -> FullShapedAuditKernel:
    """Return the only replay kernel admitted by a frozen public identity.

    A full-shaped audit must not let its caller substitute an ablation kernel:
    the published builder/arm protocol is the sole algorithm selector.  The
    lower-level replay function still receives the kernel explicitly so its
    dependencies remain transparent, but callers must prove it is this value.
    """

    match identity.builder_kind:
        case AttemptBuilderKind.FULL:
            return LiteralFullAuditKernel()
        case AttemptBuilderKind.UNIT_FLOW:
            return UnitFlowAuditKernel()
        case AttemptBuilderKind.CAPPED_FLOW:
            protocol = identity.arm_protocol
            if not isinstance(protocol, CappedFlowWeightArmProtocol):
                raise ValueError("capped-flow identity has another arm protocol")
            return CappedFlowAuditKernel(cap=protocol.CAP)
        case AttemptBuilderKind.CLIPPED_IMPORTANCE:
            protocol = identity.arm_protocol
            if not isinstance(protocol, ClippedImportanceArmProtocol):
                raise ValueError("clipped-importance identity has another arm protocol")
            return ClippedImportanceAuditKernel(clip=protocol.CLIP)
        case AttemptBuilderKind.POSTERIOR_MEAN:
            return PosteriorMeanAuditKernel()
        case AttemptBuilderKind.RESIDUAL_ONLY_PHASE:
            return ResidualOnlyAuditKernel()
        case AttemptBuilderKind.NO_BAYESIAN:
            raise TypeError("flow-only identity has a separate source/audit")
        case _ as unreachable:
            assert_never(unreachable)


def recompute_full_timeline(
    segments: tuple[CommittedLibrarySegment, ...],
    *,
    identity: PublishedAttemptIdentity,
    kernel: FullShapedAuditKernel,
    tokenizer: AuthoringTokenizerProtocol,
) -> RecomputedFullTimeline:
    """Recompute diagnostics, posterior, detector, and every committed Phi."""

    application_config = identity.application_config
    diagnostics_config = application_config.diagnostics
    calibration_config = application_config.calibration
    evolution_config = application_config.evolution
    all_steps: list[RecomputedTrainingStep] = []
    decisions: list[FullEvolutionDecision] = []
    authoring_reservation_ids_by_cycle: list[tuple[str, ...]] = []
    cells: dict[str, PosteriorCellState] = {}
    provenance = AuditPosteriorProvenance.empty()

    for segment in segments:
        diagnostics = kernel.diagnostics(
            diagnostics_config,
            library_version=segment.library_version,
        )
        calibration = (
            kernel.calibration(calibration_config)
            if not cells
            else kernel.hydrate_calibration(calibration_config, _ordered_cells(cells))
        )
        detector = kernel.detector(
            diagnostics_config=diagnostics_config,
            evolution_config=evolution_config,
            library_version=segment.library_version,
        )
        by_batch: dict[str, RecomputedTrainingStep] = {}
        detected: list[PhaseTransitionDetected] = []
        for commit in segment.training_steps:
            source = CommittedDiagnosticsSource(
                batch_id=commit.batch_id,
                optimizer_step=commit.optimizer_step,
                records=commit.records,
                stats=commit.stats,
                edge_records=commit.edge_records,
            )
            diagnostic_transition = diagnostics.preview(source)
            diagnostic = diagnostic_transition.diagnostic
            diagnostics.commit(diagnostic_transition)
            calibration_transition = calibration.preview(source=source, diagnostic=diagnostic)
            if (
                calibration_transition.posterior_batch.content_hash
                != commit.posterior_batch.content_hash
            ):
                raise AuditEvidenceMismatchError("posterior source does not recompute")
            calibration.commit(calibration_transition)
            cells = dict(calibration_transition.next_cells)
            for update in calibration_transition.posterior_batch.updates:
                provenance = provenance.append(update)
            evidence = authoring_evidence_for_diagnostic(
                {record.trajectory_id: record for record in commit.records},
                diagnostic,
            )
            step = RecomputedTrainingStep(
                batch_id=commit.batch_id,
                diagnostic=diagnostic,
                calibration_cells_after=_ordered_cells(cells),
                posterior_provenance_after=provenance,
                authoring_by_edge_id={item.edge_id: item for item in evidence},
            )
            all_steps.append(step)
            by_batch[commit.batch_id] = step
            position = commit.run_cursor_after.completed_training_steps
            if identity.run_plan.slot_kind(position) is RunSlotKind.PHASE_SEARCH:
                observed = detector.observe(diagnostic)
                if isinstance(observed, PhaseTransitionDetected):
                    detected.append(observed)
        if segment.phase is None:
            if detected:
                raise AuditEvidenceMismatchError("detected phase has no source event")
            continue
        if len(detected) != 1:
            raise AuditEvidenceMismatchError("segment phase count does not recompute")
        phase = detected[0].event
        if phase.content_hash != segment.phase.phase_event.content_hash:
            raise AuditEvidenceMismatchError("phase source does not recompute")
        window_ids = (
            *phase.previous_window.member_batch_ids,
            *phase.current_window.member_batch_ids,
        )
        try:
            window_steps = tuple(by_batch[item] for item in window_ids)
        except KeyError as error:
            raise AuditEvidenceMismatchError("phase references absent diagnostics") from error
        posterior = assemble_posterior_evidence_view(
            active_skill_ids=segment.library_state.active_skill_ids,
            cells=window_steps[-1].calibration_cells_after,
            event_ids_by_cell=_event_ids_by_cell(window_steps[-1].posterior_provenance_after),
        )
        decision = kernel.policy().decide_from_views(
            window_flow=WindowFlowView(
                phase_event=phase,
                diagnostics=tuple(item.diagnostic for item in window_steps),
                active_skill_ids=segment.library_state.active_skill_ids,
                applicability_by_skill={
                    skill_id: segment.library_state.documents[skill_id].applicability
                    for skill_id in segment.library_state.active_skill_ids
                },
                task_family_universe=identity.authoring_authority.allowed_task_families,
            ),
            posterior=posterior,
            trajectories=TrajectoryEvidenceView(
                authoring_by_edge_id={
                    edge_id: evidence
                    for item in window_steps
                    for edge_id, evidence in item.authoring_by_edge_id.items()
                }
            ),
            config=evolution_config,
        )
        if isinstance(decision, VerifiedNoOpEvolutionDecision):
            if segment.no_op is None or segment.cycle is not None:
                raise AuditEvidenceMismatchError("verified no-op source does not recompute")
            if segment.no_op.decision_reason != decision.reason:
                raise AuditEvidenceMismatchError("verified no-op reason does not recompute")
            decisions.append(decision)
            continue
        if segment.no_op is not None or segment.cycle is None:
            raise AuditEvidenceMismatchError("phase source has no committed cycle")
        authoring_reservation_ids_by_cycle.append(
            verify_cycle_decision_and_documents(
                library_before=segment.library_state,
                cycle=segment.cycle,
                decision=decision,
                phase_event=phase,
                authority=identity.authoring_authority,
                config=evolution_config,
                tokenizer=tokenizer,
                base_seed=identity.application_config.trainer.rollout.base_seed,
                cycle_ordinal=segment.cursor_after.committed_cycles,
            )
        )
        decisions.append(decision)
    return RecomputedFullTimeline(
        steps=tuple(all_steps),
        decisions=tuple(decisions),
        authoring_reservation_ids_by_cycle=tuple(authoring_reservation_ids_by_cycle),
        final_cells=_ordered_cells(cells),
        final_provenance=provenance,
    )


def _ordered_cells(cells: Mapping[str, PosteriorCellState]) -> tuple[PosteriorCellState, ...]:
    return tuple(sorted(cells.values(), key=lambda item: (item.z.content_hash, item.skill_id)))


def _event_ids_by_cell(provenance: AuditPosteriorProvenance) -> Mapping[str, tuple[str, ...]]:
    return {key: provenance.event_ids_for_cell(key) for key in sorted(set(provenance.cell_keys))}


__all__ = [
    "AuditCalibration",
    "AuditDiagnostics",
    "AuditEvolutionPolicy",
    "AuditPhaseDetector",
    "AuditPosteriorProvenance",
    "CappedFlowAuditKernel",
    "ClippedImportanceAuditKernel",
    "CommittedDiagnosticsSource",
    "FullShapedAuditKernel",
    "LiteralFullAuditKernel",
    "PosteriorMeanAuditKernel",
    "RecomputedFullTimeline",
    "RecomputedTrainingStep",
    "ResidualOnlyAuditKernel",
    "UnitFlowAuditKernel",
    "full_shaped_kernel_for_identity",
    "recompute_full_timeline",
]
