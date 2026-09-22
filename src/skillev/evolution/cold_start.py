"""Generate reachability for uninvoked public scopes; never synthetic credit.

This is the explicitly enabled zero-coverage-generate@1 method extension.
The usual whole-window importance selector is reused, including its population
and thresholds. Ordinary Phi's residual AND entropy rule is not modified.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decision import FullEvolutionDecision
    from .detector import DetectorObservation
    from .evidence import WindowFlowView
    from .loop import EvolutionPhaseDetector, EvolutionProjectionView

from skillev.diagnostics import BatchDiagnostics

from .config import EvolutionConfig
from .evidence import (
    AuthoringEdgeEvidence,
    GenerateAuthorityGroup,
    TrajectoryEvidenceView,
    generate_candidate_contribution,
    group_generate_exemplars,
    select_uncovered_generate_edges,
)


def preview_method_phase(
    detector: EvolutionPhaseDetector,
    projections: EvolutionProjectionView,
    diagnostic: BatchDiagnostics,
    config: EvolutionConfig,
) -> tuple[DetectorObservation, bool]:
    if config.cold_start is None:
        return detector.preview_observation(diagnostic), False
    from .detector import PhaseTransitionDetector

    if not isinstance(detector, PhaseTransitionDetector):
        raise ValueError("cold start requires the full-method phase detector")
    supported = projections.cold_start_supported(config)
    return detector.preview_observation(diagnostic, cold_start_supported=supported), supported


def cold_start_groups(
    diagnostics: tuple[BatchDiagnostics, ...],
    trajectories: TrajectoryEvidenceView,
    config: EvolutionConfig,
) -> tuple[GenerateAuthorityGroup, ...]:
    extension = config.cold_start
    if extension is None or not diagnostics:
        return ()
    size = diagnostics[-1].config.window_size
    if len(diagnostics) != 2 * size:
        return ()
    if any(item.library_version != diagnostics[-1].library_version for item in diagnostics):
        raise ValueError("cold start cannot mix library segments")
    covered: set[tuple[str, str, tuple[str, ...]]] = set()
    edge_coordinates: dict[str, tuple[str, str]] = {}
    population: list[float] = []
    uncovered = []
    for diagnostic in diagnostics:
        for trajectory in diagnostic.trajectories:
            for edge in trajectory.edges:
                edge_id = f"{trajectory.trajectory_id}:{edge.step_index}"
                exemplar = trajectories.authoring_by_edge_id[edge_id]
                if (
                    exemplar.absolute_log_importance != abs(edge.log_importance)
                    or exemplar.invoked_skill_ids != edge.invoked_skill_ids
                ):
                    raise ValueError("cold-start evidence differs from committed execution")
                edge_coordinates[edge_id] = (diagnostic.batch_id, trajectory.trajectory_id)
                if edge.invoked_skill_ids:
                    covered.add(
                        (exemplar.task_family, exemplar.context_id, exemplar.available_tools)
                    )
                contribution = generate_candidate_contribution(
                    exemplar=exemplar,
                    absolute_importance=abs(edge.log_importance),
                    invoked_skill_ids=edge.invoked_skill_ids,
                )
                if contribution is not None:
                    population.append(contribution.absolute_importance)
                    if contribution.uncovered is not None:
                        uncovered.append(contribution.uncovered)
    selected = select_uncovered_generate_edges(
        uncovered,
        tuple(population),
        importance_quantile=config.importance_quantile,
        minimum_absolute_log_importance=config.generate_min_absolute_log_importance,
    )
    groups = group_generate_exemplars(
        importance_edge_ids=tuple(item[0] for item in selected),
        edge_exemplars=tuple(item[2] for item in selected),
    )
    supported = []
    for group in groups:
        if (group.task_family, group.context_id, group.available_tools) in covered:
            continue
        coordinates = [edge_coordinates[edge_id] for edge_id in group.importance_edge_ids]
        sources = {
            (source[0], source[2])
            for _, trajectory_id in coordinates
            if (source := trajectories.source_by_trajectory.get(trajectory_id)) is not None
        }
        batches = {
            batch_id
            for batch_id, trajectory_id in coordinates
            if trajectories.source_by_trajectory.get(trajectory_id) is not None
        }
        if len(sources) >= extension.min_source_questions and len(batches) >= extension.min_batches:
            # Bound the mandatory per-edge author contract before any request.
            # Preserve distinct source/batch support, then fill in canonical order.
            # This changes only the declared cold-start authoring witness set,
            # not diagnostics, gradients, posterior weights, or normal Generate.
            witnesses: list[AuthoringEdgeEvidence] = []
            used_sources: set[tuple[str, str]] = set()
            used_batches: set[str] = set()
            candidates = [
                edge
                for edge in group.edge_exemplars
                if trajectories.source_by_trajectory.get(edge_coordinates[edge.edge_id][1])
                is not None
            ]
            for mode in ("source", "batch", "fill"):
                for witness in candidates:
                    batch_id, trajectory_id = edge_coordinates[witness.edge_id]
                    source = trajectories.source_by_trajectory[trajectory_id]
                    if source is None:
                        continue
                    canonical_source = (source[0], source[2])
                    if witness in witnesses or len(witnesses) >= extension.max_evidence_edges:
                        continue
                    if mode == "source" and (
                        len(used_sources) >= extension.min_source_questions
                        or canonical_source in used_sources
                    ):
                        continue
                    if mode == "batch" and (
                        len(used_batches) >= extension.min_batches or batch_id in used_batches
                    ):
                        continue
                    witnesses.append(witness)
                    used_sources.add(canonical_source)
                    used_batches.add(batch_id)
            witnesses.sort(key=lambda edge: edge.edge_id)
            supported.append(
                replace(
                    group,
                    edge_exemplars=tuple(witnesses),
                    importance_edge_ids=tuple(edge.edge_id for edge in witnesses),
                )
            )
    # Existing authority ordering is stable. No reward-based ranking or sampling.
    return tuple(supported[: extension.max_new_skills])


def decide_cold_start(
    window_flow: WindowFlowView, trajectories: TrajectoryEvidenceView, config: EvolutionConfig
) -> FullEvolutionDecision:
    from skillev.contracts import GenerateEvidence

    from .decision import EvolutionDecision, GenerateProposal, VerifiedNoOpEvolutionDecision

    groups = cold_start_groups(window_flow.diagnostics, trajectories, config)
    if not groups:
        return VerifiedNoOpEvolutionDecision(
            window_flow.phase_event.event_id, "no-supported-zero-coverage-scope"
        )
    return EvolutionDecision(
        window_flow.phase_event.event_id,
        tuple(
            GenerateProposal(
                evidence=GenerateEvidence(
                    importance_edge_ids=group.importance_edge_ids,
                    minimum_absolute_log_importance=config.generate_min_absolute_log_importance,
                    importance_quantile=config.importance_quantile,
                    importance_semantics=config.generate_importance_semantics,
                ),
                rationale_text=(
                    "zero-coverage-generate@1: residual plateau with supported zero-invocation "
                    "scope; Generate only, no inherited posterior or fabricated invocation. "
                    "Write a task-relevant reusable procedure with a concrete applicability cue, "
                    "executable steps and a public outcome check, not a generic slogan "
                    "or task answer."
                ),
                edge_exemplars=group.edge_exemplars,
            )
            for group in groups
        ),
    )
