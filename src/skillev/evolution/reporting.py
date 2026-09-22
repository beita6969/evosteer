"""Answer-free reachability telemetry; never an input to phase decisions."""

from skillev.contracts import JsonValue
from skillev.diagnostics import BatchDiagnostics

from .config import EvolutionConfig
from .detector import ActiveDetectorSegment, DetectorRuntimeState
from .evidence import AuthoringEdgeEvidence, is_generate_candidate_action


def phase_coverage_report(
    diagnostic: BatchDiagnostics,
    state: DetectorRuntimeState,
    exemplars: tuple[AuthoringEdgeEvidence, ...],
    config: EvolutionConfig,
) -> dict[str, JsonValue]:
    edges = tuple(edge for trajectory in diagnostic.trajectories for edge in trajectory.edges)
    invoked = sum(bool(edge.invoked_skill_ids) for edge in edges)
    return {
        "cold_start_extension": config.cold_start.to_value() if config.cold_start else None,
        "cold_start_required_committed_batches": 2 * diagnostic.config.window_size
        if config.cold_start
        else None,
        "cold_start_coverage_note": (
            "one observed call excludes its scope from zero-coverage Generate; "
            "this does not establish useful or sufficient coverage"
        )
        if config.cold_start
        else None,
        "edge_count": len(edges),
        "invoking_edge_count": invoked,
        "invocation_edge_fraction": invoked / len(edges) if edges else 0.0,
        "skill_invocation_counts": {
            item.skill_id: item.invoking_edge_count for item in diagnostic.skill_flows
        },
        "entropy_evidence": state.entropy_evidence.to_value()
        if isinstance(state, ActiveDetectorSegment)
        else None,
        "residual_window": diagnostic.residual_window.to_value(),
        "uncovered_high_importance_edges": [
            {
                "edge_id": item.edge_id,
                "task_family": item.task_family,
                "context_id": item.context_id,
                "absolute_log_importance": item.absolute_log_importance,
                "batch_importance_quantile": item.log_importance_quantile,
            }
            for item in exemplars
            if not item.invoked_skill_ids
            and is_generate_candidate_action(item.action_kind)
            and item.absolute_log_importance >= config.generate_min_absolute_log_importance
            and item.log_importance_quantile >= config.importance_quantile
        ],
        "generate_coverage_semantics": "no-explicit-invocation-not-no-applicable-skill@2",
        "uninvoked_exposed_edge_count": sum(
            not item.invoked_skill_ids and bool(item.exposed_skill_ids) for item in exemplars
        ),
        "uninvoked_no_exposed_skill_edge_count": sum(
            not item.invoked_skill_ids and item.exposed_skill_ids == () for item in exemplars
        ),
        "exposure_unknown_edge_count": sum(item.exposed_skill_ids is None for item in exemplars),
        "candidate_scope": "batch-telemetry-not-phase-window-selection",
        "no_op_cutoff": state.no_op_closure.to_value()
        if isinstance(state, ActiveDetectorSegment) and state.no_op_closure
        else None,
    }
