"""Read-only reporting of the resolved full-method controls and committed state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skillev.calibration import EXTRACTOR_VERSION
from skillev.contracts import JsonValue
from skillev.evolution.authoring import (
    AUTHORING_TEMPLATE_GENERATE,
    AUTHORING_TEMPLATE_REFINE,
    AUTHORING_TEMPLATE_RETAIN,
    AUTHORING_TEMPLATE_SPLIT,
)
from skillev.evolution.detector import ActiveDetectorSegment
from skillev.policy import AdapterRole
from skillev.runtime import EventType
from skillev.training.calibration_reporting import (
    posterior_evidence_composition,
    prequential_calibration,
)
from skillev.training.coverage_reporting import batch_coverage
from skillev.training.skill_lifecycle import skill_lifecycle_report

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


def resolved_method_state(application: SKILLEVApplication) -> dict[str, JsonValue]:
    """Expose actual parsed settings without copying native evaluator payloads.

    This is telemetry, never an input to retrieval, answer selection or Phi.
    Counts distinguish invocations from distinct trajectories, not IID samples.
    """
    state = application.projections.runtime_state()
    provenance = state.posterior_provenance
    trajectories_by_cell: dict[str, set[str]] = {}
    sources_by_cell: dict[str, set[tuple[str, str, str]]] = {}
    source_unknown_by_cell: dict[str, set[str]] = {}
    contexts = {
        item.trajectory_id: item
        for batch in provenance.batches
        for item in batch.trajectory_contexts
    }
    for update in provenance.updates:
        key = update.z.cell_key(update.skill_id)
        trajectories_by_cell.setdefault(key, set()).add(update.trajectory_id)
        context = contexts.get(update.trajectory_id)
        if context is not None and context.source_key is not None:
            sources_by_cell.setdefault(key, set()).add(context.source_key)
        else:
            source_unknown_by_cell.setdefault(key, set()).add(update.trajectory_id)
    detector = application.detector.state
    latest_batch = provenance.batches[-1] if provenance.batches else None
    backbone = application.backbone
    return {
        "application_config": application.public_identity.application_config.to_value(),
        "feature_extractor_version": EXTRACTOR_VERSION,
        "authoring_templates": {
            "retain": AUTHORING_TEMPLATE_RETAIN,
            "refine": AUTHORING_TEMPLATE_REFINE,
            "split": AUTHORING_TEMPLATE_SPLIT,
            "generate": AUTHORING_TEMPLATE_GENERATE,
        },
        "optimizer_step": application.training_loop.optimizer_step,
        "policy_snapshot_id": application.training_loop.policy_snapshot_id,
        "forward_adapter_version": backbone.adapter_version(AdapterRole.FORWARD_POLICY),
        "backward_adapter_version": backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
        "z_version": backbone.z_version,
        "library_version": application.library.current_version,
        "active_skill_ids": list(application.library.active_skill_ids),
        "projection_revision": state.revision,
        "terminal_evaluation_conditions": application.snapshot_identity.to_value().get(
            "terminal_evaluation_conditions"
        ),
        "task_feature_mapping_version": application.snapshot_identity.task_feature_mapping_version,
        "posterior_batch_count": len(provenance.batches),
        "posterior_update_event_count": len(provenance.updates),
        "phase_detection_count": application.emitter.log.committed_event_count(
            EventType.EVOLUTION_PHASE_OPENED
        ),
        "phase_no_op_count": application.emitter.log.committed_event_count(
            EventType.EVOLUTION_NO_OP_COMMITTED
        ),
        "actual_library_mutation_count": application.run_progress.state.committed_cycles,
        "committed_proposal_count": application.run_progress.state.committed_actions,
        "last_evidence_batch": None if latest_batch is None else latest_batch.posterior.batch_id,
        "last_evidence_policy": None if latest_batch is None else latest_batch.policy_snapshot_id,
        "last_evidence_library": None if latest_batch is None else latest_batch.library_version,
        "posterior_cells": [
            {
                "skill_id": cell.skill_id,
                "z": cell.z.to_value(),
                "alpha": cell.alpha,
                "beta_count": cell.beta_count,
                "update_count": cell.update_count,
                "cumulative_flow_weight": cell.cumulative_flow_weight,
                "distinct_trajectory_count": len(
                    trajectories_by_cell.get(cell.z.cell_key(cell.skill_id), ())
                ),
                "distinct_source_question_count": len(
                    sources_by_cell.get(cell.z.cell_key(cell.skill_id), ())
                ),
                "source_unknown_trajectory_count": len(
                    source_unknown_by_cell.get(cell.z.cell_key(cell.skill_id), ())
                ),
            }
            for cell in state.calibration_cells
        ],
        "prequential_calibration": prequential_calibration(provenance),
        "posterior_evidence_composition": posterior_evidence_composition(provenance),
        "skill_coverage": [batch_coverage(batch) for batch in provenance.batches],
        "skill_lifecycle": skill_lifecycle_report(application.library, provenance),
        "phase_status": (
            detector.phase_status.value
            if isinstance(detector, ActiveDetectorSegment)
            else "awaiting"
        ),
        "no_op_closure": (
            detector.no_op_closure.to_value()
            if isinstance(detector, ActiveDetectorSegment) and detector.no_op_closure is not None
            else None
        ),
        "residual_statistic": "mean-raw-delta-squared",
        "optimization_statistic": "mean-horizon-normalized-delta-squared",
    }
