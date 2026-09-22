"""Explicit ablation modules without eager model-runtime imports."""

from __future__ import annotations

from importlib import import_module

_MODULES = {
    "builders": (
        "ArmApplicationInputs",
        "FlowOnlyArmApplicationInputs",
        "FlowOnlyApplication",
        "build_capped_flow_weight_application",
        "build_clipped_importance_application",
        "build_no_bayesian_calibration_application",
        "build_no_flow_weighting_application",
        "build_posterior_mean_decision_application",
        "build_residual_only_phase_application",
    ),
    "capped_flow_weight": (
        "CappedFlowCalibrationEngine",
        "CappedFlowWeightConfig",
        "capped_flow_weights",
        "capped_flow_weights_for",
    ),
    "clipped_importance": (
        "ClippedEdgeFlowDiagnostic",
        "ClippedOnlineFlowDiagnostics",
        "ClippedTrajectoryFlowDiagnostic",
        "clipped_trajectory_flow",
        "observe_clipped_batch",
    ),
    "no_bayesian_calibration": (
        "FlowOnlyEvidencePack",
        "FlowOnlyEvolutionPolicy",
        "FlowOnlyProjectionPipeline",
        "FlowOnlyProjectionTransition",
        "FlowOnlySkillEvidence",
        "FlowOnlyTrainingSource",
        "build_flow_only_evidence_pack",
    ),
    "no_bayesian_contracts": (
        "FlowOnlyActionResult",
        "FlowOnlyCycleResult",
        "FlowOnlyDecision",
        "FlowOnlyGenerateEvidence",
        "FlowOnlyGenerateProposal",
        "FlowOnlyRetainEvidence",
        "FlowOnlyRetainProposal",
        "FlowOnlyLibraryInitialized",
        "FlowOnlyPhaseCheckpointPublished",
        "FlowOnlyPhaseOpened",
        "FlowOnlyTrainingStepCommit",
    ),
    "no_bayesian_loop": (
        "ArmFlowOnlyResultSink",
        "FlowOnlyEvolutionLoop",
        "FlowOnlyResultSink",
        "FlowOnlyRunSummary",
    ),
    "no_bayesian_training": ("FlowOnlyTrainingLoop",),
    "no_flow_weighting": (
        "UnitFlowCalibrationEngine",
        "unit_flow_weights_for",
    ),
    "posterior_mean_decision": ("PosteriorMeanEvolutionPolicy",),
    "residual_only_phase": ("ResidualOnlyPhaseDetector",),
}
_EXPORTS = {
    name: (f"skillev.experiments.arms.{module}", name)
    for module, names in _MODULES.items()
    for name in names
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
