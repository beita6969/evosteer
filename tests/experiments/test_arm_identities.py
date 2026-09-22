from __future__ import annotations

import inspect
import math

import pytest

import skillev.experiments.arms as arms
from skillev.calibration import CalibrationConfig, CalibrationEngine, flow_weights
from skillev.contracts import (
    ContextFeature,
    FailureMode,
    HorizonBucket,
    PosteriorCellState,
    TokenBucket,
)
from skillev.diagnostics import (
    DiagnosticsConfig,
    EdgeFlowDiagnostic,
    OnlineFlowDiagnostics,
    TrajectoryFlowDiagnostic,
    assemble_batch_flow_input,
    trajectory_flow,
)
from skillev.evolution import (
    EvolutionConfig,
    FullEvolutionPolicy,
    NoPhaseTransition,
    NoSplitModality,
    ObservedSkillFlow,
    ObservedSkillPosterior,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
    PosteriorCellEvidence,
    SkillEvidence,
    VerifiedNoOpEvolutionDecision,
)
from skillev.experiments.arms import (
    CappedFlowWeightConfig,
    FlowOnlyProjectionPipeline,
    FlowOnlyTrainingSource,
    PosteriorMeanEvolutionPolicy,
    ResidualOnlyPhaseDetector,
    capped_flow_weights,
    clipped_trajectory_flow,
    unit_flow_weights_for,
)
from skillev.training import MethodProjectionPipeline
from tests.v3_helpers import make_authoring_edge, make_phase_event, make_source


def _flows() -> tuple[TrajectoryFlowDiagnostic, ...]:
    return (
        TrajectoryFlowDiagnostic(
            trajectory_id="trajectory-a",
            horizon=2,
            edges=(
                EdgeFlowDiagnostic(
                    trajectory_id="trajectory-a",
                    step_index=1,
                    log_importance=0.0,
                    sample_log_state_weight=0.0,
                    invoked_skill_ids=("skill-alpha",),
                ),
                EdgeFlowDiagnostic(
                    trajectory_id="trajectory-a",
                    step_index=2,
                    log_importance=math.log(9.0),
                    sample_log_state_weight=math.log(9.0),
                    invoked_skill_ids=("skill-alpha",),
                ),
            ),
            terminal_sample_log_state_weight=math.log(9.0),
        ),
    )


def test_weight_arms_change_only_the_declared_kernel() -> None:
    flows = _flows()
    full = flow_weights(flows)

    assert full == {
        ("trajectory-a", 1): pytest.approx(0.2),
        ("trajectory-a", 2): pytest.approx(1.8),
    }
    assert unit_flow_weights_for(flows) == {
        ("trajectory-a", 1): 1.0,
        ("trajectory-a", 2): 1.0,
    }
    assert capped_flow_weights(
        flows,
        CappedFlowWeightConfig(cap=1.0),
    ) == {
        ("trajectory-a", 1): pytest.approx(0.2),
        ("trajectory-a", 2): 1.0,
    }
    assert flow_weights(flows) == full


def test_clipped_importance_isolated_from_raw_full_trajectory_flow() -> None:
    source = make_source(
        artifacts=(make_source().artifacts[0],),
        importances=((15.0,),),
        deltas=(0.5,),
    )
    batch = assemble_batch_flow_input(
        source.stats,
        {artifact.record.trajectory_id: artifact.record for artifact in source.artifacts},
        source.edge_records,
    )
    trajectory = batch.trajectories[0]

    full = trajectory_flow(trajectory)
    clipped = clipped_trajectory_flow(trajectory, clip=10.0)

    assert full.edges[0].sample_log_state_weight == 15.0
    assert full.edges[0].log_importance == 15.0
    assert clipped.edges[0].raw_log_importance == 15.0
    assert clipped.edges[0].clipped_log_importance == 10.0
    assert clipped.edges[0].sample_log_state_weight == 10.0


def test_no_bayesian_arm_changes_only_posterior_projection() -> None:
    source = make_source()
    diagnostics_config = DiagnosticsConfig(window_size=1)
    calibration_config = CalibrationConfig()
    full = MethodProjectionPipeline.fresh(
        diagnostics_config=diagnostics_config,
        calibration=CalibrationEngine(calibration_config),
        library_version=source.stats.library_version,
    )
    flow_only = FlowOnlyProjectionPipeline.fresh(
        diagnostics_config,
        library_version=source.stats.library_version,
    )

    full_transition = full.preview(source)
    flow_only_transition = flow_only.preview(
        FlowOnlyTrainingSource(
            batch_id=source.batch_id,
            optimizer_step=source.optimizer_step,
            records=tuple(artifact.record for artifact in source.artifacts),
            stats=source.stats,
            edge_records=source.edge_records,
        )
    )

    assert flow_only_transition.diagnostic.to_value() == full_transition.diagnostic.to_value()
    assert full_transition.posterior_batch.updates
    assert not hasattr(flow_only_transition, "posterior_batch")
    flow_only.commit(flow_only_transition)
    assert "calibration_cells" not in flow_only.runtime_state().to_value()


def _decision_skill() -> SkillEvidence:
    z = ContextFeature(
        context="debug-family",
        failure_mode=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
    )
    cell = PosteriorCellState(
        skill_id="skill-alpha",
        z=z,
        alpha_0=1.0,
        beta_0=1.0,
        alpha=2.0,
        beta_count=1.0,
        update_count=1,
        last_event_id="posterior-alpha",
    )
    exemplar = make_authoring_edge("trajectory-a:1")
    return SkillEvidence(
        skill_id="skill-alpha",
        flow=ObservedSkillFlow(
            skill_id="skill-alpha",
            log_skill_marginal_flow=1.0,
            flow_quantile=1.0,
            invoking_trajectory_count=1,
            invoking_edge_count=1,
            invoking_edge_ids=(exemplar.edge_id,),
        ),
        posterior=ObservedSkillPosterior(
            skill_id="skill-alpha",
            cells=(PosteriorCellEvidence(cell, ("posterior-alpha",)),),
        ),
        split_modality=NoSplitModality(),
        target_context_keys=(z.cell_key("skill-alpha"),),
        edge_exemplars=(exemplar,),
    )


def test_posterior_mean_arm_changes_only_the_decision_statistic() -> None:
    from skillev.evolution import EvidencePack

    pack = EvidencePack(
        phase_event=make_phase_event(),
        skills=(_decision_skill(),),
        uncovered_importance_edges=(),
        uncovered_exemplars=(),
    )
    config = EvolutionConfig(generate_min_absolute_log_importance=0.1)

    assert isinstance(
        FullEvolutionPolicy().decide(pack, config),
        VerifiedNoOpEvolutionDecision,
    )
    decision = PosteriorMeanEvolutionPolicy().decide(pack, config)

    assert len(decision.proposals) == 1
    assert decision.proposals[0].target_skill_id == "skill-alpha"


def test_residual_only_arm_removes_only_the_entropy_conjunct() -> None:
    diagnostics_config = DiagnosticsConfig(window_size=1, stagnation_rho=0.05)
    diagnostics = OnlineFlowDiagnostics.fresh(
        diagnostics_config,
        library_version="library-v1",
    )
    values = []
    for source in (
        make_source(batch_id="batch-1", optimizer_step=1, deltas=(1.0,)),
        make_source(batch_id="batch-2", optimizer_step=2, deltas=(0.99,)),
    ):
        transition = diagnostics.preview(source)
        diagnostics.commit(transition)
        values.append(transition.diagnostic)

    evolution_config = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        entropy_window=1,
        required_consecutive_drops=1,
    )
    full = PhaseTransitionDetector.fresh(
        evolution_config=evolution_config,
        diagnostics_config=diagnostics_config,
        library_version="library-v1",
    )
    residual_only = ResidualOnlyPhaseDetector.fresh(
        diagnostics_config,
        library_version="library-v1",
    )

    assert isinstance(full.observe(values[0]), NoPhaseTransition)
    assert isinstance(residual_only.observe(values[0]), NoPhaseTransition)
    assert isinstance(full.observe(values[1]), NoPhaseTransition)
    assert isinstance(residual_only.observe(values[1]), PhaseTransitionDetected)


def test_six_arm_builders_are_explicit_functions_not_a_runtime_switch() -> None:
    builders = (
        arms.build_no_bayesian_calibration_application,
        arms.build_no_flow_weighting_application,
        arms.build_capped_flow_weight_application,
        arms.build_clipped_importance_application,
        arms.build_posterior_mean_decision_application,
        arms.build_residual_only_phase_application,
    )

    assert len(set(builders)) == 6
    assert not hasattr(arms, "build_application")
    assert all(tuple(inspect.signature(builder).parameters) == ("inputs",) for builder in builders)
