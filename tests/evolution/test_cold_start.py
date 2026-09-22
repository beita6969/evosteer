"""Opt-in Generate reachability without weakening normal Phi or inventing use."""

from dataclasses import replace

import pytest

from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import PhaseTransitionEvent, PhaseTriggerRule
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    EvolutionConfig,
    NoPhaseTransition,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
)
from skillev.evolution.cold_start import cold_start_groups
from skillev.evolution.cold_start_config import ColdStartConfig
from skillev.evolution.decision import FullEvolutionPolicy, GenerateProposal
from skillev.evolution.detector import detector_state_from_value
from skillev.evolution.evidence import (
    TrajectoryEvidenceView,
    WindowFlowView,
    assemble_posterior_evidence_view,
)
from tests.evolution.test_detector_v3 import _diagnostic
from tests.training.test_bayesian_chain import pipeline, restore
from tests.training.test_evidence_reporting import artifact
from tests.v3_helpers import make_source


def prepared_window(
    *, same_question=False, other_domain_call=False, library_version="library-v1", high_per_batch=1
):
    projection, _ = pipeline()
    if library_version != "library-v1":
        projection = type(projection).fresh(
            diagnostics_config=DiagnosticsConfig(window_size=1),
            calibration=CalibrationEngine(CalibrationConfig(default_k=0.5)),
            library_version=library_version,
        )
    for step in (1, 2):
        items = tuple(
            artifact(
                f"t{step}-{i}",
                question="repeat" if same_question else f"source-{step}-{i}",
                calls=False,
            )
            for i in range(10)
        )
        importances = tuple((2.0 if i >= 10 - high_per_batch else 0.1,) for i in range(10))
        if other_domain_call:
            called = artifact(f"called-{step}", calls=True)
            h0 = replace(
                called.record.initial_context,
                meta={**called.record.initial_context.meta, "task_family": "another-family"},
            )
            called = replace(
                called,
                record=replace(called.record, initial_context=h0, task_family="another-family"),
                initial_context=replace(called.initial_context, contract=h0),
            )
            items += (called,)
            importances += ((0.2,),)
        if library_version != "library-v1":
            rebound = []
            for item in items:
                h0 = replace(
                    item.record.initial_context,
                    meta={**item.record.initial_context.meta, "library_version": library_version},
                )
                rebound.append(
                    replace(
                        item,
                        record=replace(item.record, initial_context=h0),
                        initial_context=replace(item.initial_context, contract=h0),
                        manifest=replace(item.manifest, library_version=library_version),
                    )
                )
            items = tuple(rebound)
        projection.commit(
            projection.preview(
                make_source(
                    artifacts=items,
                    library_version=library_version,
                    batch_id=f"batch-{step}",
                    optimizer_step=step,
                    importances=importances,
                )
            )
        )
    return projection


def config():
    return EvolutionConfig(
        generate_min_absolute_log_importance=0.5,
        entropy_window=1,
        required_consecutive_drops=1,
        cold_start=ColdStartConfig(),
    )


def detect(projection, cfg):
    detector = PhaseTransitionDetector.fresh(
        evolution_config=cfg,
        diagnostics_config=projection.latest_diagnostic.config,
        library_version=projection.latest_diagnostic.library_version,
    )
    for batch in projection.runtime_state().current_segment_batches:
        observation = detector.preview_observation(
            batch.diagnostic, cold_start_supported=projection.cold_start_supported(cfg)
        )
        detector.commit_observation(observation)
    return detector, observation


def test_zero_coverage_requires_explicit_extension_and_whole_window_support():
    projection = prepared_window()
    assert projection.calibration_cells == ()
    assert not projection.cold_start_supported(replace(config(), cold_start=None))
    _, default = detect(projection, replace(config(), cold_start=None))
    assert isinstance(default, NoPhaseTransition)
    assert not prepared_window(same_question=True).cold_start_supported(config())
    assert projection.cold_start_supported(config())
    assert restore(projection).cold_start_supported(config())
    _, detected = detect(projection, config())
    assert isinstance(detected, PhaseTransitionDetected)
    assert detected.event.trigger_rule is PhaseTriggerRule.ZERO_COVERAGE_COLD_START
    assert not detected.event.entropy_condition_met
    assert PhaseTransitionEvent.from_value(detected.event.to_value()) == detected.event
    frozen = projection.freeze_for_phase(detected.event)
    trajectories = TrajectoryEvidenceView(frozen.authoring_by_edge_id, frozen.source_by_trajectory)
    window = WindowFlowView(detected.event, frozen.diagnostics, (), {}, ("domain/debug",))
    decision = FullEvolutionPolicy().decide_from_views(
        window_flow=window,
        posterior=assemble_posterior_evidence_view(
            active_skill_ids=(), cells=(), event_ids_by_cell={}
        ),
        trajectories=trajectories,
        config=config(),
    )
    assert len(decision.proposals) == 1
    assert isinstance(decision.proposals[0], GenerateProposal)
    assert all(not edge.invoked_skill_ids for edge in decision.proposals[0].edge_exemplars)
    assert projection.calibration_cells == ()  # authoring inputs never earn retroactive credit
    assert not cold_start_groups(
        frozen.diagnostics, replace(trajectories, source_by_trajectory={}), config()
    )


def test_calls_in_another_scope_do_not_starve_uncovered_domain():
    projection = prepared_window(other_domain_call=True)
    assert projection.cold_start_supported(config())
    _, detected = detect(projection, config())
    frozen = projection.freeze_for_phase(detected.event)
    view = TrajectoryEvidenceView(frozen.authoring_by_edge_id, frozen.source_by_trajectory)
    groups = cold_start_groups(frozen.diagnostics, view, config())
    assert len(groups) == 1
    assert groups[0].task_family != "another-family"
    # Even a low-importance invocation in this scope excludes cold start. It is
    # not enough to check only the selected high-importance candidate edges.
    stored = dict(view.authoring_by_edge_id)
    for key, edge in stored.items():
        if edge.invoked_skill_ids:
            stored[key] = replace(edge, task_family=groups[0].task_family)
    assert not cold_start_groups(
        frozen.diagnostics, replace(view, authoring_by_edge_id=stored), config()
    )


def test_normal_phi_has_priority_and_noop_cutoff_survives_restore():
    projection = prepared_window()
    detector, detected = detect(projection, config())
    detector.close_no_op(detected.event, "test-no-op")
    restored = PhaseTransitionDetector.from_runtime_state(
        evolution_config=config(),
        diagnostics_config=projection.latest_diagnostic.config,
        expected_library_version="library-v1",
        state=detector_state_from_value(detector.state.to_value()),
    )
    for target in (detector, restored):
        with pytest.raises(ValueError):
            target.preview_observation(projection.latest_diagnostic, cold_start_supported=True)
        for step in (3, 4):
            result = target.preview_observation(
                _diagnostic(step, {}, config=projection.latest_diagnostic.config, stagnant=True),
                cold_start_supported=True,
            )
            target.commit_observation(result)
            assert isinstance(result, NoPhaseTransition if step == 3 else PhaseTransitionDetected)
    assert detector.state == restored.state
    normal = PhaseTransitionDetector.fresh(
        evolution_config=config(),
        diagnostics_config=projection.latest_diagnostic.config,
        library_version="library-v1",
    )
    normal.observe(_diagnostic(1, {"a": 1, "b": 1}, config=projection.latest_diagnostic.config))
    result = normal.preview_observation(
        _diagnostic(2, {"a": 2}, config=projection.latest_diagnostic.config, stagnant=True),
        cold_start_supported=True,
    )
    assert result.event.trigger_rule is PhaseTriggerRule.RESIDUAL_AND_ENTROPY


def test_disabled_configuration_serialization_remains_unchanged():
    old = replace(config(), cold_start=None)
    assert "cold_start" not in old.to_value()
    assert EvolutionConfig.from_value(old.to_value()) == old
    assert EvolutionConfig.from_value(config().to_value()) == config()


def test_cold_start_bounds_author_witnesses_without_losing_source_batch_support():
    projection = prepared_window(high_per_batch=6)
    cfg = replace(config(), importance_quantile=0.1)
    _, detected = detect(projection, cfg)
    frozen = projection.freeze_for_phase(detected.event)
    groups = cold_start_groups(
        frozen.diagnostics,
        TrajectoryEvidenceView(frozen.authoring_by_edge_id, frozen.source_by_trajectory),
        cfg,
    )
    assert len(groups) == 1
    assert len(groups[0].edge_exemplars) == cfg.cold_start.max_evidence_edges
    assert any(e.edge_id.startswith("t1-") for e in groups[0].edge_exemplars)
    assert any(e.edge_id.startswith("t2-") for e in groups[0].edge_exemplars)
    assert projection.calibration_cells == ()


def test_population_aliases_cannot_satisfy_two_independent_cold_start_sources():
    projection = prepared_window()
    _, detected = detect(projection, config())
    frozen = projection.freeze_for_phase(detected.event)
    aliases = {
        trajectory: ("benchmark", f"alias-{index}", "one-original-question")
        for index, trajectory in enumerate(frozen.source_by_trajectory)
    }
    assert not cold_start_groups(
        frozen.diagnostics, TrajectoryEvidenceView(frozen.authoring_by_edge_id, aliases), config()
    )


def test_declared_w50_cold_start_waits_for_full_hundred_committed_batches():
    """Controlled CPU support fixture, not natural model adoption or W50 efficacy."""
    diagnostics = DiagnosticsConfig(window_size=50)
    projection = type(prepared_window()).fresh(
        diagnostics_config=diagnostics,
        calibration=CalibrationEngine(CalibrationConfig(default_k=0.5)),
        library_version="library-v1",
    )
    detector = PhaseTransitionDetector.fresh(
        evolution_config=config(), diagnostics_config=diagnostics, library_version="library-v1"
    )
    for step in range(1, 101):
        item = artifact(f"w50-{step}", question=f"source-{step}", calls=False)
        projection.commit(
            projection.preview(
                make_source(
                    artifacts=(item,),
                    batch_id=f"w50-batch-{step}",
                    optimizer_step=step,
                    importances=((2.0 if step >= 99 else 0.1,),),
                )
            )
        )
        supported = projection.cold_start_supported(config())
        observed = detector.preview_observation(
            projection.latest_diagnostic, cold_start_supported=supported
        )
        detector.commit_observation(observed)
        if step < 100:
            assert not supported
            assert isinstance(observed, NoPhaseTransition)
    assert supported
    assert isinstance(observed, PhaseTransitionDetected)
    assert observed.event.trigger_rule is PhaseTriggerRule.ZERO_COVERAGE_COLD_START
    assert projection.calibration_cells == ()
