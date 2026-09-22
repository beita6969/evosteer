from __future__ import annotations

import math
from dataclasses import replace

import pytest

from skillev.calibration import (
    CalibrationConfig,
    CalibrationEngine,
    ConfidenceMultiplier,
    extract_z,
    flow_weights,
)
from skillev.contracts import SuccessRule
from skillev.diagnostics import DiagnosticsConfig, EdgeFlowDiagnostic, OnlineFlowDiagnostics
from skillev.evolution.evidence import query_cell
from skillev.training.calibration_reporting import prequential_calibration
from skillev.training.projections import FullProjectionRuntimeState, MethodProjectionPipeline
from tests.v3_helpers import make_artifact, make_phase_event, make_source


def pipeline() -> tuple[MethodProjectionPipeline, CalibrationEngine]:
    engine = CalibrationEngine(CalibrationConfig(default_k=0.5))
    projection = MethodProjectionPipeline.fresh(
        diagnostics_config=DiagnosticsConfig(window_size=1),
        calibration=engine,
        library_version="library-v1",
    )
    return projection, engine


def test_prequential_report_never_uses_earlier_labels_from_the_same_batch() -> None:
    projection, _ = pipeline()
    artifacts = tuple(make_artifact(f"prequential-{i}", reward_success=True) for i in range(4))
    projection.commit(projection.preview(make_source(artifacts=artifacts)))
    report = prequential_calibration(projection.runtime_state().posterior_provenance)[0]
    cell = report["cells"][0]
    assert cell["pre_batch_mean"] == 0.5
    assert cell["brier_score"] == 0.25
    assert cell["distinct_trajectory_count"] == 4
    assert projection.calibration_cells[0].mean() > cell["pre_batch_mean"]
    assert prequential_calibration(restore(projection).runtime_state().posterior_provenance) == [
        report
    ]


def restore(projection: MethodProjectionPipeline) -> MethodProjectionPipeline:
    return MethodProjectionPipeline.from_runtime_state(
        diagnostics_config=DiagnosticsConfig(window_size=1),
        calibration_config=CalibrationConfig(default_k=0.5),
        state=FullProjectionRuntimeState.from_value(projection.runtime_state().to_value()),
    )


def test_native_metric_ttb_reward_and_binary_label_remain_independent() -> None:
    artifact = make_artifact("native", reward_success=False, reward_value=0.8)
    reward = replace(
        artifact.record.reward,
        success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
        native_payload={"raw_score": -0.4, "exact_match": False},
    )
    artifact = replace(artifact, record=replace(artifact.record, reward=reward))
    source = make_source(artifacts=(artifact,))
    projection, engine = pipeline()
    transition = projection.preview(source)
    assert transition.source.stats.mean_reward == 0.8
    update = transition.posterior_batch.updates[0]
    assert not update.outcome
    assert update.alpha_after == update.alpha_before
    assert update.beta_count_after == update.beta_count_before + update.flow_weight
    z = extract_z(artifact.record, 1)
    assert not engine.query_configured("skill-alpha", z).observed
    projection.commit(transition)
    assert engine.query_configured("skill-alpha", z) == projection.query_configured(
        "skill-alpha", z
    )
    assert engine.query_configured("skill-alpha", z).mean < 0.5
    assert restore(projection).query_configured("skill-alpha", z) == engine.query_configured(
        "skill-alpha", z
    )
    # An external reference to the calculator cannot publish a second posterior.
    other = make_source(batch_id="later", optimizer_step=2)
    calculated = engine.preview(source=other, diagnostic=projection.preview(other).diagnostic)
    with pytest.raises(RuntimeError):
        engine.commit(calculated)


def test_whole_batch_weights_differ_from_worker_local_normalization() -> None:
    artifacts = tuple(make_artifact(f"independent-{i}") for i in range(4))
    source = make_source(artifacts=artifacts, importances=((0.0,), (0.0,), (0.0,), (math.log(9),)))
    projection, _ = pipeline()
    transition = projection.preview(source)
    weights = [event.flow_weight for event in transition.posterior_batch.updates]
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3, 3])
    flows = transition.diagnostic.trajectories
    shard_weights = {**flow_weights(flows[:2]), **flow_weights(flows[2:])}
    assert list(shard_weights.values()) != pytest.approx(weights)
    projection.commit(transition)
    cell = projection.calibration_cells[0]
    assert cell.update_count == 4
    assert cell.cumulative_flow_weight == pytest.approx(4)
    assert query_cell(cell, 2) == projection.query_at_k(
        cell.skill_id, cell.z, ConfidenceMultiplier(2)
    )
    assert projection.query_configured(cell.skill_id, cell.z).k == 0.5


def test_hand_computed_prefix_samples_skill_trajectory_denominator_and_beta() -> None:
    artifacts = (
        make_artifact(
            "two-calls", skills_by_step=(("skill-alpha",), ("skill-alpha",)), reward_success=True
        ),
        make_artifact("one-call", skills_by_step=(("skill-alpha",),), reward_success=False),
    )
    source = make_source(
        artifacts=artifacts, importances=((math.log(2), math.log(3)), (math.log(4),))
    )
    projection, _ = pipeline()
    transition = projection.preview(source)
    samples = [
        math.exp(edge.sample_log_state_weight)
        for trajectory in transition.diagnostic.trajectories
        for edge in trajectory.edges
    ]
    assert samples == pytest.approx([2, 6, 4])
    # Sum invoking states, divide by TWO invoking trajectories, not three calls.
    assert math.exp(transition.diagnostic.skill_flows[0].log_skill_flow) == pytest.approx(6)
    assert [event.flow_weight for event in transition.posterior_batch.updates] == pytest.approx(
        [0.5, 1.5, 1]
    )
    projection.commit(transition)
    cell = projection.calibration_cells[0]
    assert (cell.alpha, cell.beta_count) == pytest.approx((3, 2))


def test_retries_stale_previews_and_reset_previews_cannot_overwrite_evidence() -> None:
    projection, _ = pipeline()
    first = make_source(batch_id="one")
    pending = projection.preview(first)
    stale = projection.preview(make_source(batch_id="two", optimizer_step=2))
    projection.commit(pending)
    before = projection.runtime_state()
    for target in (projection, restore(projection)):
        with pytest.raises(ValueError):
            target.preview(first)
        with pytest.raises(ValueError):
            target.preview(
                make_source(batch_id="renamed-retry", optimizer_step=2, artifacts=first.artifacts)
            )
    with pytest.raises(ValueError):
        projection.commit(pending)
    with pytest.raises(ValueError):
        projection.commit(stale)
    assert projection.runtime_state() is before
    reset = projection.preview_reset(old_version="library-v1", new_version="library-v2")
    projection.commit(projection.preview(make_source(batch_id="two", optimizer_step=2)))
    with pytest.raises(ValueError):
        projection.commit_reset(reset)


def test_empty_invocation_batch_is_durable_and_cannot_be_counted_twice() -> None:
    artifact = make_artifact("retrieved-not-called", skills_by_step=((),))
    initial = replace(
        artifact.record.initial_context,
        retrieved_skill_ids=("skill-alpha",),
        active_skill_ids=("skill-alpha",),
    )
    artifact = replace(
        artifact,
        record=replace(artifact.record, initial_context=initial),
        initial_context=replace(artifact.initial_context, contract=initial),
    )
    source = make_source(artifacts=(artifact,))
    projection, _ = pipeline()
    transition = projection.preview(source)
    assert transition.posterior_batch.updates == ()
    projection.commit(transition)
    assert projection.calibration_cells == ()
    recovered = restore(projection)
    assert recovered.posterior_provenance.batches[0].trajectory_ids == ("retrieved-not-called",)
    with pytest.raises(ValueError):
        recovered.preview(source)


def test_restore_checks_arithmetic_not_just_event_count() -> None:
    projection, _ = pipeline()
    projection.commit(projection.preview(make_source()))
    state = projection.runtime_state()
    cell = state.calibration_cells[0]
    wrong = replace(state, calibration_cells=(replace(cell, alpha=cell.alpha + 1),))
    with pytest.raises(ValueError):
        MethodProjectionPipeline.from_runtime_state(
            diagnostics_config=DiagnosticsConfig(window_size=1),
            calibration_config=CalibrationConfig(default_k=0.5),
            state=wrong,
        )
    event = state.posterior_provenance.updates[0]
    assert event.alpha_before == cell.alpha_0
    assert event.alpha_after == cell.alpha


def test_phase_reads_one_frozen_posterior_and_diagnostic_cutoff() -> None:
    projection, _ = pipeline()
    for step in (1, 2):
        projection.commit(
            projection.preview(make_source(batch_id=f"batch-{step}", optimizer_step=step))
        )
    phase = make_phase_event()
    captured = projection.freeze_for_phase(phase)
    projection.commit(projection.preview(make_source(batch_id="batch-3", optimizer_step=3)))
    assert captured.calibration_cells[0].update_count == 2
    assert projection.calibration_cells[0].update_count == 3
    assert [item.optimizer_step for item in captured.diagnostics] == [1, 2]
    assert sum(map(len, captured.event_ids_by_cell.values())) == 2
    with pytest.raises(TypeError):
        captured.event_ids_by_cell["another"] = ()
    with pytest.raises(ValueError):
        projection.freeze_for_phase(phase)


def test_numerical_failure_keeps_the_entire_projection_uncommitted() -> None:
    projection, _ = pipeline()
    before = projection.runtime_state()
    source = make_source(
        artifacts=(make_artifact("tiny-flow"), make_artifact("large-flow")),
        importances=((-1_000.0,), (0.0,)),
    )
    with pytest.raises(ValueError):
        projection.preview(source)
    assert projection.runtime_state() is before
    assert projection.posterior_provenance.batches == ()


@pytest.mark.parametrize(
    "field",
    [
        "batch_id",
        "policy_snapshot_id",
        "library_version",
        "action_token_count",
        "backward_adapter_version",
    ],
)
def test_edge_scoring_context_cannot_mix_workers_or_action_denominators(field: str) -> None:
    source = make_source(artifacts=(make_artifact("a"), make_artifact("b")))
    edge = source.edge_records[1]
    assert edge.context is not None
    if field == "backward_adapter_version":
        bad = replace(edge, backward_adapter_version="backward@other")
    else:
        bad = replace(
            edge,
            context=replace(
                edge.context, **{field: 2 if field == "action_token_count" else "other"}
            ),
        )
    with pytest.raises(ValueError):
        pipeline()[0].preview(replace(source, edge_records=(source.edge_records[0], bad)))


def test_residual_cannot_be_joined_to_recomputed_edge_scores() -> None:
    source = make_source()
    edge = source.edge_records[0]
    edge = replace(
        edge,
        forward_logprob_per_token=edge.forward_logprob_per_token + 1,
        step_importance=edge.step_importance + 1,
    )
    with pytest.raises(ValueError):
        pipeline()[0].preview(replace(source, edge_records=(edge,)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_flow_reports_source_instead_of_clipping(value: float) -> None:
    with pytest.raises(ValueError):
        EdgeFlowDiagnostic("trajectory-bad", 1, 0.0, value, ("skill-alpha",))


def test_engine_stale_commit_is_rejected_even_for_zero_invocation_batch() -> None:
    engine = CalibrationEngine(CalibrationConfig())
    source = make_source(artifacts=(make_artifact("empty", skills_by_step=((),)),))
    diagnostic = (
        OnlineFlowDiagnostics.fresh(DiagnosticsConfig(), library_version="library-v1")
        .preview(source)
        .diagnostic
    )
    transition = engine.preview(source=source, diagnostic=diagnostic)
    engine.commit(transition)
    with pytest.raises(ValueError):
        engine.commit(transition)


def test_library_reset_keeps_history_without_copying_parent_confidence() -> None:
    projection, _ = pipeline()
    projection.commit(projection.preview(make_source()))
    old = projection.calibration_cells
    projection.reset_library_segment("library-v1", "library-v2")
    assert projection.calibration_cells == old
    assert projection.diagnostics_history == ()
    for skill in ("new-refined-id", "split-left-id", "split-right-id", "generated-id"):
        query = projection.query_configured(skill, old[0].z)
        assert not query.observed
        assert query.cumulative_flow_weight == 0
    assert restore(projection).calibration_cells == old


def test_prepared_backward_span_cannot_change_tokens_at_the_same_length(make_training_harness):
    import asyncio

    from skillev.scoring.edge_plan import prepare_edge_plan
    from skillev.training.step_math import compute_ttb_artifact_contribution

    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    artifact = batch.artifacts[0]
    plan = prepare_edge_plan(
        harness.backbone.tokenizer, artifact.record, artifact.initial_context.text
    )
    backward = plan.edges[-1]
    wrong_ids = (backward.action_ids[0] + 1, *backward.action_ids[1:])
    bad_plan = replace(plan, edges=(*plan.edges[:-1], replace(backward, action_ids=wrong_ids)))
    with pytest.raises(ValueError):
        compute_ttb_artifact_contribution(
            backbone=harness.backbone,
            parameters=harness.backbone.parameter_groups(),
            artifact=artifact,
            position=0,
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            policy_snapshot_id=batch.policy_snapshot_id,
            library_version=batch.library_version,
            global_batch_size=len(batch.artifacts),
            temperature_beta=harness.config.method.temperature_beta,
            prepared_edges=bad_plan,
        )
    assert all(
        parameter.grad is None
        for group in (
            harness.backbone.parameter_groups().forward,
            harness.backbone.parameter_groups().backward,
            harness.backbone.parameter_groups().z_head,
        )
        for parameter in group
    )


def test_weighted_beta_arithmetic_and_fixed_complete_batch_order() -> None:
    source = make_source(
        artifacts=(
            make_artifact("success", reward_success=True),
            make_artifact("failure", reward_success=False),
        ),
        importances=((math.log(3),), (0.0,)),
    )
    projection, _ = pipeline()
    transition = projection.preview(source)
    shuffled_edges = projection.preview(replace(source, edge_records=source.edge_records[::-1]))
    assert shuffled_edges.posterior_batch == transition.posterior_batch
    projection.commit(transition)
    cell = projection.calibration_cells[0]
    assert (cell.alpha, cell.beta_count) == pytest.approx((2.5, 1.5))
    with pytest.raises(ValueError):
        projection.preview(source)
    assert projection.calibration_cells == (cell,)
    with pytest.raises(ValueError):
        replace(source, artifacts=source.artifacts[::-1])


def test_equal_length_different_scored_action_is_rejected() -> None:
    source = make_source()
    edge = source.edge_records[0]
    assert edge.context is not None
    ids = edge.context.action_token_ids
    altered = replace(edge, context=replace(edge.context, action_token_ids=ids[::-1]))
    with pytest.raises(ValueError):
        pipeline()[0].preview(replace(source, edge_records=(altered,)))
