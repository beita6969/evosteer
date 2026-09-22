from __future__ import annotations

import math
from dataclasses import replace

import pytest

from skillev.calibration import (
    EXTRACTOR_VERSION,
    CalibrationConfig,
    CalibrationEngine,
    ConfidenceMultiplier,
    calibration_updates_for_batch,
    extract_calibration_outcome,
    extract_context,
    extract_failure_mode,
    extract_horizon_bucket,
    extract_token_bucket,
    extract_z,
    flow_weights,
    prior_cell,
)
from skillev.contracts import FailureMode, HorizonBucket, PosteriorCellState, TokenBucket
from skillev.diagnostics import DiagnosticsConfig, OnlineFlowDiagnostics
from tests.v3_helpers import make_artifact, make_source


def _diagnostic(source):
    diagnostics = OnlineFlowDiagnostics.fresh(
        DiagnosticsConfig(window_size=1),
        library_version=source.stats.library_version,
    )
    return diagnostics.preview(source).diagnostic


def test_full_config_has_only_prior_and_confidence_controls() -> None:
    config = CalibrationConfig(alpha_0=2.0, beta_0=3.0, default_k=0.5)
    assert config.to_value() == {
        "alpha_0": 2.0,
        "beta_0": 3.0,
        "default_k": 0.5,
        "format": "skillev-calibration@3",
    }
    assert EXTRACTOR_VERSION == "ttb-z-extractors@3"
    with pytest.raises(TypeError):
        CalibrationConfig(enabled=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CalibrationConfig(flow_weight_cap=10.0)  # type: ignore[call-arg]


def test_four_extractors_use_the_canonical_contract_boundaries() -> None:
    artifact = make_artifact(
        "trajectory-z",
        skills_by_step=(("skill-alpha",),) * 4,
        statuses=("success", "schema_invalid", "success", "success"),
    )
    record = artifact.record
    assert extract_context(record) == "debug/task-family"
    assert extract_failure_mode(record, 2) is FailureMode.SCHEMA_INVALID
    assert extract_token_bucket(record) is TokenBucket.LE_1K
    assert extract_horizon_bucket(record) is HorizonBucket.H4_TO_8
    assert extract_z(record, 2).failure_mode is FailureMode.SCHEMA_INVALID
    assert all(extract_calibration_outcome(record, step) for step in range(1, 5))


def test_context_uses_namespaced_task_family_not_episode_or_retrieval_context() -> None:
    first = make_artifact("trajectory-context-a").record
    source = make_artifact("trajectory-context-b").record
    environment_id = "another-debug-environment"
    second = replace(
        source,
        environment_id=environment_id,
        initial_context=replace(
            source.initial_context,
            meta={**source.initial_context.meta, "environment_id": environment_id},
        ),
        reward=replace(source.reward, environment_id=environment_id),
    )

    assert first.environment_id != second.environment_id
    assert extract_context(first) == extract_context(second) == "debug/task-family"
    assert extract_context(first) != "other/debug-task-family"


def test_flow_weights_are_uncapped_mean_normalized_and_edge_scoped() -> None:
    first = make_artifact(
        "trajectory-a",
        skills_by_step=(("alpha",), ("gamma",), ()),
    )
    source = make_source(
        artifacts=(first,),
        importances=((0.0, math.log(3.0), 1_000.0),),
    )
    weights = flow_weights(_diagnostic(source).trajectories)
    assert weights == {
        ("trajectory-a", 1): 0.5,
        ("trajectory-a", 2): 1.5,
    }
    assert math.fsum(weights.values()) / len(weights) == 1.0
    no_invocation = make_source(
        artifacts=(make_artifact("trajectory-no-credit", skills_by_step=((),)),),
        importances=((10.0,),),
    )
    assert flow_weights(_diagnostic(no_invocation).trajectories) == {}


def test_updates_use_terminal_success_fixed_order_and_rolling_prior() -> None:
    success = make_artifact(
        "trajectory-a",
        skills_by_step=(("alpha",), ("beta",), ("alpha",)),
        reward_success=True,
    )
    failure = make_artifact(
        "trajectory-b",
        skills_by_step=(("alpha",),),
        reward_success=False,
    )
    source = make_source(
        artifacts=(success, failure),
        importances=((0.0, 0.0, 0.0), (0.0,)),
        deltas=(0.2, 0.3),
    )
    updates = calibration_updates_for_batch(
        batch_id=source.batch_id,
        flows=_diagnostic(source).trajectories,
        records_by_id={
            artifact.record.trajectory_id: artifact.record for artifact in source.artifacts
        },
        cells_before={},
        config=CalibrationConfig(),
    )
    assert tuple((item.trajectory_id, item.step_index, item.skill_id) for item in updates) == (
        ("trajectory-a", 1, "alpha"),
        ("trajectory-a", 2, "beta"),
        ("trajectory-a", 3, "alpha"),
        ("trajectory-b", 1, "alpha"),
    )
    assert tuple(item.outcome for item in updates) == (True, True, True, False)
    assert updates[2].alpha_after > updates[0].alpha_after


def test_underflow_is_reported_instead_of_silently_zeroing_invocation_evidence() -> None:
    source = make_source(
        artifacts=(make_artifact("small"), make_artifact("large")),
        importances=((-1_000.0,), (0.0,)),
    )
    with pytest.raises(ValueError) as error:
        flow_weights(_diagnostic(source).trajectories)
    assert "small:1" in str(error.value)


def test_log_normalization_retains_representable_subnormal_weight() -> None:
    source = make_source(
        artifacts=(make_artifact("small"), make_artifact("large")),
        importances=((-745.5,), (0.0,)),
    )
    weights = flow_weights(_diagnostic(source).trajectories)
    assert 0.0 < weights["small", 1] < 1e-300
    assert math.fsum(weights.values()) == pytest.approx(2.0)


def test_update_generation_is_pure_repeatable_and_reordered_chain_is_rejected() -> None:
    artifact = make_artifact(
        "trajectory-pure-calibration",
        skills_by_step=(("skill-alpha",), ("skill-alpha",)),
        reward_success=True,
    )
    source = make_source(
        artifacts=(artifact,),
        importances=((0.1, 0.2),),
    )
    flows = _diagnostic(source).trajectories
    records = {artifact.record.trajectory_id: artifact.record}
    records_snapshot = {key: value.to_value() for key, value in records.items()}
    flows_snapshot = tuple(item.to_value() for item in flows)
    cells: dict[str, PosteriorCellState] = {}

    first = calibration_updates_for_batch(
        batch_id=source.batch_id,
        flows=flows,
        records_by_id=records,
        cells_before=cells,
        config=CalibrationConfig(),
    )
    second = calibration_updates_for_batch(
        batch_id=source.batch_id,
        flows=flows,
        records_by_id=records,
        cells_before=cells,
        config=CalibrationConfig(),
    )

    assert first == second
    assert records_snapshot == {key: value.to_value() for key, value in records.items()}
    assert flows_snapshot == tuple(item.to_value() for item in flows)
    assert cells == {}
    assert len(first) == 2
    prior = prior_cell(
        skill_id=first[0].skill_id,
        z=first[0].z,
        config=CalibrationConfig(),
    )
    with pytest.raises(ValueError):
        prior.apply(first[1])


def test_engine_preview_then_commit_and_unseen_prior_query() -> None:
    source = make_source()
    diagnostic = _diagnostic(source)
    engine = CalibrationEngine(CalibrationConfig())
    z = extract_z(source.artifacts[0].record, 1)
    unseen = engine.query_configured("skill-alpha", z)
    assert not unseen.observed
    assert unseen.mean == 0.5
    assert unseen.lcb == pytest.approx(0.21132486540518713)

    transition = engine.preview(source=source, diagnostic=diagnostic)
    assert engine.all_cells() == ()
    assert transition.posterior_batch.batch_id == source.batch_id
    engine.commit(transition)
    assert engine.query_configured("skill-alpha", z).observed
    assert engine.all_cells()

    explicit = engine.query_at_k("skill-alpha", z, ConfidenceMultiplier(0.5))
    assert explicit.k == 0.5
    with pytest.raises(TypeError):
        engine.query_at_k("skill-alpha", z, 0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConfidenceMultiplier("0.5")  # type: ignore[arg-type]


def test_active_calibration_has_no_replay_or_disabled_mode_surface() -> None:
    import skillev.calibration as calibration

    assert not hasattr(calibration, "rebuild_posterior_store")
    assert not hasattr(calibration, "recompute_update_stream")
    assert not hasattr(CalibrationEngine, "from_events")
