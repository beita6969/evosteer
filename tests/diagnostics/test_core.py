from __future__ import annotations

import math

import pytest

from skillev.diagnostics import (
    ComparableResidualWindows,
    DiagnosticsConfig,
    FreshDiagnosticsSegment,
    LibrarySegmentMismatchError,
    OnlineFlowDiagnostics,
    ZeroPreviousResidual,
    assemble_batch_flow_input,
    observe_batch,
    reset_diagnostics_segment,
    skill_marginal_flows,
    trajectory_flow,
)
from tests.v3_helpers import make_artifact, make_source


def test_raw_running_log_flow_and_skill_denominator_match_paper_math() -> None:
    first = make_artifact(
        "trajectory-a",
        skills_by_step=(("zeta",), ("alpha",)),
    )
    second = make_artifact(
        "trajectory-b",
        skills_by_step=(("alpha",),),
    )
    source = make_source(
        artifacts=(first, second),
        importances=((2.0, -5.0), (1.0,)),
        deltas=(0.2, 0.3),
    )
    batch = assemble_batch_flow_input(
        source.stats,
        {item.record.trajectory_id: item.record for item in source.artifacts},
        source.edge_records,
    )
    flows = tuple(trajectory_flow(item) for item in batch.trajectories)

    assert tuple(edge.sample_log_state_weight for edge in flows[0].edges) == (
        2.0,
        -3.0,
    )
    stats = skill_marginal_flows(flows)
    alpha = stats[0]
    expected = math.log(math.exp(-3.0) + math.exp(1.0)) - math.log(2)
    assert alpha.skill_id == "alpha"
    assert alpha.invoking_trajectory_count == 2
    assert alpha.invoking_edge_count == 2
    assert alpha.log_skill_flow == expected
    assert stats[1].skill_id == "zeta"
    assert stats[1].log_skill_flow == 2.0


def test_full_diagnostics_has_no_importance_clip_control() -> None:
    assert DiagnosticsConfig().to_value() == {
        "format": "skillev-diagnostics@4",
        "stagnation_rho": 0.05,
        "window_size": 50,
    }
    with pytest.raises(TypeError):
        DiagnosticsConfig(importance_clip=3.0)  # type: ignore[call-arg]


def test_stagnation_uses_raw_delta_squared_and_resets_by_library() -> None:
    config = DiagnosticsConfig(window_size=1, stagnation_rho=0.1)
    state = FreshDiagnosticsSegment(expected_library_version="library-v1")
    first = make_source(
        batch_id="batch-1",
        optimizer_step=1,
        deltas=(2.0,),
    )
    second = make_source(
        batch_id="batch-2",
        optimizer_step=2,
        deltas=(math.sqrt(3.8),),
    )
    for source in (first, second):
        batch = assemble_batch_flow_input(
            source.stats,
            {item.record.trajectory_id: item.record for item in source.artifacts},
            source.edge_records,
        )
        state, diagnostic = observe_batch(state, batch, config)
    assert isinstance(diagnostic.residual_window, ComparableResidualWindows)
    assert diagnostic.residual_window.previous_window_mean_delta_squared == 4.0
    assert math.isclose(
        diagnostic.residual_window.current_window_mean_delta_squared,
        3.8,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert diagnostic.residual_window.stagnant

    changed = make_source(
        batch_id="batch-3",
        optimizer_step=3,
        library_version="library-v2",
        artifacts=(make_artifact("trajectory-v2", library_version="library-v2"),),
        deltas=(0.1,),
    )
    changed_batch = assemble_batch_flow_input(
        changed.stats,
        {item.record.trajectory_id: item.record for item in changed.artifacts},
        changed.edge_records,
    )
    with pytest.raises(LibrarySegmentMismatchError):
        observe_batch(state, changed_batch, config)
    state = reset_diagnostics_segment(
        state,
        old_library_version="library-v1",
        new_library_version="library-v2",
    )
    state, changed_diagnostic = observe_batch(state, changed_batch, config)
    assert state.library_version == "library-v2"
    assert len(state.recent_squared_residual_batches) == 1
    assert changed_diagnostic.residual_window.kind == "insufficient-window"


def test_zero_previous_residual_is_a_distinct_non_triggering_end_to_end_state() -> None:
    config = DiagnosticsConfig(window_size=1)
    state = FreshDiagnosticsSegment(expected_library_version="library-v1")
    for step, delta in ((1, 0.0), (2, 1.0)):
        source = make_source(
            batch_id=f"zero-residual-{step}",
            optimizer_step=step,
            deltas=(delta,),
        )
        batch = assemble_batch_flow_input(
            source.stats,
            {item.record.trajectory_id: item.record for item in source.artifacts},
            source.edge_records,
        )
        state, diagnostic = observe_batch(state, batch, config)

    assert isinstance(diagnostic.residual_window, ZeroPreviousResidual)
    assert diagnostic.residual_window.previous_window_mean_delta_squared == 0.0
    assert diagnostic.residual_window.current_window_mean_delta_squared == 1.0
    assert not diagnostic.residual_window.residual_condition_met


def test_residual_windows_pool_trajectories_instead_of_averaging_batch_means() -> None:
    config = DiagnosticsConfig(window_size=2)
    state = FreshDiagnosticsSegment(expected_library_version="library-v1")
    layouts = (
        ((0.0,), 1),
        ((2.0, 2.0, 2.0), 3),
        ((math.sqrt(2.0),), 1),
        ((math.sqrt(2.0),) * 3, 3),
    )
    for step, (deltas, trajectory_count) in enumerate(layouts, start=1):
        artifacts = tuple(
            make_artifact(f"pooled-{step}-{index}") for index in range(trajectory_count)
        )
        source = make_source(
            batch_id=f"pooled-batch-{step}",
            optimizer_step=step,
            artifacts=artifacts,
            deltas=deltas,
        )
        batch = assemble_batch_flow_input(
            source.stats,
            {item.record.trajectory_id: item.record for item in source.artifacts},
            source.edge_records,
        )
        state, diagnostic = observe_batch(state, batch, config)

    assert isinstance(diagnostic.residual_window, ComparableResidualWindows)
    assert diagnostic.residual_window.previous_window_mean_delta_squared == 3.0
    assert diagnostic.residual_window.current_window_mean_delta_squared == pytest.approx(2.0)
    assert diagnostic.residual_window.previous_window_mean_delta_squared != 2.0


def test_flow_and_batch_fold_are_pure_and_repeatable() -> None:
    source = make_source(
        artifacts=(
            make_artifact(
                "purity-trajectory",
                skills_by_step=(("skill-alpha",), ("skill-beta",)),
            ),
        ),
        importances=((0.25, -0.5),),
    )
    source_snapshot = (
        tuple(item.to_value() for item in source.artifacts),
        source.stats.to_value(),
        tuple(item.to_value() for item in source.edge_records),
    )
    batch = assemble_batch_flow_input(
        source.stats,
        {item.record.trajectory_id: item.record for item in source.artifacts},
        source.edge_records,
    )
    trajectory = batch.trajectories[0]
    assert trajectory_flow(trajectory) == trajectory_flow(trajectory)

    state = FreshDiagnosticsSegment(expected_library_version="library-v1")
    config = DiagnosticsConfig(window_size=2)
    assert observe_batch(state, batch, config) == observe_batch(state, batch, config)
    assert state == FreshDiagnosticsSegment(expected_library_version="library-v1")
    assert source_snapshot == (
        tuple(item.to_value() for item in source.artifacts),
        source.stats.to_value(),
        tuple(item.to_value() for item in source.edge_records),
    )


def test_online_preview_is_immutable_until_commit() -> None:
    diagnostics = OnlineFlowDiagnostics.fresh(
        DiagnosticsConfig(window_size=1),
        library_version="library-v1",
    )
    source = make_source()
    transition = diagnostics.preview(source)

    assert diagnostics.state == FreshDiagnosticsSegment("library-v1")
    with pytest.raises(RuntimeError):
        _ = diagnostics.latest

    diagnostics.commit(transition)
    assert diagnostics.latest == transition.diagnostic
    assert diagnostics.state == transition.next_state
