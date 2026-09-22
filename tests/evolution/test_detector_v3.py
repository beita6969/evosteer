from __future__ import annotations

import math

import pytest

from skillev.contracts import PhaseTriggerRule
from skillev.diagnostics import (
    BatchDiagnostics,
    ComparableResidualWindows,
    DiagnosticsConfig,
    SkillFlowStat,
)
from skillev.evolution import (
    ActiveDetectorSegment,
    AwaitingDetectorSegment,
    EvolutionConfig,
    NoPhaseTransition,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
)
from skillev.evolution.detector import (
    NoPhaseTransitionReason,
    PhaseStatus,
    detector_state_from_value,
)


def test_no_op_consumes_cutoff_and_rearms_only_on_fresh_comparison_after_restore() -> None:
    diagnostics = DiagnosticsConfig(window_size=1)
    evolution = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        entropy_window=1,
        required_consecutive_drops=1,
    )
    detector = PhaseTransitionDetector.fresh(
        evolution_config=evolution,
        diagnostics_config=diagnostics,
        library_version="library-v1",
    )
    detector.observe(_diagnostic(1, {"a": 1, "b": 1}, config=diagnostics))
    detected = detector.observe(_diagnostic(2, {"a": 2}, config=diagnostics, stagnant=True))
    assert isinstance(detected, PhaseTransitionDetected)
    assert detector.state.phase_status == "detected"
    detector.close_no_op(detected.event, "no-action-admitted")
    restored = PhaseTransitionDetector.from_runtime_state(
        evolution_config=evolution,
        diagnostics_config=diagnostics,
        expected_library_version="library-v1",
        state=detector_state_from_value(detector.state.to_value()),
    )
    for target in (detector, restored):
        assert target.state.phase_status is PhaseStatus.NO_OP_WAITING
        assert target.state.no_op_closure.batch_id == "batch-2"
        with pytest.raises(ValueError):
            target.observe(_diagnostic(2, {"a": 2}, config=diagnostics, stagnant=True))
        waiting = target.observe(
            _diagnostic(3, {"a": 1, "b": 1}, config=diagnostics, stagnant=True)
        )
        assert isinstance(waiting, NoPhaseTransition)
        assert waiting.reason is NoPhaseTransitionReason.NO_OP_WAITING
        observation = target.observe(_diagnostic(4, {"a": 2}, config=diagnostics, stagnant=True))
        assert isinstance(observation, PhaseTransitionDetected)
        assert target.state.phase_status is PhaseStatus.DETECTED
        assert observation.event.event_id != detected.event.event_id
        assert tuple(item.optimizer_step for item in observation.triggering_window) == (3, 4)
    assert detector.state == restored.state


def _diagnostic(
    step: int,
    counts: dict[str, int],
    *,
    config: DiagnosticsConfig,
    library_version: str = "library-v1",
    stagnant: bool = False,
) -> BatchDiagnostics:
    current = 0.99 if stagnant else 0.5
    return BatchDiagnostics(
        batch_id=f"batch-{step}",
        optimizer_step=step,
        library_version=library_version,
        trajectories=(),
        skill_flows=tuple(
            SkillFlowStat(
                skill_id=skill_id,
                invoking_trajectory_count=1,
                invoking_edge_count=count,
                log_skill_flow=0.0,
            )
            for skill_id, count in sorted(counts.items())
        ),
        residual_window=ComparableResidualWindows(
            library_version=library_version,
            window_size=config.window_size,
            previous_window_mean_delta_squared=1.0,
            current_window_mean_delta_squared=current,
            relative_improvement=1.0 - current,
            rho=config.stagnation_rho,
            stagnant=stagnant,
        ),
        config=config,
    )


def test_full_detector_requires_residual_and_strict_entropy_drops() -> None:
    diagnostics = DiagnosticsConfig(window_size=1, stagnation_rho=0.05)
    detector = PhaseTransitionDetector.fresh(
        evolution_config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            entropy_window=1,
            required_consecutive_drops=2,
        ),
        diagnostics_config=diagnostics,
        library_version="library-v1",
    )

    assert isinstance(
        detector.observe(_diagnostic(1, {"a": 1, "b": 1}, config=diagnostics)),
        NoPhaseTransition,
    )
    assert isinstance(
        detector.observe(_diagnostic(2, {"a": 3, "b": 1}, config=diagnostics)),
        NoPhaseTransition,
    )
    detected = detector.observe(_diagnostic(3, {"a": 4}, config=diagnostics, stagnant=True))

    assert isinstance(detected, PhaseTransitionDetected)
    event = detected.event
    assert event.trigger_rule is PhaseTriggerRule.RESIDUAL_AND_ENTROPY
    assert event.residual_condition_met
    assert event.entropy_condition_met
    assert tuple(item.entropy for item in event.entropy_series) == (
        math.log(2.0),
        -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)),
        0.0,
    )


def test_zero_use_entropy_is_explicit_and_library_change_requires_reset() -> None:
    diagnostics = DiagnosticsConfig(window_size=1)
    detector = PhaseTransitionDetector.fresh(
        evolution_config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            entropy_window=1,
            required_consecutive_drops=1,
        ),
        diagnostics_config=diagnostics,
        library_version="library-v1",
    )

    assert isinstance(
        detector.observe(_diagnostic(1, {}, config=diagnostics)),
        NoPhaseTransition,
    )
    state = detector.state
    assert isinstance(state, ActiveDetectorSegment)
    assert state.batch_records[0].skill_invocation_counts == ()

    with pytest.raises(ValueError):
        detector.observe(
            _diagnostic(
                2,
                {"a": 1},
                config=diagnostics,
                library_version="library-v2",
            )
        )

    detector.reset_for_library("library-v1", "library-v2")
    state = detector.state
    assert isinstance(state, AwaitingDetectorSegment)
    assert state.expected_library_version == "library-v2"
    assert isinstance(
        detector.observe(
            _diagnostic(
                2,
                {"a": 1},
                config=diagnostics,
                library_version="library-v2",
            )
        ),
        NoPhaseTransition,
    )
    state = detector.state
    assert isinstance(state, ActiveDetectorSegment)
    assert state.library_version == "library-v2"
    assert len(state.batch_records) == 1


def test_zero_invocations_never_bootstrap_generate_even_on_a_stagnant_window() -> None:
    config = DiagnosticsConfig(window_size=1)
    detector = PhaseTransitionDetector.fresh(
        evolution_config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            entropy_window=1,
            required_consecutive_drops=1,
        ),
        diagnostics_config=config,
        library_version="library-v1",
    )
    detector.observe(_diagnostic(1, {}, config=config))
    for step in (2, 3, 4):
        observation = detector.observe(_diagnostic(step, {}, config=config, stagnant=True))
        assert isinstance(observation, NoPhaseTransition)
        assert observation.reason is NoPhaseTransitionReason.NO_INVOCATION_COVERAGE
        assert detector.state.phase_status is PhaseStatus.SEARCHING


def test_losing_all_invocations_is_not_a_measured_entropy_drop() -> None:
    config = DiagnosticsConfig(window_size=1)
    evolution = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        entropy_window=1,
        required_consecutive_drops=2,
    )
    detector = PhaseTransitionDetector.fresh(
        evolution_config=evolution, diagnostics_config=config, library_version="library-v1"
    )
    detector.observe(_diagnostic(1, {"a": 1, "b": 1}, config=config))
    detector.observe(_diagnostic(2, {"a": 3, "b": 1}, config=config))
    # Previously log(2) -> 0.56 -> empty=0 incorrectly opened a joint boundary.
    stopped = detector.observe(_diagnostic(3, {}, config=config, stagnant=True))
    assert isinstance(stopped, NoPhaseTransition)
    assert stopped.reason is NoPhaseTransitionReason.NO_INVOCATION_COVERAGE
    restored = PhaseTransitionDetector.from_runtime_state(
        evolution_config=evolution,
        diagnostics_config=config,
        expected_library_version="library-v1",
        state=detector_state_from_value(detector.state.to_value()),
    )
    # A later genuinely observed concentrated distribution can still qualify.
    for target in (detector, restored):
        assert isinstance(
            target.observe(_diagnostic(4, {"a": 1, "b": 1}, config=config, stagnant=True)),
            NoPhaseTransition,
        )
        target.observe(_diagnostic(5, {"a": 3, "b": 1}, config=config, stagnant=True))
        assert isinstance(
            target.observe(_diagnostic(6, {"a": 4}, config=config, stagnant=True)),
            PhaseTransitionDetected,
        )


def test_detector_rejects_diagnostics_from_another_configuration() -> None:
    expected = DiagnosticsConfig(window_size=1)
    detector = PhaseTransitionDetector.fresh(
        evolution_config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        diagnostics_config=expected,
        library_version="library-v1",
    )

    with pytest.raises(ValueError):
        detector.observe(
            _diagnostic(
                1,
                {"a": 1},
                config=DiagnosticsConfig(window_size=2),
            )
        )


def test_detector_preview_is_inert_until_explicit_commit() -> None:
    diagnostics = DiagnosticsConfig(window_size=1)
    detector = PhaseTransitionDetector.fresh(
        evolution_config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        diagnostics_config=diagnostics,
        library_version="library-v1",
    )
    before = detector.state

    observation = detector.preview_observation(_diagnostic(1, {"a": 1}, config=diagnostics))

    assert detector.state == before
    detector.commit_observation(observation)
    assert detector.state == observation.next_state
