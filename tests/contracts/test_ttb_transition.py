from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts.canonical import canonical_json, parse_canonical_json
from skillev.contracts.ttb_transition import (
    EntropyObservation,
    PhaseTransitionEvent,
    PhaseTriggerRule,
    WindowStats,
)


def _window(
    *,
    start: int = 0,
    end: int = 1,
    msr: float = 0.5,
    member_batch_ids: tuple[str, ...] = ("batch-0", "batch-1"),
) -> WindowStats:
    return WindowStats(
        start_optimizer_step=start,
        end_optimizer_step=end,
        batch_count=len(member_batch_ids),
        mean_squared_residual=msr,
        member_batch_ids=member_batch_ids,
    )


def _phase_event() -> PhaseTransitionEvent:
    return PhaseTransitionEvent(
        event_id="phase-1",
        triggered_at_step=3,
        library_version="library-v1",
        previous_window=_window(),
        current_window=_window(
            start=2,
            end=3,
            msr=0.49,
            member_batch_ids=("batch-2", "batch-3"),
        ),
        relative_improvement=0.02,
        rho=0.05,
        residual_condition_met=True,
        entropy_series=(
            EntropyObservation(1, 1.0, 3, 3),
            EntropyObservation(2, 0.8, 3, 3),
            EntropyObservation(3, 0.6, 2, 2),
        ),
        required_consecutive_drops=2,
        entropy_condition_met=True,
        triggered=True,
    )


def test_window_stats_accept_valid_evidence_and_round_trip() -> None:
    window = _window()
    rebuilt = WindowStats.from_value(parse_canonical_json(canonical_json(window.to_value())))

    assert rebuilt == window
    assert rebuilt.content_hash == window.content_hash


@pytest.mark.parametrize(
    "mutation",
    [
        {"start_optimizer_step": -1},
        {"start_optimizer_step": True},
        {"end_optimizer_step": -1},
        {"end_optimizer_step": True},
        {"batch_count": 0, "member_batch_ids": ()},
        {"batch_count": True},
        {"batch_count": 1},
        {"member_batch_ids": ["batch-0", "batch-1"]},
        {"member_batch_ids": ("batch-0", "batch-0")},
        {"member_batch_ids": ("batch-0", "")},
        {"mean_squared_residual": -0.1},
        {"mean_squared_residual": float("nan")},
        {"mean_squared_residual": float("inf")},
    ],
)
def test_window_stats_reject_each_broken_invariant(mutation: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "start_optimizer_step": 0,
        "end_optimizer_step": 1,
        "batch_count": 2,
        "mean_squared_residual": 0.5,
        "member_batch_ids": ("batch-0", "batch-1"),
    }
    fields.update(mutation)

    with pytest.raises(ValueError):
        WindowStats(**fields)  # type: ignore[arg-type]


def test_entropy_observation_accepts_valid_evidence_and_round_trip() -> None:
    observation = EntropyObservation(
        window_end_step=2, entropy=0.5, total_invocations=2, distinct_skills=2
    )
    rebuilt = EntropyObservation.from_value(
        parse_canonical_json(canonical_json(observation.to_value()))
    )

    assert rebuilt == observation
    assert rebuilt.content_hash == observation.content_hash


@pytest.mark.parametrize(
    "mutation",
    [
        {"window_end_step": -1},
        {"window_end_step": True},
        {"entropy": -0.1},
        {"entropy": True},
        {"entropy": float("nan")},
        {"entropy": float("inf")},
        {"distinct_skills": 0},
        {"distinct_skills": True},
    ],
)
def test_entropy_observation_rejects_each_broken_invariant(
    mutation: dict[str, object],
) -> None:
    fields: dict[str, object] = {
        "window_end_step": 2,
        "entropy": 0.5,
        "total_invocations": 2,
        "distinct_skills": 2,
    }
    fields.update(mutation)

    with pytest.raises(ValueError):
        EntropyObservation(**fields)  # type: ignore[arg-type]


def test_entropy_observation_accepts_explicit_zero_use_window() -> None:
    observation = EntropyObservation(
        window_end_step=4,
        entropy=0.0,
        total_invocations=0,
        distinct_skills=0,
    )

    assert EntropyObservation.from_value(observation.to_value()) == observation


@pytest.mark.parametrize(
    ("entropy", "total_invocations", "distinct_skills"),
    [(0.1, 0, 0), (0.0, 0, 1), (0.0, 1, 0), (0.0, 1, 2)],
)
def test_entropy_observation_rejects_inconsistent_use_counts(
    entropy: float,
    total_invocations: int,
    distinct_skills: int,
) -> None:
    with pytest.raises(ValueError):
        EntropyObservation(
            window_end_step=4,
            entropy=entropy,
            total_invocations=total_invocations,
            distinct_skills=distinct_skills,
        )


def test_phase_transition_accepts_conjoined_evidence_and_round_trip() -> None:
    event = _phase_event()
    rebuilt = PhaseTransitionEvent.from_value(
        parse_canonical_json(canonical_json(event.to_value()))
    )

    assert rebuilt == event
    assert rebuilt.content_hash == event.content_hash
    assert rebuilt.trigger_rule is PhaseTriggerRule.RESIDUAL_AND_ENTROPY


def test_residual_only_transition_persists_rule_and_observed_entropy_without_gating() -> None:
    event = replace(
        _phase_event(),
        trigger_rule=PhaseTriggerRule.RESIDUAL_ONLY,
        entropy_series=(
            EntropyObservation(1, 1.0, 2, 2),
            EntropyObservation(2, 1.0, 2, 2),
        ),
        entropy_condition_met=False,
        triggered=True,
    )
    rebuilt = PhaseTransitionEvent.from_value(
        parse_canonical_json(canonical_json(event.to_value()))
    )

    assert rebuilt == event
    assert rebuilt.trigger_rule is PhaseTriggerRule.RESIDUAL_ONLY
    assert rebuilt.residual_condition_met
    assert not rebuilt.entropy_condition_met
    assert rebuilt.triggered


def test_residual_only_transition_cannot_fabricate_entropy_observation_result() -> None:
    with pytest.raises(ValueError):
        replace(
            _phase_event(),
            trigger_rule=PhaseTriggerRule.RESIDUAL_ONLY,
            entropy_series=(
                EntropyObservation(1, 1.0, 2, 2),
                EntropyObservation(2, 1.0, 2, 2),
            ),
            entropy_condition_met=True,
        )


def test_residual_only_transition_still_requires_residual_condition() -> None:
    with pytest.raises(ValueError):
        replace(
            _phase_event(),
            trigger_rule=PhaseTriggerRule.RESIDUAL_ONLY,
            current_window=_window(
                start=2,
                end=3,
                msr=0.2,
                member_batch_ids=("batch-2", "batch-3"),
            ),
            relative_improvement=0.6,
            residual_condition_met=False,
            triggered=True,
        )


def test_phase_transition_defaults_to_two_consecutive_entropy_drops() -> None:
    template = _phase_event()
    event = PhaseTransitionEvent(
        event_id=template.event_id,
        triggered_at_step=template.triggered_at_step,
        library_version=template.library_version,
        previous_window=template.previous_window,
        current_window=template.current_window,
        relative_improvement=template.relative_improvement,
        rho=template.rho,
        residual_condition_met=template.residual_condition_met,
        entropy_series=template.entropy_series,
        entropy_condition_met=template.entropy_condition_met,
        triggered=template.triggered,
    )

    assert event.required_consecutive_drops == 2


def test_transition_numeric_fields_normalize_before_hashing() -> None:
    window = _window(msr=1)  # type: ignore[arg-type]
    observation = EntropyObservation(
        window_end_step=1, entropy=1, total_invocations=2, distinct_skills=2
    )
    phase = replace(_phase_event(), rho=1)

    for record, record_type in (
        (window, WindowStats),
        (observation, EntropyObservation),
        (phase, PhaseTransitionEvent),
    ):
        rebuilt = record_type.from_value(parse_canonical_json(canonical_json(record.to_value())))
        assert rebuilt == record
        assert rebuilt.content_hash == record.content_hash


@pytest.mark.parametrize(
    "mutation",
    [
        {"event_id": ""},
        {"event_id": 1},
        {"library_version": ""},
        {"library_version": 1},
        {"triggered_at_step": -1},
        {"triggered_at_step": True},
        {"previous_window": "not-a-window"},
        {
            "current_window": _window(
                start=1,
                end=2,
                msr=0.49,
                member_batch_ids=("batch-2", "batch-3"),
            )
        },
        {"previous_window": _window(msr=0.0)},
        {"relative_improvement": 0.021},
        {"relative_improvement": float("nan")},
        {"rho": 0.0},
        {"rho": float("inf")},
        {"residual_condition_met": False},
        {"residual_condition_met": 1},
        {"trigger_rule": "residual-only"},
        {"required_consecutive_drops": 0},
        {"required_consecutive_drops": True},
        {
            "entropy_series": [
                EntropyObservation(1, 1.0, 2, 2),
                EntropyObservation(2, 0.8, 2, 2),
                EntropyObservation(3, 0.6, 2, 2),
            ]
        },
        {"entropy_series": ("not-an-observation",)},
        {
            "entropy_series": (
                EntropyObservation(1, 1.0, 2, 2),
                EntropyObservation(2, 0.8, 2, 2),
            )
        },
        {
            "entropy_series": (
                EntropyObservation(1, 1.0, 2, 2),
                EntropyObservation(1, 0.8, 2, 2),
                EntropyObservation(3, 0.6, 2, 2),
            )
        },
        {
            "entropy_series": (
                EntropyObservation(1, 1.0, 2, 2),
                EntropyObservation(2, 0.8, 2, 2),
                EntropyObservation(3, 0.9, 2, 2),
            )
        },
        {"entropy_condition_met": False},
        {"entropy_condition_met": 1},
        {"triggered": False},
        {"triggered": 1},
    ],
)
def test_phase_transition_rejects_each_broken_invariant(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        replace(_phase_event(), **mutation)


def test_phase_transition_rejects_a_well_formed_non_trigger() -> None:
    with pytest.raises(ValueError):
        PhaseTransitionEvent(
            event_id="phase-no-trigger",
            triggered_at_step=3,
            library_version="library-v1",
            previous_window=_window(),
            current_window=_window(
                start=2,
                end=3,
                msr=0.2,
                member_batch_ids=("batch-2", "batch-3"),
            ),
            relative_improvement=0.6,
            rho=0.05,
            residual_condition_met=False,
            entropy_series=(
                EntropyObservation(1, 1.0, 2, 2),
                EntropyObservation(2, 0.8, 2, 2),
                EntropyObservation(3, 0.6, 2, 2),
            ),
            required_consecutive_drops=2,
            entropy_condition_met=True,
            triggered=False,
        )


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (WindowStats.from_value, {"batch_count": 1}),
        (EntropyObservation.from_value, {"entropy": 0.1}),
        (PhaseTransitionEvent.from_value, {"event_id": "phase-1"}),
    ],
)
def test_transition_from_value_rejects_incomplete_records(
    factory: object,
    payload: object,
) -> None:
    with pytest.raises(ValueError):
        factory(payload)  # type: ignore[operator]
