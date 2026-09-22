from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from skillev.contracts.canonical import canonical_json, parse_canonical_json
from skillev.contracts.ttb_calibration import (
    ContextFeature,
    FailureMode,
    HorizonBucket,
    PosteriorBatchUpdate,
    PosteriorCellState,
    PosteriorUpdateEvent,
    TokenBucket,
)


def _context(
    *,
    context: str = "synthetic/tool_chain",
    failure_mode: FailureMode = FailureMode.SUCCESS,
    token_bucket: TokenBucket = TokenBucket.K1_TO_4K,
    horizon_bucket: HorizonBucket = HorizonBucket.H4_TO_8,
) -> ContextFeature:
    return ContextFeature(
        context=context,
        failure_mode=failure_mode,
        token_bucket=token_bucket,
        horizon_bucket=horizon_bucket,
    )


def _event(
    *,
    event_id: str = "event-1",
    skill_id: str = "skill-1",
    z: ContextFeature | None = None,
    trajectory_id: str = "trajectory-1",
    step_index: int = 2,
    outcome: bool = True,
    flow_weight: float = 0.5,
    alpha_after: float = 1.5,
    beta_count_after: float = 1.0,
) -> PosteriorUpdateEvent:
    return PosteriorUpdateEvent(
        alpha_before=alpha_after - (flow_weight if outcome else 0.0),
        beta_count_before=beta_count_after - (0.0 if outcome else flow_weight),
        event_id=event_id,
        skill_id=skill_id,
        z=_context() if z is None else z,
        trajectory_id=trajectory_id,
        step_index=step_index,
        outcome=outcome,
        flow_weight=flow_weight,
        alpha_after=alpha_after,
        beta_count_after=beta_count_after,
    )


def _prior(*, z: ContextFeature | None = None) -> PosteriorCellState:
    return PosteriorCellState(
        skill_id="skill-1",
        z=_context() if z is None else z,
        alpha=1.0,
        beta_count=1.0,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("success", FailureMode.SUCCESS),
        ("tool_error", FailureMode.TOOL_ERROR),
        ("schema_invalid", FailureMode.SCHEMA_INVALID),
        ("timeout", FailureMode.TIMEOUT),
        ("parse_error", FailureMode.PARSE_ERROR),
        ("other", FailureMode.OTHER),
    ],
)
def test_failure_mode_mapping_accepts_closed_text_statuses(
    status: str,
    expected: FailureMode,
) -> None:
    assert FailureMode.from_observation_status(status) is expected


@pytest.mark.parametrize("status", ["unrecognized", "", "success "])
def test_failure_mode_rejects_unknown_text_status(status: str) -> None:
    with pytest.raises(ValueError):
        FailureMode.from_observation_status(status)


def test_failure_mode_rejects_a_non_text_status() -> None:
    with pytest.raises(ValueError):
        FailureMode.from_observation_status(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, TokenBucket.LE_1K),
        (1_000, TokenBucket.LE_1K),
        (1_001, TokenBucket.K1_TO_4K),
        (4_000, TokenBucket.K1_TO_4K),
        (4_001, TokenBucket.GT_4K),
    ],
)
def test_token_bucket_boundaries(count: int, expected: TokenBucket) -> None:
    assert TokenBucket.from_count(count) is expected


def test_token_bucket_rejects_invalid_counts() -> None:
    assert TokenBucket.from_count(0) is TokenBucket.LE_1K
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            TokenBucket.from_count(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (1, HorizonBucket.LE_3),
        (3, HorizonBucket.LE_3),
        (4, HorizonBucket.H4_TO_8),
        (8, HorizonBucket.H4_TO_8),
        (9, HorizonBucket.GT_8),
    ],
)
def test_horizon_bucket_boundaries(
    horizon: int,
    expected: HorizonBucket,
) -> None:
    assert HorizonBucket.from_horizon(horizon) is expected


def test_horizon_bucket_rejects_invalid_horizons() -> None:
    assert HorizonBucket.from_horizon(1) is HorizonBucket.LE_3
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            HorizonBucket.from_horizon(value)  # type: ignore[arg-type]


def test_context_feature_accepts_a_complete_categorical_coordinate() -> None:
    feature = _context()

    assert feature.context == "synthetic/tool_chain"
    assert feature.failure_mode is FailureMode.SUCCESS
    assert feature.token_bucket is TokenBucket.K1_TO_4K
    assert feature.horizon_bucket is HorizonBucket.H4_TO_8


def test_context_feature_rejects_empty_or_untyped_coordinates() -> None:
    for context in ("", "   "):
        with pytest.raises(ValueError):
            _context(context=context)
    with pytest.raises(ValueError):
        ContextFeature(
            context="synthetic/tool_chain",
            failure_mode="success",  # type: ignore[arg-type]
            token_bucket=TokenBucket.LE_1K,
            horizon_bucket=HorizonBucket.LE_3,
        )
    with pytest.raises(ValueError):
        ContextFeature(
            context="synthetic/tool_chain",
            failure_mode=FailureMode.SUCCESS,
            token_bucket="le_1k",  # type: ignore[arg-type]
            horizon_bucket=HorizonBucket.LE_3,
        )
    with pytest.raises(ValueError):
        ContextFeature(
            context="synthetic/tool_chain",
            failure_mode=FailureMode.SUCCESS,
            token_bucket=TokenBucket.LE_1K,
            horizon_bucket="le_3",  # type: ignore[arg-type]
        )


def test_cell_key_uses_both_skill_and_structured_context() -> None:
    feature = _context(context="a|b")
    same = feature.cell_key("skill|c")

    assert same == feature.cell_key("skill|c")
    assert same != feature.cell_key("skill")
    assert same != _context(context="a").cell_key("b|skill|c")
    with pytest.raises(ValueError):
        feature.cell_key(" ")


def test_update_event_accepts_valid_zero_and_positive_weights() -> None:
    assert _event().flow_weight == pytest.approx(0.5)
    zero = _event(flow_weight=0.0, alpha_after=1.0)
    assert zero.flow_weight == 0.0


def test_posterior_batch_update_round_trips_ordered_updates() -> None:
    first = _event(event_id="event-1")
    second = _event(
        event_id="event-2",
        trajectory_id="trajectory-2",
        outcome=False,
        alpha_after=1.0,
        beta_count_after=1.5,
    )
    batch = PosteriorBatchUpdate(batch_id="batch-1", updates=(first, second))

    rebuilt = PosteriorBatchUpdate.from_value(
        parse_canonical_json(canonical_json(batch.to_value()))
    )

    assert rebuilt == batch
    assert rebuilt.content_hash == batch.content_hash


def test_posterior_batch_update_allows_empty_updates_and_rejects_duplicates() -> None:
    assert PosteriorBatchUpdate(batch_id="batch-1", updates=()).updates == ()
    duplicate = _event()
    with pytest.raises(ValueError):
        PosteriorBatchUpdate(batch_id="batch-1", updates=(duplicate, duplicate))


def test_update_event_requires_a_context_feature() -> None:
    assert isinstance(_event().z, ContextFeature)
    with pytest.raises(ValueError):
        replace(_event(), z="not-a-context")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "event_id",
        "skill_id",
        "trajectory_id",
    ],
)
def test_update_event_rejects_empty_identity_fields(field: str) -> None:
    assert _event().event_id
    with pytest.raises(ValueError):
        replace(_event(), **{field: " "})


def test_update_event_rejects_invalid_step_or_outcome_types() -> None:
    assert _event(step_index=1, outcome=False, alpha_after=1.0, beta_count_after=1.5)
    with pytest.raises(ValueError):
        _event(step_index=0)
    with pytest.raises(ValueError):
        replace(_event(), outcome=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "flow_weight",
    [-0.1, math.nan, math.inf, -math.inf, True, "weight"],
)
def test_update_event_rejects_invalid_flow_weights(flow_weight: object) -> None:
    assert _event(flow_weight=0.0, alpha_after=1.0)
    with pytest.raises(ValueError):
        replace(_event(), flow_weight=flow_weight)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha_after", 0.0),
        ("alpha_after", -0.1),
        ("alpha_after", math.nan),
        ("alpha_after", math.inf),
        ("beta_count_after", 0.0),
        ("beta_count_after", -0.1),
        ("beta_count_after", math.nan),
        ("beta_count_after", math.inf),
    ],
)
def test_update_event_requires_finite_positive_after_counts(
    field: str,
    value: float,
) -> None:
    assert _event().alpha_after > 0
    with pytest.raises(ValueError):
        replace(_event(), **{field: value})


def test_recorded_before_counts_do_not_override_the_configured_prior() -> None:
    event = _event(flow_weight=0.1, alpha_after=0.25, beta_count_after=0.75)

    assert event.alpha_after > 0
    with pytest.raises(ValueError):
        _prior().apply(event)


def test_pure_prior_cell_accepts_positive_matching_counts() -> None:
    state = PosteriorCellState(
        skill_id="skill-1",
        z=_context(),
        alpha=2.0,
        beta_count=3.0,
        alpha_0=2.0,
        beta_0=3.0,
    )

    assert state.update_count == 0
    assert state.last_event_id is None


def test_cell_requires_a_skill_identity_and_context_feature() -> None:
    assert _prior().skill_id
    assert isinstance(_prior().z, ContextFeature)
    with pytest.raises(ValueError):
        replace(_prior(), skill_id=" ")
    with pytest.raises(ValueError):
        replace(_prior(), z="not-a-context")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha_0", 0.0),
        ("alpha_0", -1.0),
        ("alpha_0", math.nan),
        ("beta_0", 0.0),
        ("beta_0", -1.0),
        ("beta_0", math.inf),
    ],
)
def test_cell_rejects_invalid_prior_counts(field: str, value: float) -> None:
    assert _prior().alpha_0 > 0
    with pytest.raises(ValueError):
        replace(_prior(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 0.5),
        ("beta_count", 0.5),
        ("alpha", math.nan),
        ("beta_count", math.inf),
    ],
)
def test_cell_rejects_counts_below_prior_or_nonfinite(
    field: str,
    value: float,
) -> None:
    assert _prior().alpha >= _prior().alpha_0
    with pytest.raises(ValueError):
        replace(_prior(), **{field: value})


def test_never_updated_cell_must_equal_its_prior() -> None:
    assert _prior().alpha == _prior().alpha_0
    with pytest.raises(ValueError):
        replace(_prior(), alpha=1.1)
    with pytest.raises(ValueError):
        replace(_prior(), beta_count=1.1)


def test_update_count_and_last_event_reference_move_together() -> None:
    updated = PosteriorCellState(
        skill_id="skill-1",
        z=_context(),
        alpha=1.0,
        beta_count=1.0,
        update_count=1,
        last_event_id="event-1",
    )
    assert updated.update_count == 1
    assert updated.last_event_id == "event-1"

    with pytest.raises(ValueError):
        replace(_prior(), update_count=1)
    with pytest.raises(ValueError):
        replace(_prior(), last_event_id="event-1")
    with pytest.raises(ValueError):
        replace(updated, update_count=-1)
    with pytest.raises(ValueError):
        replace(updated, update_count=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(updated, last_event_id=" ")


def test_posterior_statistics_are_derived_from_counts() -> None:
    state = PosteriorCellState(
        skill_id="skill-1",
        z=_context(),
        alpha=3.0,
        beta_count=2.0,
        update_count=1,
        last_event_id="event-1",
    )
    expected_variance = 3.0 * 2.0 / ((3.0 + 2.0) ** 2 * (3.0 + 2.0 + 1.0))

    assert state.mean() == pytest.approx(3.0 / 5.0)
    assert state.variance() == pytest.approx(expected_variance)
    assert state.lcb(2.0) == pytest.approx(state.mean() - 2.0 * math.sqrt(expected_variance))
    assert state.ucb(2.0) == pytest.approx(state.mean() + 2.0 * math.sqrt(expected_variance))


@pytest.mark.parametrize("k", [-0.1, math.nan, math.inf])
def test_confidence_bounds_reject_invalid_multipliers(k: float) -> None:
    state = _prior()
    assert state.lcb(0.0) == state.mean()
    assert state.ucb(0.0) == state.mean()
    with pytest.raises(ValueError):
        state.lcb(k)
    with pytest.raises(ValueError):
        state.ucb(k)


def test_apply_purely_updates_the_outcome_selected_count() -> None:
    prior = _prior()
    successful = prior.apply(_event())
    failed = successful.apply(
        _event(
            event_id="event-2",
            outcome=False,
            flow_weight=0.25,
            alpha_after=1.5,
            beta_count_after=1.25,
        )
    )

    assert prior.alpha == prior.beta_count == 1.0
    assert prior.update_count == 0
    assert successful.alpha == pytest.approx(1.5)
    assert successful.beta_count == pytest.approx(1.0)
    assert successful.update_count == 1
    assert failed.alpha == pytest.approx(1.5)
    assert failed.beta_count == pytest.approx(1.25)
    assert failed.update_count == 2
    assert failed.last_event_id == "event-2"


def test_apply_accepts_zero_weight_as_a_replayable_event() -> None:
    updated = _prior().apply(
        _event(
            flow_weight=0.0,
            alpha_after=1.0,
            beta_count_after=1.0,
        )
    )

    assert updated.alpha == updated.beta_count == 1.0
    assert updated.update_count == 1


def test_apply_rejects_an_event_for_another_cell() -> None:
    prior = _prior()
    assert prior.apply(_event()).update_count == 1
    with pytest.raises(ValueError):
        prior.apply(_event(skill_id="skill-2"))
    with pytest.raises(ValueError):
        prior.apply(_event(z=_context(context="different/task")))
    with pytest.raises(ValueError):
        prior.apply("not-an-event")  # type: ignore[arg-type]


def test_apply_rejects_after_counts_that_do_not_equal_the_weighted_increment() -> None:
    prior = _prior()
    assert prior.apply(_event()).alpha == pytest.approx(1.5)
    with pytest.raises(ValueError):
        prior.apply(_event(alpha_after=1.4))
    with pytest.raises(ValueError):
        prior.apply(_event(beta_count_after=1.1))


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (_context(), "context"),
        (_event(), "event_id"),
        (_prior(), "skill_id"),
    ],
)
def test_contracts_are_frozen_and_slotted(value: object, field: str) -> None:
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(value, field, "change")


@pytest.mark.parametrize(
    "value",
    [
        _context(),
        _event(),
        _prior(),
    ],
)
def test_canonical_round_trip_preserves_value_and_content_hash(value: object) -> None:
    encoded = canonical_json(value.to_value())  # type: ignore[attr-defined]
    decoded = parse_canonical_json(encoded)
    rebuilt = type(value).from_value(decoded)

    assert rebuilt == value
    assert rebuilt.content_hash == value.content_hash


def test_content_hashes_change_when_contract_content_changes() -> None:
    context = _context()
    event = _event()
    state = _prior()

    assert context.content_hash != replace(context, context="different/task").content_hash
    assert event.content_hash != replace(event, event_id="event-2").content_hash
    assert state.content_hash != state.apply(event).content_hash


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (ContextFeature.from_value, {"context": "missing-coordinates"}),
        (
            PosteriorUpdateEvent.from_value,
            {"event_id": "missing-event-fields"},
        ),
        (
            PosteriorCellState.from_value,
            {"skill_id": "missing-state-fields"},
        ),
    ],
)
def test_from_value_rejects_incomplete_records(factory: object, value: object) -> None:
    with pytest.raises(ValueError):
        factory(value)  # type: ignore[operator]
