from __future__ import annotations

import math
from dataclasses import replace

import pytest

from skillev.contracts.canonical import canonical_json, parse_canonical_json
from skillev.contracts.ttb_training import (
    EdgeLogprobRecord,
    TrajectoryResidual,
    TTBBatchStats,
)


def _edge() -> EdgeLogprobRecord:
    return EdgeLogprobRecord(
        trajectory_id="trajectory-001",
        step_index=1,
        forward_logprob_per_token=-0.8,
        backward_logprob_per_token=-1.1,
        step_importance=0.3,
        forward_adapter_version="forward@12",
        backward_adapter_version="backward@12",
        scoring_stack_id="training-stack",
    )


def _first_residual() -> TrajectoryResidual:
    return TrajectoryResidual(
        trajectory_id="trajectory-001",
        log_z=0.2,
        sum_forward=-1.2,
        sum_backward=-1.0,
        log_shifted_reward=-0.1,
        raw_reward=0.8,
        temperature_beta=0.5,
        delta=0.05,
        horizon=2,
    )


def _second_residual() -> TrajectoryResidual:
    return TrajectoryResidual(
        trajectory_id="trajectory-002",
        log_z=0.4,
        sum_forward=-0.7,
        sum_backward=-0.5,
        log_shifted_reward=-0.2,
        raw_reward=0.4,
        temperature_beta=1.0,
        delta=0.4,
        horizon=1,
    )


def _batch(
    residuals: tuple[TrajectoryResidual, ...] | None = None,
    *,
    created_at: str = "2026-07-19T09:30:00Z",
) -> TTBBatchStats:
    members = residuals if residuals is not None else (_first_residual(), _second_residual())
    batch_loss = math.fsum((member.delta / member.horizon) ** 2 for member in members) / len(
        members
    )
    mean_reward = math.fsum(member.raw_reward for member in members) / len(members)
    return TTBBatchStats(
        batch_id="batch-012",
        optimizer_step=12,
        library_version="library@3",
        residuals=members,
        batch_loss=batch_loss,
        mean_reward=mean_reward,
        created_at=created_at,
    )


def test_valid_edge_logprob_record_preserves_natural_log_mean_units() -> None:
    edge = _edge()

    assert edge.step_importance == pytest.approx(
        edge.forward_logprob_per_token - edge.backward_logprob_per_token
    )
    assert edge.scoring_stack_id == "training-stack"


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("trajectory_id", ""),
        ("forward_adapter_version", ""),
        ("backward_adapter_version", "   "),
        ("scoring_stack_id", ""),
        ("scoring_stack_id", "vllm-serving"),
    ],
)
def test_edge_logprob_record_rejects_invalid_identity_or_version_fields(
    field_name: str,
    bad_value: object,
) -> None:
    assert _edge().trajectory_id

    with pytest.raises(ValueError):
        replace(_edge(), **{field_name: bad_value})


@pytest.mark.parametrize("bad_step_index", [0, -1])
def test_edge_logprob_record_rejects_nonpositive_step_index(bad_step_index: int) -> None:
    assert _edge().step_index == 1

    with pytest.raises(ValueError):
        replace(_edge(), step_index=bad_step_index)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("forward_logprob_per_token", float("nan")),
        ("forward_logprob_per_token", float("inf")),
        ("backward_logprob_per_token", float("-inf")),
        ("step_importance", float("nan")),
    ],
)
def test_edge_logprob_record_rejects_nonfinite_scores(
    field_name: str,
    bad_value: float,
) -> None:
    assert math.isfinite(_edge().step_importance)

    with pytest.raises(ValueError):
        replace(_edge(), **{field_name: bad_value})


def test_edge_logprob_record_rejects_inconsistent_importance() -> None:
    assert _edge().step_importance == pytest.approx(0.3)

    with pytest.raises(ValueError):
        replace(_edge(), step_importance=0.31)


def test_valid_trajectory_residual_reconstructs_delta() -> None:
    residual = _first_residual()
    reconstructed = (
        residual.log_z
        + residual.sum_forward
        - residual.temperature_beta * residual.log_shifted_reward
        - residual.sum_backward
    )

    assert residual.delta == pytest.approx(reconstructed)


@pytest.mark.parametrize(
    "field_name",
    [
        "log_z",
        "sum_forward",
        "sum_backward",
        "log_shifted_reward",
        "raw_reward",
        "temperature_beta",
        "delta",
    ],
)
def test_trajectory_residual_rejects_every_nonfinite_float(field_name: str) -> None:
    assert math.isfinite(_first_residual().delta)

    with pytest.raises(ValueError):
        replace(_first_residual(), **{field_name: float("nan")})


@pytest.mark.parametrize("bad_reward", [-0.01, 1.01])
def test_trajectory_residual_rejects_raw_reward_outside_unit_interval(
    bad_reward: float,
) -> None:
    assert 0.0 <= _first_residual().raw_reward <= 1.0

    with pytest.raises(ValueError):
        replace(_first_residual(), raw_reward=bad_reward)


@pytest.mark.parametrize("bad_beta", [0.0, -0.1])
def test_trajectory_residual_rejects_nonpositive_temperature(bad_beta: float) -> None:
    assert _first_residual().temperature_beta > 0.0

    with pytest.raises(ValueError):
        replace(_first_residual(), temperature_beta=bad_beta)


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_trajectory_residual_rejects_nonpositive_horizon(bad_horizon: int) -> None:
    assert _first_residual().horizon >= 1

    with pytest.raises(ValueError):
        replace(_first_residual(), horizon=bad_horizon)


def test_trajectory_residual_rejects_empty_identity() -> None:
    assert _first_residual().trajectory_id

    with pytest.raises(ValueError):
        replace(_first_residual(), trajectory_id="")


def test_trajectory_residual_rejects_inconsistent_delta() -> None:
    assert _first_residual().delta == pytest.approx(0.05)

    with pytest.raises(ValueError):
        replace(_first_residual(), delta=0.051)


def test_trajectory_residual_rejects_nonfinite_recomputed_delta() -> None:
    with pytest.raises(ValueError):
        TrajectoryResidual(
            trajectory_id="trajectory-overflow",
            log_z=1e308,
            sum_forward=1e308,
            sum_backward=0.0,
            log_shifted_reward=1e308,
            raw_reward=1.0,
            temperature_beta=1e308,
            delta=0.0,
            horizon=1,
        )


def test_valid_batch_stats_are_reconstructible_from_members() -> None:
    batch = _batch()

    expected_loss = math.fsum(
        (member.delta / member.horizon) ** 2 for member in batch.residuals
    ) / len(batch.residuals)
    expected_reward = math.fsum(member.raw_reward for member in batch.residuals) / len(
        batch.residuals
    )
    assert batch.batch_loss == pytest.approx(expected_loss)
    assert batch.mean_reward == pytest.approx(expected_reward)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("batch_id", ""),
        ("library_version", " "),
        ("created_at", ""),
        ("created_at", "2026-07-19"),
    ],
)
def test_batch_stats_rejects_invalid_audit_identity_fields(
    field_name: str,
    bad_value: str,
) -> None:
    assert _batch().batch_id

    with pytest.raises(ValueError):
        replace(_batch(), **{field_name: bad_value})


def test_batch_stats_rejects_negative_optimizer_step() -> None:
    assert _batch().optimizer_step >= 0

    with pytest.raises(ValueError):
        replace(_batch(), optimizer_step=-1)


def test_batch_stats_rejects_empty_membership() -> None:
    assert _batch().residuals

    with pytest.raises(ValueError):
        TTBBatchStats(
            batch_id="batch-empty",
            optimizer_step=12,
            library_version="library@3",
            residuals=(),
            batch_loss=0.0,
            mean_reward=0.0,
            created_at="2026-07-19T09:30:00Z",
        )


def test_batch_stats_requires_a_tuple_of_residual_records() -> None:
    with pytest.raises(ValueError):
        replace(_batch(), residuals=[_first_residual()])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(_batch(), residuals=("not-a-residual",))  # type: ignore[arg-type]


def test_batch_stats_rejects_duplicate_trajectory_members() -> None:
    first = _first_residual()
    assert _batch((first, _second_residual())).residuals

    with pytest.raises(ValueError):
        _batch((first, first))


@pytest.mark.parametrize("field_name", ["batch_loss", "mean_reward"])
def test_batch_stats_rejects_nonfinite_summary(field_name: str) -> None:
    assert math.isfinite(_batch().batch_loss)

    with pytest.raises(ValueError):
        replace(_batch(), **{field_name: float("nan")})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("batch_loss", 0.0),
        ("mean_reward", 0.0),
    ],
)
def test_batch_stats_rejects_summary_not_reconstructed_from_members(
    field_name: str,
    bad_value: float,
) -> None:
    assert _batch().batch_loss > 0.0
    assert _batch().mean_reward > 0.0

    with pytest.raises(ValueError):
        replace(_batch(), **{field_name: bad_value})


def test_batch_stats_rejects_a_finite_residual_whose_squared_loss_overflows() -> None:
    huge = TrajectoryResidual(
        trajectory_id="trajectory-huge",
        log_z=1e308,
        sum_forward=0.0,
        sum_backward=0.0,
        log_shifted_reward=0.0,
        raw_reward=1.0,
        temperature_beta=1.0,
        delta=1e308,
        horizon=1,
    )

    with pytest.raises(ValueError):
        TTBBatchStats(
            batch_id="batch-huge",
            optimizer_step=1,
            library_version="library@3",
            residuals=(huge,),
            batch_loss=0.0,
            mean_reward=1.0,
            created_at="2026-07-19T09:30:00Z",
        )


@pytest.mark.parametrize(
    ("record", "record_type"),
    [
        (_edge(), EdgeLogprobRecord),
        (_first_residual(), TrajectoryResidual),
        (_batch(), TTBBatchStats),
    ],
)
def test_training_contract_canonical_round_trip_preserves_hash(
    record: EdgeLogprobRecord | TrajectoryResidual | TTBBatchStats,
    record_type: type[EdgeLogprobRecord] | type[TrajectoryResidual] | type[TTBBatchStats],
) -> None:
    payload = canonical_json(record.to_value())
    rebuilt = record_type.from_value(parse_canonical_json(payload))

    assert rebuilt == record
    assert rebuilt.content_hash == record.content_hash


def test_batch_content_hash_excludes_audit_timestamp() -> None:
    first = _batch(created_at="2026-07-19T09:30:00Z")
    later = _batch(created_at="2026-07-19T09:31:00Z")

    assert first != later
    assert first.content_hash == later.content_hash


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (EdgeLogprobRecord.from_value, {"trajectory_id": "trajectory-001"}),
        (TrajectoryResidual.from_value, {"trajectory_id": "trajectory-001"}),
        (TTBBatchStats.from_value, {"batch_id": "batch-001"}),
        (
            EdgeLogprobRecord.from_value,
            {**_edge().to_value(), "unexpected": "field"},
        ),
        (
            EdgeLogprobRecord.from_value,
            {**_edge().to_value(), "forward_logprob_per_token": 10**400},
        ),
    ],
)
def test_training_from_value_rejects_incompatible_field_sets(
    factory: object,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        factory(value)  # type: ignore[operator]
