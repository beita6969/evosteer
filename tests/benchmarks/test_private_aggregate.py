from __future__ import annotations

import pytest
from skillev_private.benchmarks import PrivateAggregateAccumulatorFactory

from skillev.contracts import canonical_json, stable_hash
from skillev.experiments import (
    FROZEN_REWARD_BUCKET_SCHEME,
    AblationArm,
    AggregatePurpose,
    Benchmark,
    SamplingPeriod,
)


def _sink():
    return PrivateAggregateAccumulatorFactory().create(
        protocol_hash=stable_hash({"protocol": "fixture"}),
        arm=AblationArm.FULL,
        benchmark=Benchmark.WEBSHOP,
        purpose=AggregatePurpose.IID_EVALUATION,
    )


def test_private_aggregate_returns_only_sufficient_moments() -> None:
    sink = _sink()
    private_canary = "PRIVATE-ITEM-IDENTITY-CANARY"
    sink.record_private(item_id=private_canary, reward=0.25, success=False)
    sink.record_private(item_id="private-item-2", reward=1.0, success=True)

    outcome = sink.aggregate()

    assert outcome.sample_count == 2
    assert outcome.reward_sum == 1.25
    assert outcome.reward_squared_sum == 1.0625
    assert outcome.success_count == 1
    assert sink.unique_item_count == 2
    assert private_canary not in canonical_json(
        {
            "benchmark_id": outcome.benchmark_id,
            "reward_squared_sum": outcome.reward_squared_sum,
            "reward_sum": outcome.reward_sum,
            "sample_count": outcome.sample_count,
            "success_count": outcome.success_count,
        }
    )

    buckets = sink.reward_bucket_observations(
        period=SamplingPeriod.BEFORE_TRAINING,
        scheme=FROZEN_REWARD_BUCKET_SCHEME,
    )
    assert tuple(item.sample_count for item in buckets) == (1, 1)
    assert tuple(item.reward_sum for item in buckets) == (0.25, 1.0)
    assert private_canary not in canonical_json(
        [
            {
                "bucket_id": item.bucket_id,
                "period": item.period.value,
                "reward_squared_sum": item.reward_squared_sum,
                "reward_sum": item.reward_sum,
                "sample_count": item.sample_count,
            }
            for item in buckets
        ]
    )


@pytest.mark.parametrize("reward", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_private_aggregate_rejects_invalid_rewards(reward: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _sink().record_private(item_id="private-item", reward=reward, success=False)  # type: ignore[arg-type]


def test_private_aggregate_rejects_duplicate_item_identity() -> None:
    sink = _sink()
    sink.record_private(item_id="private-item", reward=0.0, success=False)
    with pytest.raises(ValueError):
        sink.record_private(item_id="private-item", reward=1.0, success=True)
