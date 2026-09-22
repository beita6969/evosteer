"""Private per-item accumulation with an aggregate-only public boundary.

The benchmark orchestrator is intentionally unable to inspect individual
rewards.  This module owns those values long enough to compute the sufficient
aggregate moments, then returns only :class:`BenchmarkAggregateOutcome`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from skillev.contracts import validate_sha256
from skillev.experiments import (
    AblationArm,
    AggregatePurpose,
    Benchmark,
    BenchmarkAggregateOutcome,
    RewardBucketObservation,
    RewardBucketScheme,
    SamplingPeriod,
)


def _reward_bucket_observations(
    values: tuple[float, ...],
    *,
    period: SamplingPeriod,
    scheme: RewardBucketScheme,
) -> tuple[RewardBucketObservation, ...]:
    """Aggregate private scalar rewards without exposing per-item values."""

    values_by_bucket: dict[str, list[float]] = {bucket.bucket_id: [] for bucket in scheme.buckets}
    last_index = len(scheme.buckets) - 1
    for value in values:
        for index, bucket in enumerate(scheme.buckets):
            if bucket.lower <= value < bucket.upper or (
                index == last_index and bucket.lower <= value <= bucket.upper
            ):
                values_by_bucket[bucket.bucket_id].append(value)
                break
        else:
            raise ValueError("private reward does not belong to the registered scheme")
    return tuple(
        RewardBucketObservation(
            period=period,
            bucket_id=bucket.bucket_id,
            sample_count=len(values_by_bucket[bucket.bucket_id]),
            reward_sum=math.fsum(values_by_bucket[bucket.bucket_id]),
            reward_squared_sum=math.fsum(
                value * value for value in values_by_bucket[bucket.bucket_id]
            ),
        )
        for bucket in scheme.buckets
    )


@dataclass(slots=True)
class PrivateAggregateAccumulator:
    """One private accumulator scoped to an exact experiment cell."""

    protocol_hash: str
    arm: AblationArm
    benchmark: Benchmark
    purpose: AggregatePurpose
    _item_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _sample_count: int = field(default=0, init=False, repr=False)
    _reward_sum: float = field(default=0.0, init=False, repr=False)
    _reward_squared_sum: float = field(default=0.0, init=False, repr=False)
    _success_count: int = field(default=0, init=False, repr=False)
    _rewards: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_sha256(self.protocol_hash)
        if not isinstance(self.arm, AblationArm):
            raise TypeError("aggregate arm must be AblationArm")
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("aggregate benchmark must be Benchmark")
        if not isinstance(self.purpose, AggregatePurpose):
            raise TypeError("aggregate purpose must be AggregatePurpose")

    @property
    def benchmark_id(self) -> str:
        return self.benchmark.value

    @property
    def unique_item_count(self) -> int:
        return len(self._item_ids)

    def record_private(self, *, item_id: str, reward: float, success: bool) -> None:
        """Consume exactly one private result without exposing it publicly."""

        if type(item_id) is not str or not item_id.strip():
            raise ValueError("private aggregate item_id must be non-empty text")
        if item_id in self._item_ids:
            raise ValueError("private aggregate received a duplicate item identity")
        if isinstance(reward, bool) or not isinstance(reward, int | float):
            raise TypeError("private aggregate reward must be numeric")
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward) or not 0.0 <= numeric_reward <= 1.0:
            raise ValueError("private aggregate reward must lie in [0, 1]")
        if type(success) is not bool:
            raise TypeError("private aggregate success must be a boolean")

        self._item_ids.add(item_id)
        self._sample_count += 1
        self._reward_sum = math.fsum((self._reward_sum, numeric_reward))
        self._reward_squared_sum = math.fsum(
            (self._reward_squared_sum, numeric_reward * numeric_reward)
        )
        self._success_count += int(success)
        self._rewards.append(numeric_reward)

    def aggregate(self) -> BenchmarkAggregateOutcome:
        """Return the only public representation of accumulated outcomes."""

        return BenchmarkAggregateOutcome(
            benchmark_id=self.benchmark_id,
            sample_count=self._sample_count,
            reward_sum=self._reward_sum,
            reward_squared_sum=self._reward_squared_sum,
            success_count=self._success_count,
        )

    def reward_bucket_observations(
        self,
        *,
        period: SamplingPeriod,
        scheme: RewardBucketScheme,
    ) -> tuple[RewardBucketObservation, ...]:
        """Return bucket moments without exposing any individual reward."""

        return _reward_bucket_observations(
            tuple(self._rewards),
            period=period,
            scheme=scheme,
        )


@dataclass(frozen=True, slots=True)
class PrivateAggregateAccumulatorFactory:
    """Create independent accumulators for orchestrator-requested cells."""

    def create(
        self,
        *,
        protocol_hash: str,
        arm: AblationArm,
        benchmark: Benchmark,
        purpose: AggregatePurpose,
    ) -> PrivateAggregateAccumulator:
        return PrivateAggregateAccumulator(
            protocol_hash=protocol_hash,
            arm=arm,
            benchmark=benchmark,
            purpose=purpose,
        )


__all__ = ["PrivateAggregateAccumulator", "PrivateAggregateAccumulatorFactory"]
