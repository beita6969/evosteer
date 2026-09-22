"""Coverage-aware aggregation for direct-reference task verdicts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .config import (
    AggregateMetricSpec,
    DirectBenchmark,
    DirectParityPolicy,
    MetricContract,
    MetricScale,
    SeedAggregationMode,
    SeedAggregationSpec,
)


class ResultAvailability(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"


class CandidateOutcomeKind(StrEnum):
    SCORED = "scored"
    PARSE_FAILURE = "parse-failure"
    INVALID_ACTION = "invalid-action"
    GENERATION_INFRASTRUCTURE = "generation-infrastructure"
    SCORER_INFRASTRUCTURE = "scorer-infrastructure"
    ENVIRONMENT_INFRASTRUCTURE = "environment-infrastructure"


@dataclass(frozen=True, slots=True)
class BenchmarkCoverage:
    planned_count: int
    final_record_count: int
    candidate_response_count: int
    definitive_verdict_count: int
    generation_infrastructure_failures: int = 0
    scorer_infrastructure_failures: int = 0
    environment_infrastructure_failures: int = 0

    def __post_init__(self) -> None:
        values = (
            self.planned_count,
            self.final_record_count,
            self.candidate_response_count,
            self.definitive_verdict_count,
            self.generation_infrastructure_failures,
            self.scorer_infrastructure_failures,
            self.environment_infrastructure_failures,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("coverage counts must be non-negative integers")
        if self.planned_count <= 0:
            raise ValueError("planned_count must be positive")
        if self.final_record_count > self.planned_count:
            raise ValueError("final records cannot exceed planned population")
        if self.candidate_response_count > self.final_record_count:
            raise ValueError("candidate responses cannot exceed final records")
        if self.definitive_verdict_count > self.final_record_count:
            raise ValueError("verdicts cannot exceed final records")
        if (
            self.candidate_response_count
            + self.generation_infrastructure_failures
            + self.environment_infrastructure_failures
            != self.final_record_count
        ):
            raise ValueError(
                "candidate responses plus pre-candidate infrastructure failures "
                "must equal final records"
            )
        if (
            self.definitive_verdict_count + self.scorer_infrastructure_failures
            > self.candidate_response_count
        ):
            raise ValueError("verdicts and scorer failures exceed candidate responses")
        if self.infrastructure_failures > self.planned_count:
            raise ValueError("infrastructure failures exceed planned population")

    @property
    def infrastructure_failures(self) -> int:
        return (
            self.generation_infrastructure_failures
            + self.scorer_infrastructure_failures
            + self.environment_infrastructure_failures
        )

    @property
    def availability(self) -> ResultAvailability:
        if self.final_record_count == 0:
            return ResultAvailability.UNAVAILABLE
        if (
            self.final_record_count == self.planned_count
            and self.definitive_verdict_count == self.planned_count
            and self.infrastructure_failures == 0
        ):
            return ResultAvailability.COMPLETE
        return ResultAvailability.INCOMPLETE


@dataclass(frozen=True, slots=True)
class TaskMetricVerdict:
    task_id: str
    benchmark: DirectBenchmark
    kind: CandidateOutcomeKind
    metrics: Mapping[str, float]
    definitive: bool

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task verdict requires task_id")
        definitive_kinds = {
            CandidateOutcomeKind.SCORED,
            CandidateOutcomeKind.PARSE_FAILURE,
            CandidateOutcomeKind.INVALID_ACTION,
        }
        if self.definitive != (self.kind in definitive_kinds):
            raise ValueError("candidate outcome kind and definitive status disagree")
        if self.definitive:
            if not self.metrics:
                raise ValueError("definitive verdict requires metric values")
            if any(
                not math.isfinite(value) or not 0 <= value <= 100 for value in self.metrics.values()
            ):
                raise ValueError("task metrics must be finite non-negative percentage values")
        elif self.metrics:
            raise ValueError("non-definitive verdict cannot carry formal metrics")


@dataclass(frozen=True, slots=True)
class AggregatedMetric:
    metric_id: str
    reference_percent: Decimal
    diagnostic_observed_percent: Decimal | None
    formal_observed_percent: Decimal | None

    @property
    def formal_gap_pp(self) -> Decimal | None:
        if self.formal_observed_percent is None:
            return None
        return abs(self.formal_observed_percent - self.reference_percent)

    def numeric_passes(self, *, policy: DirectParityPolicy, coverage: BenchmarkCoverage) -> bool:
        return (
            coverage_satisfies(coverage, policy)
            and self.formal_gap_pp is not None
            and self.formal_gap_pp < policy.max_gap_pp_exclusive
        )


@dataclass(frozen=True, slots=True)
class SeedAggregatedValue:
    diagnostic_percent: Decimal
    formal_percent: Decimal | None
    dispersion_percent: Decimal | None


def aggregate_seed_runs(
    spec: SeedAggregationSpec,
    values: Mapping[int, Decimal],
) -> SeedAggregatedValue:
    """Aggregate only declared seeds; unknown paper semantics cannot be formal."""

    if set(values) != set(spec.seeds):
        raise ValueError("seed results must exactly match the declared seed set")
    ordered = tuple(values[seed] for seed in spec.seeds)
    mean = sum(ordered, Decimal("0")) / Decimal(len(ordered))
    if spec.mode is SeedAggregationMode.UNKNOWN:
        return SeedAggregatedValue(mean, None, None)
    if spec.mode is SeedAggregationMode.SINGLE_RUN:
        return SeedAggregatedValue(mean, mean, None)
    divisor = len(ordered) - 1 if spec.dispersion == "sample-std" else len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / Decimal(divisor)
    return SeedAggregatedValue(mean, mean, variance.sqrt())


def _metric_to_percent(value: float, scale: MetricScale) -> float:
    if not math.isfinite(value):
        raise ValueError("metric value must be finite")
    if scale is MetricScale.UNIT_INTERVAL:
        if not 0.0 <= value <= 1.0:
            raise ValueError("unit-interval metric must lie in [0, 1]")
        return value * 100.0
    if not 0.0 <= value <= 100.0:
        raise ValueError("percentage metric must lie in [0, 100]")
    return value


def _percent(values: list[float], denominator: int) -> Decimal:
    if denominator <= 0:
        raise ValueError("metric denominator must be positive")
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("metric total must be finite")
    return Decimal(str(total / denominator))


def coverage_satisfies(coverage: BenchmarkCoverage, policy: DirectParityPolicy) -> bool:
    if (
        policy.require_complete_population
        and coverage.availability is not ResultAvailability.COMPLETE
    ):
        return False
    return not (
        policy.require_zero_infrastructure_failures and coverage.infrastructure_failures != 0
    )


def aggregate_metric(
    contract: MetricContract,
    verdicts: tuple[TaskMetricVerdict, ...],
    coverage: BenchmarkCoverage,
) -> AggregatedMetric:
    definitive = tuple(verdict for verdict in verdicts if verdict.definitive)
    values = [
        _metric_to_percent(verdict.metrics[contract.metric_id], contract.source_scale)
        for verdict in definitive
    ]
    diagnostic = _percent(values, len(definitive)) if definitive else None
    formal = (
        _percent(values, coverage.planned_count)
        if coverage.availability is ResultAvailability.COMPLETE
        else None
    )
    return AggregatedMetric(
        metric_id=contract.metric_id,
        reference_percent=contract.reference_percent,
        diagnostic_observed_percent=diagnostic,
        formal_observed_percent=formal,
    )


def compute_published_aggregate(
    spec: AggregateMetricSpec,
    metrics: Mapping[tuple[DirectBenchmark, str], AggregatedMetric],
) -> AggregatedMetric:
    """Compute an equal-weight table row only when every component is present."""

    diagnostic_values: list[float] = []
    formal_values: list[float] = []
    for component in spec.components:
        metric = metrics[(component.benchmark, component.metric_id)]
        if metric.diagnostic_observed_percent is not None:
            diagnostic_values.append(float(metric.diagnostic_observed_percent))
        if metric.formal_observed_percent is not None:
            formal_values.append(float(metric.formal_observed_percent))
    diagnostic = (
        Decimal(str(math.fsum(diagnostic_values) / len(spec.components)))
        if len(diagnostic_values) == len(spec.components)
        else None
    )
    formal = (
        Decimal(str(math.fsum(formal_values) / len(spec.components)))
        if len(formal_values) == len(spec.components)
        else None
    )
    return AggregatedMetric(spec.aggregate_id, spec.reference_percent, diagnostic, formal)
