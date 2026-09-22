"""Answer-free, coverage-aware reporting for direct-reference runs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .aggregation import BenchmarkCoverage, ResultAvailability, coverage_satisfies
from .config import (
    BenchmarkComparability,
    ComparabilityEvidence,
    DirectBenchmark,
    DirectParityPolicy,
)


class EvaluationScope(StrEnum):
    IID = "iid"
    OOD = "ood"
    ALL = "all"


class GateStatus(StrEnum):
    GO = "go"
    NO_GO = "no-go"
    NOT_EVALUATED = "not-evaluated"


@dataclass(frozen=True, slots=True)
class MetricResult:
    metric_id: str
    reference_percent: Decimal
    diagnostic_observed_percent: Decimal | None
    formal_observed_percent: Decimal | None

    @property
    def gap_pp(self) -> Decimal | None:
        if self.formal_observed_percent is None:
            return None
        return abs(self.formal_observed_percent - self.reference_percent)

    def numeric_status(self, *, policy: DirectParityPolicy, coverage: BenchmarkCoverage) -> str:
        if coverage.availability is ResultAvailability.UNAVAILABLE:
            return "UNAVAILABLE"
        if coverage.availability is ResultAvailability.INCOMPLETE:
            return "INCOMPLETE"
        gap = self.gap_pp
        if gap is None:
            return "UNAVAILABLE"
        if not coverage_satisfies(coverage, policy):
            return "INCOMPLETE"
        return "PASS" if gap < policy.max_gap_pp_exclusive else "FAIL"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    benchmark: DirectBenchmark
    comparability: BenchmarkComparability
    evidence: ComparabilityEvidence
    population: str
    prompt_profile: str
    decoding_profile: str
    parser_profile: str
    scorer_profile: str
    seed_aggregation: str
    coverage: BenchmarkCoverage
    metrics: tuple[MetricResult, ...]
    submission_rate_percent: Decimal
    scorer_reach_rate_percent: Decimal | None
    valid_native_action_rate_percent: Decimal | None = None

    def numeric_pass(self, policy: DirectParityPolicy) -> bool:
        return bool(self.metrics) and all(
            metric.numeric_status(policy=policy, coverage=self.coverage) == "PASS"
            for metric in self.metrics
        )

    def scientific_pass(self, policy: DirectParityPolicy) -> bool:
        return self.numeric_pass(policy) and self.evidence.is_exact

    def to_public_dict(self, policy: DirectParityPolicy) -> dict[str, object]:
        coverage = self.coverage
        return {
            "benchmark": self.benchmark.value,
            "comparability": self.comparability.value,
            "comparability_evidence": {
                "population": self.evidence.population.value,
                "prompt": self.evidence.prompt.value,
                "decoding": self.evidence.decoding.value,
                "seed_aggregation": self.evidence.seed_aggregation.value,
                "scorer": self.evidence.scorer.value,
                "environment": self.evidence.environment.value,
            },
            "population": self.population,
            "prompt_profile": self.prompt_profile,
            "decoding_profile": self.decoding_profile,
            "parser_profile": self.parser_profile,
            "scorer_profile": self.scorer_profile,
            "seed_aggregation": self.seed_aggregation,
            "availability": coverage.availability.value,
            "coverage": {
                "planned_count": coverage.planned_count,
                "final_record_count": coverage.final_record_count,
                "candidate_response_count": coverage.candidate_response_count,
                "definitive_verdict_count": coverage.definitive_verdict_count,
                "generation_infrastructure_failures": coverage.generation_infrastructure_failures,
                "scorer_infrastructure_failures": coverage.scorer_infrastructure_failures,
                "environment_infrastructure_failures": coverage.environment_infrastructure_failures,
            },
            "metrics": [
                {
                    "metric": metric.metric_id,
                    "reference_percent": str(metric.reference_percent),
                    "diagnostic_observed_percent": (
                        str(metric.diagnostic_observed_percent)
                        if metric.diagnostic_observed_percent is not None
                        else None
                    ),
                    "formal_observed_percent": (
                        str(metric.formal_observed_percent)
                        if metric.formal_observed_percent is not None
                        else None
                    ),
                    "absolute_gap_pp": str(metric.gap_pp) if metric.gap_pp is not None else None,
                    "numeric_status": metric.numeric_status(policy=policy, coverage=coverage),
                }
                for metric in self.metrics
            ],
            "submission_rate_percent": str(self.submission_rate_percent),
            "scorer_reach_rate_percent": (
                str(self.scorer_reach_rate_percent)
                if self.scorer_reach_rate_percent is not None
                else None
            ),
            "valid_native_action_rate_percent": (
                str(self.valid_native_action_rate_percent)
                if self.valid_native_action_rate_percent is not None
                else None
            ),
            "numeric_pass": self.numeric_pass(policy),
            "scientific_pass": self.scientific_pass(policy),
        }


@dataclass(frozen=True, slots=True)
class DirectParityGateResult:
    scope: EvaluationScope
    benchmark_results: tuple[BenchmarkResult, ...]
    aggregate_results: tuple[MetricResult, ...]
    engineering_check_passed: bool
    protocol_semantics_known: bool
    policy: DirectParityPolicy
    expected_all_benchmarks: frozenset[DirectBenchmark]

    @property
    def scope_numeric_pass(self) -> bool:
        return bool(self.benchmark_results) and all(
            result.numeric_pass(self.policy) for result in self.benchmark_results
        )

    @property
    def scope_scientific_pass(self) -> bool:
        return (
            self.scope_numeric_pass
            and self.protocol_semantics_known
            and all(result.scientific_pass(self.policy) for result in self.benchmark_results)
        )

    @property
    def formal_training_status(self) -> GateStatus:
        if self.scope is not EvaluationScope.ALL:
            return GateStatus.NOT_EVALUATED
        actual = frozenset(result.benchmark for result in self.benchmark_results)
        if not self.expected_all_benchmarks or actual != self.expected_all_benchmarks:
            return GateStatus.NOT_EVALUATED
        return (
            GateStatus.GO
            if self.scope_scientific_pass and self.engineering_check_passed
            else GateStatus.NO_GO
        )
