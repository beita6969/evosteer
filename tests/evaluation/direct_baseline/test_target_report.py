from decimal import Decimal

from skillev.evaluation.direct_baseline.aggregation import BenchmarkCoverage
from skillev.evaluation.direct_baseline.config import (
    BenchmarkComparability,
    ComparabilityEvidence,
    DirectBenchmark,
    DirectParityPolicy,
    EvidenceStatus,
    ParityMetric,
)
from skillev.evaluation.direct_baseline.reporting import (
    BenchmarkResult,
    DirectParityGateResult,
    EvaluationScope,
    GateStatus,
    MetricResult,
)


def _policy() -> DirectParityPolicy:
    return DirectParityPolicy(
        ParityMetric.ABSOLUTE_PERCENTAGE_POINT_GAP,
        Decimal("7.0"),
        True,
        True,
        True,
    )


def _coverage(count: int = 128) -> BenchmarkCoverage:
    return BenchmarkCoverage(count, count, count, count)


def test_exact_metric_uses_strict_seven_point_gate() -> None:
    passing = MetricResult("f1", Decimal("75.70"), Decimal("68.700001"), Decimal("68.700001"))
    boundary = MetricResult("f1", Decimal("75.70"), Decimal("68.70"), Decimal("68.70"))
    assert passing.numeric_status(policy=_policy(), coverage=_coverage()) == "PASS"
    assert boundary.numeric_status(policy=_policy(), coverage=_coverage()) == "FAIL"


def test_incomplete_result_cannot_be_numeric_pass() -> None:
    metric = MetricResult("f1", Decimal("75.70"), Decimal("75.70"), None)
    coverage = BenchmarkCoverage(128, 128, 128, 126, scorer_infrastructure_failures=2)
    assert metric.numeric_status(policy=_policy(), coverage=coverage) == "INCOMPLETE"


def _benchmark_result(benchmark: DirectBenchmark) -> BenchmarkResult:
    evidence = ComparabilityEvidence(
        *(EvidenceStatus.EXACT for _ in range(5)), EvidenceStatus.EXACT
    )
    return BenchmarkResult(
        benchmark=benchmark,
        comparability=BenchmarkComparability.EXACT_PAPER,
        evidence=evidence,
        population="population",
        prompt_profile="prompt",
        decoding_profile="decoding",
        parser_profile="parser",
        scorer_profile="scorer",
        seed_aggregation="single-run",
        coverage=BenchmarkCoverage(1, 1, 1, 1),
        metrics=(MetricResult("metric", Decimal("50"), Decimal("50"), Decimal("50")),),
        submission_rate_percent=Decimal("100"),
        scorer_reach_rate_percent=Decimal("100"),
    )


def _gate(scope: EvaluationScope, count: int, engineering: bool) -> DirectParityGateResult:
    expected = frozenset(DirectBenchmark)
    results = tuple(_benchmark_result(item) for item in tuple(DirectBenchmark)[:count])
    return DirectParityGateResult(
        scope=scope,
        benchmark_results=results,
        aggregate_results=(),
        engineering_check_passed=engineering,
        protocol_semantics_known=True,
        policy=_policy(),
        expected_all_benchmarks=expected,
    )


def test_role_specific_and_incomplete_reports_cannot_emit_formal_go() -> None:
    complete = len(tuple(DirectBenchmark))
    assert (
        _gate(EvaluationScope.IID, complete, True).formal_training_status
        is GateStatus.NOT_EVALUATED
    )
    assert (
        _gate(EvaluationScope.OOD, complete, True).formal_training_status
        is GateStatus.NOT_EVALUATED
    )
    assert (
        _gate(EvaluationScope.ALL, complete - 1, True).formal_training_status
        is GateStatus.NOT_EVALUATED
    )


def test_all_scope_requires_engineering_pass_for_go() -> None:
    complete = len(tuple(DirectBenchmark))
    assert _gate(EvaluationScope.ALL, complete, False).formal_training_status is GateStatus.NO_GO
    assert _gate(EvaluationScope.ALL, complete, True).formal_training_status is GateStatus.GO
