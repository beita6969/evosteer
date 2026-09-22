from decimal import Decimal

import pytest

from skillev.evaluation.direct_baseline.aggregation import (
    AggregatedMetric,
    BenchmarkCoverage,
    compute_published_aggregate,
)
from skillev.evaluation.direct_baseline.config import (
    AggregateComponent,
    AggregateMetricSpec,
    DirectBenchmark,
)


def test_published_aggregate_requires_all_formal_components() -> None:
    spec = AggregateMetricSpec(
        "avg",
        Decimal("50"),
        (
            AggregateComponent(DirectBenchmark.HOTPOT_QA, "em"),
            AggregateComponent(DirectBenchmark.TRIVIA_QA, "em"),
        ),
    )
    complete = AggregatedMetric("em", Decimal("1"), Decimal("40"), Decimal("40"))
    incomplete = AggregatedMetric("em", Decimal("1"), Decimal("60"), None)
    result = compute_published_aggregate(
        spec,
        {
            (DirectBenchmark.HOTPOT_QA, "em"): complete,
            (DirectBenchmark.TRIVIA_QA, "em"): incomplete,
        },
    )
    assert result.diagnostic_observed_percent == Decimal("50.0")
    assert result.formal_observed_percent is None


def test_coverage_requires_candidate_and_infrastructure_conservation() -> None:
    with pytest.raises(ValueError, match="must equal final records"):
        BenchmarkCoverage(
            planned_count=128,
            final_record_count=128,
            candidate_response_count=126,
            definitive_verdict_count=126,
            generation_infrastructure_failures=1,
        )
