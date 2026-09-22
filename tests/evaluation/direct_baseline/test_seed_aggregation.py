from decimal import Decimal

import pytest

from skillev.evaluation.direct_baseline.aggregation import aggregate_seed_runs
from skillev.evaluation.direct_baseline.config import SeedAggregationMode, SeedAggregationSpec


def test_single_and_unknown_seed_semantics_remain_distinct() -> None:
    single = aggregate_seed_runs(
        SeedAggregationSpec(SeedAggregationMode.SINGLE_RUN, (42,)),
        {42: Decimal("50")},
    )
    unknown = aggregate_seed_runs(
        SeedAggregationSpec(SeedAggregationMode.UNKNOWN, (42,)),
        {42: Decimal("50")},
    )
    assert single.formal_percent == Decimal("50")
    assert unknown.diagnostic_percent == Decimal("50")
    assert unknown.formal_percent is None


def test_multi_seed_mean_and_sample_dispersion_use_all_declared_seeds() -> None:
    result = aggregate_seed_runs(
        SeedAggregationSpec(SeedAggregationMode.MEAN_OVER_RUNS, (1, 2), "sample-std"),
        {1: Decimal("40"), 2: Decimal("60")},
    )
    assert result.formal_percent == Decimal("50")
    assert result.dispersion_percent is not None
    with pytest.raises(ValueError):
        aggregate_seed_runs(
            SeedAggregationSpec(SeedAggregationMode.MEAN_OVER_RUNS, (1, 2), "sample-std"),
            {1: Decimal("40")},
        )
