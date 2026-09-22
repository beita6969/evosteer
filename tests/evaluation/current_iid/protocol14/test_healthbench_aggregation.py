import pytest

from skillev.evaluation.healthbench_official import (
    HealthBenchTransportError,
    aggregate_healthbench_seed_metrics,
)


def test_healthbench_single_seed_identity_has_no_dispersion() -> None:
    result = aggregate_healthbench_seed_metrics(
        metrics_by_seed={40: {"overall_score": 0.4}},
        terminal_count_by_seed={40: 128},
        infrastructure_failures_by_seed={40: 0},
    )
    metric = result["metrics"]["overall_score"]
    assert result["reducer"] == "identity"
    assert result["dispersion"] is None
    assert metric["mean_percent"] == pytest.approx(40.0)
    assert metric["population_std_percent"] is None
    assert metric["per_seed_percent"] == {"40": 40.0}


def test_healthbench_seed_aggregate_rejects_grader_infrastructure() -> None:
    with pytest.raises(HealthBenchTransportError):
        aggregate_healthbench_seed_metrics(
            metrics_by_seed={40: {"overall_score": 0.4}},
            terminal_count_by_seed={40: 128},
            infrastructure_failures_by_seed={40: 1},
        )
