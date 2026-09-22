from skillev.evaluation.step0_receipts import (
    PairedMetricComparison,
    StrictBackboneImprovementTarget,
    mcnemar_exact_two_sided,
    paired_bootstrap_interval,
    paired_outcome_counts,
)


def test_non_regression_gate_is_strictly_less_than_seven_points() -> None:
    assert PairedMetricComparison("f1", 75.0, 68.01).passes_non_regression
    assert not PairedMetricComparison("f1", 75.0, 68.0).passes_non_regression


def test_improvement_preserves_signed_paired_delta() -> None:
    comparison = PairedMetricComparison("accuracy", 72.0, 80.0)
    assert comparison.delta_pp == 8.0
    assert comparison.passes_non_regression


def test_step0_promotion_requires_strict_backbone_improvement() -> None:
    tied = StrictBackboneImprovementTarget("accuracy", 86.67, 86.67)
    improved = StrictBackboneImprovementTarget("accuracy", 86.67, 86.68)

    assert not tied.passes
    assert improved.passes


def test_paired_bootstrap_operates_on_within_record_differences() -> None:
    interval = paired_bootstrap_interval(
        (0.0, 0.5, 1.0, 0.5),
        (0.5, 0.5, 1.0, 1.0),
        resamples=500,
    )

    assert interval.delta == 0.25
    assert interval.lower <= interval.delta <= interval.upper


def test_mcnemar_uses_only_discordant_pairs() -> None:
    counts = paired_outcome_counts(
        (True, True, True, False, False, False),
        (True, False, False, True, True, False),
    )

    assert mcnemar_exact_two_sided(counts) == 1.0
