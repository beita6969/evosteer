import pytest

from skillev.evaluation.direct_baseline.aggregation import (
    BenchmarkCoverage,
    ResultAvailability,
)


def test_coverage_distinguishes_complete_incomplete_and_unavailable() -> None:
    assert BenchmarkCoverage(128, 128, 128, 128).availability is ResultAvailability.COMPLETE
    assert BenchmarkCoverage(128, 128, 128, 126).availability is ResultAvailability.INCOMPLETE
    assert BenchmarkCoverage(128, 0, 0, 0).availability is ResultAvailability.UNAVAILABLE


def test_candidate_failure_is_a_complete_definitive_record() -> None:
    coverage = BenchmarkCoverage(128, 128, 128, 128)
    assert coverage.availability is ResultAvailability.COMPLETE


@pytest.mark.parametrize(
    "coverage",
    [
        (128, 128, 127, 127, 0, 0, 0),
        (128, 128, 128, 127, 0, 2, 0),
        (128, 128, 0, 0, 64, 0, 65),
    ],
)
def test_coverage_rejects_impossible_conservation(coverage: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        BenchmarkCoverage(*coverage)
