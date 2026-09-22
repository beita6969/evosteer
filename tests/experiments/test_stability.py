from __future__ import annotations

import pytest

from skillev.experiments.stability import LibraryCycleObservation, multi_cycle_stability


def _cycle(
    index: int,
    *skills: tuple[str, str],
) -> LibraryCycleObservation:
    return LibraryCycleObservation(index, f"library-v{index}", tuple(sorted(skills)))


def test_multi_cycle_stability_measures_growth_churn_and_reappearance() -> None:
    summary = multi_cycle_stability(
        (
            _cycle(0, ("a", "hash-a0"), ("b", "hash-b")),
            _cycle(1, ("a", "hash-a1"), ("c", "hash-c")),
            _cycle(2, ("a", "hash-a1"), ("b", "hash-b"), ("d", "hash-d")),
            _cycle(3, ("a", "hash-a1"), ("b", "hash-b"), ("d", "hash-d")),
        )
    )

    assert summary.library_change_count == 3
    assert summary.initial_active_count == 2
    assert summary.final_active_count == 3
    assert summary.maximum_active_count == 3
    assert summary.total_additions == 3
    assert summary.total_removals == 2
    assert summary.total_content_replacements == 1
    assert summary.reappeared_skill_count == 1
    assert summary.duplicate_active_content_count == 0
    assert summary.maximum_growth_factor == 1.5
    assert tuple(metric.churn_rate for metric in summary.transitions) == (1.0, 0.75, 0.0)


def test_stability_counts_duplicate_active_content_without_task_data() -> None:
    summary = multi_cycle_stability(
        (
            _cycle(0, ("a", "same")),
            _cycle(1, ("a", "same"), ("b", "same")),
        )
    )
    assert summary.duplicate_active_content_count == 1


@pytest.mark.parametrize(
    "observations",
    [
        (),
        (_cycle(0, ("a", "h")),),
        (_cycle(0, ("a", "h")), LibraryCycleObservation(2, "v2", (("a", "h"),))),
        (_cycle(0, ("a", "h")), LibraryCycleObservation(1, "library-v0", (("a", "h"),))),
    ],
)
def test_stability_rejects_incomplete_or_noncommitted_sequences(
    observations: tuple[LibraryCycleObservation, ...],
) -> None:
    with pytest.raises(ValueError):
        multi_cycle_stability(observations)
