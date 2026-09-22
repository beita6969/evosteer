"""Behavioral contracts for answer-free frozen training progress counts."""

from __future__ import annotations

import pytest

from skillev.experiments import TrainingEvolutionCounts


def test_training_evolution_counts_round_trip_and_order_constraints() -> None:
    counts = TrainingEvolutionCounts(
        training_step_count=9,
        phase_count=2,
        cycle_count=2,
        action_count=3,
    )

    assert TrainingEvolutionCounts.from_value(counts.to_value()) == counts

    with pytest.raises(ValueError):
        TrainingEvolutionCounts(
            training_step_count=9,
            phase_count=1,
            cycle_count=2,
            action_count=2,
        )
    with pytest.raises(ValueError):
        TrainingEvolutionCounts(
            training_step_count=9,
            phase_count=2,
            cycle_count=2,
            action_count=1,
        )
