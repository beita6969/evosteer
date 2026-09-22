"""Declared quality identities work through the same persisted seed coordinate."""

from dataclasses import replace

import pytest

from skillev.contracts import ScientificSamplingCoordinate


def test_explicit_sampling_plan_and_source_order_round_trip():
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash="source-holdout-seed0@1",
        schedule_purpose="diagnostic-collect-only",
        ordered_sequence_hash="fixed-quality-source-order@1",
        sequence_position=0,
        task_id="synthetic-question",
        optimizer_step_or_anchor_ordinal=0,
    )
    assert ScientificSamplingCoordinate.from_value(coordinate.to_value()) == coordinate
    assert replace(coordinate, sequence_position=1) != coordinate
    assert replace(coordinate, ordered_sequence_hash="different-order@1") != coordinate


@pytest.mark.parametrize("field", ["sampling_schedule_hash", "ordered_sequence_hash"])
@pytest.mark.parametrize("invalid", ["", "   ", None, 1])
def test_sampling_identifiers_cannot_be_missing(field, invalid):
    values = {
        "sampling_schedule_hash": "source-holdout-seed0@1",
        "schedule_purpose": "diagnostic-collect-only",
        "ordered_sequence_hash": "fixed-quality-source-order@1",
        "sequence_position": 0,
        "task_id": "synthetic-question",
        "optimizer_step_or_anchor_ordinal": 0,
    }
    values[field] = invalid
    with pytest.raises(ValueError):
        ScientificSamplingCoordinate(**values)
