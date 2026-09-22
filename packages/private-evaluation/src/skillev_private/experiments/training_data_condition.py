"""Inline data declarations and actual source order in existing run conditions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from skillev.contracts import JsonValue, normalize_json
from skillev.training.run_condition import EffectiveRunCondition
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord


def inline_data_condition(value: object) -> dict[str, JsonValue] | None:
    """Accept an inline declaration, never a path to be dereferenced."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("data_condition must be an inline JSON object or null")
    normalized = normalize_json(value)
    assert isinstance(normalized, dict)
    return normalized


def data_condition_scientific(
    declaration: dict[str, JsonValue] | None,
    selected: tuple[Protocol13TrainingRecord, ...],
) -> dict[str, JsonValue]:
    """Project only provenance coordinates, never a record's public/private body."""
    declared = inline_data_condition(declaration)
    if declared is None:
        return {}
    return {
        "data_condition": {
            "format": "training-data-condition@1",
            "declaration": declared,
            "ordered_selected_sources": [
                {
                    "benchmark_id": record.episode.benchmark.value,
                    "population_id": record.episode.population_id,
                    "source_question_id": record.episode.source_id,
                    "occurrence_id": record.episode.episode_id,
                }
                for record in selected
            ],
        }
    }


def require_same_run_data_condition(
    root: Path,
    current: EffectiveRunCondition,
    *,
    declared_data_by_condition: Mapping[str, JsonValue] | None = None,
) -> None:
    """Reject data changes on resume, using the run's existing process declarations.

    Other science (including an explicitly permitted horizon transition) stays
    with its existing authority. No new state file or identity is manufactured.
    A legacy run without an effective declaration has no declared data condition.
    """
    current_data = current.scientific.get("data_condition")
    paths = sorted(root.glob("effective-condition-process-*.json"))
    if declared_data_by_condition is not None:
        if current_data != declared_data_by_condition.get(current.condition_id):
            raise ValueError("current data differs from its declared domain schedule")
        for path in paths:
            previous = EffectiveRunCondition.from_value(json.loads(path.read_text()))
            if (
                previous.condition_id not in declared_data_by_condition
                or previous.scientific.get("data_condition")
                != declared_data_by_condition[previous.condition_id]
            ):
                raise ValueError("historical data differs from its original declared schedule")
        return
    previous_values = (
        EffectiveRunCondition.from_value(
            json.loads(path.read_text(encoding="utf-8"))
        ).scientific.get("data_condition")
        for path in paths
    )
    # No historical file means a legacy absent declaration, not permission to
    # silently introduce a different dataset under an existing run identity.
    for previous_data in previous_values if paths else (None,):
        if previous_data is None and current_data is None:
            continue
        previous = EffectiveRunCondition.create(
            condition_id=current.condition_id,
            scientific={"data_condition": previous_data},
            execution={},
        )
        proposed = EffectiveRunCondition.create(
            condition_id=current.condition_id,
            scientific={"data_condition": current_data},
            execution={},
        )
        previous.require_same_science(proposed)
