"""Separate source identity, panel selection and development exposure metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .input_metric_contracts import PublicTaskView


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    panel_id: str
    population_id: str | None
    benchmark_id: str
    source_revision: str | None
    input_profile: str
    exposure_status: str
    source_split: str | None = None

    @classmethod
    def for_entry(
        cls,
        entry: PublicTaskView,
        provenance: Mapping[str, object],
        *,
        panel_id: str,
        exposure_status: str,
    ) -> SourceIdentity:
        source = provenance.get(entry.benchmark, {})
        source = source if isinstance(source, Mapping) else {}

        def first(*keys: str) -> str | None:
            return next(
                (str(source[k]) for k in keys if isinstance(source.get(k), str) and source[k]), None
            )

        return cls(
            panel_id,
            first("population_id", "dataset", "source"),
            entry.benchmark,
            first("source_revision", "dataset_revision", "revision", "snapshot", "release"),
            entry.input_profile,
            exposure_status,
            first("split"),
        )
