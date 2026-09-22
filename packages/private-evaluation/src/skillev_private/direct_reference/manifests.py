"""Explicit, non-hash population identity manifests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.evaluation.direct_baseline.config import DirectBenchmark


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"population manifest {label} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class PopulationEntry:
    task_id: str
    source_identity: str
    source_split: str
    source_position: int
    option_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not self.task_id.strip()
            or not self.source_identity.strip()
            or not self.source_split.strip()
        ):
            raise ValueError("population identity fields must be non-empty")
        if self.source_position < 0:
            raise ValueError("source_position must be non-negative")


@dataclass(frozen=True, slots=True)
class PopulationManifest:
    format_version: str
    population_id: str
    benchmark: DirectBenchmark
    dataset_revision: str
    selection_rule: str
    entries: tuple[PopulationEntry, ...]

    def __post_init__(self) -> None:
        expected = 30 if self.benchmark is DirectBenchmark.AIME_2026 else 128
        if len(self.entries) != expected:
            raise ValueError(f"formal population must contain {expected} entries")
        task_ids = tuple(entry.task_id for entry in self.entries)
        source_ids = tuple(entry.source_identity for entry in self.entries)
        if len(set(task_ids)) != len(task_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("population task and source identities must be unique")


def load_population_manifest(path: Path) -> PopulationManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("population manifest must be an object")
    raw = cast(dict[str, object], value)
    if raw.get("format") != "skillev-direct-population-manifest@1":
        raise ValueError("population manifest format is incompatible")
    rows = raw.get("entries")
    if type(rows) is not list:
        raise ValueError("population manifest entries must be an array")
    entries: list[PopulationEntry] = []
    for value in cast(list[object], rows):
        if type(value) is not dict:
            raise ValueError("population entry must be an object")
        item = cast(dict[str, object], value)
        order = item.get("option_order")
        if order is not None and (
            type(order) is not list or any(type(choice) is not str for choice in order)
        ):
            raise ValueError("population option order must be text or null")
        entries.append(
            PopulationEntry(
                task_id=str(item["task_id"]),
                source_identity=str(item["source_identity"]),
                source_split=str(item["source_split"]),
                source_position=_integer(item["source_position"], "source position"),
                option_order=tuple(cast(list[str], order)) if order is not None else None,
            )
        )
    return PopulationManifest(
        format_version=str(raw["format"]),
        population_id=str(raw["population_id"]),
        benchmark=DirectBenchmark(str(raw["benchmark"])),
        dataset_revision=str(raw["dataset_revision"]),
        selection_rule=str(raw["selection_rule"]),
        entries=tuple(entries),
    )


def population_manifest_value(manifest: PopulationManifest) -> dict[str, object]:
    """Return the stable private JSON representation without content hashes."""

    return {
        "format": manifest.format_version,
        "population_id": manifest.population_id,
        "benchmark": manifest.benchmark.value,
        "dataset_revision": manifest.dataset_revision,
        "selection_rule": manifest.selection_rule,
        "entries": [
            {
                "task_id": entry.task_id,
                "source_identity": entry.source_identity,
                "source_split": entry.source_split,
                "source_position": entry.source_position,
                "option_order": list(entry.option_order) if entry.option_order else None,
            }
            for entry in manifest.entries
        ],
    }


def write_population_manifest(path: Path, manifest: PopulationManifest) -> None:
    """Create a manifest once; formal populations are never silently replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_population_manifest(path)
        if existing != manifest:
            raise FileExistsError("population manifest already exists with different contents")
        return
    path.write_text(
        json.dumps(population_manifest_value(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_manifest_task_ids(
    manifest: PopulationManifest,
    *,
    population_id: str,
    task_ids: tuple[str, ...],
    source_identities: tuple[str, ...] | None = None,
    dataset_revision: str | None = None,
    selection_rule: str | None = None,
) -> None:
    if manifest.population_id != population_id:
        raise ValueError("manifest population differs from executable protocol")
    if dataset_revision is not None and manifest.dataset_revision != dataset_revision:
        raise ValueError("manifest dataset revision differs from executable protocol")
    if selection_rule is not None and manifest.selection_rule != selection_rule:
        raise ValueError("manifest selection rule differs from executable protocol")
    if tuple(entry.task_id for entry in manifest.entries) != task_ids:
        raise ValueError("runtime task order differs from frozen population manifest")
    if (
        source_identities is not None
        and tuple(entry.source_identity for entry in manifest.entries) != source_identities
    ):
        raise ValueError("runtime source identities differ from frozen population manifest")


def select_manifest_rows(
    rows: list[object],
    manifest: PopulationManifest,
    *,
    source_identity: Callable[[object, int], str],
) -> list[object]:
    by_identity: dict[str, object] = {}
    for position, row in enumerate(rows):
        identity = source_identity(row, position)
        if identity in by_identity:
            raise ValueError("source identities must be unique")
        by_identity[identity] = row
    selected: list[object] = []
    for entry in manifest.entries:
        try:
            selected.append(by_identity[entry.source_identity])
        except KeyError as exc:
            raise ValueError(f"manifest source identity missing: {entry.source_identity}") from exc
    return selected
