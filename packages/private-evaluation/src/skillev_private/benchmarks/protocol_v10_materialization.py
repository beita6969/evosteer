"""Server-only materialization boundary for Protocol 10 populations.

Dataset-specific readers produce :class:`ProtocolV10SourceRecord` objects in
memory.  This module writes the answer-free rollout projection separately from
the verifier-only payload and builds the frozen SkillFlow-style 9 x 512
selection.  Neither output belongs in Git.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from skillev.contracts import JsonValue, normalize_json
from skillev.experiments.protocol_v10 import BenchmarkPopulation
from skillev.rollout import RolloutTask

from .protocol_v10_identity import ProtocolV10TaskIdentityMaterializer
from .protocol_v10_population import (
    PopulationOverlapIdentityMaterializer,
    PrivateBenchmarkPopulation,
    PrivatePopulationItem,
    ProtocolV10PopulationCatalog,
    ProtocolV10TrainingSelection,
    build_protocol_v10_training_selection,
    load_protocol_v10_population_catalog,
    write_protocol_v10_population_file,
)

PRIVATE_RECORD_FORMAT = "skillev-private-protocol-v10-source-record@1"
MATERIALIZED_CATALOG_FORMAT = "skillev-private-protocol-v10-catalog@1"


@dataclass(frozen=True, slots=True)
class ProtocolV10SourceRecord:
    """One trusted source row before public/private separation.

    ``private_payload`` may contain answers, rubrics, tests, simulator routes,
    or workbook paths.  It is deliberately omitted from ``RolloutTask`` and
    from the population file consumed by the policy.
    """

    source_id: str
    task: RolloutTask
    private_payload: JsonValue

    def __post_init__(self) -> None:
        if not self.source_id.strip() or self.task.task_id != self.source_id:
            raise ValueError("source record identity differs from its rollout task")
        object.__setattr__(self, "private_payload", normalize_json(self.private_payload))

    def private_value(self) -> dict[str, JsonValue]:
        return {
            "format": PRIVATE_RECORD_FORMAT,
            "private_payload": self.private_payload,
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class MaterializedProtocolV10Population:
    population: PrivateBenchmarkPopulation
    private_records: tuple[ProtocolV10SourceRecord, ...]

    def __post_init__(self) -> None:
        public_ids = tuple(item.source_id for item in self.population.items)
        private_ids = tuple(item.source_id for item in self.private_records)
        if public_ids != private_ids:
            raise ValueError("public and private population records differ")


@dataclass(frozen=True, slots=True)
class LoadedProtocolV10Catalog:
    """One complete server-private catalog plus its exact training routes."""

    catalog: ProtocolV10PopulationCatalog
    selection: ProtocolV10TrainingSelection
    private_record_files: dict[str, Path]

    def __post_init__(self) -> None:
        population_ids = {item.spec.population_id for item in self.catalog.populations}
        if set(self.private_record_files) != population_ids:
            raise ValueError("loaded Protocol 10 private routes are incomplete")
        if self.selection != build_protocol_v10_training_selection(self.catalog):
            raise ValueError("loaded Protocol 10 selection differs from its catalog")


def load_protocol_v10_materialized_catalog(
    protocol: object,
    root: Path,
) -> LoadedProtocolV10Catalog:
    """Load only a fully published catalog and rebuild its frozen selection."""

    from skillev.experiments.protocol_v10 import ActiveBenchmarkProtocolV10

    from .protocol_v10_identity import ProtocolV10TaskIdentityMaterializer

    if not isinstance(protocol, ActiveBenchmarkProtocolV10):
        raise TypeError("catalog loader requires the active Protocol 10")
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("materialized Protocol 10 root must be an absolute directory")
    summary_path = root / "catalog.json"
    selection_path = root / "training-selection.json"
    if not summary_path.is_file() or not selection_path.is_file():
        raise ValueError("materialized Protocol 10 catalog is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("format") != MATERIALIZED_CATALOG_FORMAT
        or summary.get("training_episode_count") != 4_608
        or not isinstance(summary.get("populations"), list)
    ):
        raise ValueError("materialized Protocol 10 catalog summary is incompatible")
    expected = tuple(
        population for benchmark in protocol.benchmarks for population in benchmark.populations
    )
    expected_ids = {population.population_id for population in expected}
    summary_ids = {
        row.get("population_id") for row in summary["populations"] if isinstance(row, dict)
    }
    if summary_ids != expected_ids or len(summary["populations"]) != len(expected):
        raise ValueError("materialized Protocol 10 summary does not cover the protocol")
    public_files = {
        population_id: (root / "populations" / f"{population_id}.jsonl").resolve()
        for population_id in expected_ids
    }
    private_files = {
        population_id: (root / "private-records" / f"{population_id}.jsonl").resolve()
        for population_id in expected_ids
    }
    if any(not path.is_file() for path in (*public_files.values(), *private_files.values())):
        raise ValueError("materialized Protocol 10 population route is absent")
    materializers: dict[str, PopulationOverlapIdentityMaterializer] = {
        population_id: ProtocolV10TaskIdentityMaterializer() for population_id in expected_ids
    }
    catalog = load_protocol_v10_population_catalog(protocol, public_files, materializers)
    selection = ProtocolV10TrainingSelection.read(selection_path)
    return LoadedProtocolV10Catalog(catalog, selection, private_files)


def materialize_protocol_v10_population(
    spec: BenchmarkPopulation,
    records: tuple[ProtocolV10SourceRecord, ...],
    *,
    identity_materializer: ProtocolV10TaskIdentityMaterializer | None = None,
) -> MaterializedProtocolV10Population:
    """Separate trusted source records into answer-free and verifier-only data."""

    if not records:
        raise ValueError("Protocol 10 source population cannot be empty")
    source_ids = tuple(record.source_id for record in records)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Protocol 10 source population repeats a source ID")
    materializer = identity_materializer or ProtocolV10TaskIdentityMaterializer()
    items = tuple(
        PrivatePopulationItem(
            source_id=record.source_id,
            task=record.task,
            overlap_keys=materializer.materialize(
                spec=spec,
                source_id=record.source_id,
                task=record.task,
            ),
        )
        for record in records
    )
    return MaterializedProtocolV10Population(
        PrivateBenchmarkPopulation(spec, items, materializer.version),
        records,
    )


def _write_private_records_once(
    records: tuple[ProtocolV10SourceRecord, ...],
    path: Path,
) -> None:
    if not path.is_absolute():
        raise ValueError("private source-record path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = path.with_name(f".{path.name}.staging")
    if path.exists() or staging.exists():
        raise FileExistsError(path if path.exists() else staging)
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record.private_value(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def publish_protocol_v10_materialized_catalog(
    *,
    protocol: object,
    populations: tuple[MaterializedProtocolV10Population, ...],
    output_root: Path,
) -> ProtocolV10PopulationCatalog:
    """Publish a complete catalog under one server-private directory."""

    from skillev.experiments.protocol_v10 import ActiveBenchmarkProtocolV10

    if not isinstance(protocol, ActiveBenchmarkProtocolV10):
        raise TypeError("catalog publisher requires the active Protocol 10")
    if not output_root.is_absolute() or output_root.exists():
        raise ValueError("Protocol 10 output root must be a new absolute directory")
    catalog = ProtocolV10PopulationCatalog(
        protocol,
        tuple(item.population for item in populations),
    )
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, mode=0o700)
    try:
        public_root = staging / "populations"
        private_root = staging / "private-records"
        public_root.mkdir(mode=0o700)
        private_root.mkdir(mode=0o700)
        summary: list[dict[str, JsonValue]] = []
        for materialized in populations:
            population_id = materialized.population.spec.population_id
            write_protocol_v10_population_file(
                materialized.population,
                (public_root / f"{population_id}.jsonl").resolve(),
            )
            _write_private_records_once(
                materialized.private_records,
                (private_root / f"{population_id}.jsonl").resolve(),
            )
            summary.append(
                {
                    "benchmark": materialized.population.spec.benchmark.value,
                    "population_id": population_id,
                    "role": materialized.population.spec.role.value,
                    "unique_item_count": len(materialized.population.items),
                }
            )
        selection = build_protocol_v10_training_selection(catalog)
        selection.write_once(staging / "training-selection.json")
        (staging / "catalog.json").write_text(
            json.dumps(
                {
                    "format": MATERIALIZED_CATALOG_FORMAT,
                    "populations": summary,
                    "training_episode_count": len(selection.episodes),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    except BaseException:
        # Preserve a failed staging tree for diagnosis; it is never accepted as
        # a catalog because only the final directory name is consumed.
        raise
    return catalog


__all__ = [
    "MATERIALIZED_CATALOG_FORMAT",
    "PRIVATE_RECORD_FORMAT",
    "LoadedProtocolV10Catalog",
    "MaterializedProtocolV10Population",
    "ProtocolV10SourceRecord",
    "load_protocol_v10_materialized_catalog",
    "materialize_protocol_v10_population",
    "publish_protocol_v10_materialized_catalog",
]
