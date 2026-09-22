"""Freeze exact private benchmark files into one production catalog bundle.

The caller names every accepted file and source revision explicitly.  This
module performs no discovery, download, extraction, conversion, or benchmark
selection.  It only hashes the declared files and persists answer-free catalog
identities that the production loaders can consume later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.experiments import Benchmark

from .process_catalog import (
    ProductionProcessBenchmarkSource,
    ProductionProcessCatalogConfig,
)
from .production_catalog import (
    ProductionBenchmarkSource,
    ProductionCatalogConfig,
    ProductionSourceFormat,
)
from .snapshot import create_private_dataset_snapshot

FROZEN_PRODUCTION_CATALOG_BUNDLE_FORMAT = "skillev-private-production-catalog-bundle@2"
FROZEN_PRODUCTION_CATALOG_PUBLIC_FREEZE_FORMAT = "skillev-production-catalog-public-freeze@1"


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkFreezeSpec:
    """Exact non-process source declaration before its byte snapshot exists."""

    benchmark: Benchmark
    dataset_revision: str
    split: str
    source_format: ProductionSourceFormat
    relative_files: tuple[str, ...]
    relative_file_splits: tuple[str, ...]
    auxiliary_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProductionProcessBenchmarkFreezeSpec:
    """Exact process source declaration before its byte snapshot exists."""

    benchmark: Benchmark
    dataset_revision: str
    split: str
    task_manifest_relative_path: str
    snapshot_relative_files: tuple[str, ...]
    environment_source_revision: str


@dataclass(frozen=True, slots=True)
class FrozenProductionCatalogBundle:
    """Both catalog configurations tied to the same private dataset root."""

    non_process: ProductionCatalogConfig
    process: ProductionProcessCatalogConfig

    def __post_init__(self) -> None:
        if self.non_process.dataset_root != self.process.dataset_root:
            raise ValueError("production catalog roots must be identical")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": FROZEN_PRODUCTION_CATALOG_BUNDLE_FORMAT,
            "non_process": self.non_process.to_value(),
            "process": self.process.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> FrozenProductionCatalogBundle:
        data = normalize_json(value)
        if not isinstance(data, dict) or set(data) != {"format", "non_process", "process"}:
            raise ValueError("frozen production catalog bundle has an invalid field set")
        if data["format"] != FROZEN_PRODUCTION_CATALOG_BUNDLE_FORMAT:
            raise ValueError("unsupported frozen production catalog bundle format")
        return cls(
            non_process=ProductionCatalogConfig.from_value(data["non_process"]),
            process=ProductionProcessCatalogConfig.from_value(data["process"]),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def public_freeze_value(self) -> dict[str, JsonValue]:
        """Return the path-free catalog identity carried by formal attempts."""

        return {
            "format": FROZEN_PRODUCTION_CATALOG_PUBLIC_FREEZE_FORMAT,
            "non_process_sources": [source.to_value() for source in self.non_process.sources],
            "process_sources": [source.to_value() for source in self.process.sources],
            "retrieval_index_id": self.non_process.retrieval_index_id,
            "retrieval_index_relative_path": self.non_process.retrieval_index_relative_path,
        }

    @property
    def public_freeze_hash(self) -> str:
        return stable_hash(self.public_freeze_value())


def _freeze_non_process_source(
    dataset_root: Path,
    spec: ProductionBenchmarkFreezeSpec,
) -> ProductionBenchmarkSource:
    snapshot = create_private_dataset_snapshot(
        name=spec.benchmark.value,
        version=spec.dataset_revision,
        root=dataset_root,
        relative_files=tuple(sorted((*spec.relative_files, *spec.auxiliary_files))),
    )
    return ProductionBenchmarkSource(
        benchmark=spec.benchmark,
        dataset_revision=spec.dataset_revision,
        split=spec.split,
        source_format=spec.source_format,
        relative_files=spec.relative_files,
        snapshot=snapshot,
        relative_file_splits=spec.relative_file_splits,
        auxiliary_files=spec.auxiliary_files,
    )


def _freeze_process_source(
    dataset_root: Path,
    spec: ProductionProcessBenchmarkFreezeSpec,
) -> ProductionProcessBenchmarkSource:
    snapshot = create_private_dataset_snapshot(
        name=spec.benchmark.value,
        version=spec.dataset_revision,
        root=dataset_root,
        relative_files=spec.snapshot_relative_files,
    )
    return ProductionProcessBenchmarkSource(
        benchmark=spec.benchmark,
        dataset_revision=spec.dataset_revision,
        split=spec.split,
        task_manifest_relative_path=spec.task_manifest_relative_path,
        snapshot_relative_files=spec.snapshot_relative_files,
        snapshot=snapshot,
        environment_source_revision=spec.environment_source_revision,
    )


def freeze_production_catalog_bundle(
    *,
    dataset_root: Path,
    non_process_specs: tuple[ProductionBenchmarkFreezeSpec, ...],
    process_specs: tuple[ProductionProcessBenchmarkFreezeSpec, ...],
    retrieval_index_relative_path: str,
    retrieval_index_id: str,
) -> FrozenProductionCatalogBundle:
    """Hash the exact declared file sets and assemble both loader configs."""

    if not isinstance(dataset_root, Path) or not dataset_root.is_absolute():
        raise ValueError("dataset_root must be an absolute Path")
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    non_process = ProductionCatalogConfig(
        dataset_root=dataset_root,
        sources=tuple(_freeze_non_process_source(dataset_root, spec) for spec in non_process_specs),
        retrieval_index_relative_path=retrieval_index_relative_path,
        retrieval_index_id=retrieval_index_id,
    )
    process = ProductionProcessCatalogConfig(
        dataset_root=dataset_root,
        sources=tuple(_freeze_process_source(dataset_root, spec) for spec in process_specs),
    )
    return FrozenProductionCatalogBundle(non_process=non_process, process=process)


def publish_frozen_production_catalog_bundle(
    bundle: FrozenProductionCatalogBundle,
    output_path: Path,
) -> None:
    """Write one canonical config file without replacing an existing artifact."""

    if not isinstance(bundle, FrozenProductionCatalogBundle):
        raise TypeError("bundle must be FrozenProductionCatalogBundle")
    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("output_path must be an absolute Path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(bundle.to_value()))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_frozen_production_catalog_bundle(
    path: Path,
) -> FrozenProductionCatalogBundle:
    """Load a previously published canonical catalog bundle."""

    with path.open(encoding="utf-8") as stream:
        return FrozenProductionCatalogBundle.from_value(json.load(stream))


__all__ = [
    "FROZEN_PRODUCTION_CATALOG_BUNDLE_FORMAT",
    "FROZEN_PRODUCTION_CATALOG_PUBLIC_FREEZE_FORMAT",
    "FrozenProductionCatalogBundle",
    "ProductionBenchmarkFreezeSpec",
    "ProductionProcessBenchmarkFreezeSpec",
    "freeze_production_catalog_bundle",
    "load_frozen_production_catalog_bundle",
    "publish_frozen_production_catalog_bundle",
]
