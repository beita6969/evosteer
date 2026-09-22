from __future__ import annotations

from pathlib import Path

import pytest
from skillev_private.benchmarks.catalog_freeze import (
    FrozenProductionCatalogBundle,
    ProductionBenchmarkFreezeSpec,
    ProductionProcessBenchmarkFreezeSpec,
    freeze_production_catalog_bundle,
    load_frozen_production_catalog_bundle,
    publish_frozen_production_catalog_bundle,
)
from skillev_private.benchmarks.production_catalog import ProductionSourceFormat

from skillev.contracts import canonical_json
from skillev.experiments import Benchmark

_NON_PROCESS_FORMATS = {
    Benchmark.HOTPOT_QA: ProductionSourceFormat.PARQUET,
    Benchmark.TRIVIA_QA: ProductionSourceFormat.JSONL,
    Benchmark.AIME_2026: ProductionSourceFormat.PARQUET,
    Benchmark.MED_QA: ProductionSourceFormat.JSONL,
    Benchmark.BIRD_SQL: ProductionSourceFormat.JSONL,
    Benchmark.MBPP_PLUS: ProductionSourceFormat.JSONL,
    Benchmark.MUSIQUE: ProductionSourceFormat.JSONL,
    Benchmark.NQ_OPEN: ProductionSourceFormat.JSONL,
    Benchmark.MATH_HARD: ProductionSourceFormat.PARQUET,
    Benchmark.GPQA_DIAMOND: ProductionSourceFormat.CSV,
    Benchmark.MIND2WEB: ProductionSourceFormat.JSON_ARRAY,
    Benchmark.TABLEBENCH: ProductionSourceFormat.JSONL,
    Benchmark.HUMANEVAL_PLUS: ProductionSourceFormat.JSONL,
}
_PROCESS_BENCHMARKS = (
    Benchmark.WEBSHOP,
    Benchmark.ALFWORLD,
    Benchmark.APPWORLD,
    Benchmark.SCIENCE_WORLD,
    Benchmark.BFCL_V3,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _freeze_fixture(
    root: Path,
    *,
    initialize_files: bool = True,
) -> FrozenProductionCatalogBundle:
    non_process_specs: list[ProductionBenchmarkFreezeSpec] = []
    for benchmark, source_format in _NON_PROCESS_FORMATS.items():
        relative_files = (f"non-process/{benchmark.value}.source",)
        auxiliary_files = (
            (f"non-process/{benchmark.value}.auxiliary",) if benchmark is Benchmark.MIND2WEB else ()
        )
        if initialize_files:
            _write(root / relative_files[0], f"{benchmark.value}\n".encode())
            for auxiliary_file in auxiliary_files:
                _write(root / auxiliary_file, f"{benchmark.value}-auxiliary\n".encode())
        non_process_specs.append(
            ProductionBenchmarkFreezeSpec(
                benchmark=benchmark,
                dataset_revision=f"{benchmark.value}@fixture",
                split="fixture",
                source_format=source_format,
                relative_files=relative_files,
                relative_file_splits=("fixture",),
                auxiliary_files=auxiliary_files,
            )
        )

    process_specs: list[ProductionProcessBenchmarkFreezeSpec] = []
    for benchmark in _PROCESS_BENCHMARKS:
        manifest = f"process/{benchmark.value}/tasks.jsonl"
        asset = f"process/{benchmark.value}/environment.asset"
        files = tuple(sorted((manifest, asset)))
        if initialize_files:
            _write(root / manifest, b'{"public":"fixture"}\n')
            _write(root / asset, f"{benchmark.value}-environment\n".encode())
        process_specs.append(
            ProductionProcessBenchmarkFreezeSpec(
                benchmark=benchmark,
                dataset_revision=f"{benchmark.value}@fixture",
                split="fixture",
                task_manifest_relative_path=manifest,
                snapshot_relative_files=files,
                environment_source_revision="a" * 40,
            )
        )

    return freeze_production_catalog_bundle(
        dataset_root=root,
        non_process_specs=tuple(non_process_specs),
        process_specs=tuple(process_specs),
        retrieval_index_relative_path="retrieval/atlas.sqlite3",
        retrieval_index_id=f"sha256:{'b' * 64}",
    )


def test_freezes_and_publishes_one_canonical_catalog_bundle(tmp_path: Path) -> None:
    bundle = _freeze_fixture(tmp_path)
    output = tmp_path / "production-catalog.json"

    publish_frozen_production_catalog_bundle(bundle, output)
    restored = load_frozen_production_catalog_bundle(output)

    assert restored == bundle
    assert restored.content_hash == bundle.content_hash
    assert restored.non_process.dataset_root == tmp_path
    assert restored.process.dataset_root == tmp_path
    mind2web = next(
        source for source in restored.non_process.sources if source.benchmark is Benchmark.MIND2WEB
    )
    assert mind2web.auxiliary_files == ("non-process/mind2web.auxiliary",)


def test_publisher_never_replaces_an_existing_bundle(tmp_path: Path) -> None:
    bundle = _freeze_fixture(tmp_path)
    output = tmp_path / "production-catalog.json"
    publish_frozen_production_catalog_bundle(bundle, output)
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        publish_frozen_production_catalog_bundle(bundle, output)

    assert output.read_bytes() == original


def test_changed_declared_source_bytes_change_the_bundle_identity(tmp_path: Path) -> None:
    first = _freeze_fixture(tmp_path)
    source = tmp_path / "non-process" / "hotpotqa.source"
    source.write_bytes(source.read_bytes() + b"changed\n")

    second = _freeze_fixture(tmp_path, initialize_files=False)

    assert second.content_hash != first.content_hash
    assert second.non_process.sources[0].snapshot != first.non_process.sources[0].snapshot


def test_bundle_rejects_mismatched_private_roots(tmp_path: Path) -> None:
    first = _freeze_fixture(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()

    with pytest.raises(ValueError):
        FrozenProductionCatalogBundle(
            non_process=first.non_process,
            process=type(first.process)(
                dataset_root=other_root,
                sources=first.process.sources,
            ),
        )


def test_public_catalog_freeze_hash_is_path_free_but_binds_all_frozen_sources(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-private-root"
    second_root = tmp_path / "second-private-root"
    first_root.mkdir()
    second_root.mkdir()
    first = _freeze_fixture(first_root)
    second = _freeze_fixture(second_root)

    assert first.content_hash != second.content_hash
    assert first.public_freeze_hash == second.public_freeze_hash
    assert first_root.as_posix() not in canonical_json(first.public_freeze_value())
    assert second_root.as_posix() not in canonical_json(second.public_freeze_value())
