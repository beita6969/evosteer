from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.acquisition import (
    AcquiredArtifactReceipt,
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
)
from skillev_private.benchmarks.acquisition_git import (
    AcquiredRepositoryReceipt,
    RepositoryAcquisitionReceipt,
)
from skillev_private.benchmarks.archive_preparation import (
    LockedArchiveBatchReceipt,
    PreparedLockedArchive,
    locked_archive_preparations,
)
from skillev_private.benchmarks.catalog_freeze import (
    freeze_production_catalog_bundle,
    load_frozen_production_catalog_bundle,
    publish_frozen_production_catalog_bundle,
)
from skillev_private.benchmarks.catalog_preparation import (
    LockedRetrievalInfrastructure,
    freeze_locked_production_catalog,
)
from skillev_private.benchmarks.non_process_preparation import (
    LockedNonProcessPreparationReceipt,
    locked_non_process_source_plans,
)
from skillev_private.benchmarks.process_preparation import (
    ALFWorldProcessTaskRecord,
    ExternalProcessTaskRecord,
    LockedProcessPreparationReceipt,
    ProcessTaskManifestInput,
    ScienceWorldProcessTaskRecord,
    WebShopProcessTaskRecord,
    load_locked_process_preparation_receipt,
    prepare_locked_process_benchmarks,
    publish_locked_process_preparation_receipt,
)
from skillev_private.benchmarks.retrieval_preparation import (
    RetrievalPreparationManifest,
)

from skillev.benchmarks import DocumentPassage, RetrievalIndexManifest, build_retrieval_index
from skillev.contracts import canonical_json
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

_GIT_SOURCE_KINDS = {
    "pinned-atlas-preparation",
    "pinned-git-dataset",
    "pinned-git-environment",
    "pinned-git-file",
    "pinned-huggingface-environment",
}


def _lock() -> BenchmarkAcquisitionLock:
    path = Path(__file__).parents[2] / "benchmark-acquisition-lock.json"
    return BenchmarkAcquisitionLock.from_value(json.loads(path.read_text(encoding="utf-8")))


def _receipt_chain(
    lock: BenchmarkAcquisitionLock,
    root: Path,
) -> tuple[
    BenchmarkAcquisitionReceipt,
    LockedArchiveBatchReceipt,
    RepositoryAcquisitionReceipt,
]:
    acquired = BenchmarkAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=root,
        artifacts=tuple(
            AcquiredArtifactReceipt(
                benchmark=entry.benchmark,
                locator=artifact.locator,
                relative_path=artifact.relative_path,
                size_bytes=artifact.expected_size or 1,
                sha256=artifact.expected_sha256 or "a" * 64,
            )
            for entry in lock.benchmarks
            for artifact in entry.source.artifacts
        ),
    )
    archive = LockedArchiveBatchReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquired.content_hash,
        archives=tuple(
            PreparedLockedArchive(
                benchmark=plan.benchmark,
                artifact_relative_path=plan.artifact_relative_path,
                output_relative_path=plan.output_relative_path,
                preparation_content_hash=f"sha256:{'b' * 64}",
            )
            for plan in locked_archive_preparations(lock)
        ),
    )
    repositories = RepositoryAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=root,
        repositories=tuple(
            AcquiredRepositoryReceipt(
                benchmark=entry.benchmark,
                repository=entry.source.repository,
                revision=entry.source.revision,
                relative_checkout_path=f"{entry.benchmark.value}/repository",
            )
            for entry in lock.benchmarks
            if entry.source.kind in _GIT_SOURCE_KINDS
        ),
    )
    return acquired, archive, repositories


def _revision(lock: BenchmarkAcquisitionLock, benchmark: Benchmark) -> str:
    return next(entry.source.revision for entry in lock.benchmarks if entry.benchmark is benchmark)


def _write(path: Path, payload: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _inputs(
    lock: BenchmarkAcquisitionLock,
    root: Path,
) -> tuple[ProcessTaskManifestInput, ...]:
    asset_paths = {
        Benchmark.WEBSHOP: (
            "webshop/environment/products.json",
            "webshop/environment/search-index/segments",
        ),
        Benchmark.ALFWORLD: (
            "alfworld/environment/base_config.yaml",
            "alfworld/environment/game-1/game.tw-pddl",
            "alfworld/environment/game-1/traj_data.json",
        ),
        Benchmark.SCIENCE_WORLD: ("scienceworld/environment/scienceworld.jar",),
        Benchmark.APPWORLD: ("appworld/environment/manifest.json",),
        Benchmark.BFCL_V3: ("bfcl-v3/environment/manifest.json",),
    }
    for paths in asset_paths.values():
        for path in paths:
            _write(root / path)
    webshop_revision = _revision(lock, Benchmark.WEBSHOP)
    alfworld_revision = _revision(lock, Benchmark.ALFWORLD)
    scienceworld_revision = _revision(lock, Benchmark.SCIENCE_WORLD)

    def external(benchmark: Benchmark, split: str) -> ProcessTaskManifestInput:
        revision = _revision(lock, benchmark)
        task = RolloutTask(
            task_id=f"{benchmark.value}-task-0001",
            environment_id=f"{benchmark.value}-environment",
            task_family=f"{benchmark.value}/fixture",
            context_id=f"{benchmark.value}-context-0001",
            query=f"Complete the public {benchmark.value} fixture task.",
            available_tools=("submit",),
            public_context={"benchmark_id": benchmark.value},
        )
        return ProcessTaskManifestInput(
            benchmark=benchmark,
            dataset_revision=revision,
            split=split,
            task_manifest_relative_path=f"_derived/process/{benchmark.value}.jsonl",
            asset_relative_files=asset_paths[benchmark],
            environment_source_revision=revision,
            records=(
                ExternalProcessTaskRecord(
                    benchmark=benchmark,
                    source_split=split,
                    task=task,
                ),
            ),
        )

    return (
        ProcessTaskManifestInput(
            benchmark=Benchmark.WEBSHOP,
            dataset_revision=webshop_revision,
            split="train",
            task_manifest_relative_path="_derived/process/webshop.jsonl",
            asset_relative_files=asset_paths[Benchmark.WEBSHOP],
            environment_source_revision=webshop_revision,
            records=(
                WebShopProcessTaskRecord(
                    task_id="webshop-task-0001",
                    task_family="webshop/home",
                    query="Find the requested public product.",
                    public_context={"available_actions": ["search"]},
                    goal_id="goal-1",
                    session_id="session-1",
                    goal_index=1,
                ),
                WebShopProcessTaskRecord(
                    task_id="webshop-task-0002",
                    task_family="webshop/office",
                    query="Find another requested public product.",
                    public_context={"available_actions": ["search"]},
                    goal_id="goal-2",
                    session_id="session-2",
                    goal_index=2,
                ),
            ),
        ),
        ProcessTaskManifestInput(
            benchmark=Benchmark.ALFWORLD,
            dataset_revision=alfworld_revision,
            split="train",
            task_manifest_relative_path="_derived/process/alfworld.jsonl",
            asset_relative_files=asset_paths[Benchmark.ALFWORLD],
            environment_source_revision=alfworld_revision,
            records=(
                ALFWorldProcessTaskRecord(
                    task_id="alfworld-task-0001",
                    task_family="alfworld/pick-and-place",
                    query="Put the public object in its requested receptacle.",
                    game_id="game-1",
                    initial_observation="A public fixture room.",
                    admissible_commands=("look", "take object"),
                    max_steps=50,
                ),
            ),
        ),
        external(Benchmark.APPWORLD, "train"),
        ProcessTaskManifestInput(
            benchmark=Benchmark.SCIENCE_WORLD,
            dataset_revision=scienceworld_revision,
            split="test",
            task_manifest_relative_path="_derived/process/scienceworld.jsonl",
            asset_relative_files=asset_paths[Benchmark.SCIENCE_WORLD],
            environment_source_revision=scienceworld_revision,
            records=(
                ScienceWorldProcessTaskRecord(
                    task_id="scienceworld-task-0001",
                    task_family="scienceworld/chemistry",
                    query="Complete the public science task.",
                    task_name="chemistry-task",
                    variation_index=1,
                    initial_observation="A public fixture laboratory.",
                    max_steps=100,
                ),
            ),
        ),
        external(Benchmark.BFCL_V3, "test"),
    )


@pytest.mark.parametrize("record_index", [0, 1, 3])
def test_process_manifest_records_require_their_benchmark_namespace(
    tmp_path: Path,
    record_index: int,
) -> None:
    records = _inputs(_lock(), tmp_path)[record_index].records
    record = records[0]

    with pytest.raises(ValueError):
        replace(record, task_family="other/family")


@pytest.mark.parametrize("record_index", [2, 4])
def test_external_process_records_require_their_benchmark_namespace(
    tmp_path: Path,
    record_index: int,
) -> None:
    record = _inputs(_lock(), tmp_path)[record_index].records[0]
    assert isinstance(record, ExternalProcessTaskRecord)

    with pytest.raises(ValueError):
        replace(record, task=replace(record.task, task_family="other/family"))


def _prepare(
    tmp_path: Path,
    *,
    lock: BenchmarkAcquisitionLock | None = None,
) -> tuple[
    BenchmarkAcquisitionLock,
    Path,
    tuple[ProcessTaskManifestInput, ...],
    LockedProcessPreparationReceipt,
]:
    lock = _lock() if lock is None else lock
    root = (tmp_path / "private-datasets").resolve()
    root.mkdir()
    acquired, archive, repositories = _receipt_chain(lock, root)
    inputs = _inputs(lock, root)
    receipt = prepare_locked_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquired,
        archive_batch_receipt=archive,
        repository_receipt=repositories,
        target_root=root,
        inputs=inputs,
    )
    return lock, root, inputs, receipt


def _tiny_retrieval_identity(raw_sha256: str) -> RetrievalIndexManifest:
    return RetrievalIndexManifest.create(
        (
            DocumentPassage(
                passage_id="fixture-passage-1",
                document_id="fixture-document-1",
                title="Public fixture title",
                text="Public fixture passage.",
            ),
        ),
        corpus_name="atlas-dpr-wikipedia-psgs-w100",
        corpus_version=raw_sha256,
    )


def _lock_with_retrieval(
    *,
    raw_sha256: str,
    raw_size: int,
    manifest: RetrievalIndexManifest,
    index_id: str | None = None,
) -> BenchmarkAcquisitionLock:
    base = _lock()
    return BenchmarkAcquisitionLock(
        benchmarks=base.benchmarks,
        shared_infrastructure={
            "atlas_wikipedia_fts5": {
                "corpus_source": "official-dpr-wikipedia-passages",
                "index_id": manifest.index_id if index_id is None else index_id,
                "passage_count": manifest.passage_count,
                "retrieval_backend": manifest.retrieval_backend,
                "source_locator": "https://example.invalid/pinned-source.tsv.gz",
                "source_sha256": raw_sha256.removeprefix("sha256:"),
                "source_size": raw_size,
            }
        },
    )


def _non_process_receipt(
    lock: BenchmarkAcquisitionLock,
    root: Path,
) -> LockedNonProcessPreparationReceipt:
    acquired, archive, _ = _receipt_chain(lock, root)
    sources = locked_non_process_source_plans(lock)
    for source in sources:
        for relative_path in (*source.relative_files, *source.auxiliary_files):
            _write(root / relative_path)
    return LockedNonProcessPreparationReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquired.content_hash,
        archive_batch_receipt_hash=archive.content_hash,
        atlas_preparation_content_hash=f"sha256:{'d' * 64}",
        medqa_preparation_content_hash=f"sha256:{'e' * 64}",
        external_preparation_content_hash=f"sha256:{'f' * 64}",
        target_root=root,
        sources=sources,
    )


def _write_tiny_retrieval(
    *,
    root: Path,
    raw_sha256: str,
    raw_size: int,
) -> tuple[Path, Path, RetrievalIndexManifest]:
    index_path = root / "retrieval/atlas.sqlite3"
    index_path.parent.mkdir(parents=True)
    manifest = build_retrieval_index(
        index_path,
        (
            DocumentPassage(
                passage_id="fixture-passage-1",
                document_id="fixture-document-1",
                title="Public fixture title",
                text="Public fixture passage.",
            ),
        ),
        corpus_name="atlas-dpr-wikipedia-psgs-w100",
        corpus_version=raw_sha256,
    )
    index_bytes = index_path.read_bytes()
    prepared = RetrievalPreparationManifest.create(
        input_raw_sha256=raw_sha256,
        input_size_bytes=raw_size,
        index_file_sha256=f"sha256:{hashlib.sha256(index_bytes).hexdigest()}",
        index_file_size_bytes=len(index_bytes),
        index_manifest=manifest,
    )
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest_path.write_text(canonical_json(prepared.to_value()) + "\n", encoding="utf-8")
    return index_path, manifest_path, manifest


def test_prepares_five_canonical_process_manifests_and_round_trips_receipt(
    tmp_path: Path,
) -> None:
    lock, root, inputs, first = _prepare(tmp_path)
    acquired, archive, repositories = _receipt_chain(lock, root)
    second = prepare_locked_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquired,
        archive_batch_receipt=archive,
        repository_receipt=repositories,
        target_root=root,
        inputs=inputs,
    )

    assert first == second
    assert tuple(source.task_count for source in first.sources) == (2, 1, 1, 1, 1)
    for item in inputs:
        path = root / item.task_manifest_relative_path
        assert path.read_bytes() == item.manifest_bytes()
        assert tuple(json.loads(line) for line in path.read_text().splitlines()) == tuple(
            record.to_value() for record in item.records
        )
    output = (tmp_path / "process-preparation.json").resolve()
    publish_locked_process_preparation_receipt(first, output)
    assert load_locked_process_preparation_receipt(output) == first
    with pytest.raises(FileExistsError):
        publish_locked_process_preparation_receipt(second, output)


def test_existing_manifest_change_is_rejected_without_replacement(tmp_path: Path) -> None:
    lock, root, inputs, _ = _prepare(tmp_path)
    acquired, archive, repositories = _receipt_chain(lock, root)
    manifest = root / inputs[0].task_manifest_relative_path
    changed = manifest.read_bytes() + b'{"changed":true}\n'
    manifest.write_bytes(changed)

    with pytest.raises(ValueError):
        prepare_locked_process_benchmarks(
            lock=lock,
            acquisition_receipt=acquired,
            archive_batch_receipt=archive,
            repository_receipt=repositories,
            target_root=root,
            inputs=inputs,
        )

    assert manifest.read_bytes() == changed


def test_process_preparation_requires_every_explicit_deployment_asset(
    tmp_path: Path,
) -> None:
    lock = _lock()
    root = (tmp_path / "private-datasets").resolve()
    root.mkdir()
    acquired, archive, repositories = _receipt_chain(lock, root)
    inputs = _inputs(lock, root)
    missing = root / inputs[1].asset_relative_files[0]
    moved = missing.with_name(missing.name + ".preserved")
    missing.rename(moved)

    with pytest.raises(FileNotFoundError):
        prepare_locked_process_benchmarks(
            lock=lock,
            acquisition_receipt=acquired,
            archive_batch_receipt=archive,
            repository_receipt=repositories,
            target_root=root,
            inputs=inputs,
        )

    assert moved.is_file()


def test_process_receipt_drives_complete_eighteen_source_freeze(
    tmp_path: Path,
) -> None:
    lock, root, _, process = _prepare(tmp_path)
    non_process = locked_non_process_source_plans(lock)
    for plan in non_process:
        for relative_path in (*plan.relative_files, *plan.auxiliary_files):
            _write(root / relative_path)
    retrieval = "retrieval/atlas.sqlite3"
    _write(root / retrieval)

    bundle = freeze_production_catalog_bundle(
        dataset_root=root,
        non_process_specs=tuple(plan.freeze_spec() for plan in non_process),
        process_specs=process.freeze_specs(),
        retrieval_index_relative_path=retrieval,
        retrieval_index_id=f"sha256:{'c' * 64}",
    )
    output = (tmp_path / "production-catalog.json").resolve()
    publish_frozen_production_catalog_bundle(bundle, output)
    restored = load_frozen_production_catalog_bundle(output)

    assert restored == bundle
    assert len(restored.non_process.sources) == 13
    assert len(restored.process.sources) == 5


def test_locked_receipts_and_verified_retrieval_freeze_one_catalog(
    tmp_path: Path,
) -> None:
    raw_sha256 = f"sha256:{'a' * 64}"
    raw_size = 321
    expected_manifest = _tiny_retrieval_identity(raw_sha256)
    lock = _lock_with_retrieval(
        raw_sha256=raw_sha256,
        raw_size=raw_size,
        manifest=expected_manifest,
    )
    lock, root, _, process = _prepare(tmp_path, lock=lock)
    non_process = _non_process_receipt(lock, root)
    index_path, manifest_path, actual_manifest = _write_tiny_retrieval(
        root=root,
        raw_sha256=raw_sha256,
        raw_size=raw_size,
    )

    locked_retrieval = LockedRetrievalInfrastructure.from_lock(lock)
    bundle = freeze_locked_production_catalog(
        lock=lock,
        non_process_receipt=non_process,
        process_receipt=process,
        retrieval_index_path=index_path,
        retrieval_manifest_path=manifest_path,
    )

    assert locked_retrieval.index_id == actual_manifest.index_id
    assert len(bundle.non_process.sources) == 13
    assert len(bundle.process.sources) == 5
    assert bundle.non_process.retrieval_index_relative_path == "retrieval/atlas.sqlite3"
    assert bundle.non_process.retrieval_index_id == actual_manifest.index_id


def test_catalog_freeze_rejects_retrieval_identity_not_committed_by_lock(
    tmp_path: Path,
) -> None:
    raw_sha256 = f"sha256:{'a' * 64}"
    raw_size = 321
    expected_manifest = _tiny_retrieval_identity(raw_sha256)
    lock = _lock_with_retrieval(
        raw_sha256=raw_sha256,
        raw_size=raw_size,
        manifest=expected_manifest,
        index_id=f"sha256:{'f' * 64}",
    )
    lock, root, _, process = _prepare(tmp_path, lock=lock)
    non_process = _non_process_receipt(lock, root)
    index_path, manifest_path, _ = _write_tiny_retrieval(
        root=root,
        raw_sha256=raw_sha256,
        raw_size=raw_size,
    )

    with pytest.raises(ValueError):
        freeze_locked_production_catalog(
            lock=lock,
            non_process_receipt=non_process,
            process_receipt=process,
            retrieval_index_path=index_path,
            retrieval_manifest_path=manifest_path,
        )
