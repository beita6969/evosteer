"""Command line entrypoint for the result-blind benchmark source lock.

The command never discovers sources or benchmark records.  It only inspects or
materializes the exact repository revisions and artifact locators already
committed in ``benchmark-acquisition-lock.json``.  All destination and receipt
paths are explicit so callers can keep the complete private tree on the
designated E-hosted ext4 volume.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

from skillev.contracts import JsonValue, canonical_json
from skillev.experiments import Benchmark

from .acquisition import (
    BenchmarkAcquisitionReceipt,
    CurlLockedArtifactFetcher,
    LockedArtifactFetcher,
    acquire_locked_benchmark_artifacts,
    inspect_acquisition_progress,
    load_benchmark_acquisition_lock,
    load_benchmark_acquisition_receipt,
    publish_benchmark_acquisition_receipt,
)
from .acquisition_git import (
    GitLockedRepositoryFetcher,
    LockedRepositoryFetcher,
    RepositoryAcquisitionReceipt,
    acquire_locked_benchmark_repositories,
    load_repository_acquisition_receipt,
    publish_repository_acquisition_receipt,
)
from .archive_preparation import (
    LockedArchiveBatchReceipt,
    load_locked_archive_batch_receipt,
    locked_archive_preparations,
    prepare_locked_archives,
    publish_locked_archive_batch_receipt,
)
from .catalog_freeze import publish_frozen_production_catalog_bundle
from .catalog_preparation import freeze_locked_production_catalog
from .non_process_preparation import (
    LockedNonProcessPreparationReceipt,
    load_locked_non_process_preparation_receipt,
    prepare_locked_non_process_benchmarks,
    publish_locked_non_process_preparation_receipt,
)
from .official_process_preparation import (
    OfficialProcessPreparationDeployment,
    OfficialProcessPreparationFactory,
)
from .process_preparation import (
    LockedProcessPreparationReceipt,
    load_locked_process_preparation_receipt,
    prepare_locked_process_benchmarks,
    publish_locked_process_preparation_receipt,
)

ACQUISITION_COMMAND_FORMAT = "skillev-private-acquisition-command@1"
MIND2WEB_ZIP_PASSWORD_ENV = "SKILLEV_MIND2WEB_ZIP_PASSWORD"  # noqa: S105
ProcessPreparationFactoryLoader = Callable[[str], OfficialProcessPreparationFactory]


def _process_preparation_factory(reference: str) -> OfficialProcessPreparationFactory:
    module_name, separator, symbol_name = reference.partition(":")
    if separator != ":" or not module_name or not symbol_name or ":" in symbol_name:
        raise ValueError("process preparation factory must use the explicit module:symbol form")
    module: ModuleType = importlib.import_module(module_name)
    value = getattr(module, symbol_name)
    if not callable(getattr(value, "build", None)):
        raise TypeError(
            "process preparation factory symbol must implement build(lock, target_root)"
        )
    return cast(OfficialProcessPreparationFactory, value)


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"required file is absent: {path}")
    return path


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"required directory is absent: {path}")
    return path


def _new_receipt_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError(f"receipt parent is absent: {path.parent}")
    if path.exists():
        raise argparse.ArgumentTypeError(f"receipt path already exists: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillev-benchmark-acquire",
        description="Inspect or acquire the exact result-blind benchmark source lock",
    )
    parser.add_argument("--lock", required=True, type=_existing_file)
    parser.add_argument("--target-root", required=True, type=_existing_directory)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status")
    status.add_argument("--verify-digests", action="store_true")

    repositories = commands.add_parser("repositories")
    repositories.add_argument("--receipt", required=True, type=_new_receipt_path)

    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("--receipt", required=True, type=_new_receipt_path)

    prepare_archives = commands.add_parser("prepare-archives")
    prepare_archives.add_argument(
        "--acquisition-receipt",
        required=True,
        type=_existing_file,
    )
    prepare_archives.add_argument("--receipt", required=True, type=_new_receipt_path)

    prepare_non_process = commands.add_parser("prepare-non-process")
    prepare_non_process.add_argument(
        "--acquisition-receipt",
        required=True,
        type=_existing_file,
    )
    prepare_non_process.add_argument(
        "--archive-receipt",
        required=True,
        type=_existing_file,
    )
    prepare_non_process.add_argument("--receipt", required=True, type=_new_receipt_path)

    prepare_process = commands.add_parser("prepare-process")
    prepare_process.add_argument(
        "--acquisition-receipt",
        required=True,
        type=_existing_file,
    )
    prepare_process.add_argument(
        "--archive-receipt",
        required=True,
        type=_existing_file,
    )
    prepare_process.add_argument(
        "--repository-receipt",
        required=True,
        type=_existing_file,
    )
    prepare_process.add_argument(
        "--factory",
        required=True,
        help="explicit private deployment module:symbol",
    )
    prepare_process.add_argument("--receipt", required=True, type=_new_receipt_path)

    freeze_catalog = commands.add_parser("freeze-catalog")
    freeze_catalog.add_argument(
        "--non-process-receipt",
        required=True,
        type=_existing_file,
    )
    freeze_catalog.add_argument(
        "--process-receipt",
        required=True,
        type=_existing_file,
    )
    freeze_catalog.add_argument(
        "--retrieval-index",
        required=True,
        type=_existing_file,
    )
    freeze_catalog.add_argument(
        "--retrieval-manifest",
        required=True,
        type=_existing_file,
    )
    freeze_catalog.add_argument("--output", required=True, type=_new_receipt_path)
    return parser


def acquisition_status(
    *,
    lock_path: Path,
    target_root: Path,
    verify_digests: bool,
) -> dict[str, JsonValue]:
    """Return a content-free snapshot of exact locked-path progress."""

    lock = load_benchmark_acquisition_lock(lock_path)
    progress = inspect_acquisition_progress(
        lock,
        target_root=target_root,
        verify_digests=verify_digests,
    )
    return {
        "acquisition_lock_hash": lock.content_hash,
        "artifact_count": progress.artifact_count,
        "complete_count": progress.complete_count,
        "format": ACQUISITION_COMMAND_FORMAT,
        "invalid_count": progress.invalid_count,
        "known_expected_bytes": progress.known_expected_bytes,
        "operation": "status",
        "present_bytes": progress.present_bytes,
        "target_root": target_root.as_posix(),
        "verified_digests": verify_digests,
    }


def acquire_artifacts(
    *,
    lock_path: Path,
    target_root: Path,
    receipt_path: Path,
    fetcher: LockedArtifactFetcher,
) -> dict[str, JsonValue]:
    """Acquire all locked artifacts and publish one immutable private receipt."""

    lock = load_benchmark_acquisition_lock(lock_path)
    receipt: BenchmarkAcquisitionReceipt = acquire_locked_benchmark_artifacts(
        lock,
        target_root=target_root,
        fetcher=fetcher,
    )
    publish_benchmark_acquisition_receipt(receipt, receipt_path)
    return {
        "acquisition_lock_hash": lock.content_hash,
        "artifact_count": len(receipt.artifacts),
        "format": ACQUISITION_COMMAND_FORMAT,
        "operation": "artifacts",
        "receipt_content_hash": receipt.content_hash,
        "receipt_path": receipt_path.as_posix(),
        "target_root": target_root.as_posix(),
    }


def acquire_repositories(
    *,
    lock_path: Path,
    target_root: Path,
    receipt_path: Path,
    fetcher: LockedRepositoryFetcher,
) -> dict[str, JsonValue]:
    """Acquire all pinned source repositories and publish their private receipt."""

    lock = load_benchmark_acquisition_lock(lock_path)
    receipt: RepositoryAcquisitionReceipt = acquire_locked_benchmark_repositories(
        lock,
        target_root=target_root,
        fetcher=fetcher,
    )
    publish_repository_acquisition_receipt(receipt, receipt_path)
    return {
        "acquisition_lock_hash": lock.content_hash,
        "format": ACQUISITION_COMMAND_FORMAT,
        "operation": "repositories",
        "receipt_content_hash": receipt.content_hash,
        "receipt_path": receipt_path.as_posix(),
        "repository_count": len(receipt.repositories),
        "target_root": target_root.as_posix(),
    }


def prepare_archives(
    *,
    lock_path: Path,
    target_root: Path,
    acquisition_receipt_path: Path,
    receipt_path: Path,
    zip_passwords: dict[Benchmark, bytes],
) -> dict[str, JsonValue]:
    """Prepare all explicitly declared archives from a complete source receipt."""

    lock = load_benchmark_acquisition_lock(lock_path)
    acquisition_receipt = load_benchmark_acquisition_receipt(acquisition_receipt_path)
    receipt: LockedArchiveBatchReceipt = prepare_locked_archives(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        target_root=target_root,
        zip_passwords=zip_passwords,
    )
    publish_locked_archive_batch_receipt(receipt, receipt_path)
    return {
        "acquisition_lock_hash": lock.content_hash,
        "archive_count": len(receipt.archives),
        "format": ACQUISITION_COMMAND_FORMAT,
        "operation": "prepare-archives",
        "receipt_content_hash": receipt.content_hash,
        "receipt_path": receipt_path.as_posix(),
        "target_root": target_root.as_posix(),
    }


def prepare_non_process(
    *,
    lock_path: Path,
    target_root: Path,
    acquisition_receipt_path: Path,
    archive_receipt_path: Path,
    receipt_path: Path,
) -> dict[str, JsonValue]:
    """Prepare and bind all thirteen non-process production sources."""

    lock = load_benchmark_acquisition_lock(lock_path)
    acquisition_receipt = load_benchmark_acquisition_receipt(acquisition_receipt_path)
    archive_receipt = load_locked_archive_batch_receipt(archive_receipt_path)
    receipt: LockedNonProcessPreparationReceipt = prepare_locked_non_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        archive_batch_receipt=archive_receipt,
        target_root=target_root,
    )
    publish_locked_non_process_preparation_receipt(receipt, receipt_path)
    return {
        "acquisition_lock_hash": lock.content_hash,
        "benchmark_count": len(receipt.sources),
        "format": ACQUISITION_COMMAND_FORMAT,
        "operation": "prepare-non-process",
        "receipt_content_hash": receipt.content_hash,
        "receipt_path": receipt_path.as_posix(),
        "target_root": target_root.as_posix(),
    }


def prepare_process(
    *,
    lock_path: Path,
    target_root: Path,
    acquisition_receipt_path: Path,
    archive_receipt_path: Path,
    repository_receipt_path: Path,
    receipt_path: Path,
    factory: OfficialProcessPreparationFactory,
) -> dict[str, JsonValue]:
    """Enumerate and persist all seven pinned official process inventories."""

    lock = load_benchmark_acquisition_lock(lock_path)
    acquisition_receipt = load_benchmark_acquisition_receipt(acquisition_receipt_path)
    archive_receipt = load_locked_archive_batch_receipt(archive_receipt_path)
    repository_receipt = load_repository_acquisition_receipt(repository_receipt_path)
    deployment = factory.build(lock=lock, target_root=target_root)
    if not isinstance(deployment, OfficialProcessPreparationDeployment):
        raise TypeError(
            "process preparation factory must return OfficialProcessPreparationDeployment"
        )
    receipt: LockedProcessPreparationReceipt = prepare_locked_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        archive_batch_receipt=archive_receipt,
        repository_receipt=repository_receipt,
        target_root=target_root,
        inputs=deployment.manifest_inputs(),
    )
    publish_locked_process_preparation_receipt(receipt, receipt_path)
    return {
        "acquisition_lock_hash": lock.content_hash,
        "benchmark_count": len(receipt.sources),
        "format": ACQUISITION_COMMAND_FORMAT,
        "operation": "prepare-process",
        "receipt_content_hash": receipt.content_hash,
        "receipt_path": receipt_path.as_posix(),
        "target_root": target_root.as_posix(),
        "task_count": sum(source.task_count for source in receipt.sources),
    }


def freeze_catalog(
    *,
    lock_path: Path,
    target_root: Path,
    non_process_receipt_path: Path,
    process_receipt_path: Path,
    retrieval_index_path: Path,
    retrieval_manifest_path: Path,
    output_path: Path,
) -> dict[str, JsonValue]:
    """Freeze prepared sources and the locked lexical index into one bundle."""

    lock = load_benchmark_acquisition_lock(lock_path)
    non_process_receipt = load_locked_non_process_preparation_receipt(non_process_receipt_path)
    process_receipt = load_locked_process_preparation_receipt(process_receipt_path)
    if non_process_receipt.target_root != target_root or process_receipt.target_root != target_root:
        raise ValueError("preparation receipts differ from the requested target root")
    bundle = freeze_locked_production_catalog(
        lock=lock,
        non_process_receipt=non_process_receipt,
        process_receipt=process_receipt,
        retrieval_index_path=retrieval_index_path,
        retrieval_manifest_path=retrieval_manifest_path,
    )
    publish_frozen_production_catalog_bundle(bundle, output_path)
    return {
        "acquisition_lock_hash": lock.content_hash,
        "bundle_content_hash": bundle.content_hash,
        "format": ACQUISITION_COMMAND_FORMAT,
        "non_process_benchmark_count": len(bundle.non_process.sources),
        "operation": "freeze-catalog",
        "output_path": output_path.as_posix(),
        "process_benchmark_count": len(bundle.process.sources),
        "retrieval_index_id": bundle.non_process.retrieval_index_id,
        "target_root": target_root.as_posix(),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    process_factory_loader: ProcessPreparationFactoryLoader = _process_preparation_factory,
) -> int:
    args = _parser().parse_args(argv)
    lock_path = cast(Path, args.lock)
    target_root = cast(Path, args.target_root)
    command = cast(str, args.command)
    if command == "status":
        result = acquisition_status(
            lock_path=lock_path,
            target_root=target_root,
            verify_digests=cast(bool, args.verify_digests),
        )
    elif command == "repositories":
        result = acquire_repositories(
            lock_path=lock_path,
            target_root=target_root,
            receipt_path=cast(Path, args.receipt),
            fetcher=GitLockedRepositoryFetcher(),
        )
    elif command == "artifacts":
        result = acquire_artifacts(
            lock_path=lock_path,
            target_root=target_root,
            receipt_path=cast(Path, args.receipt),
            fetcher=CurlLockedArtifactFetcher(),
        )
    elif command == "prepare-archives":
        mind2web_password = os.environ.get(MIND2WEB_ZIP_PASSWORD_ENV)
        lock = load_benchmark_acquisition_lock(lock_path)
        needs_password = any(
            plan.password_required
            and not (target_root / plan.benchmark.value / Path(plan.output_relative_path)).exists()
            for plan in locked_archive_preparations(lock)
        )
        if needs_password and (mind2web_password is None or not mind2web_password):
            raise ValueError(
                f"{MIND2WEB_ZIP_PASSWORD_ENV} must provide the official archive password"
            )
        passwords = (
            {Benchmark.MIND2WEB: mind2web_password.encode()}
            if mind2web_password is not None and mind2web_password
            else {}
        )
        result = prepare_archives(
            lock_path=lock_path,
            target_root=target_root,
            acquisition_receipt_path=cast(Path, args.acquisition_receipt),
            receipt_path=cast(Path, args.receipt),
            zip_passwords=passwords,
        )
    elif command == "prepare-non-process":
        result = prepare_non_process(
            lock_path=lock_path,
            target_root=target_root,
            acquisition_receipt_path=cast(Path, args.acquisition_receipt),
            archive_receipt_path=cast(Path, args.archive_receipt),
            receipt_path=cast(Path, args.receipt),
        )
    elif command == "prepare-process":
        result = prepare_process(
            lock_path=lock_path,
            target_root=target_root,
            acquisition_receipt_path=cast(Path, args.acquisition_receipt),
            archive_receipt_path=cast(Path, args.archive_receipt),
            repository_receipt_path=cast(Path, args.repository_receipt),
            receipt_path=cast(Path, args.receipt),
            factory=process_factory_loader(cast(str, args.factory)),
        )
    elif command == "freeze-catalog":
        result = freeze_catalog(
            lock_path=lock_path,
            target_root=target_root,
            non_process_receipt_path=cast(Path, args.non_process_receipt),
            process_receipt_path=cast(Path, args.process_receipt),
            retrieval_index_path=cast(Path, args.retrieval_index),
            retrieval_manifest_path=cast(Path, args.retrieval_manifest),
            output_path=cast(Path, args.output),
        )
    else:  # pragma: no cover - argparse owns the closed command set
        raise AssertionError(f"unhandled acquisition command {command!r}")
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACQUISITION_COMMAND_FORMAT",
    "MIND2WEB_ZIP_PASSWORD_ENV",
    "ProcessPreparationFactoryLoader",
    "acquire_artifacts",
    "acquire_repositories",
    "acquisition_status",
    "freeze_catalog",
    "main",
    "prepare_archives",
    "prepare_non_process",
    "prepare_process",
]
