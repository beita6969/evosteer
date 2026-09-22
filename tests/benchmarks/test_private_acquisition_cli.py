from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from skillev_private.benchmarks import acquisition_cli
from skillev_private.benchmarks.acquisition import (
    AcquiredArtifactReceipt,
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
    LockedAcquisitionArtifact,
    LockedBenchmarkSource,
)
from skillev_private.benchmarks.acquisition_git import (
    AcquiredRepositoryReceipt,
    RepositoryAcquisitionReceipt,
)
from skillev_private.benchmarks.non_process_preparation import (
    LockedNonProcessPreparationReceipt,
    locked_non_process_source_plans,
)
from skillev_private.benchmarks.official_process_preparation import (
    OfficialProcessPreparationDeployment,
)

from skillev.experiments import Benchmark


def _lock_path() -> Path:
    return (Path(__file__).parents[2] / "benchmark-acquisition-lock.json").resolve()


class _UnusedArtifactFetcher:
    def fetch(
        self,
        *,
        benchmark: Benchmark,
        source: LockedBenchmarkSource,
        artifact: LockedAcquisitionArtifact,
        destination: Path,
    ) -> None:
        raise AssertionError((benchmark, source, artifact, destination))


class _UnusedRepositoryFetcher:
    def fetch(
        self,
        *,
        source: LockedBenchmarkSource,
        destination: Path,
    ) -> None:
        raise AssertionError((source, destination))


class _UnusedProcessPreparationFactory:
    def build(
        self,
        *,
        lock: BenchmarkAcquisitionLock,
        target_root: Path,
    ) -> OfficialProcessPreparationDeployment:
        raise AssertionError((lock, target_root))


def test_process_factory_loader_requires_one_explicit_module_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _UnusedProcessPreparationFactory()
    module = ModuleType("private_process_deployment")
    module.__dict__["FACTORY"] = factory
    monkeypatch.setattr(
        acquisition_cli.importlib,
        "import_module",
        lambda module_name: module,
    )

    loaded = acquisition_cli._process_preparation_factory("private_process_deployment:FACTORY")

    assert loaded is factory
    with pytest.raises(ValueError):
        acquisition_cli._process_preparation_factory("private_process_deployment")


def test_status_reports_only_content_free_locked_path_progress(tmp_path: Path) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()

    status = acquisition_cli.acquisition_status(
        lock_path=_lock_path(),
        target_root=target,
        verify_digests=False,
    )

    assert status["operation"] == "status"
    assert status["artifact_count"] == 28
    assert status["complete_count"] == 0
    assert status["invalid_count"] == 0
    assert status["present_bytes"] == 0
    assert cast(int, status["known_expected_bytes"]) > 0
    assert "benchmarks" not in status


def test_artifact_command_routes_explicit_paths_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    receipt_path = (tmp_path / "artifact-receipt.json").resolve()
    captured: dict[str, object] = {}

    def fake_acquire(
        lock: BenchmarkAcquisitionLock,
        *,
        target_root: Path,
        fetcher: object,
    ) -> BenchmarkAcquisitionReceipt:
        captured.update(lock=lock, target_root=target_root, fetcher=fetcher)
        return BenchmarkAcquisitionReceipt(
            acquisition_lock_hash=lock.content_hash,
            target_root=target_root,
            artifacts=(
                AcquiredArtifactReceipt(
                    benchmark=Benchmark.AIME_2026,
                    locator="https://example.invalid/pinned",
                    relative_path="data/pinned.parquet",
                    size_bytes=7,
                    sha256="0" * 64,
                ),
            ),
        )

    monkeypatch.setattr(acquisition_cli, "acquire_locked_benchmark_artifacts", fake_acquire)
    fetcher = _UnusedArtifactFetcher()
    result = acquisition_cli.acquire_artifacts(
        lock_path=_lock_path(),
        target_root=target,
        receipt_path=receipt_path,
        fetcher=fetcher,
    )

    assert captured["target_root"] == target
    assert captured["fetcher"] is fetcher
    assert result["artifact_count"] == 1
    assert result["receipt_path"] == receipt_path.as_posix()
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["artifacts"][0]["benchmark"] == Benchmark.AIME_2026.value
    original = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        acquisition_cli.acquire_artifacts(
            lock_path=_lock_path(),
            target_root=target,
            receipt_path=receipt_path,
            fetcher=fetcher,
        )
    assert receipt_path.read_bytes() == original


def test_repository_command_routes_explicit_paths_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    receipt_path = (tmp_path / "repository-receipt.json").resolve()
    captured: dict[str, object] = {}

    def fake_acquire(
        lock: BenchmarkAcquisitionLock,
        *,
        target_root: Path,
        fetcher: object,
    ) -> RepositoryAcquisitionReceipt:
        captured.update(lock=lock, target_root=target_root, fetcher=fetcher)
        return RepositoryAcquisitionReceipt(
            acquisition_lock_hash=lock.content_hash,
            target_root=target_root,
            repositories=(
                AcquiredRepositoryReceipt(
                    benchmark=Benchmark.WEBSHOP,
                    repository="https://example.invalid/pinned.git",
                    revision="1" * 40,
                    relative_checkout_path="webshop/repository",
                ),
            ),
        )

    monkeypatch.setattr(acquisition_cli, "acquire_locked_benchmark_repositories", fake_acquire)
    fetcher = _UnusedRepositoryFetcher()
    result = acquisition_cli.acquire_repositories(
        lock_path=_lock_path(),
        target_root=target,
        receipt_path=receipt_path,
        fetcher=fetcher,
    )

    assert captured["target_root"] == target
    assert captured["fetcher"] is fetcher
    assert result["repository_count"] == 1
    assert result["receipt_path"] == receipt_path.as_posix()
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["repositories"][0]["benchmark"] == Benchmark.WEBSHOP.value
    original = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        acquisition_cli.acquire_repositories(
            lock_path=_lock_path(),
            target_root=target,
            receipt_path=receipt_path,
            fetcher=fetcher,
        )
    assert receipt_path.read_bytes() == original


def test_non_process_command_routes_both_source_receipts_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    output = (tmp_path / "non-process-receipt.json").resolve()
    lock = acquisition_cli.load_benchmark_acquisition_lock(_lock_path())
    receipt = LockedNonProcessPreparationReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=f"sha256:{'a' * 64}",
        archive_batch_receipt_hash=f"sha256:{'b' * 64}",
        atlas_preparation_content_hash=f"sha256:{'c' * 64}",
        medqa_preparation_content_hash=f"sha256:{'d' * 64}",
        external_preparation_content_hash=f"sha256:{'e' * 64}",
        target_root=target,
        sources=locked_non_process_source_plans(lock),
    )
    acquired = object()
    archives = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        acquisition_cli,
        "load_benchmark_acquisition_receipt",
        lambda path: acquired,
    )
    monkeypatch.setattr(
        acquisition_cli,
        "load_locked_archive_batch_receipt",
        lambda path: archives,
    )

    def fake_prepare(**kwargs: object) -> LockedNonProcessPreparationReceipt:
        captured.update(kwargs)
        return receipt

    monkeypatch.setattr(
        acquisition_cli,
        "prepare_locked_non_process_benchmarks",
        fake_prepare,
    )
    result = acquisition_cli.prepare_non_process(
        lock_path=_lock_path(),
        target_root=target,
        acquisition_receipt_path=tmp_path / "acquired.json",
        archive_receipt_path=tmp_path / "archives.json",
        receipt_path=output,
    )

    assert captured == {
        "lock": lock,
        "acquisition_receipt": acquired,
        "archive_batch_receipt": archives,
        "target_root": target,
    }
    assert result["benchmark_count"] == 13
    assert result["operation"] == "prepare-non-process"
    assert json.loads(output.read_text(encoding="utf-8"))["content_hash"] == (receipt.content_hash)


def test_prepare_archives_main_reads_password_without_printing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    acquired = tmp_path / "acquired.json"
    acquired.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "archives.json"
    secret = "fixture-cli-secret"
    captured: dict[str, object] = {}

    def fake_prepare_archives(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "format": acquisition_cli.ACQUISITION_COMMAND_FORMAT,
            "operation": "prepare-archives",
        }

    monkeypatch.setattr(acquisition_cli, "prepare_archives", fake_prepare_archives)
    monkeypatch.setenv(acquisition_cli.MIND2WEB_ZIP_PASSWORD_ENV, secret)

    assert (
        acquisition_cli.main(
            (
                "--lock",
                str(_lock_path()),
                "--target-root",
                str(target),
                "prepare-archives",
                "--acquisition-receipt",
                str(acquired),
                "--receipt",
                str(receipt),
            )
        )
        == 0
    )

    assert captured["zip_passwords"] == {Benchmark.MIND2WEB: secret.encode()}
    assert secret not in capsys.readouterr().out


def test_prepare_archives_main_does_not_require_password_for_verified_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    for plan in acquisition_cli.locked_archive_preparations(
        acquisition_cli.load_benchmark_acquisition_lock(_lock_path())
    ):
        if plan.password_required:
            (target / plan.benchmark.value / plan.output_relative_path).mkdir(parents=True)
    acquired = tmp_path / "acquired.json"
    acquired.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "archives.json"
    captured: dict[str, object] = {}

    def fake_prepare_archives(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "format": acquisition_cli.ACQUISITION_COMMAND_FORMAT,
            "operation": "prepare-archives",
        }

    monkeypatch.setattr(acquisition_cli, "prepare_archives", fake_prepare_archives)
    monkeypatch.delenv(acquisition_cli.MIND2WEB_ZIP_PASSWORD_ENV, raising=False)

    assert (
        acquisition_cli.main(
            (
                "--lock",
                str(_lock_path()),
                "--target-root",
                str(target),
                "prepare-archives",
                "--acquisition-receipt",
                str(acquired),
                "--receipt",
                str(receipt),
            )
        )
        == 0
    )
    assert captured["zip_passwords"] == {}
    assert capsys.readouterr().out


def test_main_status_emits_one_canonical_json_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()

    assert (
        acquisition_cli.main(
            (
                "--lock",
                str(_lock_path()),
                "--target-root",
                str(target),
                "status",
            )
        )
        == 0
    )

    output = capsys.readouterr().out
    value = json.loads(output)
    assert value["operation"] == "status"
    assert value["target_root"] == target.as_posix()


def test_freeze_catalog_main_routes_every_explicit_preparation_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    inputs = tuple(
        (tmp_path / name).resolve()
        for name in (
            "non-process.json",
            "process.json",
            "retrieval.sqlite3",
            "retrieval.manifest.json",
        )
    )
    for path in inputs:
        path.write_text("{}\n", encoding="utf-8")
    output = (tmp_path / "production-catalog.json").resolve()
    captured: dict[str, object] = {}

    def fake_freeze_catalog(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "format": acquisition_cli.ACQUISITION_COMMAND_FORMAT,
            "operation": "freeze-catalog",
        }

    monkeypatch.setattr(acquisition_cli, "freeze_catalog", fake_freeze_catalog)
    assert (
        acquisition_cli.main(
            (
                "--lock",
                str(_lock_path()),
                "--target-root",
                str(target),
                "freeze-catalog",
                "--non-process-receipt",
                str(inputs[0]),
                "--process-receipt",
                str(inputs[1]),
                "--retrieval-index",
                str(inputs[2]),
                "--retrieval-manifest",
                str(inputs[3]),
                "--output",
                str(output),
            )
        )
        == 0
    )

    assert captured == {
        "lock_path": _lock_path(),
        "target_root": target,
        "non_process_receipt_path": inputs[0],
        "process_receipt_path": inputs[1],
        "retrieval_index_path": inputs[2],
        "retrieval_manifest_path": inputs[3],
        "output_path": output,
    }
    assert json.loads(capsys.readouterr().out)["operation"] == "freeze-catalog"


def test_prepare_process_main_loads_one_explicit_factory_and_routes_all_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    inputs = tuple(
        (tmp_path / name).resolve()
        for name in (
            "acquired.json",
            "archives.json",
            "repositories.json",
        )
    )
    for path in inputs:
        path.write_text("{}\n", encoding="utf-8")
    output = (tmp_path / "process-preparation.json").resolve()
    factory = _UnusedProcessPreparationFactory()
    loaded: list[str] = []
    captured: dict[str, object] = {}

    def load_factory(reference: str) -> _UnusedProcessPreparationFactory:
        loaded.append(reference)
        return factory

    def fake_prepare_process(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "format": acquisition_cli.ACQUISITION_COMMAND_FORMAT,
            "operation": "prepare-process",
        }

    monkeypatch.setattr(acquisition_cli, "prepare_process", fake_prepare_process)
    assert (
        acquisition_cli.main(
            (
                "--lock",
                str(_lock_path()),
                "--target-root",
                str(target),
                "prepare-process",
                "--acquisition-receipt",
                str(inputs[0]),
                "--archive-receipt",
                str(inputs[1]),
                "--repository-receipt",
                str(inputs[2]),
                "--factory",
                "private_process_deployment:FACTORY",
                "--receipt",
                str(output),
            ),
            process_factory_loader=load_factory,
        )
        == 0
    )

    assert loaded == ["private_process_deployment:FACTORY"]
    assert captured == {
        "lock_path": _lock_path(),
        "target_root": target,
        "acquisition_receipt_path": inputs[0],
        "archive_receipt_path": inputs[1],
        "repository_receipt_path": inputs[2],
        "receipt_path": output,
        "factory": factory,
    }
    assert json.loads(capsys.readouterr().out)["operation"] == "prepare-process"
