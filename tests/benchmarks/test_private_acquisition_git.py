from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from skillev_private.benchmarks.acquisition import (
    BenchmarkAcquisitionLock,
    LockedBenchmarkSource,
)
from skillev_private.benchmarks.acquisition_git import (
    acquire_locked_benchmark_repositories,
    load_repository_acquisition_receipt,
    publish_repository_acquisition_receipt,
)


def _git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(  # noqa: S603 - isolated fixture repositories and args
        ("git", *arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_repository(root: Path) -> tuple[Path, str]:
    repository = root / "fixture-source"
    repository.mkdir()
    _git("init", "-q", cwd=repository)
    _git("config", "user.email", "fixture@example.invalid", cwd=repository)
    _git("config", "user.name", "Fixture", cwd=repository)
    (repository / "source.txt").write_text("pinned source\n", encoding="utf-8")
    _git("add", "source.txt", cwd=repository)
    _git("commit", "-q", "-m", "fixture", cwd=repository)
    return repository, _git("rev-parse", "HEAD", cwd=repository)


def _fixture_lock(revision: str) -> BenchmarkAcquisitionLock:
    lock_path = Path(__file__).parents[2] / "benchmark-acquisition-lock.json"
    value = copy.deepcopy(json.loads(lock_path.read_text(encoding="utf-8")))
    entries = cast(list[dict[str, object]], value["benchmarks"])
    for entry in entries:
        source = cast(dict[str, object], entry["source"])
        if (
            str(source["kind"]).startswith("pinned-git")
            or source["kind"] == "pinned-atlas-preparation"
            or source["kind"] == "pinned-huggingface-environment"
        ):
            source["revision"] = revision
    return BenchmarkAcquisitionLock.from_value(value)


@dataclass
class _FixtureRepositoryFetcher:
    source_repository: Path
    calls: list[str] = field(default_factory=list)

    def fetch(
        self,
        *,
        source: LockedBenchmarkSource,
        destination: Path,
    ) -> None:
        self.calls.append(destination.parent.name)
        _git("clone", "-q", str(self.source_repository), str(destination))
        _git("remote", "set-url", "origin", source.repository, cwd=destination)
        _git("checkout", "--detach", source.revision, cwd=destination)


def test_acquires_each_required_git_source_once_and_publishes_receipt(
    tmp_path: Path,
) -> None:
    source_repository, revision = _fixture_repository(tmp_path)
    lock = _fixture_lock(revision)
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    first_fetcher = _FixtureRepositoryFetcher(source_repository)

    first = acquire_locked_benchmark_repositories(
        lock,
        target_root=target,
        fetcher=first_fetcher,
    )
    second_fetcher = _FixtureRepositoryFetcher(source_repository)
    second = acquire_locked_benchmark_repositories(
        lock,
        target_root=target,
        fetcher=second_fetcher,
    )

    assert len(first.repositories) == len(second.repositories) == 9
    assert len(first_fetcher.calls) == 9
    assert second_fetcher.calls == []
    assert first.to_value() == second.to_value()
    output = (tmp_path / "repository-receipt.json").resolve()
    publish_repository_acquisition_receipt(first, output)
    original = output.read_bytes()
    assert load_repository_acquisition_receipt(output) == first
    with pytest.raises(FileExistsError):
        publish_repository_acquisition_receipt(second, output)
    assert output.read_bytes() == original


def test_existing_dirty_or_wrong_revision_checkout_is_rejected(tmp_path: Path) -> None:
    source_repository, revision = _fixture_repository(tmp_path)
    lock = _fixture_lock(revision)
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    receipt = acquire_locked_benchmark_repositories(
        lock,
        target_root=target,
        fetcher=_FixtureRepositoryFetcher(source_repository),
    )
    checkout = target / receipt.repositories[0].benchmark.value / "repository"
    (checkout / "source.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError):
        acquire_locked_benchmark_repositories(
            lock,
            target_root=target,
            fetcher=_FixtureRepositoryFetcher(source_repository),
        )
