"""Acquire the exact Git source revisions named by the benchmark lock."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from skillev.contracts import (
    JsonValue,
    canonical_json,
    normalize_json,
    stable_hash,
    validate_sha256,
)
from skillev.experiments import Benchmark

from .acquisition import (
    BenchmarkAcquisitionLock,
    LockedBenchmarkSource,
    _read_published_canonical_record,
)

_GIT_SOURCE_KINDS = frozenset(
    {
        "pinned-atlas-preparation",
        "pinned-git-dataset",
        "pinned-git-environment",
        "pinned-git-file",
        "pinned-huggingface-environment",
    }
)


class LockedRepositoryFetcher(Protocol):
    """Materialize one exact repository revision into an unpublished path."""

    def fetch(
        self,
        *,
        source: LockedBenchmarkSource,
        destination: Path,
    ) -> None: ...


def _git(
    *arguments: str,
    cwd: Path | None = None,
) -> str:
    result = subprocess.run(  # noqa: S603 - exact Git operation, no shell
        ("git", *arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_checkout(source: LockedBenchmarkSource, path: Path) -> None:
    if not (path / ".git").is_dir():
        raise ValueError("repository checkout is not a Git worktree")
    if _git("rev-parse", "HEAD", cwd=path) != source.revision:
        raise ValueError("repository checkout differs from the locked revision")
    if _git("remote", "get-url", "origin", cwd=path) != source.repository:
        raise ValueError("repository checkout differs from the locked source URL")
    if _git("status", "--porcelain", "--untracked-files=all", cwd=path):
        raise ValueError("repository checkout contains uncommitted source changes")


@dataclass(frozen=True, slots=True)
class GitLockedRepositoryFetcher:
    """One-attempt, resumable Git fetch with no branch or revision fallback."""

    executable: str = "git"

    def _run(self, destination: Path, *arguments: str) -> None:
        subprocess.run(  # noqa: S603 - exact executable and args are explicit
            (self.executable, "-C", str(destination), *arguments),
            check=True,
        )

    def fetch(
        self,
        *,
        source: LockedBenchmarkSource,
        destination: Path,
    ) -> None:
        if destination.exists():
            if not (destination / ".git").is_dir():
                raise ValueError("partial repository path is not a Git worktree")
            if _git("remote", "get-url", "origin", cwd=destination) != source.repository:
                raise ValueError("partial repository has another origin")
        else:
            subprocess.run(  # noqa: S603 - exact executable and args are explicit
                (self.executable, "init", str(destination)),
                check=True,
            )
            self._run(destination, "remote", "add", "origin", source.repository)
        self._run(destination, "fetch", "--depth=1", "origin", source.revision)
        self._run(destination, "checkout", "--detach", "FETCH_HEAD")
        _verify_checkout(source, destination)


@dataclass(frozen=True, slots=True)
class AcquiredRepositoryReceipt:
    """Measured source checkout used to prepare one benchmark."""

    benchmark: Benchmark
    repository: str
    revision: str
    relative_checkout_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("repository receipt benchmark must be Benchmark")
        if not self.repository.startswith("https://"):
            raise ValueError("repository receipt URL must use HTTPS")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("repository receipt revision must be a full lowercase commit")
        path = Path(self.relative_checkout_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_checkout_path
        ):
            raise ValueError("repository receipt checkout path must be normalized and relative")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "relative_checkout_path": self.relative_checkout_path,
            "repository": self.repository,
            "revision": self.revision,
        }

    @classmethod
    def from_value(cls, value: object) -> AcquiredRepositoryReceipt:
        normalized = normalize_json(value)
        fields = {
            "benchmark",
            "relative_checkout_path",
            "repository",
            "revision",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("acquired repository receipt has an invalid field set")
        values = tuple(normalized[field] for field in fields)
        if any(type(item) is not str for item in values):
            raise TypeError("acquired repository receipt fields must be text")
        return cls(
            benchmark=Benchmark(cast(str, normalized["benchmark"])),
            repository=cast(str, normalized["repository"]),
            revision=cast(str, normalized["revision"]),
            relative_checkout_path=cast(str, normalized["relative_checkout_path"]),
        )


@dataclass(frozen=True, slots=True)
class RepositoryAcquisitionReceipt:
    """Private receipt for every Git source required by the acquisition lock."""

    acquisition_lock_hash: str
    target_root: Path
    repositories: tuple[AcquiredRepositoryReceipt, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.acquisition_lock_hash)
        if not isinstance(self.target_root, Path) or not self.target_root.is_absolute():
            raise ValueError("repository receipt target_root must be an absolute Path")
        if not self.repositories or any(
            not isinstance(repository, AcquiredRepositoryReceipt)
            for repository in self.repositories
        ):
            raise ValueError("repository receipt must contain acquired repositories")
        benchmarks = tuple(repository.benchmark for repository in self.repositories)
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError("repository receipt contains duplicate benchmark identities")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "acquisition_lock_hash": self.acquisition_lock_hash,
            "format": "skillev-private-repository-acquisition-receipt@1",
            "repositories": [repository.to_value() for repository in self.repositories],
            "target_root": self.target_root.as_posix(),
        }

    @classmethod
    def from_value(cls, value: object) -> RepositoryAcquisitionReceipt:
        normalized = normalize_json(value)
        fields = {
            "acquisition_lock_hash",
            "format",
            "repositories",
            "target_root",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("repository acquisition receipt has an invalid field set")
        if normalized["format"] != "skillev-private-repository-acquisition-receipt@1":
            raise ValueError("unsupported repository acquisition receipt format")
        raw_repositories = normalized["repositories"]
        if not isinstance(raw_repositories, list):
            raise TypeError("repository acquisition receipt repositories must be an array")
        lock_hash = normalized["acquisition_lock_hash"]
        target_root = normalized["target_root"]
        if type(lock_hash) is not str or type(target_root) is not str:
            raise TypeError("repository acquisition receipt identity fields must be text")
        return cls(
            acquisition_lock_hash=lock_hash,
            target_root=Path(target_root),
            repositories=tuple(
                AcquiredRepositoryReceipt.from_value(repository) for repository in raw_repositories
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


def acquire_locked_benchmark_repositories(
    lock: BenchmarkAcquisitionLock,
    *,
    target_root: Path,
    fetcher: LockedRepositoryFetcher,
) -> RepositoryAcquisitionReceipt:
    """Acquire each required repository once and retain its clean pinned tree."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    if not target_root.is_dir():
        raise NotADirectoryError(target_root)
    if not callable(getattr(fetcher, "fetch", None)):
        raise TypeError("fetcher must implement LockedRepositoryFetcher")

    receipts: list[AcquiredRepositoryReceipt] = []
    for entry in lock.benchmarks:
        source = entry.source
        if source.kind not in _GIT_SOURCE_KINDS:
            continue
        relative = f"{entry.benchmark.value}/repository"
        destination = target_root / relative
        if destination.is_dir():
            _verify_checkout(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            fetcher.fetch(source=source, destination=partial)
            _verify_checkout(source, partial)
            if destination.exists():
                raise FileExistsError(destination)
            partial.rename(destination)
        receipts.append(
            AcquiredRepositoryReceipt(
                benchmark=entry.benchmark,
                repository=source.repository,
                revision=source.revision,
                relative_checkout_path=relative,
            )
        )
    return RepositoryAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=target_root,
        repositories=tuple(receipts),
    )


def publish_repository_acquisition_receipt(
    receipt: RepositoryAcquisitionReceipt,
    output_path: Path,
) -> None:
    """Persist one canonical private receipt without replacing prior evidence."""

    if not isinstance(receipt, RepositoryAcquisitionReceipt):
        raise TypeError("receipt must be RepositoryAcquisitionReceipt")
    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("output_path must be an absolute Path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(receipt.to_value()))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_repository_acquisition_receipt(path: Path) -> RepositoryAcquisitionReceipt:
    """Load one canonical private pinned-repository receipt."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("repository receipt path must be an absolute Path")
    return RepositoryAcquisitionReceipt.from_value(
        _read_published_canonical_record(path, label="repository acquisition receipt")
    )


__all__ = [
    "AcquiredRepositoryReceipt",
    "GitLockedRepositoryFetcher",
    "LockedRepositoryFetcher",
    "RepositoryAcquisitionReceipt",
    "acquire_locked_benchmark_repositories",
    "load_repository_acquisition_receipt",
    "publish_repository_acquisition_receipt",
]
