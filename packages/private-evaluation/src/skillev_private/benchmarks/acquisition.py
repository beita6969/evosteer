"""Strict, answer-free acquisition metadata and local progress inspection.

The committed acquisition lock chooses every source before benchmark results
exist.  This module turns that lock into typed metadata and inspects only the
declared local artifact paths.  It does not download, discover, extract, or
interpret benchmark records.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast
from urllib.parse import quote, urlencode, urlparse

from skillev.contracts import (
    JsonValue,
    canonical_json,
    normalize_json,
    parse_canonical_json,
    stable_hash,
    validate_sha256,
)
from skillev.experiments import BENCHMARK_SPECS, FIXED_SEED, Benchmark

ACQUISITION_LOCK_FORMAT = "skillev-benchmark-acquisition-lock@1"
ACQUISITION_PRESERVATION_RULE = "preserve-every-source-and-archive"
_SHA256_HEX_LENGTH = 64
_GIT_REVISION_LENGTH = 40


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an invalid field set")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _lower_hex(value: object, *, length: int, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    return text


def _relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
        or any(marker in text for marker in ("*", "?", "["))
    ):
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    return text


def _read_published_canonical_record(path: Path, *, label: str) -> JsonValue:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} must end with exactly one newline")
    try:
        text = raw[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    value = parse_canonical_json(text)
    if raw != (canonical_json(value) + "\n").encode():
        raise ValueError(f"{label} is not a canonical published record")
    return value


@dataclass(frozen=True, slots=True)
class LockedAcquisitionArtifact:
    """One exact logical artifact from the committed acquisition lock."""

    locator: str
    relative_path: str
    expected_size: int | None
    expected_sha256: str | None

    def __post_init__(self) -> None:
        _text(self.locator, field="artifact locator")
        _relative_path(self.relative_path, field="artifact relative_path")
        if self.expected_size is not None and (
            type(self.expected_size) is not int or self.expected_size <= 0
        ):
            raise ValueError("artifact expected_size must be a positive integer or null")
        if self.expected_sha256 is not None:
            _lower_hex(
                self.expected_sha256,
                length=_SHA256_HEX_LENGTH,
                field="artifact expected_sha256",
            )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "expected_sha256": self.expected_sha256,
            "expected_size": self.expected_size,
            "locator": self.locator,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_value(cls, value: object) -> LockedAcquisitionArtifact:
        data = _object(
            value,
            fields=frozenset(
                {
                    "expected_sha256",
                    "expected_size",
                    "locator",
                    "relative_path",
                }
            ),
            label="acquisition artifact",
        )
        expected_size = data["expected_size"]
        expected_sha256 = data["expected_sha256"]
        if expected_size is not None and type(expected_size) is not int:
            raise ValueError("artifact expected_size must be an integer or null")
        if expected_sha256 is not None and type(expected_sha256) is not str:
            raise ValueError("artifact expected_sha256 must be text or null")
        return cls(
            locator=_text(data["locator"], field="artifact locator"),
            relative_path=_relative_path(
                data["relative_path"],
                field="artifact relative_path",
            ),
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )


@dataclass(frozen=True, slots=True)
class LockedBenchmarkSource:
    """Pinned repository plus its explicitly named downloadable artifacts."""

    kind: str
    repository: str
    revision: str
    artifacts: tuple[LockedAcquisitionArtifact, ...]
    prepared_files: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.kind, field="source kind")
        repository = _text(self.repository, field="source repository")
        if not repository.startswith("https://"):
            raise ValueError("source repository must use HTTPS")
        _lower_hex(
            self.revision,
            length=_GIT_REVISION_LENGTH,
            field="source revision",
        )
        if not self.artifacts:
            raise ValueError("source must declare at least one artifact")
        if any(not isinstance(item, LockedAcquisitionArtifact) for item in self.artifacts):
            raise TypeError("source artifacts contain an incompatible value")
        if not self.prepared_files:
            raise ValueError("source must declare at least one prepared file")
        for relative_path in self.prepared_files:
            _relative_path(relative_path, field="prepared file")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifacts": [artifact.to_value() for artifact in self.artifacts],
            "kind": self.kind,
            "prepared_files": list(self.prepared_files),
            "repository": self.repository,
            "revision": self.revision,
        }

    @classmethod
    def from_value(cls, value: object) -> LockedBenchmarkSource:
        data = _object(
            value,
            fields=frozenset(
                {
                    "artifacts",
                    "kind",
                    "prepared_files",
                    "repository",
                    "revision",
                }
            ),
            label="benchmark acquisition source",
        )
        raw_artifacts = data["artifacts"]
        raw_prepared = data["prepared_files"]
        if not isinstance(raw_artifacts, list):
            raise ValueError("source artifacts must be an array")
        if not isinstance(raw_prepared, list) or any(
            type(item) is not str for item in raw_prepared
        ):
            raise ValueError("source prepared_files must be a text array")
        return cls(
            kind=_text(data["kind"], field="source kind"),
            repository=_text(data["repository"], field="source repository"),
            revision=_lower_hex(
                data["revision"],
                length=_GIT_REVISION_LENGTH,
                field="source revision",
            ),
            artifacts=tuple(LockedAcquisitionArtifact.from_value(item) for item in raw_artifacts),
            prepared_files=tuple(cast(list[str], raw_prepared)),
        )


@dataclass(frozen=True, slots=True)
class LockedBenchmarkAcquisition:
    """One benchmark's exact pre-result source choice."""

    benchmark: Benchmark
    source: LockedBenchmarkSource


@dataclass(frozen=True, slots=True)
class BenchmarkAcquisitionLock:
    """Typed acquisition authority for the complete active benchmark suite."""

    benchmarks: tuple[LockedBenchmarkAcquisition, ...]
    shared_infrastructure: dict[str, JsonValue]

    def __post_init__(self) -> None:
        expected = tuple(spec.benchmark for spec in BENCHMARK_SPECS)
        if tuple(item.benchmark for item in self.benchmarks) != expected:
            raise ValueError("acquisition benchmarks must match the protocol order")

    @classmethod
    def from_value(cls, value: object) -> BenchmarkAcquisitionLock:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmarks",
                    "fixed_seed",
                    "format",
                    "preservation",
                    "shared_infrastructure",
                }
            ),
            label="benchmark acquisition lock",
        )
        if data["format"] != ACQUISITION_LOCK_FORMAT:
            raise ValueError("unsupported acquisition lock format")
        if data["fixed_seed"] != FIXED_SEED:
            raise ValueError("acquisition lock differs from the preregistered seed")
        if data["preservation"] != ACQUISITION_PRESERVATION_RULE:
            raise ValueError("acquisition lock differs from the preservation rule")
        raw_benchmarks = data["benchmarks"]
        infrastructure = data["shared_infrastructure"]
        if not isinstance(raw_benchmarks, list):
            raise ValueError("acquisition benchmarks must be an array")
        if not isinstance(infrastructure, dict):
            raise ValueError("shared infrastructure must be an object")
        if len(raw_benchmarks) != len(BENCHMARK_SPECS):
            raise ValueError("acquisition lock must cover every protocol benchmark")

        entries: list[LockedBenchmarkAcquisition] = []
        for raw_entry, protocol in zip(raw_benchmarks, BENCHMARK_SPECS, strict=True):
            entry = _object(
                raw_entry,
                fields=frozenset(
                    {
                        "benchmark",
                        "role",
                        "sampling",
                        "source",
                        "training_use",
                    }
                ),
                label="benchmark acquisition entry",
            )
            if (
                entry["benchmark"] != protocol.benchmark.value
                or entry["role"] != protocol.role.value
                or entry["training_use"] != protocol.training_use.value
                or entry["sampling"] != protocol.evaluation_sampling.to_value()
            ):
                raise ValueError("acquisition entry differs from the benchmark protocol")
            entries.append(
                LockedBenchmarkAcquisition(
                    benchmark=protocol.benchmark,
                    source=LockedBenchmarkSource.from_value(entry["source"]),
                )
            )
        return cls(tuple(entries), dict(infrastructure))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmarks": [
                {
                    "benchmark": item.benchmark.value,
                    "role": protocol.role.value,
                    "sampling": protocol.evaluation_sampling.to_value(),
                    "source": item.source.to_value(),
                    "training_use": protocol.training_use.value,
                }
                for item, protocol in zip(
                    self.benchmarks,
                    BENCHMARK_SPECS,
                    strict=True,
                )
            ],
            "fixed_seed": FIXED_SEED,
            "format": ACQUISITION_LOCK_FORMAT,
            "preservation": ACQUISITION_PRESERVATION_RULE,
            "shared_infrastructure": self.shared_infrastructure,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class ArtifactAcquisitionStatus:
    """Content-free local status for one locked artifact."""

    benchmark: Benchmark
    relative_path: str
    local_path: Path
    expected_size: int | None
    exists: bool
    size_bytes: int | None
    sha256: str | None
    size_matches: bool | None
    sha256_matches: bool | None

    @property
    def complete(self) -> bool:
        return (
            self.exists
            and self.size_bytes is not None
            and self.sha256 is not None
            and self.size_matches is not False
            and self.sha256_matches is not False
        )


@dataclass(frozen=True, slots=True)
class AcquisitionProgress:
    """Aggregate progress over all exact artifact paths in the lock."""

    artifacts: tuple[ArtifactAcquisitionStatus, ...]

    @property
    def artifact_count(self) -> int:
        return len(self.artifacts)

    @property
    def complete_count(self) -> int:
        return sum(item.complete for item in self.artifacts)

    @property
    def present_bytes(self) -> int:
        return sum(item.size_bytes or 0 for item in self.artifacts)

    @property
    def known_expected_bytes(self) -> int:
        return sum(item.expected_size or 0 for item in self.artifacts)

    @property
    def invalid_count(self) -> int:
        return sum(
            item.exists and (item.size_matches is False or item.sha256_matches is False)
            for item in self.artifacts
        )


class LockedArtifactFetcher(Protocol):
    """Fetch one locked artifact into its resumable, unpublished path."""

    def fetch(
        self,
        *,
        benchmark: Benchmark,
        source: LockedBenchmarkSource,
        artifact: LockedAcquisitionArtifact,
        destination: Path,
    ) -> None: ...


def _huggingface_url(locator: str) -> str:
    payload = locator.removeprefix("hf:")
    repository_revision, separator, relative_path = payload.partition(":")
    repository, revision_separator, revision = repository_revision.rpartition("@")
    if (
        not separator
        or not revision_separator
        or not repository
        or not revision
        or not relative_path
    ):
        raise ValueError("Hugging Face locator is malformed")
    return (
        "https://huggingface.co/datasets/"
        f"{quote(repository, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(relative_path, safe='/')}?download=true"
    )


def _gdrive_url(locator: str) -> str:
    file_id = locator.removeprefix("gdrive:")
    if not file_id or any(
        character not in "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for character in file_id
    ):
        raise ValueError("Google Drive locator is malformed")
    return "https://drive.usercontent.google.com/download?" + urlencode(
        {
            "confirm": "t",
            "export": "download",
            "id": file_id,
        }
    )


def _git_raw_url(
    source: LockedBenchmarkSource,
    artifact: LockedAcquisitionArtifact,
) -> str:
    parsed = urlparse(source.repository)
    parts = tuple(part for part in parsed.path.removesuffix(".git").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) != 2:
        raise ValueError("git-blob acquisition requires a pinned GitHub repository")
    owner, repository = parts
    return (
        "https://raw.githubusercontent.com/"
        f"{quote(owner, safe='')}/{quote(repository, safe='')}/"
        f"{source.revision}/{quote(artifact.relative_path, safe='/')}"
    )


def locked_artifact_url(
    source: LockedBenchmarkSource,
    artifact: LockedAcquisitionArtifact,
) -> str:
    """Translate one committed locator to its single network endpoint."""

    locator = artifact.locator
    if locator.startswith(("https://", "http://")):
        return locator
    if locator.startswith("hf:"):
        return _huggingface_url(locator)
    if locator.startswith("gdrive:"):
        return _gdrive_url(locator)
    if locator.startswith("git-blob:"):
        return _git_raw_url(source, artifact)
    raise ValueError("artifact locator kind is unsupported")


@dataclass(frozen=True, slots=True)
class CurlLockedArtifactFetcher:
    """One-attempt curl fetcher with resumable partials and no token logging."""

    executable: str = "curl"
    huggingface_token_environment_variable: str = "HF_TOKEN"  # noqa: S105 - environment variable name, not a credential

    def fetch(
        self,
        *,
        benchmark: Benchmark,
        source: LockedBenchmarkSource,
        artifact: LockedAcquisitionArtifact,
        destination: Path,
    ) -> None:
        del benchmark
        url = locked_artifact_url(source, artifact)
        token = os.environ.get(self.huggingface_token_environment_variable)
        config_lines = [f'url = "{url}"']
        if artifact.locator.startswith("hf:") and token:
            if any(character in token for character in ('"', "\r", "\n")):
                raise ValueError("Hugging Face token contains unsupported characters")
            config_lines.append(f'header = "Authorization: Bearer {token}"')
        subprocess.run(  # noqa: S603 - exact executable and args are explicit inputs
            (
                self.executable,
                "--config",
                "-",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--output",
                str(destination),
            ),
            input="\n".join(config_lines) + "\n",
            text=True,
            check=True,
        )
        if not destination.is_file():
            raise FileNotFoundError(destination)


@dataclass(frozen=True, slots=True)
class AcquiredArtifactReceipt:
    """Measured identity of one downloaded source artifact."""

    benchmark: Benchmark
    locator: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("acquired artifact benchmark must be Benchmark")
        _text(self.locator, field="acquired artifact locator")
        _relative_path(
            self.relative_path,
            field="acquired artifact relative_path",
        )
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("acquired artifact size_bytes must be a positive integer")
        _lower_hex(
            self.sha256,
            length=_SHA256_HEX_LENGTH,
            field="acquired artifact sha256",
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "locator": self.locator,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object) -> AcquiredArtifactReceipt:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmark",
                    "locator",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                }
            ),
            label="acquired artifact receipt",
        )
        size_bytes = data["size_bytes"]
        if type(size_bytes) is not int:
            raise TypeError("acquired artifact size_bytes must be an integer")
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="artifact benchmark")),
            locator=_text(data["locator"], field="acquired artifact locator"),
            relative_path=_relative_path(
                data["relative_path"],
                field="acquired artifact relative_path",
            ),
            size_bytes=size_bytes,
            sha256=_lower_hex(
                data["sha256"],
                length=_SHA256_HEX_LENGTH,
                field="acquired artifact sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkAcquisitionReceipt:
    """Private source receipt produced only after every locked artifact exists."""

    acquisition_lock_hash: str
    target_root: Path
    artifacts: tuple[AcquiredArtifactReceipt, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.acquisition_lock_hash)
        if not isinstance(self.target_root, Path) or not self.target_root.is_absolute():
            raise ValueError("acquisition receipt target_root must be an absolute Path")
        if not self.artifacts or any(
            not isinstance(artifact, AcquiredArtifactReceipt) for artifact in self.artifacts
        ):
            raise ValueError("acquisition receipt must contain acquired artifact receipts")
        identities = tuple(
            (artifact.benchmark, artifact.relative_path) for artifact in self.artifacts
        )
        if len(set(identities)) != len(identities):
            raise ValueError("acquisition receipt contains duplicate artifact identities")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "acquisition_lock_hash": self.acquisition_lock_hash,
            "artifacts": [artifact.to_value() for artifact in self.artifacts],
            "format": "skillev-private-benchmark-acquisition-receipt@1",
            "target_root": self.target_root.as_posix(),
        }

    @classmethod
    def from_value(cls, value: object) -> BenchmarkAcquisitionReceipt:
        data = _object(
            value,
            fields=frozenset(
                {
                    "acquisition_lock_hash",
                    "artifacts",
                    "format",
                    "target_root",
                }
            ),
            label="benchmark acquisition receipt",
        )
        if data["format"] != "skillev-private-benchmark-acquisition-receipt@1":
            raise ValueError("unsupported benchmark acquisition receipt format")
        raw_artifacts = data["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise TypeError("acquisition receipt artifacts must be an array")
        return cls(
            acquisition_lock_hash=_text(
                data["acquisition_lock_hash"],
                field="acquisition_lock_hash",
            ),
            target_root=Path(_text(data["target_root"], field="acquisition receipt target_root")),
            artifacts=tuple(
                AcquiredArtifactReceipt.from_value(artifact) for artifact in raw_artifacts
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _measure_acquired_artifact(
    *,
    benchmark: Benchmark,
    artifact: LockedAcquisitionArtifact,
    path: Path,
) -> AcquiredArtifactReceipt:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = _sha256(path)
    if artifact.expected_size is not None and size != artifact.expected_size:
        raise ValueError("downloaded artifact size differs from the acquisition lock")
    if artifact.expected_sha256 is not None and digest != artifact.expected_sha256:
        raise ValueError("downloaded artifact digest differs from the acquisition lock")
    if artifact.locator.startswith("gdrive:"):
        with path.open("rb") as stream:
            prefix = stream.read(4096).lstrip()
        suffix = artifact.relative_path.lower()
        if suffix.endswith(".zip") and not prefix.startswith(
            (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
        ):
            raise ValueError("Google Drive ZIP artifact does not have a ZIP file signature")
        if suffix.endswith(".json") and not prefix.startswith((b"{", b"[")):
            raise ValueError("Google Drive JSON artifact does not have a JSON document signature")
    if artifact.locator.startswith("git-blob:"):
        expected_blob = artifact.locator.removeprefix("git-blob:")
        if _lower_hex(
            expected_blob,
            length=_GIT_REVISION_LENGTH,
            field="git blob locator",
        ) != _git_blob_sha1(path):
            raise ValueError("downloaded artifact differs from the pinned Git blob")
    return AcquiredArtifactReceipt(
        benchmark=benchmark,
        locator=artifact.locator,
        relative_path=artifact.relative_path,
        size_bytes=size,
        sha256=digest,
    )


def acquire_locked_benchmark_artifacts(
    lock: BenchmarkAcquisitionLock,
    *,
    target_root: Path,
    fetcher: LockedArtifactFetcher,
) -> BenchmarkAcquisitionReceipt:
    """Fetch every missing locked artifact once and preserve partial failures."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    if not target_root.is_dir():
        raise NotADirectoryError(target_root)
    if not callable(getattr(fetcher, "fetch", None)):
        raise TypeError("fetcher must implement LockedArtifactFetcher")

    receipts: list[AcquiredArtifactReceipt] = []
    for entry in lock.benchmarks:
        for artifact in entry.source.artifacts:
            destination = (
                target_root / entry.benchmark.value / PurePosixPath(artifact.relative_path)
            )
            if destination.is_file():
                receipts.append(
                    _measure_acquired_artifact(
                        benchmark=entry.benchmark,
                        artifact=artifact,
                        path=destination,
                    )
                )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            fetcher.fetch(
                benchmark=entry.benchmark,
                source=entry.source,
                artifact=artifact,
                destination=partial,
            )
            measured = _measure_acquired_artifact(
                benchmark=entry.benchmark,
                artifact=artifact,
                path=partial,
            )
            if destination.exists():
                raise FileExistsError(destination)
            partial.rename(destination)
            receipts.append(measured)
    return BenchmarkAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=target_root,
        artifacts=tuple(receipts),
    )


def publish_benchmark_acquisition_receipt(
    receipt: BenchmarkAcquisitionReceipt,
    output_path: Path,
) -> None:
    """Persist one private canonical receipt without replacing prior evidence."""

    if not isinstance(receipt, BenchmarkAcquisitionReceipt):
        raise TypeError("receipt must be BenchmarkAcquisitionReceipt")
    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("output_path must be an absolute Path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(receipt.to_value()))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_benchmark_acquisition_receipt(path: Path) -> BenchmarkAcquisitionReceipt:
    """Load one canonical private receipt without reopening benchmark records."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("receipt path must be an absolute Path")
    return BenchmarkAcquisitionReceipt.from_value(
        _read_published_canonical_record(path, label="benchmark acquisition receipt")
    )


def load_benchmark_acquisition_lock(path: Path) -> BenchmarkAcquisitionLock:
    """Load the committed result-blind source authority."""

    with path.open(encoding="utf-8") as stream:
        return BenchmarkAcquisitionLock.from_value(json.load(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_acquisition_progress(
    lock: BenchmarkAcquisitionLock,
    *,
    target_root: Path,
    verify_digests: bool,
) -> AcquisitionProgress:
    """Inspect only locked paths; digest verification is explicit and streaming."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    if type(verify_digests) is not bool:
        raise TypeError("verify_digests must be a boolean")

    statuses: list[ArtifactAcquisitionStatus] = []
    for entry in lock.benchmarks:
        for artifact in entry.source.artifacts:
            local_path = target_root / entry.benchmark.value / PurePosixPath(artifact.relative_path)
            if not local_path.is_file():
                statuses.append(
                    ArtifactAcquisitionStatus(
                        benchmark=entry.benchmark,
                        relative_path=artifact.relative_path,
                        local_path=local_path,
                        expected_size=artifact.expected_size,
                        exists=False,
                        size_bytes=None,
                        sha256=None,
                        size_matches=None,
                        sha256_matches=None,
                    )
                )
                continue
            size = local_path.stat().st_size
            digest = _sha256(local_path) if verify_digests else None
            statuses.append(
                ArtifactAcquisitionStatus(
                    benchmark=entry.benchmark,
                    relative_path=artifact.relative_path,
                    local_path=local_path,
                    expected_size=artifact.expected_size,
                    exists=True,
                    size_bytes=size,
                    sha256=digest,
                    size_matches=(
                        None if artifact.expected_size is None else size == artifact.expected_size
                    ),
                    sha256_matches=(
                        None
                        if artifact.expected_sha256 is None or digest is None
                        else digest == artifact.expected_sha256
                    ),
                )
            )
    return AcquisitionProgress(tuple(statuses))


__all__ = [
    "ACQUISITION_LOCK_FORMAT",
    "ACQUISITION_PRESERVATION_RULE",
    "AcquiredArtifactReceipt",
    "AcquisitionProgress",
    "ArtifactAcquisitionStatus",
    "BenchmarkAcquisitionLock",
    "BenchmarkAcquisitionReceipt",
    "CurlLockedArtifactFetcher",
    "LockedAcquisitionArtifact",
    "LockedArtifactFetcher",
    "LockedBenchmarkAcquisition",
    "LockedBenchmarkSource",
    "acquire_locked_benchmark_artifacts",
    "inspect_acquisition_progress",
    "load_benchmark_acquisition_lock",
    "load_benchmark_acquisition_receipt",
    "locked_artifact_url",
    "publish_benchmark_acquisition_receipt",
]
