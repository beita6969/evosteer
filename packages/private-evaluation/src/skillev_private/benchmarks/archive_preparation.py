"""Memory-bounded, no-delete extraction for pinned private source archives."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tarfile
import uuid
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import IO, cast

from skillev.contracts import (
    JsonValue,
    canonical_json,
    normalize_json,
    parse_canonical_json,
    stable_hash,
    validate_sha256,
)
from skillev.experiments import Benchmark

from .acquisition import (
    AcquiredArtifactReceipt,
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
    _read_published_canonical_record,
)

ARCHIVE_PREPARATION_FORMAT = "skillev-private-archive-preparation@1"
LOCKED_ARCHIVE_BATCH_FORMAT = "skillev-private-locked-archive-batch@1"
_COPY_CHUNK_BYTES = 1024 * 1024


class ArchiveKind(str, Enum):
    ZIP = "zip"
    TAR = "tar"


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _member_path(value: str) -> PurePosixPath:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("archive member path is not normalized POSIX text")
    normalized = value.removesuffix("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts or path.as_posix() != normalized:
        raise ValueError("archive member path escapes or differs from its normalized path")
    return path


def _relative_path(value: str, *, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    return value


def _wire_object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an invalid field set")
    return normalized


@dataclass(frozen=True, slots=True)
class ExtractedArchiveMember:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, field="extracted member path")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("extracted member size must be a non-negative integer")
        _lower_sha256(self.sha256)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object) -> ExtractedArchiveMember:
        data = _wire_object(
            value,
            fields=frozenset({"relative_path", "sha256", "size_bytes"}),
            label="extracted archive member",
        )
        relative_path = data["relative_path"]
        sha256 = data["sha256"]
        size_bytes = data["size_bytes"]
        if type(relative_path) is not str or type(sha256) is not str:
            raise TypeError("extracted archive member identity fields must be text")
        if type(size_bytes) is not int:
            raise TypeError("extracted archive member size must be an integer")
        return cls(relative_path, size_bytes, sha256)


@dataclass(frozen=True, slots=True)
class ArchivePreparationReceipt:
    archive_kind: ArchiveKind
    archive_name: str
    archive_sha256: str
    archive_size_bytes: int
    extracted_file_count: int
    extracted_size_bytes: int
    members_jsonl_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.archive_kind, ArchiveKind):
            raise TypeError("archive preparation kind must be ArchiveKind")
        if (
            not self.archive_name
            or Path(self.archive_name).name != self.archive_name
            or "\x00" in self.archive_name
        ):
            raise ValueError("archive preparation name must be one file name")
        _lower_sha256(self.archive_sha256)
        _lower_sha256(self.members_jsonl_sha256)
        for name, value in (
            ("archive_size_bytes", self.archive_size_bytes),
            ("extracted_file_count", self.extracted_file_count),
            ("extracted_size_bytes", self.extracted_size_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.archive_size_bytes == 0:
            raise ValueError("archive_size_bytes must be positive")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "archive_kind": self.archive_kind.value,
            "archive_name": self.archive_name,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "extracted_file_count": self.extracted_file_count,
            "extracted_size_bytes": self.extracted_size_bytes,
            "format": ARCHIVE_PREPARATION_FORMAT,
            "members_jsonl_sha256": self.members_jsonl_sha256,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_manifest_value(cls, value: object) -> ArchivePreparationReceipt:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "archive_kind",
                    "archive_name",
                    "archive_sha256",
                    "archive_size_bytes",
                    "content_hash",
                    "extracted_file_count",
                    "extracted_size_bytes",
                    "format",
                    "members_jsonl_sha256",
                }
            ),
            label="archive preparation manifest",
        )
        if data["format"] != ARCHIVE_PREPARATION_FORMAT:
            raise ValueError("unsupported archive preparation manifest format")
        text_fields = (
            "archive_kind",
            "archive_name",
            "archive_sha256",
            "content_hash",
            "members_jsonl_sha256",
        )
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("archive preparation manifest identity fields must be text")
        integer_fields = (
            "archive_size_bytes",
            "extracted_file_count",
            "extracted_size_bytes",
        )
        if any(type(data[field]) is not int for field in integer_fields):
            raise TypeError("archive preparation manifest counters must be integers")
        receipt = cls(
            archive_kind=ArchiveKind(cast(str, data["archive_kind"])),
            archive_name=cast(str, data["archive_name"]),
            archive_sha256=cast(str, data["archive_sha256"]),
            archive_size_bytes=cast(int, data["archive_size_bytes"]),
            extracted_file_count=cast(int, data["extracted_file_count"]),
            extracted_size_bytes=cast(int, data["extracted_size_bytes"]),
            members_jsonl_sha256=cast(str, data["members_jsonl_sha256"]),
        )
        if data["content_hash"] != receipt.content_hash:
            raise ValueError("archive preparation manifest content hash differs")
        return receipt


@dataclass(frozen=True, slots=True)
class LockedArchivePreparation:
    benchmark: Benchmark
    artifact_relative_path: str
    archive_kind: ArchiveKind
    output_relative_path: str
    password_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("locked archive benchmark must be Benchmark")
        _relative_path(self.artifact_relative_path, field="locked archive source path")
        _relative_path(self.output_relative_path, field="locked archive output path")
        if not isinstance(self.archive_kind, ArchiveKind):
            raise TypeError("locked archive kind must be ArchiveKind")
        if type(self.password_required) is not bool:
            raise TypeError("locked archive password_required must be bool")
        if self.password_required and self.archive_kind is not ArchiveKind.ZIP:
            raise ValueError("only ZIP archives may require a password")


_LOCKED_ARCHIVE_PREPARATIONS = (
    LockedArchivePreparation(
        Benchmark.ALFWORLD,
        "raw/json_2.1.1_json.zip",
        ArchiveKind.ZIP,
        "prepared/json_2.1.1_json",
    ),
    LockedArchivePreparation(
        Benchmark.ALFWORLD,
        "raw/json_2.1.1_pddl.zip",
        ArchiveKind.ZIP,
        "prepared/json_2.1.1_pddl",
    ),
    LockedArchivePreparation(
        Benchmark.ALFWORLD,
        "raw/json_2.1.3_tw-pddl.zip",
        ArchiveKind.ZIP,
        "prepared/json_2.1.3_tw-pddl",
    ),
    LockedArchivePreparation(
        Benchmark.TRIVIA_QA,
        "raw/dataindex.tar.gz",
        ArchiveKind.TAR,
        "prepared/dataindex",
    ),
    LockedArchivePreparation(
        Benchmark.TRIVIA_QA,
        "raw/triviaqa-unfiltered.tar.gz",
        ArchiveKind.TAR,
        "prepared/triviaqa-unfiltered",
    ),
    LockedArchivePreparation(
        Benchmark.MED_QA,
        "raw/data_clean.zip",
        ArchiveKind.ZIP,
        "prepared/data_clean",
    ),
    LockedArchivePreparation(
        Benchmark.BIRD_SQL,
        "raw/train.zip",
        ArchiveKind.ZIP,
        "prepared/train",
    ),
    LockedArchivePreparation(
        Benchmark.BIRD_SQL,
        "raw/dev.zip",
        ArchiveKind.ZIP,
        "prepared/dev",
    ),
    LockedArchivePreparation(
        Benchmark.MUSIQUE,
        "raw/musique_v1.0.zip",
        ArchiveKind.ZIP,
        "prepared/musique_v1.0",
    ),
    LockedArchivePreparation(
        Benchmark.NQ_OPEN,
        "raw/dataindex.tar.gz",
        ArchiveKind.TAR,
        "prepared/dataindex",
    ),
    LockedArchivePreparation(
        Benchmark.MIND2WEB,
        "raw/test.zip",
        ArchiveKind.ZIP,
        "prepared/test",
        password_required=True,
    ),
)


@dataclass(frozen=True, slots=True)
class PreparedLockedArchive:
    benchmark: Benchmark
    artifact_relative_path: str
    output_relative_path: str
    preparation_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("prepared archive benchmark must be Benchmark")
        _relative_path(self.artifact_relative_path, field="prepared archive source path")
        _relative_path(self.output_relative_path, field="prepared archive output path")
        validate_sha256(self.preparation_content_hash)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifact_relative_path": self.artifact_relative_path,
            "benchmark": self.benchmark.value,
            "output_relative_path": self.output_relative_path,
            "preparation_content_hash": self.preparation_content_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> PreparedLockedArchive:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "artifact_relative_path",
                    "benchmark",
                    "output_relative_path",
                    "preparation_content_hash",
                }
            ),
            label="prepared locked archive",
        )
        if any(type(item) is not str for item in data.values()):
            raise TypeError("prepared locked archive fields must be text")
        return cls(
            benchmark=Benchmark(cast(str, data["benchmark"])),
            artifact_relative_path=cast(str, data["artifact_relative_path"]),
            output_relative_path=cast(str, data["output_relative_path"]),
            preparation_content_hash=cast(str, data["preparation_content_hash"]),
        )


@dataclass(frozen=True, slots=True)
class LockedArchiveBatchReceipt:
    acquisition_lock_hash: str
    acquisition_receipt_hash: str
    archives: tuple[PreparedLockedArchive, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.acquisition_lock_hash)
        validate_sha256(self.acquisition_receipt_hash)
        if tuple(
            (archive.benchmark, archive.artifact_relative_path, archive.output_relative_path)
            for archive in self.archives
        ) != tuple(
            (plan.benchmark, plan.artifact_relative_path, plan.output_relative_path)
            for plan in _LOCKED_ARCHIVE_PREPARATIONS
        ):
            raise ValueError("prepared archives differ from the explicit locked archive plan")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "acquisition_lock_hash": self.acquisition_lock_hash,
            "acquisition_receipt_hash": self.acquisition_receipt_hash,
            "archives": [archive.to_value() for archive in self.archives],
            "format": LOCKED_ARCHIVE_BATCH_FORMAT,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_manifest_value(cls, value: object) -> LockedArchiveBatchReceipt:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "acquisition_lock_hash",
                    "acquisition_receipt_hash",
                    "archives",
                    "content_hash",
                    "format",
                }
            ),
            label="locked archive batch receipt",
        )
        if data["format"] != LOCKED_ARCHIVE_BATCH_FORMAT:
            raise ValueError("unsupported locked archive batch receipt format")
        lock_hash = data["acquisition_lock_hash"]
        acquisition_hash = data["acquisition_receipt_hash"]
        content_hash = data["content_hash"]
        archives = data["archives"]
        if (
            type(lock_hash) is not str
            or type(acquisition_hash) is not str
            or type(content_hash) is not str
        ):
            raise TypeError("locked archive batch identity fields must be text")
        if not isinstance(archives, list):
            raise TypeError("locked archive batch archives must be an array")
        receipt = cls(
            acquisition_lock_hash=lock_hash,
            acquisition_receipt_hash=acquisition_hash,
            archives=tuple(PreparedLockedArchive.from_value(item) for item in archives),
        )
        if content_hash != receipt.content_hash:
            raise ValueError("locked archive batch content hash differs")
        return receipt


def _copy_member(source: IO[bytes], destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("xb") as output:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _zip_members(
    archive: zipfile.ZipFile,
    staging_root: Path,
    *,
    password: bytes | None,
) -> Iterator[ExtractedArchiveMember]:
    for information in archive.infolist():
        relative = _member_path(information.filename)
        unix_mode = information.external_attr >> 16
        if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise ValueError("ZIP symbolic-link members are not supported")
        destination = staging_root.joinpath(*relative.parts)
        if information.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(information, "r", pwd=password) as source:
            size_bytes, sha256 = _copy_member(source, destination)
        yield ExtractedArchiveMember(
            relative_path=relative.as_posix(),
            size_bytes=size_bytes,
            sha256=sha256,
        )


def _tar_members(
    archive: tarfile.TarFile,
    staging_root: Path,
) -> Iterator[ExtractedArchiveMember]:
    for information in archive:
        relative = _member_path(information.name)
        destination = staging_root.joinpath(*relative.parts)
        if information.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if not information.isfile():
            raise ValueError("TAR members must be regular files or directories")
        source = archive.extractfile(information)
        if source is None:
            raise ValueError("regular TAR member has no readable payload")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source:
            size_bytes, sha256 = _copy_member(source, destination)
        yield ExtractedArchiveMember(
            relative_path=relative.as_posix(),
            size_bytes=size_bytes,
            sha256=sha256,
        )


def _members(
    *,
    archive_path: Path,
    archive_kind: ArchiveKind,
    staging_root: Path,
    zip_password: bytes | None,
) -> Iterator[ExtractedArchiveMember]:
    if archive_kind is ArchiveKind.ZIP:
        with zipfile.ZipFile(archive_path, "r") as archive:
            yield from _zip_members(archive, staging_root, password=zip_password)
        return
    if archive_kind is ArchiveKind.TAR:
        with tarfile.open(archive_path, mode="r:*") as archive:
            yield from _tar_members(archive, staging_root)
        return
    raise AssertionError(f"unsupported archive kind {archive_kind!r}")


def _validate_zip_password(archive_path: Path, password: bytes | None) -> None:
    if password is not None and (type(password) is not bytes or not password):
        raise ValueError("ZIP password must be non-empty bytes")
    with zipfile.ZipFile(archive_path, "r") as archive:
        encrypted = tuple(
            information
            for information in archive.infolist()
            if not information.is_dir() and information.flag_bits & 0x1
        )
        if not encrypted:
            if password is not None:
                raise ValueError("password supplied for an unencrypted ZIP archive")
            return
        if password is None:
            raise ValueError("encrypted ZIP archive requires an explicit password")
        try:
            with archive.open(encrypted[0], "r", pwd=password) as source:
                source.read(1)
        except RuntimeError as error:
            raise ValueError("ZIP password does not unlock the archive") from error


def prepare_archive(
    *,
    archive_path: Path,
    archive_kind: ArchiveKind,
    expected_size: int,
    expected_sha256: str,
    output_root: Path,
    zip_password: bytes | None = None,
) -> ArchivePreparationReceipt:
    """Extract one exact archive into a new tree while preserving its source."""

    if not isinstance(archive_path, Path) or not archive_path.is_absolute():
        raise ValueError("archive_path must be an absolute Path")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("output_root must be an absolute Path")
    if output_root.exists():
        raise FileExistsError(output_root)
    if not isinstance(archive_kind, ArchiveKind):
        raise TypeError("archive_kind must be ArchiveKind")
    if archive_kind is ArchiveKind.TAR and zip_password is not None:
        raise ValueError("TAR archives do not accept a ZIP password")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("expected_size must be a positive integer")
    expected_digest = _lower_sha256(expected_sha256)
    measured_size = archive_path.stat().st_size
    if measured_size != expected_size:
        raise ValueError("archive size differs from the private acquisition receipt")
    measured_digest = _sha256_file(archive_path)
    if measured_digest != expected_digest:
        raise ValueError("archive digest differs from the private acquisition receipt")
    if archive_kind is ArchiveKind.ZIP:
        _validate_zip_password(archive_path, zip_password)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(f".{output_root.name}.part-{uuid.uuid4().hex}")
    staging_root.mkdir()
    members_path = staging_root / "extracted-members.jsonl"
    member_lines_digest = hashlib.sha256()
    extracted_file_count = 0
    extracted_size_bytes = 0
    with members_path.open("xb") as member_stream:
        for member in _members(
            archive_path=archive_path,
            archive_kind=archive_kind,
            staging_root=staging_root,
            zip_password=zip_password,
        ):
            line = (canonical_json(member.to_value()) + "\n").encode()
            member_stream.write(line)
            member_lines_digest.update(line)
            extracted_file_count += 1
            extracted_size_bytes += member.size_bytes
        member_stream.flush()
        os.fsync(member_stream.fileno())

    receipt = ArchivePreparationReceipt(
        archive_kind=archive_kind,
        archive_name=archive_path.name,
        archive_sha256=measured_digest,
        archive_size_bytes=measured_size,
        extracted_file_count=extracted_file_count,
        extracted_size_bytes=extracted_size_bytes,
        members_jsonl_sha256=member_lines_digest.hexdigest(),
    )
    manifest = {
        **receipt.to_value(),
        "content_hash": receipt.content_hash,
    }
    manifest_path = staging_root / "preparation.manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(manifest))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging_root.rename(output_root)
    return receipt


def verify_prepared_archive(output_root: Path) -> ArchivePreparationReceipt:
    """Verify every published extracted byte against its canonical manifest."""

    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("output_root must be an absolute Path")
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    manifest_path = output_root / "preparation.manifest.json"
    members_path = output_root / "extracted-members.jsonl"
    receipt = ArchivePreparationReceipt.from_manifest_value(
        _read_published_canonical_record(
            manifest_path,
            label="archive preparation manifest",
        )
    )
    member_lines_digest = hashlib.sha256()
    file_count = 0
    extracted_size = 0
    declared_paths: set[str] = set()
    with members_path.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.endswith(b"\n") or raw_line == b"\n":
                raise ValueError("extracted member receipt must be non-empty canonical JSONL")
            member_lines_digest.update(raw_line)
            member = ExtractedArchiveMember.from_value(
                parse_canonical_json(raw_line[:-1].decode("utf-8"))
            )
            if raw_line != (canonical_json(member.to_value()) + "\n").encode():
                raise ValueError("extracted member receipt line is not canonical")
            if member.relative_path in declared_paths:
                raise ValueError("extracted member receipt contains a duplicate path")
            declared_paths.add(member.relative_path)
            path = output_root / PurePosixPath(member.relative_path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("extracted member is absent or is not a regular file")
            if path.stat().st_size != member.size_bytes or _sha256_file(path) != member.sha256:
                raise ValueError("extracted member bytes differ from their receipt")
            file_count += 1
            extracted_size += member.size_bytes
    if (
        file_count != receipt.extracted_file_count
        or extracted_size != receipt.extracted_size_bytes
        or member_lines_digest.hexdigest() != receipt.members_jsonl_sha256
    ):
        raise ValueError("extracted archive aggregate differs from its receipt")
    actual_file_count = 0
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("prepared archive tree contains a symbolic link")
        if path.is_file():
            actual_file_count += 1
    if actual_file_count != receipt.extracted_file_count + 2:
        raise ValueError("prepared archive tree contains an unrecorded file")
    return receipt


def locked_archive_preparations(
    lock: BenchmarkAcquisitionLock,
) -> tuple[LockedArchivePreparation, ...]:
    """Return the explicit archive plan after proving it belongs to the lock."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    locked_artifacts = {
        (entry.benchmark, artifact.relative_path)
        for entry in lock.benchmarks
        for artifact in entry.source.artifacts
    }
    planned = tuple(
        (plan.benchmark, plan.artifact_relative_path) for plan in _LOCKED_ARCHIVE_PREPARATIONS
    )
    if len(set(planned)) != len(planned) or any(
        identity not in locked_artifacts for identity in planned
    ):
        raise ValueError("explicit archive plan differs from the acquisition lock")
    return _LOCKED_ARCHIVE_PREPARATIONS


def _validate_acquisition_receipt(
    *,
    lock: BenchmarkAcquisitionLock,
    receipt: BenchmarkAcquisitionReceipt,
    target_root: Path,
) -> dict[tuple[Benchmark, str], AcquiredArtifactReceipt]:
    if receipt.acquisition_lock_hash != lock.content_hash:
        raise ValueError("artifact receipt belongs to another acquisition lock")
    if receipt.target_root != target_root:
        raise ValueError("artifact receipt belongs to another target root")
    expected = tuple(
        (entry.benchmark, artifact.relative_path)
        for entry in lock.benchmarks
        for artifact in entry.source.artifacts
    )
    actual = tuple((artifact.benchmark, artifact.relative_path) for artifact in receipt.artifacts)
    if actual != expected:
        raise ValueError("artifact receipt does not cover the exact acquisition lock order")
    by_identity = {
        (artifact.benchmark, artifact.relative_path): artifact for artifact in receipt.artifacts
    }
    for entry in lock.benchmarks:
        for locked in entry.source.artifacts:
            acquired = by_identity[(entry.benchmark, locked.relative_path)]
            if acquired.locator != locked.locator:
                raise ValueError("artifact receipt locator differs from the acquisition lock")
            if locked.expected_size is not None and acquired.size_bytes != locked.expected_size:
                raise ValueError("artifact receipt size differs from the acquisition lock")
            if locked.expected_sha256 is not None and acquired.sha256 != locked.expected_sha256:
                raise ValueError("artifact receipt digest differs from the acquisition lock")
    return by_identity


def prepare_locked_archives(
    *,
    lock: BenchmarkAcquisitionLock,
    acquisition_receipt: BenchmarkAcquisitionReceipt,
    target_root: Path,
    zip_passwords: Mapping[Benchmark, bytes],
) -> LockedArchiveBatchReceipt:
    """Prepare every explicitly declared archive from one complete acquisition."""

    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    if not target_root.is_dir():
        raise NotADirectoryError(target_root)
    if not isinstance(zip_passwords, Mapping):
        raise TypeError("zip_passwords must be a benchmark-to-bytes mapping")
    password_benchmarks = frozenset(
        plan.benchmark for plan in _LOCKED_ARCHIVE_PREPARATIONS if plan.password_required
    )
    if not frozenset(zip_passwords).issubset(password_benchmarks):
        raise ValueError("zip_passwords may cover only password-protected archives")
    if any(type(password) is not bytes or not password for password in zip_passwords.values()):
        raise ValueError("every ZIP password must be non-empty bytes")
    artifacts = _validate_acquisition_receipt(
        lock=lock,
        receipt=acquisition_receipt,
        target_root=target_root,
    )
    prepared: list[PreparedLockedArchive] = []
    for plan in locked_archive_preparations(lock):
        acquired = artifacts[(plan.benchmark, plan.artifact_relative_path)]
        source = target_root / plan.benchmark.value / PurePosixPath(plan.artifact_relative_path)
        output = target_root / plan.benchmark.value / PurePosixPath(plan.output_relative_path)
        if output.exists():
            receipt = verify_prepared_archive(output)
            if (
                receipt.archive_kind is not plan.archive_kind
                or receipt.archive_name != source.name
                or receipt.archive_size_bytes != acquired.size_bytes
                or receipt.archive_sha256 != acquired.sha256
            ):
                raise ValueError("existing archive preparation belongs to another source")
        else:
            if plan.password_required and plan.benchmark not in zip_passwords:
                raise ValueError("missing password for an unprepared encrypted archive")
            receipt = prepare_archive(
                archive_path=source,
                archive_kind=plan.archive_kind,
                expected_size=acquired.size_bytes,
                expected_sha256=acquired.sha256,
                output_root=output,
                zip_password=(zip_passwords[plan.benchmark] if plan.password_required else None),
            )
        prepared.append(
            PreparedLockedArchive(
                benchmark=plan.benchmark,
                artifact_relative_path=plan.artifact_relative_path,
                output_relative_path=plan.output_relative_path,
                preparation_content_hash=receipt.content_hash,
            )
        )
    return LockedArchiveBatchReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquisition_receipt.content_hash,
        archives=tuple(prepared),
    )


def publish_locked_archive_batch_receipt(
    receipt: LockedArchiveBatchReceipt,
    output_path: Path,
) -> None:
    """Publish the complete locked archive preparation receipt without replacement."""

    if not isinstance(receipt, LockedArchiveBatchReceipt):
        raise TypeError("receipt must be LockedArchiveBatchReceipt")
    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("output_path must be an absolute Path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json({**receipt.to_value(), "content_hash": receipt.content_hash}))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_locked_archive_batch_receipt(path: Path) -> LockedArchiveBatchReceipt:
    """Load a canonical complete archive-preparation receipt."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("locked archive receipt path must be an absolute Path")
    return LockedArchiveBatchReceipt.from_manifest_value(
        _read_published_canonical_record(path, label="locked archive batch receipt")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skillev-prepare-archive",
        description="Extract one pinned private source archive without deleting its source",
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=tuple(kind.value for kind in ArchiveKind))
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = prepare_archive(
        archive_path=cast(Path, args.archive).expanduser().resolve(),
        archive_kind=ArchiveKind(cast(str, args.kind)),
        expected_size=cast(int, args.expected_size),
        expected_sha256=cast(str, args.expected_sha256),
        output_root=cast(Path, args.output_root).expanduser().resolve(),
    )
    print(canonical_json({**receipt.to_value(), "content_hash": receipt.content_hash}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_PREPARATION_FORMAT",
    "LOCKED_ARCHIVE_BATCH_FORMAT",
    "ArchiveKind",
    "ArchivePreparationReceipt",
    "ExtractedArchiveMember",
    "LockedArchiveBatchReceipt",
    "LockedArchivePreparation",
    "PreparedLockedArchive",
    "load_locked_archive_batch_receipt",
    "locked_archive_preparations",
    "main",
    "prepare_archive",
    "prepare_locked_archives",
    "publish_locked_archive_batch_receipt",
    "verify_prepared_archive",
]
