"""Content-address private dataset files without exposing their contents.

Callers provide the exact relative file set accepted for one benchmark.  The
returned public identity contains only the declared dataset name/version and a
digest over file names, sizes, and byte-level SHA-256 digests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from skillev.contracts import stable_hash
from skillev.experiments import DatasetSnapshotIdentity


def _relative_file_path(value: object) -> PurePosixPath:
    if type(value) is not str or not value:
        raise ValueError("dataset relative path must be non-empty text")
    normalized = PurePosixPath(value)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("dataset relative path must remain below the snapshot root")
    if normalized.as_posix() != value:
        raise ValueError("dataset relative path must be normalized POSIX text")
    return normalized


@dataclass(frozen=True, slots=True)
class PrivateDatasetFileDigest:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_file_path(self.relative_path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("dataset file size must be non-negative")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("dataset file digest must be lowercase SHA-256 hex")

    def to_value(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _file_digest(path: Path, *, relative_path: str) -> PrivateDatasetFileDigest:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return PrivateDatasetFileDigest(
        relative_path=relative_path,
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def create_private_dataset_snapshot(
    *,
    name: str,
    version: str,
    root: Path,
    relative_files: tuple[str, ...],
) -> DatasetSnapshotIdentity:
    """Hash an exact file set and return its answer-free public identity."""

    if type(name) is not str or not name.strip():
        raise ValueError("dataset snapshot name must be non-empty text")
    if type(version) is not str or not version.strip():
        raise ValueError("dataset snapshot version must be non-empty text")
    if not isinstance(root, Path) or not root.is_dir():
        raise NotADirectoryError(root)
    if not relative_files:
        raise ValueError("dataset snapshot requires an explicit non-empty file set")
    if tuple(sorted(relative_files)) != relative_files:
        raise ValueError("dataset snapshot file set must be lexicographically sorted")
    if len(set(relative_files)) != len(relative_files):
        raise ValueError("dataset snapshot file set must not contain duplicates")

    normalized_paths = tuple(_relative_file_path(value) for value in relative_files)
    files = tuple(
        _file_digest(root / normalized, relative_path=relative_path)
        for relative_path, normalized in zip(relative_files, normalized_paths, strict=True)
    )
    snapshot_hash = stable_hash(
        {
            "files": [file.to_value() for file in files],
            "format": "skillev-private-dataset-snapshot@1",
            "name": name,
            "version": version,
        }
    )
    return DatasetSnapshotIdentity(name=name, version=version, snapshot_hash=snapshot_hash)


__all__ = ["PrivateDatasetFileDigest", "create_private_dataset_snapshot"]
