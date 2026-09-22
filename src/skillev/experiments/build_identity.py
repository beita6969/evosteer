"""Path-free identities for an executing source package.

The source tree hash is shared by formal benchmark build measurement and the
non-benchmark real-backbone gate.  Source packages carry one small provenance
record whose own bytes are deliberately excluded from the tree hash, avoiding
a circular identity.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256

SOURCE_PACKAGE_PROVENANCE_FILE: Final = "SKILLEV_SOURCE_PROVENANCE.json"
SOURCE_PACKAGE_PROVENANCE_FORMAT: Final = "skillev-source-package-provenance@1"
SOURCE_PACKAGE_TREE_FORMAT: Final = "skillev-source-package-tree@2"

_EXCLUDED_DIRECTORIES: Final = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".private",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "test-results",
    }
)


def sha256_file(path: Path) -> str:
    """Hash one regular file without exposing its absolute path."""

    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def source_package_tree_hash(root: Path) -> str:
    """Hash source files while pruning all known runtime by-product trees."""

    if not root.is_dir():
        raise NotADirectoryError(root)
    files: list[dict[str, JsonValue]] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        names[:] = sorted(name for name in names if name not in _EXCLUDED_DIRECTORIES)
        directory_path = Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            relative = path.relative_to(root)
            if relative.as_posix() == SOURCE_PACKAGE_PROVENANCE_FILE:
                continue
            if path.is_symlink():
                raise ValueError("source package cannot contain symbolic links")
            if path.is_file():
                files.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
    if not files:
        raise ValueError("source package tree cannot be empty")
    files.sort(key=lambda item: str(item["path"]))
    return stable_hash({"files": files, "format": SOURCE_PACKAGE_TREE_FORMAT})


@dataclass(frozen=True, slots=True)
class SourcePackageProvenance:
    """Commit and byte-tree identity embedded in one immutable source archive."""

    source_commit: str
    source_tree_hash: str
    format: str = SOURCE_PACKAGE_PROVENANCE_FORMAT

    def __post_init__(self) -> None:
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("source_commit must be a lowercase Git object ID")
        validate_sha256(self.source_tree_hash)
        if self.format != SOURCE_PACKAGE_PROVENANCE_FORMAT:
            raise ValueError("unsupported source package provenance format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "source_commit": self.source_commit,
            "source_tree_hash": self.source_tree_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> SourcePackageProvenance:
        normalized = normalize_json(value)
        fields = {"format", "source_commit", "source_tree_hash"}
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("source package provenance has incompatible fields")
        if any(type(normalized[field]) is not str for field in fields):
            raise TypeError("source package provenance fields must be text")
        return cls(
            source_commit=normalized["source_commit"],
            source_tree_hash=normalized["source_tree_hash"],
            format=normalized["format"],
        )


def write_source_package_provenance(
    root: Path,
    *,
    source_commit: str,
) -> SourcePackageProvenance:
    """Write the identity of an already materialized immutable source tree."""

    provenance_path = root / SOURCE_PACKAGE_PROVENANCE_FILE
    if provenance_path.exists():
        raise FileExistsError(provenance_path)
    provenance = SourcePackageProvenance(
        source_commit=source_commit,
        source_tree_hash=source_package_tree_hash(root),
    )
    provenance_path.write_text(canonical_json(provenance.to_value()) + "\n", encoding="utf-8")
    return provenance


def read_source_package_provenance(root: Path) -> SourcePackageProvenance:
    """Read the embedded record and verify the surrounding source bytes."""

    path = root / SOURCE_PACKAGE_PROVENANCE_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    import json

    provenance = SourcePackageProvenance.from_value(json.loads(path.read_text(encoding="utf-8")))
    if source_package_tree_hash(root) != provenance.source_tree_hash:
        raise ValueError("source tree bytes differ from embedded provenance")
    return provenance


def require_source_archive_matches_execution(
    *,
    source_archive: Path,
    executing_root: Path,
    temporary_parent: Path | None = None,
) -> SourcePackageProvenance:
    """Prove that the executing tree and supplied archive name one build."""

    executing = read_source_package_provenance(executing_root)
    with tempfile.TemporaryDirectory(
        prefix="skillev-source-identity-",
        dir=temporary_parent,
    ) as temporary:
        extracted_root = Path(temporary)
        with tarfile.open(source_archive, mode="r:*") as archive:
            archive.extractall(extracted_root, filter="data")
        archived = read_source_package_provenance(extracted_root)
    if archived != executing:
        raise ValueError("executing source provenance differs from supplied archive")
    return executing


__all__ = [
    "SOURCE_PACKAGE_PROVENANCE_FILE",
    "SOURCE_PACKAGE_PROVENANCE_FORMAT",
    "SOURCE_PACKAGE_TREE_FORMAT",
    "SourcePackageProvenance",
    "read_source_package_provenance",
    "require_source_archive_matches_execution",
    "sha256_file",
    "source_package_tree_hash",
    "write_source_package_provenance",
]
