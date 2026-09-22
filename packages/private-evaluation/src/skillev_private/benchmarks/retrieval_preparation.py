"""Strict streaming preparation of lexical FTS5 over pinned Atlas passages.

Atlas/DPR names the public passage source, not a dense retriever: runtime
queries use deterministic SQLite FTS5/BM25.  The source ``.tsv.gz`` remains
private.  The only sidecar emitted beside the public SQLite retrieval index is
an aggregate, content-addressed manifest; it contains byte counts and hashes,
never a passage, title, or dataset row.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from skillev.benchmarks import (
    RetrievalBuildPhase,
    RetrievalBuildProgress,
    RetrievalIndex,
    RetrievalIndexManifest,
    build_retrieval_index_stream,
    verify_built_index,
)
from skillev.contracts import (
    JsonValue,
    canonical_json,
    normalize_json,
    parse_canonical_json,
    stable_hash,
    validate_sha256,
)

from .retrieval_corpus import iter_atlas_wikipedia_tsv

PREPARATION_FORMAT = "skillev-private-atlas-index-preparation@1"
_SOURCE_FORMAT = "atlas-dpr-wikipedia-tsv.gz@1"
_HASH_CHUNK_SIZE = 1024 * 1024
_COPY_BUFFER_BYTES = 16 * 1024 * 1024
PERFORMANCE_GATE_ROWS = (100_000, 1_000_000)
_STAGING_DIRECTORY_ENV = "SKILLEV_RETRIEVAL_STAGING_DIRECTORY"
_EXPECTED_PASSAGE_COUNT_ENV = "SKILLEV_RETRIEVAL_EXPECTED_PASSAGE_COUNT"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None else Path(_text(value, field=name))


def _environment_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be a positive decimal integer")
    return _positive_int(int(value), field=name)


def _sha256_file(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return size, f"sha256:{digest.hexdigest()}"


def _object(value: object) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise TypeError("retrieval preparation manifest must be normalized JSON")
    expected = {
        "format",
        "index_file_sha256",
        "index_file_size_bytes",
        "index_manifest",
        "input_raw_sha256",
        "input_size_bytes",
        "manifest_id",
        "source_format",
    }
    if set(normalized) != expected:
        raise ValueError("retrieval preparation manifest has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class RetrievalPreparationManifest:
    """Answer-free provenance for one verified on-disk retrieval index."""

    manifest_id: str
    input_raw_sha256: str
    input_size_bytes: int
    index_file_sha256: str
    index_file_size_bytes: int
    index_manifest: RetrievalIndexManifest
    source_format: str = _SOURCE_FORMAT
    format: str = PREPARATION_FORMAT

    def __post_init__(self) -> None:
        if self.format != PREPARATION_FORMAT:
            raise ValueError("retrieval preparation format is unsupported")
        if self.source_format != _SOURCE_FORMAT:
            raise ValueError("retrieval preparation source format is unsupported")
        validate_sha256(self.input_raw_sha256)
        validate_sha256(self.index_file_sha256)
        _positive_int(self.input_size_bytes, field="input_size_bytes")
        _positive_int(self.index_file_size_bytes, field="index_file_size_bytes")
        if not isinstance(self.index_manifest, RetrievalIndexManifest):
            raise TypeError("index_manifest must be RetrievalIndexManifest")
        validate_sha256(self.manifest_id)
        if self.manifest_id != stable_hash(self._identity_value()):
            raise ValueError("retrieval preparation manifest ID does not match its provenance")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "index_file_sha256": self.index_file_sha256,
            "index_file_size_bytes": self.index_file_size_bytes,
            "index_manifest": self.index_manifest.to_value(),
            "input_raw_sha256": self.input_raw_sha256,
            "input_size_bytes": self.input_size_bytes,
            "source_format": self.source_format,
        }

    def to_value(self) -> dict[str, JsonValue]:
        return {"manifest_id": self.manifest_id, **self._identity_value()}

    @classmethod
    def create(
        cls,
        *,
        input_raw_sha256: str,
        input_size_bytes: int,
        index_file_sha256: str,
        index_file_size_bytes: int,
        index_manifest: RetrievalIndexManifest,
    ) -> RetrievalPreparationManifest:
        identity: dict[str, JsonValue] = {
            "format": PREPARATION_FORMAT,
            "index_file_sha256": index_file_sha256,
            "index_file_size_bytes": index_file_size_bytes,
            "index_manifest": index_manifest.to_value(),
            "input_raw_sha256": input_raw_sha256,
            "input_size_bytes": input_size_bytes,
            "source_format": _SOURCE_FORMAT,
        }
        return cls(
            manifest_id=stable_hash(identity),
            input_raw_sha256=input_raw_sha256,
            input_size_bytes=input_size_bytes,
            index_file_sha256=index_file_sha256,
            index_file_size_bytes=index_file_size_bytes,
            index_manifest=index_manifest,
        )

    @classmethod
    def from_value(cls, value: object) -> RetrievalPreparationManifest:
        data = _object(value)
        return cls(
            manifest_id=_text(data["manifest_id"], field="manifest_id"),
            input_raw_sha256=_text(data["input_raw_sha256"], field="input_raw_sha256"),
            input_size_bytes=_positive_int(data["input_size_bytes"], field="input_size_bytes"),
            index_file_sha256=_text(data["index_file_sha256"], field="index_file_sha256"),
            index_file_size_bytes=_positive_int(
                data["index_file_size_bytes"], field="index_file_size_bytes"
            ),
            index_manifest=RetrievalIndexManifest.from_value(data["index_manifest"]),
            source_format=_text(data["source_format"], field="source_format"),
            format=_text(data["format"], field="format"),
        )


def load_retrieval_preparation_manifest(
    path: Path,
) -> RetrievalPreparationManifest:
    """Load one canonical aggregate retrieval-preparation sidecar."""

    if not isinstance(path, Path):
        raise TypeError("retrieval preparation manifest path must be Path")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("existing retrieval preparation manifest is unreadable") from error
    if not raw.endswith("\n") or "\n" in raw[:-1]:
        raise ValueError("existing retrieval preparation manifest is not one canonical record")
    payload = raw[:-1]
    value = parse_canonical_json(payload)
    if canonical_json(value) != payload:
        raise ValueError("existing retrieval preparation manifest is not canonical")
    return RetrievalPreparationManifest.from_value(value)


def verify_retrieval_preparation_output(
    *,
    index_path: Path,
    manifest_path: Path,
) -> RetrievalPreparationManifest:
    """Verify the published SQLite bytes and its aggregate sidecar once."""

    if not isinstance(index_path, Path) or not index_path.is_absolute():
        raise ValueError("retrieval index path must be absolute")
    prepared = load_retrieval_preparation_manifest(manifest_path)
    index_size, index_sha256 = _sha256_file(index_path)
    if index_size != prepared.index_file_size_bytes or index_sha256 != prepared.index_file_sha256:
        raise ValueError("retrieval index bytes differ from the aggregate manifest")
    verify_built_index(index_path, expected_manifest=prepared.index_manifest)
    return prepared


def _verify_existing(
    *,
    output_path: Path,
    manifest_path: Path,
    expected_raw_sha256: str,
    input_size_bytes: int,
    corpus_name: str,
    corpus_version: str,
) -> RetrievalPreparationManifest:
    if not output_path.is_file() or not manifest_path.is_file():
        raise ValueError("existing retrieval target and aggregate manifest must form one pair")
    prepared = load_retrieval_preparation_manifest(manifest_path)
    if (
        prepared.input_raw_sha256 != expected_raw_sha256
        or prepared.input_size_bytes != input_size_bytes
        or prepared.index_manifest.corpus_name != corpus_name
        or prepared.index_manifest.corpus_version != corpus_version
    ):
        raise ValueError("existing retrieval target differs from the requested pinned corpus")
    index_size, index_sha256 = _sha256_file(output_path)
    if index_size != prepared.index_file_size_bytes or index_sha256 != prepared.index_file_sha256:
        raise ValueError("existing retrieval index bytes differ from the aggregate manifest")
    with RetrievalIndex.open(output_path) as index:
        if index.manifest != prepared.index_manifest:
            raise ValueError("existing retrieval index manifest differs from its sidecar")
    return prepared


def _same_filesystem(source: Path, target_directory: Path) -> bool:
    return source.stat().st_dev == target_directory.stat().st_dev


def _publish_new_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _same_filesystem(source, target.parent):
        try:
            os.link(source, target)
        except FileExistsError as error:
            raise ValueError("retrieval preparation target appeared during construction") from error
        _fsync_directory(target.parent)
        source.unlink()
        return

    descriptor, publication_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".publish",
        dir=target.parent,
    )
    os.close(descriptor)
    publication = Path(publication_name)
    try:
        with source.open("rb") as source_stream, publication.open("wb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream, length=_COPY_BUFFER_BYTES)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        try:
            os.link(publication, target)
        except FileExistsError as error:
            raise ValueError("retrieval preparation target appeared during construction") from error
        _fsync_directory(target.parent)
    finally:
        if publication.exists():
            publication.unlink()
    source.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_manifest(path: Path, manifest: RetrievalPreparationManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(canonical_json(manifest.to_value()) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("retrieval aggregate manifest appeared during construction") from error
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_atlas_retrieval_index(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    corpus_name: str,
    corpus_version: str,
    expected_raw_sha256: str,
    staging_directory: Path | None = None,
    expected_passage_count: int | None = None,
    performance_gate_rows: int | None = None,
    minimum_ingest_rows_per_second: float | None = None,
    progress_callback: Callable[[RetrievalBuildProgress], None] | None = None,
) -> RetrievalPreparationManifest:
    """Build a lexical FTS5 index over the pinned Atlas/DPR passage source.

    This is deliberately not dense DPR retrieval: the Atlas/DPR artifact is
    the public passage source, while query execution is deterministic SQLite
    FTS5/BM25.  A performance gate builds an explicitly versioned source
    prefix and can reject throughput below the caller's launch threshold.
    """

    for value, field in (
        (input_path, "input_path"),
        (output_path, "output_path"),
        (manifest_path, "manifest_path"),
    ):
        if not isinstance(value, Path):
            raise TypeError(f"{field} must be a Path")
    if staging_directory is not None and not isinstance(staging_directory, Path):
        raise TypeError("staging_directory must be a Path or None")
    if input_path.suffixes[-2:] != [".tsv", ".gz"]:
        raise ValueError("Atlas retrieval input must end in .tsv.gz")
    corpus_name = _text(corpus_name, field="corpus_name")
    corpus_version = _text(corpus_version, field="corpus_version")
    expected_rows = (
        None
        if expected_passage_count is None
        else _positive_int(expected_passage_count, field="expected_passage_count")
    )
    if performance_gate_rows is not None:
        gate_rows = _positive_int(performance_gate_rows, field="performance_gate_rows")
        if gate_rows not in PERFORMANCE_GATE_ROWS:
            raise ValueError("retrieval performance gate must use an approved scale")
        expected_rows = gate_rows
        corpus_version = f"{corpus_version}/lexical-fts5-gate-{gate_rows}"
    if minimum_ingest_rows_per_second is not None:
        if (
            type(minimum_ingest_rows_per_second) not in (int, float)
            or not math.isfinite(float(minimum_ingest_rows_per_second))
            or minimum_ingest_rows_per_second <= 0.0
        ):
            raise ValueError("minimum_ingest_rows_per_second must be positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    validate_sha256(expected_raw_sha256)
    locations = (input_path.absolute(), output_path.absolute(), manifest_path.absolute())
    if len(set(locations)) != len(locations):
        raise ValueError("retrieval input, index, and manifest paths must be distinct")

    input_size, actual_raw_sha256 = _sha256_file(input_path)
    if actual_raw_sha256 != expected_raw_sha256:
        raise ValueError("Atlas retrieval input bytes do not match the pinned raw SHA-256")

    if output_path.exists() or manifest_path.exists():
        return _verify_existing(
            output_path=output_path,
            manifest_path=manifest_path,
            expected_raw_sha256=expected_raw_sha256,
            input_size_bytes=input_size,
            corpus_name=corpus_name,
            corpus_version=corpus_version,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_path.parent if staging_directory is None else staging_directory
    staging_root.mkdir(parents=True, exist_ok=True)
    if not staging_root.is_dir():
        raise ValueError("retrieval staging directory must be a directory")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".staging",
        dir=staging_root,
    )
    os.close(descriptor)
    staging = Path(staging_name)
    staging.unlink()
    last_ingest_progress: RetrievalBuildProgress | None = None

    def observe_progress(progress: RetrievalBuildProgress) -> None:
        nonlocal last_ingest_progress
        if progress.phase is RetrievalBuildPhase.INGEST:
            last_ingest_progress = progress
        if progress_callback is not None:
            progress_callback(progress)

    try:
        with gzip.open(input_path, mode="rt", encoding="utf-8", newline="") as lines:
            passages = iter_atlas_wikipedia_tsv(lines)
            if performance_gate_rows is not None:
                passages = islice(passages, performance_gate_rows)
            built_manifest = build_retrieval_index_stream(
                staging,
                passages,
                corpus_name=corpus_name,
                corpus_version=corpus_version,
                staging_directory=staging_root,
                expected_passage_count=expected_rows,
                progress_callback=observe_progress,
            )
        if minimum_ingest_rows_per_second is not None and (
            last_ingest_progress is None
            or last_ingest_progress.rows_per_second < minimum_ingest_rows_per_second
        ):
            raise RuntimeError("retrieval build did not satisfy the requested ingest-rate gate")
        index_size, index_sha256 = _sha256_file(staging)
        prepared = RetrievalPreparationManifest.create(
            input_raw_sha256=actual_raw_sha256,
            input_size_bytes=input_size,
            index_file_sha256=index_sha256,
            index_file_size_bytes=index_size,
            index_manifest=built_manifest,
        )
        _publish_new_file(staging, output_path)
        _publish_manifest(manifest_path, prepared)
        if load_retrieval_preparation_manifest(manifest_path) != prepared:
            raise ValueError("published retrieval aggregate manifest changed during construction")
        return prepared
    finally:
        if staging.exists():
            staging.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    environment_staging_directory = _environment_path(_STAGING_DIRECTORY_ENV)
    environment_expected_passage_count = _environment_positive_int(_EXPECTED_PASSAGE_COUNT_ENV)
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic lexical SQLite FTS5 retrieval over pinned Atlas/DPR passages "
            "(not dense DPR retrieval)."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument(
        "--expected-passage-count",
        type=int,
        default=environment_expected_passage_count,
    )
    parser.add_argument(
        "--performance-gate-rows",
        type=int,
        choices=PERFORMANCE_GATE_ROWS,
        help="build an explicitly versioned 100k/1M prefix before launching the full build",
    )
    parser.add_argument("--minimum-ingest-rows-per-second", type=float)
    parser.add_argument(
        "--staging-directory",
        type=Path,
        default=environment_staging_directory,
        help="build the disposable SQLite database in this directory (for example node-local SSD)",
    )
    arguments = parser.parse_args(argv)

    def print_progress(progress: RetrievalBuildProgress) -> None:
        eta = "unknown" if progress.eta_seconds is None else f"{progress.eta_seconds:.1f}"
        expected = "unknown" if progress.expected_rows is None else str(progress.expected_rows)
        print(
            "retrieval-build "
            f"phase={progress.phase.value} rows={progress.rows_completed}/{expected} "
            f"rate_rows_per_second={progress.rows_per_second:.1f} eta_seconds={eta}",
            file=sys.stderr,
            flush=True,
        )

    prepared = prepare_atlas_retrieval_index(
        input_path=arguments.input,
        output_path=arguments.output,
        manifest_path=arguments.manifest,
        corpus_name=arguments.corpus_name,
        corpus_version=arguments.corpus_version,
        expected_raw_sha256=arguments.raw_sha256,
        staging_directory=arguments.staging_directory,
        expected_passage_count=arguments.expected_passage_count,
        performance_gate_rows=arguments.performance_gate_rows,
        minimum_ingest_rows_per_second=arguments.minimum_ingest_rows_per_second,
        progress_callback=print_progress,
    )
    print(canonical_json(prepared.to_value()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PERFORMANCE_GATE_ROWS",
    "PREPARATION_FORMAT",
    "RetrievalPreparationManifest",
    "load_retrieval_preparation_manifest",
    "main",
    "prepare_atlas_retrieval_index",
    "verify_retrieval_preparation_output",
]
