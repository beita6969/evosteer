"""Materialize exact private task manifests for the active process benchmarks.

Official simulators are acquired and initialized separately because their
Python/JVM dependencies are mutually incompatible.  This module owns the
small, dependency-free boundary between those official enumerators and the
production catalog: callers supply already observed, answer-free task records
plus an explicit list of deployment assets, and the module writes canonical
JSONL manifests without replacing any existing file.

No benchmark source is discovered here.  Every repository revision, artifact
receipt, archive receipt, task record, and deployment asset must be supplied
explicitly and agree with the committed acquisition lock.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from skillev.benchmarks.task_family import require_benchmark_task_family
from skillev.contracts import (
    JsonValue,
    canonical_json,
    normalize_json,
    stable_hash,
    validate_sha256,
)
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

from .acquisition import (
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
    _read_published_canonical_record,
)
from .acquisition_git import RepositoryAcquisitionReceipt
from .archive_preparation import LockedArchiveBatchReceipt
from .catalog_freeze import ProductionProcessBenchmarkFreezeSpec

PROCESS_PREPARATION_FORMAT = "skillev-private-process-preparation@1"
_PROCESS_BENCHMARKS = (
    Benchmark.WEBSHOP,
    Benchmark.ALFWORLD,
    Benchmark.APPWORLD,
    Benchmark.SCIENCE_WORLD,
    Benchmark.BFCL_V3,
)
_PROCESS_SPLITS = {
    Benchmark.WEBSHOP: "train",
    Benchmark.ALFWORLD: "train",
    Benchmark.APPWORLD: "train",
    Benchmark.SCIENCE_WORLD: "test",
    Benchmark.BFCL_V3: "test",
}
_EXTERNAL_PROCESS_BENCHMARKS = frozenset(
    {
        Benchmark.APPWORLD,
        Benchmark.BFCL_V3,
    }
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


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _relative_path(value: str, *, field: str) -> str:
    text = _text(value, field=field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    return text


def _revision(value: object, *, field: str) -> str:
    revision = _text(value, field=field).lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"{field} must be a full lowercase Git commit")
    return revision


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


def _text_array(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    result = tuple(_text(item, field=field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class WebShopProcessTaskRecord:
    task_id: str
    task_family: str
    query: str
    public_context: JsonValue
    goal_id: str
    session_id: str
    goal_index: int

    def __post_init__(self) -> None:
        for field, value in (
            ("WebShop task_id", self.task_id),
            ("WebShop task_family", self.task_family),
            ("WebShop query", self.query),
            ("WebShop goal_id", self.goal_id),
            ("WebShop session_id", self.session_id),
        ):
            _text(value, field=field)
        require_benchmark_task_family(
            benchmark_id=Benchmark.WEBSHOP.value,
            task_family=self.task_family,
        )
        context = normalize_json(self.public_context)
        if not isinstance(context, dict):
            raise TypeError("WebShop public_context must be an object")
        object.__setattr__(self, "public_context", context)
        _non_negative_int(self.goal_index, field="WebShop goal_index")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "goal_id": self.goal_id,
            "goal_index": self.goal_index,
            "public_context": self.public_context,
            "query": self.query,
            "session_id": self.session_id,
            "task_family": self.task_family,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> WebShopProcessTaskRecord:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "goal_id",
                    "goal_index",
                    "public_context",
                    "query",
                    "session_id",
                    "task_family",
                    "task_id",
                }
            ),
            label="WebShop process task",
        )
        return cls(
            task_id=_text(data["task_id"], field="WebShop task_id"),
            task_family=_text(data["task_family"], field="WebShop task_family"),
            query=_text(data["query"], field="WebShop query"),
            public_context=data["public_context"],
            goal_id=_text(data["goal_id"], field="WebShop goal_id"),
            session_id=_text(data["session_id"], field="WebShop session_id"),
            goal_index=_non_negative_int(
                data["goal_index"],
                field="WebShop goal_index",
            ),
        )


@dataclass(frozen=True, slots=True)
class ALFWorldProcessTaskRecord:
    task_id: str
    task_family: str
    query: str
    game_id: str
    initial_observation: str
    admissible_commands: tuple[str, ...]
    max_steps: int

    def __post_init__(self) -> None:
        for field, value in (
            ("ALFWorld task_id", self.task_id),
            ("ALFWorld task_family", self.task_family),
            ("ALFWorld query", self.query),
            ("ALFWorld game_id", self.game_id),
            ("ALFWorld initial_observation", self.initial_observation),
        ):
            _text(value, field=field)
        require_benchmark_task_family(
            benchmark_id=Benchmark.ALFWORLD.value,
            task_family=self.task_family,
        )
        if (
            not isinstance(self.admissible_commands, tuple)
            or not self.admissible_commands
            or any(type(item) is not str for item in self.admissible_commands)
        ):
            raise ValueError("ALFWorld admissible_commands must be a non-empty text tuple")
        for command in self.admissible_commands:
            _text(command, field="ALFWorld admissible command")
        if len(set(self.admissible_commands)) != len(self.admissible_commands):
            raise ValueError("ALFWorld admissible_commands must be unique")
        _positive_int(self.max_steps, field="ALFWorld max_steps")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "admissible_commands": list(self.admissible_commands),
            "game_id": self.game_id,
            "initial_observation": self.initial_observation,
            "max_steps": self.max_steps,
            "query": self.query,
            "task_family": self.task_family,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ALFWorldProcessTaskRecord:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "admissible_commands",
                    "game_id",
                    "initial_observation",
                    "max_steps",
                    "query",
                    "task_family",
                    "task_id",
                }
            ),
            label="ALFWorld process task",
        )
        return cls(
            task_id=_text(data["task_id"], field="ALFWorld task_id"),
            task_family=_text(data["task_family"], field="ALFWorld task_family"),
            query=_text(data["query"], field="ALFWorld query"),
            game_id=_text(data["game_id"], field="ALFWorld game_id"),
            initial_observation=_text(
                data["initial_observation"],
                field="ALFWorld initial_observation",
            ),
            admissible_commands=_text_array(
                data["admissible_commands"],
                field="ALFWorld admissible_commands",
            ),
            max_steps=_positive_int(data["max_steps"], field="ALFWorld max_steps"),
        )


@dataclass(frozen=True, slots=True)
class ScienceWorldProcessTaskRecord:
    task_id: str
    task_family: str
    query: str
    task_name: str
    variation_index: int
    initial_observation: str
    max_steps: int

    def __post_init__(self) -> None:
        for field, value in (
            ("ScienceWorld task_id", self.task_id),
            ("ScienceWorld task_family", self.task_family),
            ("ScienceWorld query", self.query),
            ("ScienceWorld task_name", self.task_name),
            ("ScienceWorld initial_observation", self.initial_observation),
        ):
            _text(value, field=field)
        require_benchmark_task_family(
            benchmark_id=Benchmark.SCIENCE_WORLD.value,
            task_family=self.task_family,
        )
        _non_negative_int(
            self.variation_index,
            field="ScienceWorld variation_index",
        )
        _positive_int(self.max_steps, field="ScienceWorld max_steps")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "initial_observation": self.initial_observation,
            "max_steps": self.max_steps,
            "query": self.query,
            "task_family": self.task_family,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "variation_index": self.variation_index,
        }

    @classmethod
    def from_value(cls, value: object) -> ScienceWorldProcessTaskRecord:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "initial_observation",
                    "max_steps",
                    "query",
                    "task_family",
                    "task_id",
                    "task_name",
                    "variation_index",
                }
            ),
            label="ScienceWorld process task",
        )
        return cls(
            task_id=_text(data["task_id"], field="ScienceWorld task_id"),
            task_family=_text(data["task_family"], field="ScienceWorld task_family"),
            query=_text(data["query"], field="ScienceWorld query"),
            task_name=_text(data["task_name"], field="ScienceWorld task_name"),
            variation_index=_non_negative_int(
                data["variation_index"],
                field="ScienceWorld variation_index",
            ),
            initial_observation=_text(
                data["initial_observation"],
                field="ScienceWorld initial_observation",
            ),
            max_steps=_positive_int(
                data["max_steps"],
                field="ScienceWorld max_steps",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExternalProcessTaskRecord:
    """Answer-free public task projection for an isolated official worker."""

    benchmark: Benchmark
    source_split: str
    task: RolloutTask

    def __post_init__(self) -> None:
        if self.benchmark not in _EXTERNAL_PROCESS_BENCHMARKS:
            raise ValueError("external process task benchmark is unsupported")
        if self.source_split != _PROCESS_SPLITS[self.benchmark]:
            raise ValueError("external process task split differs from protocol")
        if not isinstance(self.task, RolloutTask):
            raise TypeError("external process record requires RolloutTask")
        require_benchmark_task_family(
            benchmark_id=self.benchmark.value,
            task_family=self.task.task_family,
        )
        context = self.task.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != self.benchmark.value:
            raise ValueError("external process task benchmark identity differs")

    @property
    def task_id(self) -> str:
        return self.task.task_id

    def to_value(self) -> dict[str, JsonValue]:
        return {"source_split": self.source_split, "task": self.task.to_value()}

    @classmethod
    def from_value(
        cls,
        benchmark: Benchmark,
        value: object,
    ) -> ExternalProcessTaskRecord:
        data = _wire_object(
            value,
            fields=frozenset({"source_split", "task"}),
            label="external process task",
        )
        return cls(
            benchmark=benchmark,
            source_split=_text(data["source_split"], field="external process split"),
            task=RolloutTask.from_value(data["task"]),
        )


ProcessTaskRecord = (
    WebShopProcessTaskRecord
    | ALFWorldProcessTaskRecord
    | ScienceWorldProcessTaskRecord
    | ExternalProcessTaskRecord
)


def _record_for_benchmark(
    benchmark: Benchmark,
    value: object,
) -> ProcessTaskRecord:
    if benchmark is Benchmark.WEBSHOP:
        return (
            value
            if isinstance(value, WebShopProcessTaskRecord)
            else WebShopProcessTaskRecord.from_value(value)
        )
    if benchmark is Benchmark.ALFWORLD:
        return (
            value
            if isinstance(value, ALFWorldProcessTaskRecord)
            else ALFWorldProcessTaskRecord.from_value(value)
        )
    if benchmark is Benchmark.SCIENCE_WORLD:
        return (
            value
            if isinstance(value, ScienceWorldProcessTaskRecord)
            else ScienceWorldProcessTaskRecord.from_value(value)
        )
    if benchmark in _EXTERNAL_PROCESS_BENCHMARKS:
        if isinstance(value, ExternalProcessTaskRecord):
            if value.benchmark is not benchmark:
                raise ValueError("external process record benchmark differs")
            return value
        return ExternalProcessTaskRecord.from_value(benchmark, value)
    raise ValueError("process task benchmark is not interactive")


@dataclass(frozen=True, slots=True)
class ProcessTaskManifestInput:
    """Answer-free task records and exact assets from one official enumerator."""

    benchmark: Benchmark
    dataset_revision: str
    split: str
    task_manifest_relative_path: str
    asset_relative_files: tuple[str, ...]
    environment_source_revision: str
    records: tuple[ProcessTaskRecord, ...]

    def __post_init__(self) -> None:
        if self.benchmark not in _PROCESS_BENCHMARKS:
            raise ValueError("process manifest input benchmark is not interactive")
        revision = _revision(
            self.dataset_revision,
            field="process dataset_revision",
        )
        environment_revision = _revision(
            self.environment_source_revision,
            field="process environment_source_revision",
        )
        if revision != environment_revision:
            raise ValueError("process dataset and environment revisions must match")
        if self.split != _PROCESS_SPLITS[self.benchmark]:
            raise ValueError("process manifest input uses an unsupported split")
        manifest = _relative_path(
            self.task_manifest_relative_path,
            field="process task manifest path",
        )
        if not manifest.endswith(".jsonl"):
            raise ValueError("process task manifest path must end in .jsonl")
        if not isinstance(self.asset_relative_files, tuple) or not self.asset_relative_files:
            raise ValueError("process manifest input requires explicit deployment assets")
        for path in self.asset_relative_files:
            _relative_path(path, field="process deployment asset path")
        if self.asset_relative_files != tuple(sorted(self.asset_relative_files)):
            raise ValueError("process deployment assets must be in lexicographic order")
        if len(set(self.asset_relative_files)) != len(self.asset_relative_files):
            raise ValueError("process deployment assets must be unique")
        if manifest in self.asset_relative_files:
            raise ValueError("task manifest cannot be supplied as a deployment asset")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("process manifest input requires task records")
        normalized_records = tuple(
            _record_for_benchmark(self.benchmark, record) for record in self.records
        )
        task_ids = tuple(record.task_id for record in normalized_records)
        if task_ids != tuple(sorted(task_ids)):
            raise ValueError("process task records must be sorted by task_id")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("process task record identities must be unique")
        private_identities: tuple[object, ...]
        if self.benchmark is Benchmark.WEBSHOP:
            private_identities = tuple(
                cast(WebShopProcessTaskRecord, record).goal_index for record in normalized_records
            )
        elif self.benchmark is Benchmark.ALFWORLD:
            private_identities = tuple(
                cast(ALFWorldProcessTaskRecord, record).game_id for record in normalized_records
            )
        elif self.benchmark is Benchmark.SCIENCE_WORLD:
            private_identities = tuple(
                (
                    cast(ScienceWorldProcessTaskRecord, record).task_name,
                    cast(ScienceWorldProcessTaskRecord, record).variation_index,
                )
                for record in normalized_records
            )
        else:
            private_identities = tuple(
                cast(ExternalProcessTaskRecord, record).task_id for record in normalized_records
            )
        if len(set(private_identities)) != len(private_identities):
            raise ValueError("process task records repeat an official environment identity")
        object.__setattr__(self, "dataset_revision", revision)
        object.__setattr__(self, "environment_source_revision", environment_revision)
        object.__setattr__(self, "records", normalized_records)

    @property
    def snapshot_relative_files(self) -> tuple[str, ...]:
        return tuple(sorted((*self.asset_relative_files, self.task_manifest_relative_path)))

    def manifest_bytes(self) -> bytes:
        return "".join(canonical_json(record.to_value()) + "\n" for record in self.records).encode(
            "utf-8"
        )


@dataclass(frozen=True, slots=True)
class LockedProcessSourcePlan:
    benchmark: Benchmark
    dataset_revision: str
    split: str
    task_manifest_relative_path: str
    snapshot_relative_files: tuple[str, ...]
    environment_source_revision: str
    task_count: int
    task_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.benchmark not in _PROCESS_BENCHMARKS:
            raise ValueError("process source plan benchmark is not interactive")
        _revision(self.dataset_revision, field="process dataset_revision")
        _revision(
            self.environment_source_revision,
            field="process environment_source_revision",
        )
        if self.split != _PROCESS_SPLITS[self.benchmark]:
            raise ValueError("process source plan uses an unsupported split")
        manifest = _relative_path(
            self.task_manifest_relative_path,
            field="process task manifest path",
        )
        if not isinstance(self.snapshot_relative_files, tuple) or not self.snapshot_relative_files:
            raise ValueError("process source plan requires snapshot files")
        for path in self.snapshot_relative_files:
            _relative_path(path, field="process snapshot path")
        if self.snapshot_relative_files != tuple(sorted(self.snapshot_relative_files)):
            raise ValueError("process snapshot paths must be in lexicographic order")
        if len(set(self.snapshot_relative_files)) != len(self.snapshot_relative_files):
            raise ValueError("process snapshot paths must be unique")
        if manifest not in self.snapshot_relative_files:
            raise ValueError("process snapshot must include its task manifest")
        _positive_int(self.task_count, field="process task_count")
        validate_sha256(self.task_manifest_sha256)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "dataset_revision": self.dataset_revision,
            "environment_source_revision": self.environment_source_revision,
            "snapshot_relative_files": list(self.snapshot_relative_files),
            "split": self.split,
            "task_count": self.task_count,
            "task_manifest_relative_path": self.task_manifest_relative_path,
            "task_manifest_sha256": self.task_manifest_sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> LockedProcessSourcePlan:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "benchmark",
                    "dataset_revision",
                    "environment_source_revision",
                    "snapshot_relative_files",
                    "split",
                    "task_count",
                    "task_manifest_relative_path",
                    "task_manifest_sha256",
                }
            ),
            label="locked process source plan",
        )
        files = data["snapshot_relative_files"]
        if not isinstance(files, list) or any(type(item) is not str for item in files):
            raise TypeError("process snapshot_relative_files must be a text array")
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="benchmark")),
            dataset_revision=_text(
                data["dataset_revision"],
                field="process dataset_revision",
            ),
            split=_text(data["split"], field="process split"),
            task_manifest_relative_path=_text(
                data["task_manifest_relative_path"],
                field="process task manifest path",
            ),
            snapshot_relative_files=tuple(cast(list[str], files)),
            environment_source_revision=_text(
                data["environment_source_revision"],
                field="process environment_source_revision",
            ),
            task_count=_positive_int(data["task_count"], field="process task_count"),
            task_manifest_sha256=_text(
                data["task_manifest_sha256"],
                field="process task manifest SHA-256",
            ),
        )

    def freeze_spec(self) -> ProductionProcessBenchmarkFreezeSpec:
        return ProductionProcessBenchmarkFreezeSpec(
            benchmark=self.benchmark,
            dataset_revision=self.dataset_revision,
            split=self.split,
            task_manifest_relative_path=self.task_manifest_relative_path,
            snapshot_relative_files=self.snapshot_relative_files,
            environment_source_revision=self.environment_source_revision,
        )


@dataclass(frozen=True, slots=True)
class LockedProcessPreparationReceipt:
    acquisition_lock_hash: str
    acquisition_receipt_hash: str
    archive_batch_receipt_hash: str
    repository_receipt_hash: str
    target_root: Path
    sources: tuple[LockedProcessSourcePlan, ...]

    def __post_init__(self) -> None:
        for value in (
            self.acquisition_lock_hash,
            self.acquisition_receipt_hash,
            self.archive_batch_receipt_hash,
            self.repository_receipt_hash,
        ):
            validate_sha256(value)
        if not isinstance(self.target_root, Path) or not self.target_root.is_absolute():
            raise ValueError("process target_root must be an absolute Path")
        if tuple(source.benchmark for source in self.sources) != _PROCESS_BENCHMARKS:
            raise ValueError("process sources must match the production loader order")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "acquisition_lock_hash": self.acquisition_lock_hash,
            "acquisition_receipt_hash": self.acquisition_receipt_hash,
            "archive_batch_receipt_hash": self.archive_batch_receipt_hash,
            "format": PROCESS_PREPARATION_FORMAT,
            "repository_receipt_hash": self.repository_receipt_hash,
            "sources": [source.to_value() for source in self.sources],
            "target_root": self.target_root.as_posix(),
        }

    @classmethod
    def from_manifest_value(
        cls,
        value: object,
    ) -> LockedProcessPreparationReceipt:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "acquisition_lock_hash",
                    "acquisition_receipt_hash",
                    "archive_batch_receipt_hash",
                    "content_hash",
                    "format",
                    "repository_receipt_hash",
                    "sources",
                    "target_root",
                }
            ),
            label="locked process preparation receipt",
        )
        if data["format"] != PROCESS_PREPARATION_FORMAT:
            raise ValueError("unsupported process preparation format")
        text_fields = (
            "acquisition_lock_hash",
            "acquisition_receipt_hash",
            "archive_batch_receipt_hash",
            "content_hash",
            "repository_receipt_hash",
            "target_root",
        )
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("process preparation identity fields must be text")
        sources = data["sources"]
        if not isinstance(sources, list):
            raise TypeError("process preparation sources must be an array")
        receipt = cls(
            acquisition_lock_hash=cast(str, data["acquisition_lock_hash"]),
            acquisition_receipt_hash=cast(str, data["acquisition_receipt_hash"]),
            archive_batch_receipt_hash=cast(
                str,
                data["archive_batch_receipt_hash"],
            ),
            repository_receipt_hash=cast(str, data["repository_receipt_hash"]),
            target_root=Path(cast(str, data["target_root"])),
            sources=tuple(LockedProcessSourcePlan.from_value(item) for item in sources),
        )
        if receipt.content_hash != data["content_hash"]:
            raise ValueError("process preparation content hash differs")
        return receipt

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def freeze_specs(self) -> tuple[ProductionProcessBenchmarkFreezeSpec, ...]:
        return tuple(source.freeze_spec() for source in self.sources)


def _validate_receipt_chain(
    *,
    lock: BenchmarkAcquisitionLock,
    acquisition_receipt: BenchmarkAcquisitionReceipt,
    archive_batch_receipt: LockedArchiveBatchReceipt,
    repository_receipt: RepositoryAcquisitionReceipt,
    target_root: Path,
) -> None:
    if acquisition_receipt.acquisition_lock_hash != lock.content_hash:
        raise ValueError("acquisition receipt belongs to a different source lock")
    if acquisition_receipt.target_root != target_root:
        raise ValueError("acquisition receipt belongs to a different target root")
    expected_artifacts = tuple(
        (entry.benchmark, artifact.locator, artifact.relative_path)
        for entry in lock.benchmarks
        for artifact in entry.source.artifacts
    )
    measured_artifacts = tuple(
        (artifact.benchmark, artifact.locator, artifact.relative_path)
        for artifact in acquisition_receipt.artifacts
    )
    if measured_artifacts != expected_artifacts:
        raise ValueError("acquisition receipt differs from the complete locked source order")
    if archive_batch_receipt.acquisition_lock_hash != lock.content_hash:
        raise ValueError("archive receipt belongs to a different source lock")
    if archive_batch_receipt.acquisition_receipt_hash != acquisition_receipt.content_hash:
        raise ValueError("archive receipt belongs to a different acquisition receipt")
    if repository_receipt.acquisition_lock_hash != lock.content_hash:
        raise ValueError("repository receipt belongs to a different source lock")
    if repository_receipt.target_root != target_root:
        raise ValueError("repository receipt belongs to a different target root")
    repositories = {
        repository.benchmark: repository for repository in repository_receipt.repositories
    }
    for benchmark in _PROCESS_BENCHMARKS:
        entry = next(item for item in lock.benchmarks if item.benchmark is benchmark)
        measured = repositories.get(benchmark)
        if entry.source.kind not in _GIT_SOURCE_KINDS:
            if measured is not None:
                raise ValueError("non-Git process source has a repository receipt")
            continue
        if (
            measured is None
            or measured.repository != entry.source.repository
            or measured.revision != entry.source.revision
        ):
            raise ValueError("process repository receipt differs from the locked source")


def _write_or_verify_manifest(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError("existing process task manifest differs from official enumeration")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_locked_process_benchmarks(
    *,
    lock: BenchmarkAcquisitionLock,
    acquisition_receipt: BenchmarkAcquisitionReceipt,
    archive_batch_receipt: LockedArchiveBatchReceipt,
    repository_receipt: RepositoryAcquisitionReceipt,
    target_root: Path,
    inputs: tuple[ProcessTaskManifestInput, ...],
) -> LockedProcessPreparationReceipt:
    """Persist the exact official task records and bind their deployment files."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(acquisition_receipt, BenchmarkAcquisitionReceipt):
        raise TypeError("acquisition_receipt must be BenchmarkAcquisitionReceipt")
    if not isinstance(archive_batch_receipt, LockedArchiveBatchReceipt):
        raise TypeError("archive_batch_receipt must be LockedArchiveBatchReceipt")
    if not isinstance(repository_receipt, RepositoryAcquisitionReceipt):
        raise TypeError("repository_receipt must be RepositoryAcquisitionReceipt")
    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    if not target_root.is_dir():
        raise NotADirectoryError(target_root)
    if tuple(item.benchmark for item in inputs) != _PROCESS_BENCHMARKS:
        raise ValueError("process inputs must match the production loader order")
    _validate_receipt_chain(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        archive_batch_receipt=archive_batch_receipt,
        repository_receipt=repository_receipt,
        target_root=target_root,
    )

    plans: list[LockedProcessSourcePlan] = []
    for item in inputs:
        locked = next(entry for entry in lock.benchmarks if entry.benchmark is item.benchmark)
        if (
            item.environment_source_revision != locked.source.revision
            or item.dataset_revision != locked.source.revision
        ):
            raise ValueError("process manifest input differs from the locked source revision")
        for relative_path in item.asset_relative_files:
            asset = target_root / PurePosixPath(relative_path)
            if not asset.is_file():
                raise FileNotFoundError(asset)
        payload = item.manifest_bytes()
        manifest_path = target_root / PurePosixPath(item.task_manifest_relative_path)
        _write_or_verify_manifest(manifest_path, payload)
        plans.append(
            LockedProcessSourcePlan(
                benchmark=item.benchmark,
                dataset_revision=item.dataset_revision,
                split=item.split,
                task_manifest_relative_path=item.task_manifest_relative_path,
                snapshot_relative_files=item.snapshot_relative_files,
                environment_source_revision=item.environment_source_revision,
                task_count=len(item.records),
                task_manifest_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
            )
        )
    return LockedProcessPreparationReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquisition_receipt.content_hash,
        archive_batch_receipt_hash=archive_batch_receipt.content_hash,
        repository_receipt_hash=repository_receipt.content_hash,
        target_root=target_root,
        sources=tuple(plans),
    )


def publish_locked_process_preparation_receipt(
    receipt: LockedProcessPreparationReceipt,
    output_path: Path,
) -> None:
    if not isinstance(receipt, LockedProcessPreparationReceipt):
        raise TypeError("receipt must be LockedProcessPreparationReceipt")
    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("output_path must be an absolute Path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    value = {**receipt.to_value(), "content_hash": receipt.content_hash}
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_locked_process_preparation_receipt(
    path: Path,
) -> LockedProcessPreparationReceipt:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("receipt path must be an absolute Path")
    return LockedProcessPreparationReceipt.from_manifest_value(
        _read_published_canonical_record(
            path,
            label="locked process preparation receipt",
        )
    )


__all__ = [
    "PROCESS_PREPARATION_FORMAT",
    "ALFWorldProcessTaskRecord",
    "ExternalProcessTaskRecord",
    "LockedProcessPreparationReceipt",
    "LockedProcessSourcePlan",
    "ProcessTaskManifestInput",
    "ScienceWorldProcessTaskRecord",
    "WebShopProcessTaskRecord",
    "load_locked_process_preparation_receipt",
    "prepare_locked_process_benchmarks",
    "publish_locked_process_preparation_receipt",
]
