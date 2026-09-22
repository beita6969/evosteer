"""Prepare the exact thirteen non-process benchmark source declarations.

Downloaded artifacts and extracted archives remain at their immutable
acquisition paths.  This module performs the one required conversion shared by
TriviaQA and NQ-Open, then binds every production loader input to an explicit
relative file.  It never discovers substitute files and never copies or
deletes source data.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

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
    BenchmarkAcquisitionReceipt,
    _read_published_canonical_record,
)
from .archive_preparation import (
    ArchiveKind,
    ArchivePreparationReceipt,
    LockedArchiveBatchReceipt,
    prepare_archive,
    verify_prepared_archive,
)
from .atlas_qa_preparation import prepare_atlas_qa, verify_atlas_qa_preparation
from .catalog_freeze import ProductionBenchmarkFreezeSpec
from .external_materialization import (
    materialize_bird_sql,
    materialize_evalplus,
    materialize_tablebench,
)
from .medqa_preparation import (
    prepare_medqa_us_four_option,
    verify_medqa_us_four_option_preparation,
)
from .production_catalog import ProductionSourceFormat

NON_PROCESS_PREPARATION_FORMAT = "skillev-private-non-process-preparation@4"
_ATLAS_OUTPUT_RELATIVE_PATH = "_derived/atlas-qa"
_MEDQA_OUTPUT_RELATIVE_PATH = "_derived/medqa-us-four-option"


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
class LockedNonProcessSourcePlan:
    """One exact source declaration consumed by the production catalog."""

    benchmark: Benchmark
    dataset_revision: str
    split: str
    source_format: ProductionSourceFormat
    relative_files: tuple[str, ...]
    relative_file_splits: tuple[str, ...]
    auxiliary_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("non-process source benchmark must be Benchmark")
        if (
            type(self.dataset_revision) is not str
            or len(self.dataset_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.dataset_revision)
        ):
            raise ValueError("non-process dataset revision must be a full lowercase Git commit")
        if type(self.split) is not str or not self.split.strip() or "\x00" in self.split:
            raise ValueError("non-process source split must be non-empty text")
        if not isinstance(self.source_format, ProductionSourceFormat):
            raise TypeError("non-process source_format must be ProductionSourceFormat")
        if not self.relative_files:
            raise ValueError("non-process source requires explicit files")
        for value in self.relative_files:
            _relative_path(value, field="non-process source path")
        if self.relative_files != tuple(sorted(self.relative_files)):
            raise ValueError("non-process source files must be in lexicographic order")
        if len(set(self.relative_files)) != len(self.relative_files):
            raise ValueError("non-process source files must be unique")
        for value in self.auxiliary_files:
            _relative_path(value, field="non-process auxiliary path")
        if self.auxiliary_files != tuple(sorted(self.auxiliary_files)):
            raise ValueError("non-process auxiliary files must be in lexicographic order")
        if len(set(self.auxiliary_files)) != len(self.auxiliary_files):
            raise ValueError("non-process auxiliary files must be unique")
        if set(self.relative_files).intersection(self.auxiliary_files):
            raise ValueError("non-process data and auxiliary files must be disjoint")
        if len(self.relative_file_splits) != len(self.relative_files):
            raise ValueError("non-process file splits must align with source files")
        if any(
            type(value) is not str or not value.strip() or "\x00" in value
            for value in self.relative_file_splits
        ):
            raise ValueError("non-process file split must be non-empty text")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "auxiliary_files": list(self.auxiliary_files),
            "benchmark": self.benchmark.value,
            "dataset_revision": self.dataset_revision,
            "relative_file_splits": list(self.relative_file_splits),
            "relative_files": list(self.relative_files),
            "source_format": self.source_format.value,
            "split": self.split,
        }

    @classmethod
    def from_value(cls, value: object) -> LockedNonProcessSourcePlan:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "auxiliary_files",
                    "benchmark",
                    "dataset_revision",
                    "relative_file_splits",
                    "relative_files",
                    "source_format",
                    "split",
                }
            ),
            label="locked non-process source plan",
        )
        raw_files = data["relative_files"]
        raw_auxiliary_files = data["auxiliary_files"]
        raw_splits = data["relative_file_splits"]
        if not isinstance(raw_files, list) or any(type(item) is not str for item in raw_files):
            raise TypeError("non-process relative_files must be a text array")
        if not isinstance(raw_splits, list) or any(type(item) is not str for item in raw_splits):
            raise TypeError("non-process relative_file_splits must be a text array")
        if not isinstance(raw_auxiliary_files, list) or any(
            type(item) is not str for item in raw_auxiliary_files
        ):
            raise TypeError("non-process auxiliary_files must be a text array")
        text_fields = ("benchmark", "dataset_revision", "source_format", "split")
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("non-process source identity fields must be text")
        return cls(
            benchmark=Benchmark(cast(str, data["benchmark"])),
            dataset_revision=cast(str, data["dataset_revision"]),
            split=cast(str, data["split"]),
            source_format=ProductionSourceFormat(cast(str, data["source_format"])),
            relative_files=tuple(cast(list[str], raw_files)),
            relative_file_splits=tuple(cast(list[str], raw_splits)),
            auxiliary_files=tuple(cast(list[str], raw_auxiliary_files)),
        )

    def freeze_spec(self) -> ProductionBenchmarkFreezeSpec:
        return ProductionBenchmarkFreezeSpec(
            benchmark=self.benchmark,
            dataset_revision=self.dataset_revision,
            split=self.split,
            source_format=self.source_format,
            relative_files=self.relative_files,
            relative_file_splits=self.relative_file_splits,
            auxiliary_files=self.auxiliary_files,
        )


def _mind2web_files() -> tuple[str, ...]:
    paths = [f"mind2web/prepared/test/test_domain/test_domain_{index}.json" for index in range(10)]
    paths.extend(f"mind2web/prepared/test/test_task/test_task_{index}.json" for index in range(3))
    paths.extend(
        f"mind2web/prepared/test/test_website/test_website_{index}.json" for index in range(2)
    )
    return tuple(paths)


def _source_revisions(lock: BenchmarkAcquisitionLock) -> dict[Benchmark, str]:
    return {entry.benchmark: entry.source.revision for entry in lock.benchmarks}


def locked_non_process_source_plans(
    lock: BenchmarkAcquisitionLock,
) -> tuple[LockedNonProcessSourcePlan, ...]:
    """Translate the committed acquisition authority to exact loader files."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    revision = _source_revisions(lock)
    atlas = _ATLAS_OUTPUT_RELATIVE_PATH
    mind2web_files = _mind2web_files()
    plans = (
        LockedNonProcessSourcePlan(
            Benchmark.HOTPOT_QA,
            revision[Benchmark.HOTPOT_QA],
            "validation",
            ProductionSourceFormat.PARQUET,
            ("hotpotqa/distractor/validation-00000-of-00001.parquet",),
            ("validation",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.TRIVIA_QA,
            revision[Benchmark.TRIVIA_QA],
            "dev",
            ProductionSourceFormat.JSONL,
            (f"{atlas}/triviaqa_data/dev.jsonl",),
            ("dev",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.AIME_2026,
            revision[Benchmark.AIME_2026],
            "train",
            ProductionSourceFormat.PARQUET,
            ("aime-2026/data/train-00000-of-00001.parquet",),
            ("train",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.MED_QA,
            revision[Benchmark.MED_QA],
            "official",
            ProductionSourceFormat.JSONL,
            (
                f"{_MEDQA_OUTPUT_RELATIVE_PATH}/dev.jsonl",
                f"{_MEDQA_OUTPUT_RELATIVE_PATH}/test.jsonl",
                f"{_MEDQA_OUTPUT_RELATIVE_PATH}/train.jsonl",
            ),
            ("dev", "test", "train"),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.BIRD_SQL,
            revision[Benchmark.BIRD_SQL],
            "train+dev",
            ProductionSourceFormat.JSONL,
            ("_derived/bird-sql/tasks.jsonl",),
            ("train+dev",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.MBPP_PLUS,
            revision[Benchmark.MBPP_PLUS],
            "test",
            ProductionSourceFormat.JSONL,
            ("_derived/mbpp-plus/tasks.jsonl",),
            ("test",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.MUSIQUE,
            revision[Benchmark.MUSIQUE],
            "dev",
            ProductionSourceFormat.JSONL,
            ("musique/prepared/musique_v1.0/data/musique_ans_v1.0_dev.jsonl",),
            ("dev",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.NQ_OPEN,
            revision[Benchmark.NQ_OPEN],
            "dev",
            ProductionSourceFormat.JSONL,
            (f"{atlas}/nq_data/dev.jsonl",),
            ("dev",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.MATH_HARD,
            revision[Benchmark.MATH_HARD],
            "test",
            ProductionSourceFormat.PARQUET,
            ("math-hard/default/test/0000.parquet",),
            ("test",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.GPQA_DIAMOND,
            revision[Benchmark.GPQA_DIAMOND],
            "train",
            ProductionSourceFormat.CSV,
            ("gpqa-diamond/gpqa_diamond.csv",),
            ("train",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.MIND2WEB,
            revision[Benchmark.MIND2WEB],
            "official",
            ProductionSourceFormat.JSON_ARRAY,
            mind2web_files,
            (
                *(("test_domain",) * 10),
                *(("test_task",) * 3),
                *(("test_website",) * 2),
            ),
            ("mind2web/raw/scores_all_data.pkl",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.TABLEBENCH,
            revision[Benchmark.TABLEBENCH],
            "test",
            ProductionSourceFormat.JSONL,
            ("_derived/tablebench/tasks.jsonl",),
            ("test",),
        ),
        LockedNonProcessSourcePlan(
            Benchmark.HUMANEVAL_PLUS,
            revision[Benchmark.HUMANEVAL_PLUS],
            "test",
            ProductionSourceFormat.JSONL,
            ("_derived/humaneval-plus/tasks.jsonl",),
            ("test",),
        ),
    )
    return plans


@dataclass(frozen=True, slots=True)
class LockedNonProcessPreparationReceipt:
    """Complete private receipt for the thirteen production source plans."""

    acquisition_lock_hash: str
    acquisition_receipt_hash: str
    archive_batch_receipt_hash: str
    atlas_preparation_content_hash: str
    medqa_preparation_content_hash: str
    external_preparation_content_hash: str
    target_root: Path
    sources: tuple[LockedNonProcessSourcePlan, ...]

    def __post_init__(self) -> None:
        for value in (
            self.acquisition_lock_hash,
            self.acquisition_receipt_hash,
            self.archive_batch_receipt_hash,
            self.atlas_preparation_content_hash,
            self.medqa_preparation_content_hash,
            self.external_preparation_content_hash,
        ):
            validate_sha256(value)
        if not isinstance(self.target_root, Path) or not self.target_root.is_absolute():
            raise ValueError("non-process target_root must be an absolute Path")
        expected = (
            Benchmark.HOTPOT_QA,
            Benchmark.TRIVIA_QA,
            Benchmark.AIME_2026,
            Benchmark.MED_QA,
            Benchmark.BIRD_SQL,
            Benchmark.MBPP_PLUS,
            Benchmark.MUSIQUE,
            Benchmark.NQ_OPEN,
            Benchmark.MATH_HARD,
            Benchmark.GPQA_DIAMOND,
            Benchmark.MIND2WEB,
            Benchmark.TABLEBENCH,
            Benchmark.HUMANEVAL_PLUS,
        )
        if tuple(source.benchmark for source in self.sources) != expected:
            raise ValueError("non-process sources must match the production loader order")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "acquisition_lock_hash": self.acquisition_lock_hash,
            "acquisition_receipt_hash": self.acquisition_receipt_hash,
            "archive_batch_receipt_hash": self.archive_batch_receipt_hash,
            "atlas_preparation_content_hash": self.atlas_preparation_content_hash,
            "format": NON_PROCESS_PREPARATION_FORMAT,
            "medqa_preparation_content_hash": self.medqa_preparation_content_hash,
            "external_preparation_content_hash": self.external_preparation_content_hash,
            "sources": [source.to_value() for source in self.sources],
            "target_root": self.target_root.as_posix(),
        }

    @classmethod
    def from_manifest_value(
        cls,
        value: object,
    ) -> LockedNonProcessPreparationReceipt:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "acquisition_lock_hash",
                    "acquisition_receipt_hash",
                    "archive_batch_receipt_hash",
                    "atlas_preparation_content_hash",
                    "content_hash",
                    "format",
                    "medqa_preparation_content_hash",
                    "external_preparation_content_hash",
                    "sources",
                    "target_root",
                }
            ),
            label="locked non-process preparation receipt",
        )
        if data["format"] != NON_PROCESS_PREPARATION_FORMAT:
            raise ValueError("unsupported non-process preparation format")
        text_fields = (
            "acquisition_lock_hash",
            "acquisition_receipt_hash",
            "archive_batch_receipt_hash",
            "atlas_preparation_content_hash",
            "content_hash",
            "medqa_preparation_content_hash",
            "external_preparation_content_hash",
            "target_root",
        )
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("non-process preparation identity fields must be text")
        raw_sources = data["sources"]
        if not isinstance(raw_sources, list):
            raise TypeError("non-process preparation sources must be an array")
        receipt = cls(
            acquisition_lock_hash=cast(str, data["acquisition_lock_hash"]),
            acquisition_receipt_hash=cast(str, data["acquisition_receipt_hash"]),
            archive_batch_receipt_hash=cast(str, data["archive_batch_receipt_hash"]),
            atlas_preparation_content_hash=cast(
                str,
                data["atlas_preparation_content_hash"],
            ),
            medqa_preparation_content_hash=cast(
                str,
                data["medqa_preparation_content_hash"],
            ),
            external_preparation_content_hash=cast(
                str,
                data["external_preparation_content_hash"],
            ),
            target_root=Path(cast(str, data["target_root"])),
            sources=tuple(LockedNonProcessSourcePlan.from_value(source) for source in raw_sources),
        )
        if receipt.content_hash != data["content_hash"]:
            raise ValueError("non-process preparation content hash differs")
        return receipt

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def freeze_specs(self) -> tuple[ProductionBenchmarkFreezeSpec, ...]:
        return tuple(source.freeze_spec() for source in self.sources)


def _validate_receipt_chain(
    *,
    lock: BenchmarkAcquisitionLock,
    acquisition_receipt: BenchmarkAcquisitionReceipt,
    archive_batch_receipt: LockedArchiveBatchReceipt,
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


def _prepare_or_verify_atlas(target_root: Path) -> dict[str, JsonValue]:
    output_root = target_root / _ATLAS_OUTPUT_RELATIVE_PATH
    if output_root.exists():
        return verify_atlas_qa_preparation(output_root)
    return prepare_atlas_qa(
        dataindex_root=target_root / "triviaqa/prepared/dataindex",
        triviaqa_unfiltered_root=(
            target_root / "triviaqa/prepared/triviaqa-unfiltered/triviaqa-unfiltered"
        ),
        nq_train_jsonl=target_root / "nq-open/raw/NQ-open.train.jsonl",
        nq_dev_jsonl=target_root / "nq-open/raw/NQ-open.dev.jsonl",
        output_root=output_root,
    )


def _prepare_or_verify_medqa(
    *,
    target_root: Path,
    dataset_revision: str,
) -> dict[str, JsonValue]:
    output_root = target_root / _MEDQA_OUTPUT_RELATIVE_PATH
    if output_root.exists():
        return verify_medqa_us_four_option_preparation(
            output_root=output_root,
            dataset_revision=dataset_revision,
        )
    return prepare_medqa_us_four_option(
        source_root=(target_root / "medqa/prepared/data_clean/data_clean/questions/US/4_options"),
        output_root=output_root,
        dataset_revision=dataset_revision,
    )


def _unique_named_file(root: Path, name: str) -> Path:
    matches = tuple(
        path
        for path in root.rglob(name)
        if path.is_file() and "__MACOSX" not in path.relative_to(root).parts
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} below {root}")
    return matches[0]


def _unique_named_directory(root: Path, name: str) -> Path:
    matches = tuple(
        path
        for path in root.rglob(name)
        if path.is_dir() and "__MACOSX" not in path.relative_to(root).parts
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} directory below {root}")
    return matches[0]


def _prepare_nested_bird_database_archive(
    *,
    split_root: Path,
    archive_name: str,
    databases_directory_name: str,
    output_root: Path,
) -> tuple[Path, ArchivePreparationReceipt]:
    archive_path = _unique_named_file(split_root, archive_name)
    if output_root.exists():
        receipt = verify_prepared_archive(output_root)
    else:
        with archive_path.open("rb") as stream:
            archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = prepare_archive(
            archive_path=archive_path,
            archive_kind=ArchiveKind.ZIP,
            expected_size=archive_path.stat().st_size,
            expected_sha256=archive_sha256,
            output_root=output_root,
        )
    return (
        _unique_named_directory(output_root, databases_directory_name),
        receipt,
    )


def _prepare_external_completion_sources(
    *,
    target_root: Path,
    revisions: dict[Benchmark, str],
) -> str:
    bird_root = target_root / Benchmark.BIRD_SQL.value / "prepared"
    train_databases, train_archive = _prepare_nested_bird_database_archive(
        split_root=bird_root / "train",
        archive_name="train_databases.zip",
        databases_directory_name="train_databases",
        output_root=bird_root / "train/_nested-databases",
    )
    dev_databases, dev_archive = _prepare_nested_bird_database_archive(
        split_root=bird_root / "dev",
        archive_name="dev_databases.zip",
        databases_directory_name="dev_databases",
        output_root=bird_root / "dev/_nested-databases",
    )
    bird = materialize_bird_sql(
        train_json=_unique_named_file(bird_root / "train", "train.json"),
        train_databases=train_databases,
        dev_json=_unique_named_file(bird_root / "dev", "dev.json"),
        dev_databases=dev_databases,
        dataset_revision=revisions[Benchmark.BIRD_SQL],
        public_manifest=target_root / "_derived/bird-sql/tasks.jsonl",
        private_manifest=target_root / "bird-sql/private/cases.jsonl",
    )
    mbpp = materialize_evalplus(
        benchmark=Benchmark.MBPP_PLUS,
        source_parquet=target_root / "mbpp-plus/data/test.parquet",
        dataset_revision=revisions[Benchmark.MBPP_PLUS],
        public_manifest=target_root / "_derived/mbpp-plus/tasks.jsonl",
        private_manifest=target_root / "mbpp-plus/private/cases.jsonl",
    )
    tablebench = materialize_tablebench(
        source_jsonl=target_root / "tablebench/raw/TableBench.jsonl",
        dataset_revision=revisions[Benchmark.TABLEBENCH],
        public_manifest=target_root / "_derived/tablebench/tasks.jsonl",
        private_manifest=target_root / "tablebench/private/cases.jsonl",
    )
    humaneval = materialize_evalplus(
        benchmark=Benchmark.HUMANEVAL_PLUS,
        source_parquet=target_root / "humaneval-plus/data/test.parquet",
        dataset_revision=revisions[Benchmark.HUMANEVAL_PLUS],
        public_manifest=target_root / "_derived/humaneval-plus/tasks.jsonl",
        private_manifest=target_root / "humaneval-plus/private/cases.jsonl",
    )
    return stable_hash(
        {
            "bird_nested_archives": [
                train_archive.content_hash,
                dev_archive.content_hash,
            ],
            "materializations": [item.to_value() for item in (bird, mbpp, tablebench, humaneval)],
        }
    )


def prepare_locked_non_process_benchmarks(
    *,
    lock: BenchmarkAcquisitionLock,
    acquisition_receipt: BenchmarkAcquisitionReceipt,
    archive_batch_receipt: LockedArchiveBatchReceipt,
    target_root: Path,
) -> LockedNonProcessPreparationReceipt:
    """Prepare Atlas QA and bind every non-process loader to exact files."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(acquisition_receipt, BenchmarkAcquisitionReceipt):
        raise TypeError("acquisition_receipt must be BenchmarkAcquisitionReceipt")
    if not isinstance(archive_batch_receipt, LockedArchiveBatchReceipt):
        raise TypeError("archive_batch_receipt must be LockedArchiveBatchReceipt")
    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    if not target_root.is_dir():
        raise NotADirectoryError(target_root)
    _validate_receipt_chain(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        archive_batch_receipt=archive_batch_receipt,
        target_root=target_root,
    )

    atlas_manifest = _prepare_or_verify_atlas(target_root)
    atlas_hash = atlas_manifest["content_hash"]
    if type(atlas_hash) is not str:
        raise TypeError("Atlas preparation content hash must be text")
    medqa_revision = _source_revisions(lock)[Benchmark.MED_QA]
    medqa_manifest = _prepare_or_verify_medqa(
        target_root=target_root,
        dataset_revision=medqa_revision,
    )
    medqa_hash = medqa_manifest["content_hash"]
    if type(medqa_hash) is not str:
        raise TypeError("MedQA preparation content hash must be text")
    external_hash = _prepare_external_completion_sources(
        target_root=target_root,
        revisions=_source_revisions(lock),
    )
    plans = locked_non_process_source_plans(lock)
    for plan in plans:
        for relative_path in (*plan.relative_files, *plan.auxiliary_files):
            path = target_root / PurePosixPath(relative_path)
            if not path.is_file():
                raise FileNotFoundError(path)
    return LockedNonProcessPreparationReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquisition_receipt.content_hash,
        archive_batch_receipt_hash=archive_batch_receipt.content_hash,
        atlas_preparation_content_hash=atlas_hash,
        medqa_preparation_content_hash=medqa_hash,
        external_preparation_content_hash=external_hash,
        target_root=target_root,
        sources=plans,
    )


def publish_locked_non_process_preparation_receipt(
    receipt: LockedNonProcessPreparationReceipt,
    output_path: Path,
) -> None:
    """Publish one canonical private receipt without replacing prior evidence."""

    if not isinstance(receipt, LockedNonProcessPreparationReceipt):
        raise TypeError("receipt must be LockedNonProcessPreparationReceipt")
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


def load_locked_non_process_preparation_receipt(
    path: Path,
) -> LockedNonProcessPreparationReceipt:
    """Load one canonical private non-process preparation receipt."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("receipt path must be an absolute Path")
    return LockedNonProcessPreparationReceipt.from_manifest_value(
        _read_published_canonical_record(
            path,
            label="locked non-process preparation receipt",
        )
    )


__all__ = [
    "NON_PROCESS_PREPARATION_FORMAT",
    "LockedNonProcessPreparationReceipt",
    "LockedNonProcessSourcePlan",
    "load_locked_non_process_preparation_receipt",
    "locked_non_process_source_plans",
    "prepare_locked_non_process_benchmarks",
    "publish_locked_non_process_preparation_receipt",
]
