"""Strict production loading for the private, non-process benchmark suite.

The loader admits only explicitly named source files whose bytes match the
snapshot identity frozen in the experiment protocol.  Dataset answers remain
inside private case objects and only their answer-free ``RolloutTask``
projections enter :class:`PrivateBenchmarkWorkload`.

Parquet decoding is an injected dependency because the model-facing package
must not depend on a dataframe stack.  The injection is deliberately narrow:
it returns exact row dictionaries and all benchmark-specific schema checking
still happens in the existing converters in this package.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import ijson  # type: ignore[import-untyped]

from skillev.benchmarks import QABenchmark, RetrievalIndex
from skillev.contracts import JsonValue, normalize_json, stable_hash, validate_sha256
from skillev.experiments import Benchmark, DatasetSnapshotIdentity
from skillev.rollout import RolloutTask

from .catalog import PrivateBenchmarkCatalog, PrivateBenchmarkWorkload, PrivateSessionFactory
from .code_math import MathHardSessionFactory
from .converters import (
    convert_atlas_nq_row,
    convert_atlas_triviaqa_row,
    convert_gpqa_diamond_row,
    convert_hotpotqa_row,
    convert_math_hard_row,
    convert_matharena_aime_2026_row,
    convert_medqa_us_4option_row,
    convert_mind2web_row,
    convert_musique_row,
)
from .math_hard_official import SympyMathEquivalenceBackend
from .mind2web import PrivateMind2WebStepSessionFactory
from .mind2web_scores import load_mind2web_candidate_rankings
from .snapshot import create_private_dataset_snapshot
from .source_cases import PrivateMathCase, PrivateMind2WebStepCase
from .static import (
    PrivateRetrievalBenchmarkSessionFactory,
    PrivateStaticBenchmarkCase,
    PrivateStaticBenchmarkSessionFactory,
)


class ProductionSourceFormat(StrEnum):
    """Closed set of wire formats selected explicitly by each source spec."""

    JSONL = "jsonl"
    GZIP_JSONL = "gzip-jsonl"
    JSON_ARRAY = "json-array"
    JSON_OBJECT = "json-object"
    CSV = "csv"
    PARQUET = "parquet"


PRODUCTION_CATALOG_CONFIG_FORMAT = "skillev-private-production-catalog-config@3"


_SUPPORTED_BENCHMARKS = (
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

_EXTERNAL_MANIFEST_BENCHMARKS = frozenset(
    {
        Benchmark.BIRD_SQL,
        Benchmark.MBPP_PLUS,
        Benchmark.TABLEBENCH,
        Benchmark.HUMANEVAL_PLUS,
    }
)

_FORMATS_BY_BENCHMARK = {
    Benchmark.HOTPOT_QA: frozenset({ProductionSourceFormat.PARQUET}),
    Benchmark.TRIVIA_QA: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.AIME_2026: frozenset({ProductionSourceFormat.PARQUET}),
    Benchmark.MED_QA: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.BIRD_SQL: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.MBPP_PLUS: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.MUSIQUE: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.NQ_OPEN: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.MATH_HARD: frozenset({ProductionSourceFormat.PARQUET}),
    Benchmark.GPQA_DIAMOND: frozenset({ProductionSourceFormat.CSV}),
    Benchmark.MIND2WEB: frozenset(
        {ProductionSourceFormat.JSON_ARRAY, ProductionSourceFormat.JSON_OBJECT}
    ),
    Benchmark.TABLEBENCH: frozenset({ProductionSourceFormat.JSONL}),
    Benchmark.HUMANEVAL_PLUS: frozenset({ProductionSourceFormat.JSONL}),
}


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _relative_path(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
        or any(marker in text for marker in ("*", "?", "["))
    ):
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")
    return text


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
class ProductionBenchmarkSource:
    """One exact source file set and its frozen public identity."""

    benchmark: Benchmark
    dataset_revision: str
    split: str
    source_format: ProductionSourceFormat
    relative_files: tuple[str, ...]
    snapshot: DatasetSnapshotIdentity
    relative_file_splits: tuple[str, ...]
    auxiliary_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.benchmark not in _SUPPORTED_BENCHMARKS:
            raise ValueError("production source benchmark is unsupported by this loader")
        _text(self.dataset_revision, field_name="dataset_revision")
        _text(self.split, field_name="split")
        if not isinstance(self.source_format, ProductionSourceFormat):
            raise TypeError("source_format must be ProductionSourceFormat")
        if self.source_format not in _FORMATS_BY_BENCHMARK[self.benchmark]:
            raise ValueError("benchmark source format differs from its declared official wire")
        if not self.relative_files:
            raise ValueError("production source requires explicit files")
        for path in self.relative_files:
            _relative_path(path, field_name="source relative path")
        if self.relative_files != tuple(sorted(self.relative_files)):
            raise ValueError("source files must be in lexicographic order")
        if len(set(self.relative_files)) != len(self.relative_files):
            raise ValueError("source files must be unique")
        for path in self.auxiliary_files:
            _relative_path(path, field_name="source auxiliary path")
        if self.auxiliary_files != tuple(sorted(self.auxiliary_files)):
            raise ValueError("source auxiliary files must be in lexicographic order")
        if len(set(self.auxiliary_files)) != len(self.auxiliary_files):
            raise ValueError("source auxiliary files must be unique")
        if set(self.relative_files).intersection(self.auxiliary_files):
            raise ValueError("source data and auxiliary file sets must be disjoint")
        if type(self.relative_file_splits) is not tuple or len(self.relative_file_splits) != len(
            self.relative_files
        ):
            raise ValueError("per-file split identities must align with source files")
        for split in self.relative_file_splits:
            _text(split, field_name="source file split")
        if not isinstance(self.snapshot, DatasetSnapshotIdentity):
            raise TypeError("source snapshot must be DatasetSnapshotIdentity")
        if (
            self.snapshot.name != self.benchmark.value
            or self.snapshot.version != self.dataset_revision
        ):
            raise ValueError("source revision differs from its frozen snapshot identity")

    @property
    def file_splits(self) -> tuple[str, ...]:
        return self.relative_file_splits

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "auxiliary_files": list(self.auxiliary_files),
            "benchmark": self.benchmark.value,
            "dataset_revision": self.dataset_revision,
            "relative_file_splits": list(self.relative_file_splits),
            "relative_files": list(self.relative_files),
            "snapshot": self.snapshot.to_value(),
            "source_format": self.source_format.value,
            "split": self.split,
        }

    @classmethod
    def from_value(cls, value: object) -> ProductionBenchmarkSource:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "auxiliary_files",
                    "benchmark",
                    "dataset_revision",
                    "relative_file_splits",
                    "relative_files",
                    "snapshot",
                    "source_format",
                    "split",
                }
            ),
            label="production benchmark source",
        )
        raw_files = data["relative_files"]
        raw_auxiliary_files = data["auxiliary_files"]
        raw_splits = data["relative_file_splits"]
        if not isinstance(raw_files, list) or any(type(item) is not str for item in raw_files):
            raise ValueError("production source relative_files must be a text array")
        if not isinstance(raw_auxiliary_files, list) or any(
            type(item) is not str for item in raw_auxiliary_files
        ):
            raise ValueError("production source auxiliary_files must be a text array")
        if not isinstance(raw_splits, list) or any(type(item) is not str for item in raw_splits):
            raise ValueError("production source relative_file_splits must be a text array")
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field_name="benchmark")),
            dataset_revision=_text(
                data["dataset_revision"],
                field_name="dataset_revision",
            ),
            split=_text(data["split"], field_name="split"),
            source_format=ProductionSourceFormat(
                _text(data["source_format"], field_name="source_format")
            ),
            relative_files=tuple(cast(list[str], raw_files)),
            snapshot=DatasetSnapshotIdentity.from_value(data["snapshot"]),
            relative_file_splits=tuple(cast(list[str], raw_splits)),
            auxiliary_files=tuple(cast(list[str], raw_auxiliary_files)),
        )

    @property
    def snapshot_files(self) -> tuple[str, ...]:
        return tuple(sorted((*self.relative_files, *self.auxiliary_files)))


@dataclass(frozen=True, slots=True)
class ProductionCatalogConfig:
    """Complete non-process catalog input with no path discovery rules."""

    dataset_root: Path
    sources: tuple[ProductionBenchmarkSource, ...]
    retrieval_index_relative_path: str
    retrieval_index_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_root, Path) or not self.dataset_root.is_absolute():
            raise ValueError("dataset_root must be an absolute Path")
        if tuple(source.benchmark for source in self.sources) != _SUPPORTED_BENCHMARKS:
            raise ValueError("production sources must match the complete declared order")
        _relative_path(
            self.retrieval_index_relative_path,
            field_name="retrieval_index_relative_path",
        )
        validate_sha256(self.retrieval_index_id)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "dataset_root": self.dataset_root.as_posix(),
            "format": PRODUCTION_CATALOG_CONFIG_FORMAT,
            "retrieval_index_id": self.retrieval_index_id,
            "retrieval_index_relative_path": self.retrieval_index_relative_path,
            "sources": [source.to_value() for source in self.sources],
        }

    @classmethod
    def from_value(cls, value: object) -> ProductionCatalogConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "dataset_root",
                    "format",
                    "retrieval_index_id",
                    "retrieval_index_relative_path",
                    "sources",
                }
            ),
            label="production catalog config",
        )
        if data["format"] != PRODUCTION_CATALOG_CONFIG_FORMAT:
            raise ValueError("unsupported production catalog config format")
        raw_sources = data["sources"]
        if not isinstance(raw_sources, list):
            raise ValueError("production catalog sources must be an array")
        return cls(
            dataset_root=Path(_text(data["dataset_root"], field_name="dataset_root")),
            sources=tuple(ProductionBenchmarkSource.from_value(item) for item in raw_sources),
            retrieval_index_relative_path=_text(
                data["retrieval_index_relative_path"],
                field_name="retrieval_index_relative_path",
            ),
            retrieval_index_id=_text(
                data["retrieval_index_id"],
                field_name="retrieval_index_id",
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


class ParquetRowReader(Protocol):
    """Read one explicitly named Parquet shard without changing row values."""

    def read_rows(self, path: Path) -> tuple[dict[str, object], ...]: ...


class ExternalSessionFactoryBuilder(Protocol):
    """Bind one frozen public task population to its private evaluator."""

    def build(self, tasks: tuple[RolloutTask, ...]) -> PrivateSessionFactory: ...


@dataclass(frozen=True, slots=True)
class PyArrowParquetRowReader:
    """Read exact Parquet rows through the production private dependency.

    ``to_pylist`` preserves Arrow row order and scalar/container types rather
    than routing records through pandas.  Benchmark-specific converters remain
    responsible for their exact schemas; this boundary only rejects an empty
    table or a non-object row representation.
    """

    def read_rows(self, path: Path) -> tuple[dict[str, object], ...]:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Parquet source must be an absolute Path")
        if not path.is_file():
            raise FileNotFoundError(path)

        import pyarrow.parquet as parquet  # type: ignore[import-untyped]

        table = parquet.read_table(path)
        raw_rows = table.to_pylist()
        if type(raw_rows) is not list or not raw_rows:
            raise ValueError("official Parquet source must contain records")
        if any(
            type(row) is not dict or any(type(field_name) is not str for field_name in row)
            for row in raw_rows
        ):
            raise TypeError("official Parquet rows must be exact string-keyed objects")
        return tuple(cast(dict[str, object], row) for row in raw_rows)


@dataclass(frozen=True, slots=True)
class ProductionCatalogDependencies:
    parquet_reader: ParquetRowReader = field(repr=False)
    external_session_factory_builders: tuple[
        tuple[Benchmark, ExternalSessionFactoryBuilder], ...
    ] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.parquet_reader, "read_rows", None)):
            raise TypeError("production parquet dependency must implement read_rows")
        benchmarks = tuple(item[0] for item in self.external_session_factory_builders)
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError("external benchmark session factories must be unique")
        if any(benchmark not in _EXTERNAL_MANIFEST_BENCHMARKS for benchmark in benchmarks):
            raise ValueError("external session factory belongs to an unsupported benchmark")
        if any(
            not callable(getattr(builder, "build", None))
            for _, builder in self.external_session_factory_builders
        ):
            raise TypeError("external benchmark session factory builder must implement build")

    def external_factory(
        self,
        benchmark: Benchmark,
        tasks: tuple[RolloutTask, ...],
    ) -> PrivateSessionFactory:
        matches = tuple(
            builder
            for candidate, builder in self.external_session_factory_builders
            if candidate is benchmark
        )
        if len(matches) != 1:
            raise ValueError(f"{benchmark.value} requires one private session factory builder")
        factory = matches[0].build(tasks)
        if not callable(getattr(factory, "create", None)):
            raise TypeError("external session factory builder returned an invalid factory")
        return factory


@dataclass(slots=True)
class LoadedProductionCatalog:
    """Catalog plus the read-only retrieval handle owned by its factories."""

    catalog: PrivateBenchmarkCatalog
    retrieval_index: RetrievalIndex = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if not self._closed:
            self.retrieval_index.close()
            self._closed = True

    def __enter__(self) -> LoadedProductionCatalog:
        if self._closed:
            raise RuntimeError("production catalog is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("official JSON source contains a duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"official JSON source contains unsupported constant {value}")


def _parse_json(text: str, *, source: Path) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"official JSON source is invalid: {source.name}") from error


class _DuplicateRejectingDict(dict[str, object]):
    def __setitem__(self, key: str, value: object) -> None:
        if key in self:
            raise ValueError("official JSON source contains a duplicate object key")
        super().__setitem__(key, value)


def _plain_streamed_json(value: object) -> object:
    if type(value) is _DuplicateRejectingDict:
        return {key: _plain_streamed_json(item) for key, item in value.items()}
    if type(value) is list:
        return [_plain_streamed_json(item) for item in cast(list[object], value)]
    return value


def _iter_json_lines(lines: Iterable[str], *, source: Path) -> Iterator[dict[str, object]]:
    count = 0
    for raw_line in lines:
        line = raw_line.removesuffix("\n").removesuffix("\r")
        if not line:
            raise ValueError("official JSONL source must contain non-empty records")
        value = _parse_json(line, source=source)
        if type(value) is not dict:
            raise TypeError("official JSONL record must be an object")
        count += 1
        yield cast(dict[str, object], value)
    if count == 0:
        raise ValueError("official JSONL source must contain records")


def _iter_streamed_json(
    path: Path,
    *,
    prefix: str,
) -> Iterator[dict[str, object]]:
    count = 0
    try:
        with path.open("rb") as stream:
            for value in ijson.items(
                stream,
                prefix,
                map_type=_DuplicateRejectingDict,
                use_float=True,
            ):
                if type(value) is not _DuplicateRejectingDict:
                    raise TypeError("official JSON source record must be an object")
                count += 1
                yield cast(dict[str, object], _plain_streamed_json(value))
    except (ijson.JSONError, SystemError) as error:
        raise ValueError(f"official streamed JSON source is invalid: {path.name}") from error
    if count == 0:
        raise ValueError("official JSON source must contain records")


def _iter_one_file(
    path: Path,
    *,
    source_format: ProductionSourceFormat,
    parquet_reader: ParquetRowReader,
) -> Iterator[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if source_format is ProductionSourceFormat.PARQUET:
        rows = parquet_reader.read_rows(path)
        if type(rows) is not tuple or not rows or any(type(row) is not dict for row in rows):
            raise TypeError("Parquet reader must return a non-empty tuple of exact row objects")
        yield from rows
        return
    if source_format is ProductionSourceFormat.GZIP_JSONL:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as stream:
            yield from _iter_json_lines(stream, source=path)
        return
    if source_format is ProductionSourceFormat.JSONL:
        with path.open("r", encoding="utf-8", newline="") as stream:
            yield from _iter_json_lines(stream, source=path)
        return
    if source_format is ProductionSourceFormat.CSV:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, dialect="excel", strict=True)
            if reader.fieldnames is None or any(
                type(field) is not str for field in reader.fieldnames
            ):
                raise ValueError("official CSV source is missing its exact header")
            count = 0
            for row in reader:
                if None in row:
                    raise ValueError("official CSV source has an incompatible row width")
                count += 1
                yield dict(row)
            if count == 0:
                raise ValueError("official CSV source must contain records")
        return
    if source_format is ProductionSourceFormat.JSON_OBJECT:
        yield from _iter_streamed_json(path, prefix="")
        return
    if source_format is not ProductionSourceFormat.JSON_ARRAY:
        raise AssertionError("unhandled production source format")
    yield from _iter_streamed_json(path, prefix="item")


def _source_rows(
    root: Path,
    source: ProductionBenchmarkSource,
    *,
    parquet_reader: ParquetRowReader,
) -> Iterator[tuple[str, dict[str, object]]]:
    actual_snapshot = create_private_dataset_snapshot(
        name=source.benchmark.value,
        version=source.dataset_revision,
        root=root,
        relative_files=source.snapshot_files,
    )
    if actual_snapshot != source.snapshot:
        raise ValueError("production source bytes differ from the frozen snapshot")
    for relative_path, split in zip(
        source.relative_files,
        source.file_splits,
        strict=True,
    ):
        for row in _iter_one_file(
            root / PurePosixPath(relative_path),
            source_format=source.source_format,
            parquet_reader=parquet_reader,
        ):
            yield split, row


def _plain_tasks(cases: tuple[PrivateStaticBenchmarkCase, ...]) -> tuple[RolloutTask, ...]:
    return tuple(case.public.to_rollout_task() for case in cases)


def _retrieval_tasks(
    cases: tuple[PrivateStaticBenchmarkCase, ...],
    index: RetrievalIndex,
) -> tuple[RolloutTask, ...]:
    return tuple(case.public.to_retrieval_rollout_task(index.manifest) for case in cases)


def _static_workload(
    benchmark: Benchmark,
    cases: tuple[PrivateStaticBenchmarkCase, ...],
) -> PrivateBenchmarkWorkload:
    return PrivateBenchmarkWorkload(
        benchmark,
        _plain_tasks(cases),
        PrivateStaticBenchmarkSessionFactory(cases),
    )


def _external_manifest_workload(
    benchmark: Benchmark,
    rows: Iterable[tuple[str, dict[str, object]]],
    dependencies: ProductionCatalogDependencies,
) -> PrivateBenchmarkWorkload:
    """Load answer-free task projections; verifier truth stays in the injected factory."""

    tasks: list[RolloutTask] = []
    for row_split, row in rows:
        if set(row) != {"source_split", "task"}:
            raise ValueError("external benchmark manifest row has an invalid field set")
        if row["source_split"] != row_split:
            raise ValueError("external benchmark manifest split differs from source identity")
        task = RolloutTask.from_value(row["task"])
        context = task.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != benchmark.value:
            raise ValueError("external benchmark task has an incompatible public identity")
        tasks.append(task)
    if not tasks:
        raise ValueError("external benchmark manifest must contain tasks")
    return PrivateBenchmarkWorkload(
        benchmark=benchmark,
        tasks=tuple(tasks),
        session_factory=dependencies.external_factory(benchmark, tuple(tasks)),
    )


def _load_workload(
    source: ProductionBenchmarkSource,
    rows: Iterable[tuple[str, dict[str, object]]],
    *,
    dataset_root: Path,
    index: RetrievalIndex,
    dependencies: ProductionCatalogDependencies,
) -> PrivateBenchmarkWorkload:
    revision = source.dataset_revision
    benchmark = source.benchmark

    if benchmark in _EXTERNAL_MANIFEST_BENCHMARKS:
        return _external_manifest_workload(benchmark, rows, dependencies)

    if benchmark is Benchmark.HOTPOT_QA:
        cases = tuple(
            convert_hotpotqa_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return PrivateBenchmarkWorkload(
            benchmark,
            _retrieval_tasks(cases, index),
            PrivateRetrievalBenchmarkSessionFactory(cases, index, QABenchmark.HOTPOT_QA),
        )
    if benchmark is Benchmark.TRIVIA_QA:
        cases = tuple(
            convert_atlas_triviaqa_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return PrivateBenchmarkWorkload(
            benchmark,
            _retrieval_tasks(cases, index),
            PrivateRetrievalBenchmarkSessionFactory(cases, index, QABenchmark.TRIVIA_QA),
        )
    if benchmark is Benchmark.NQ_OPEN:
        cases = tuple(
            convert_atlas_nq_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return PrivateBenchmarkWorkload(
            benchmark,
            _retrieval_tasks(cases, index),
            PrivateRetrievalBenchmarkSessionFactory(
                cases,
                index,
                QABenchmark.NATURAL_QUESTIONS,
            ),
        )
    if benchmark is Benchmark.MED_QA:
        cases = tuple(
            convert_medqa_us_4option_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return _static_workload(benchmark, cases)
    if benchmark is Benchmark.AIME_2026:
        cases = tuple(
            convert_matharena_aime_2026_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return _static_workload(benchmark, cases)
    if benchmark is Benchmark.MUSIQUE:
        cases = tuple(
            convert_musique_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return _static_workload(benchmark, cases)
    if benchmark is Benchmark.GPQA_DIAMOND:
        cases = tuple(
            convert_gpqa_diamond_row(
                row,
                dataset_revision=revision,
                split=row_split,
            ).to_static_case()
            for row_split, row in rows
        )
        return _static_workload(benchmark, cases)
    if benchmark is Benchmark.MATH_HARD:
        math_cases: tuple[PrivateMathCase, ...] = tuple(
            convert_math_hard_row(row, dataset_revision=revision, split=row_split)
            for row_split, row in rows
        )
        return PrivateBenchmarkWorkload(
            benchmark,
            tuple(case.public.to_rollout_task() for case in math_cases),
            MathHardSessionFactory(math_cases, SympyMathEquivalenceBackend()),
        )
    if benchmark is Benchmark.MIND2WEB:
        if len(source.auxiliary_files) != 1:
            raise ValueError("Mind2Web requires exactly one candidate-score artifact")
        candidate_rankings = load_mind2web_candidate_rankings(
            (dataset_root / PurePosixPath(source.auxiliary_files[0])).resolve()
        )
        mind_cases: tuple[PrivateMind2WebStepCase, ...] = tuple(
            case
            for row_split, row in rows
            for case in convert_mind2web_row(
                row,
                dataset_revision=revision,
                split=row_split,
                candidate_rankings=candidate_rankings,
            )
        )
        return PrivateBenchmarkWorkload(
            benchmark,
            tuple(case.public.to_rollout_task() for case in mind_cases),
            PrivateMind2WebStepSessionFactory(mind_cases),
        )
    raise AssertionError("unhandled production benchmark")


def load_production_benchmark_catalog(
    config: ProductionCatalogConfig,
    dependencies: ProductionCatalogDependencies,
) -> LoadedProductionCatalog:
    """Load all eleven non-process workloads from exact frozen source bytes."""

    if not isinstance(config, ProductionCatalogConfig):
        raise TypeError("production catalog requires ProductionCatalogConfig")
    if not isinstance(dependencies, ProductionCatalogDependencies):
        raise TypeError("production catalog requires ProductionCatalogDependencies")
    if not config.dataset_root.is_dir():
        raise NotADirectoryError(config.dataset_root)
    index_path = config.dataset_root / PurePosixPath(config.retrieval_index_relative_path)
    with ExitStack() as cleanup:
        index = cleanup.enter_context(RetrievalIndex.open(index_path))
        if index.manifest.index_id != config.retrieval_index_id:
            raise ValueError("retrieval index differs from its frozen identity")
        workloads = tuple(
            _load_workload(
                source,
                _source_rows(
                    config.dataset_root,
                    source,
                    parquet_reader=dependencies.parquet_reader,
                ),
                dataset_root=config.dataset_root,
                index=index,
                dependencies=dependencies,
            )
            for source in config.sources
        )
        catalog = PrivateBenchmarkCatalog(workloads)
        cleanup.pop_all()
    return LoadedProductionCatalog(catalog, index)


__all__ = [
    "PRODUCTION_CATALOG_CONFIG_FORMAT",
    "ExternalSessionFactoryBuilder",
    "LoadedProductionCatalog",
    "ParquetRowReader",
    "ProductionBenchmarkSource",
    "ProductionCatalogConfig",
    "ProductionCatalogDependencies",
    "ProductionSourceFormat",
    "PyArrowParquetRowReader",
    "load_production_benchmark_catalog",
]
