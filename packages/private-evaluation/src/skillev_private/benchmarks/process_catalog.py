"""Strict production catalog assembly for the active interactive simulators.

WebShop, ALFWorld, and ScienceWorld cannot be decoded as ordinary static
rows: each public task must be paired with a deployment-only simulator
identity.  This module loads those pairings from explicitly named, frozen
JSONL manifests and wires the existing public/private session adapters to the
pinned official process factories.

The loader performs no source discovery, download, environment probing, or
substitution.  Every task manifest and every deployment asset used by a
factory must be part of the source's declared byte snapshot.  Private goal,
game, and variation identities never enter the public ``RolloutTask``.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import cast

from skillev.benchmarks import (
    ALFWorldPublicItem,
    ScienceWorldPublicItem,
    WebShopPublicItem,
)
from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.experiments import ACTIVE_BENCHMARKS, FIXED_SEED, Benchmark, DatasetSnapshotIdentity
from skillev.rollout import RolloutTask

from .alfworld import PrivateALFWorldCase, PrivateALFWorldSessionFactory
from .alfworld_official import OfficialALFWorldEpisodeFactory, OfficialALFWorldTask
from .catalog import PrivateBenchmarkCatalog, PrivateBenchmarkWorkload, PrivateSessionFactory
from .official_process import (
    JsonArrayWebShopDeployment,
    OfficialALFWorldProcessFactory,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    SQLiteWebShopDeployment,
)
from .production_catalog import (
    LoadedProductionCatalog,
    ProductionCatalogConfig,
    ProductionCatalogDependencies,
    load_production_benchmark_catalog,
)
from .scienceworld import PrivateScienceWorldCase, PrivateScienceWorldSessionFactory
from .scienceworld_official import (
    OfficialScienceWorldEpisodeFactory,
    OfficialScienceWorldTask,
)
from .snapshot import create_private_dataset_snapshot
from .webshop import PrivateWebShopCase, PrivateWebShopSessionFactory
from .webshop_official import OfficialWebShopEpisodeFactory, OfficialWebShopGoal

_PROCESS_BENCHMARKS = (
    Benchmark.WEBSHOP,
    Benchmark.ALFWORLD,
    Benchmark.APPWORLD,
    Benchmark.SCIENCE_WORLD,
    Benchmark.BFCL_V3,
)
_NON_PROCESS_BENCHMARKS = (
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
_EXTERNAL_PROCESS_BENCHMARKS = frozenset(
    {
        Benchmark.APPWORLD,
        Benchmark.BFCL_V3,
    }
)
_TRAINING_PROCESS_BENCHMARKS = frozenset(
    {
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.APPWORLD,
    }
)
_COMPLETE_BENCHMARK_ORDER = ACTIVE_BENCHMARKS
_REVISION_LENGTH = 40
PRODUCTION_PROCESS_CATALOG_CONFIG_FORMAT = "skillev-private-production-process-catalog-config@2"

_WEBSHOP_FIELDS = frozenset(
    {
        "goal_id",
        "goal_index",
        "public_context",
        "query",
        "session_id",
        "task_family",
        "task_id",
    }
)
_ALFWORLD_FIELDS = frozenset(
    {
        "admissible_commands",
        "game_id",
        "initial_observation",
        "max_steps",
        "query",
        "task_family",
        "task_id",
    }
)
_SCIENCEWORLD_FIELDS = frozenset(
    {
        "initial_observation",
        "max_steps",
        "query",
        "task_family",
        "task_id",
        "task_name",
        "variation_index",
    }
)


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _relative_path(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")
    return text


def _revision(value: object) -> str:
    revision = _text(value, field_name="environment_source_revision").lower()
    if len(revision) != _REVISION_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("environment_source_revision must be a full Git commit")
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


@dataclass(frozen=True, slots=True)
class ProductionProcessBenchmarkSource:
    """One exact process-task manifest and its deployment-asset snapshot."""

    benchmark: Benchmark
    dataset_revision: str
    split: str
    task_manifest_relative_path: str
    snapshot_relative_files: tuple[str, ...]
    snapshot: DatasetSnapshotIdentity
    environment_source_revision: str

    def __post_init__(self) -> None:
        if self.benchmark not in _PROCESS_BENCHMARKS:
            raise ValueError("process source benchmark is not interactive")
        _text(self.dataset_revision, field_name="dataset_revision")
        _text(self.split, field_name="split")
        manifest = _relative_path(
            self.task_manifest_relative_path,
            field_name="task_manifest_relative_path",
        )
        if not manifest.endswith(".jsonl"):
            raise ValueError("process task manifest must be JSONL")
        if not self.snapshot_relative_files:
            raise ValueError("process source requires explicit snapshot files")
        for value in self.snapshot_relative_files:
            _relative_path(value, field_name="snapshot relative path")
        if self.snapshot_relative_files != tuple(sorted(self.snapshot_relative_files)):
            raise ValueError("process snapshot files must be in lexicographic order")
        if len(set(self.snapshot_relative_files)) != len(self.snapshot_relative_files):
            raise ValueError("process snapshot files must be unique")
        if manifest not in self.snapshot_relative_files:
            raise ValueError("process task manifest must belong to its frozen snapshot")
        if not isinstance(self.snapshot, DatasetSnapshotIdentity):
            raise TypeError("process source snapshot must be DatasetSnapshotIdentity")
        if (
            self.snapshot.name != self.benchmark.value
            or self.snapshot.version != self.dataset_revision
        ):
            raise ValueError("process source identity differs from its frozen snapshot")
        object.__setattr__(
            self,
            "environment_source_revision",
            _revision(self.environment_source_revision),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "dataset_revision": self.dataset_revision,
            "environment_source_revision": self.environment_source_revision,
            "snapshot": self.snapshot.to_value(),
            "snapshot_relative_files": list(self.snapshot_relative_files),
            "split": self.split,
            "task_manifest_relative_path": self.task_manifest_relative_path,
        }

    @classmethod
    def from_value(cls, value: object) -> ProductionProcessBenchmarkSource:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "benchmark",
                    "dataset_revision",
                    "environment_source_revision",
                    "snapshot",
                    "snapshot_relative_files",
                    "split",
                    "task_manifest_relative_path",
                }
            ),
            label="production process benchmark source",
        )
        raw_files = data["snapshot_relative_files"]
        if not isinstance(raw_files, list) or any(type(item) is not str for item in raw_files):
            raise ValueError("process snapshot_relative_files must be a text array")
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field_name="benchmark")),
            dataset_revision=_text(
                data["dataset_revision"],
                field_name="dataset_revision",
            ),
            split=_text(data["split"], field_name="split"),
            task_manifest_relative_path=_text(
                data["task_manifest_relative_path"],
                field_name="task_manifest_relative_path",
            ),
            snapshot_relative_files=tuple(cast(list[str], raw_files)),
            snapshot=DatasetSnapshotIdentity.from_value(data["snapshot"]),
            environment_source_revision=_text(
                data["environment_source_revision"],
                field_name="environment_source_revision",
            ),
        )


@dataclass(frozen=True, slots=True)
class ProductionProcessCatalogConfig:
    """Complete process-catalog inputs with no inferred files or revisions."""

    dataset_root: Path
    sources: tuple[ProductionProcessBenchmarkSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_root, Path) or not self.dataset_root.is_absolute():
            raise ValueError("process dataset_root must be an absolute Path")
        if tuple(source.benchmark for source in self.sources) != _PROCESS_BENCHMARKS:
            raise ValueError("process sources must match the complete declared order")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "dataset_root": self.dataset_root.as_posix(),
            "format": PRODUCTION_PROCESS_CATALOG_CONFIG_FORMAT,
            "sources": [source.to_value() for source in self.sources],
        }

    @classmethod
    def from_value(cls, value: object) -> ProductionProcessCatalogConfig:
        data = _wire_object(
            value,
            fields=frozenset({"dataset_root", "format", "sources"}),
            label="production process catalog config",
        )
        if data["format"] != PRODUCTION_PROCESS_CATALOG_CONFIG_FORMAT:
            raise ValueError("unsupported production process catalog config format")
        raw_sources = data["sources"]
        if not isinstance(raw_sources, list):
            raise ValueError("production process sources must be an array")
        return cls(
            dataset_root=Path(_text(data["dataset_root"], field_name="dataset_root")),
            sources=tuple(
                ProductionProcessBenchmarkSource.from_value(item) for item in raw_sources
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class ProductionProcessCatalogDependencies:
    """Pinned official deployments plus externally isolated benchmark workers."""

    webshop: OfficialWebShopProcessFactory = field(repr=False)
    alfworld: OfficialALFWorldProcessFactory = field(repr=False)
    scienceworld: OfficialScienceWorldProcessFactory = field(repr=False)
    external_session_factories: tuple[tuple[Benchmark, PrivateSessionFactory], ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.webshop, OfficialWebShopProcessFactory):
            raise TypeError("webshop dependency must be OfficialWebShopProcessFactory")
        if not isinstance(self.alfworld, OfficialALFWorldProcessFactory):
            raise TypeError("alfworld dependency must be OfficialALFWorldProcessFactory")
        if not isinstance(self.scienceworld, OfficialScienceWorldProcessFactory):
            raise TypeError("scienceworld dependency must be OfficialScienceWorldProcessFactory")
        if (
            self.webshop.deployment.seed != FIXED_SEED
            or self.alfworld.seed != FIXED_SEED
            or self.scienceworld.seed != FIXED_SEED
        ):
            raise ValueError("process deployments must use the preregistered single seed")
        benchmarks = tuple(item[0] for item in self.external_session_factories)
        if any(benchmark not in _EXTERNAL_PROCESS_BENCHMARKS for benchmark in benchmarks):
            raise ValueError("external process factory benchmark is not declared")
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError("external process factory benchmarks must be unique")
        if any(
            not callable(getattr(factory, "create", None))
            for _, factory in self.external_session_factories
        ):
            raise TypeError("external process factory must implement create")

    def external_factory(self, benchmark: Benchmark) -> PrivateSessionFactory:
        matches = tuple(
            factory
            for candidate, factory in self.external_session_factories
            if candidate is benchmark
        )
        if len(matches) != 1:
            raise ValueError(f"{benchmark.value} requires one external process factory")
        return matches[0]


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("process task manifest contains a duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"process task manifest contains unsupported constant {value}")


def _manifest_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("process task manifest must contain non-empty JSONL records")
    rows: list[dict[str, object]] = []
    for line in lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=_duplicate_rejecting_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("process task manifest is invalid JSONL") from error
        if type(value) is not dict:
            raise TypeError("process task manifest records must be objects")
        rows.append(cast(dict[str, object], value))
    return tuple(rows)


def _exact_row(
    row: dict[str, object],
    *,
    fields: frozenset[str],
    benchmark: Benchmark,
) -> dict[str, object]:
    if set(row) != fields:
        raise ValueError(f"{benchmark.value} task record has an incompatible field set")
    return row


def _string_array(value: object, *, field_name: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    result = tuple(_text(item, field_name=field_name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be unique")
    return result


def _public_context(value: object) -> JsonValue:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TypeError("WebShop public_context must be an object")
    return normalized


def _source_rows(
    root: Path,
    source: ProductionProcessBenchmarkSource,
) -> tuple[dict[str, object], ...]:
    actual_snapshot = create_private_dataset_snapshot(
        name=source.benchmark.value,
        version=source.dataset_revision,
        root=root,
        relative_files=source.snapshot_relative_files,
    )
    if actual_snapshot != source.snapshot:
        raise ValueError("process source bytes differ from the frozen snapshot")
    return _manifest_rows(root / PurePosixPath(source.task_manifest_relative_path))


def _relative_to_root(root: Path, path: Path, *, label: str) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must remain below process dataset_root") from error
    return relative.as_posix()


def _require_pinned_file(
    root: Path,
    source: ProductionProcessBenchmarkSource,
    path: Path,
    *,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = _relative_to_root(root, path, label=label)
    if relative not in source.snapshot_relative_files:
        raise ValueError(f"{label} is not part of the process source snapshot")


def _require_pinned_directory(
    root: Path,
    source: ProductionProcessBenchmarkSource,
    path: Path,
    *,
    label: str,
) -> None:
    if not path.is_dir():
        raise NotADirectoryError(path)
    files = tuple(candidate for candidate in sorted(path.rglob("*")) if candidate.is_file())
    if not files:
        raise ValueError(f"{label} cannot be empty")
    for candidate in files:
        _require_pinned_file(root, source, candidate, label=label)


def _validate_task_order(task_ids: tuple[str, ...], *, benchmark: Benchmark) -> None:
    if task_ids != tuple(sorted(task_ids)):
        raise ValueError(f"{benchmark.value} task manifest must be sorted by task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{benchmark.value} task identities must be unique")


def _webshop_workload(
    root: Path,
    source: ProductionProcessBenchmarkSource,
    rows: tuple[dict[str, object], ...],
    deployment: OfficialWebShopProcessFactory,
) -> PrivateBenchmarkWorkload:
    selected = deployment.deployment
    if selected.runtime.source_revision != source.environment_source_revision:
        raise ValueError("WebShop process revision differs from its declared source")
    _require_pinned_directory(
        root,
        source,
        selected.index_path,
        label="WebShop search index",
    )
    match selected:
        case JsonArrayWebShopDeployment():
            _require_pinned_file(
                root,
                source,
                selected.products_path,
                label="WebShop products file",
            )
        case SQLiteWebShopDeployment():
            _require_pinned_file(
                root,
                source,
                selected.store_path,
                label="WebShop product store",
            )
            _require_pinned_file(
                root,
                source,
                selected.goals_path,
                label="WebShop goal stream",
            )
        case _ as unmatched:
            from typing import assert_never

            assert_never(unmatched)
    cases: list[PrivateWebShopCase] = []
    for raw in rows:
        row = _exact_row(raw, fields=_WEBSHOP_FIELDS, benchmark=Benchmark.WEBSHOP)
        item = WebShopPublicItem(
            dataset_revision=source.dataset_revision,
            environment_snapshot_id=source.snapshot.snapshot_hash,
            split=source.split,
            task_id=_text(row["task_id"], field_name="WebShop task_id"),
            task_family=_text(row["task_family"], field_name="WebShop task_family"),
            query=_text(row["query"], field_name="WebShop query"),
            public_context=_public_context(row["public_context"]),
        )
        goal = OfficialWebShopGoal(
            task_id=item.task_id,
            environment_id=item.environment_id,
            goal_id=_text(row["goal_id"], field_name="WebShop goal_id"),
            session_id=_text(row["session_id"], field_name="WebShop session_id"),
            payload=_non_negative_int(row["goal_index"], field_name="WebShop goal_index"),
        )
        cases.append(PrivateWebShopCase(item, goal))
    frozen_cases = tuple(cases)
    _validate_task_order(
        tuple(case.public.task_id for case in frozen_cases),
        benchmark=Benchmark.WEBSHOP,
    )
    return PrivateBenchmarkWorkload(
        Benchmark.WEBSHOP,
        tuple(case.public.to_rollout_task() for case in frozen_cases),
        PrivateWebShopSessionFactory(
            frozen_cases,
            OfficialWebShopEpisodeFactory(deployment),
        ),
    )


def _alfworld_workload(
    root: Path,
    source: ProductionProcessBenchmarkSource,
    rows: tuple[dict[str, object], ...],
    deployment: OfficialALFWorldProcessFactory,
) -> PrivateBenchmarkWorkload:
    if deployment.runtime.source_revision != source.environment_source_revision:
        raise ValueError("ALFWorld process revision differs from its declared source")
    _require_pinned_file(
        root,
        source,
        deployment.config_path,
        label="ALFWorld configuration",
    )
    cases: list[PrivateALFWorldCase] = []
    observed_games: list[str] = []
    for raw in rows:
        row = _exact_row(raw, fields=_ALFWORLD_FIELDS, benchmark=Benchmark.ALFWORLD)
        game_id = _text(row["game_id"], field_name="ALFWorld game_id")
        game = deployment.games.get(game_id)
        if game is None:
            raise ValueError("ALFWorld task has no exact game deployment")
        if game.train_eval != source.split:
            raise ValueError("ALFWorld game split differs from its declared source")
        query = _text(row["query"], field_name="ALFWorld query")
        if game.instruction_text != query:
            raise ValueError("ALFWorld public query differs from the deployed instruction")
        _require_pinned_directory(
            root,
            source,
            game.data_directory,
            label="ALFWorld game directory",
        )
        max_steps = _positive_int(row["max_steps"], field_name="ALFWorld max_steps")
        item = ALFWorldPublicItem(
            dataset_revision=source.dataset_revision,
            environment_snapshot_id=source.snapshot.snapshot_hash,
            split=source.split,
            task_id=_text(row["task_id"], field_name="ALFWorld task_id"),
            task_family=_text(row["task_family"], field_name="ALFWorld task_family"),
            query=query,
            public_context={
                "admissible_commands": list(
                    _string_array(
                        row["admissible_commands"],
                        field_name="ALFWorld admissible_commands",
                    )
                ),
                "initial_observation": _text(
                    row["initial_observation"],
                    field_name="ALFWorld initial_observation",
                ),
            },
            seed=deployment.seed,
            max_steps=max_steps,
        )
        private_task = OfficialALFWorldTask(
            task_id=item.task_id,
            environment_id=item.environment_id,
            game_id=game_id,
            seed=item.seed,
            max_steps=item.max_steps,
            payload={"game_id": game_id},
        )
        observed_games.append(game_id)
        cases.append(PrivateALFWorldCase(item, private_task))
    frozen_cases = tuple(cases)
    _validate_task_order(
        tuple(case.public.task_id for case in frozen_cases),
        benchmark=Benchmark.ALFWORLD,
    )
    if len(set(observed_games)) != len(observed_games):
        raise ValueError("ALFWorld task manifest repeats a game deployment")
    if set(observed_games) != set(deployment.games):
        raise ValueError("ALFWorld deployment games differ from the task manifest")
    return PrivateBenchmarkWorkload(
        Benchmark.ALFWORLD,
        tuple(case.public.to_rollout_task() for case in frozen_cases),
        PrivateALFWorldSessionFactory(
            frozen_cases,
            OfficialALFWorldEpisodeFactory(deployment),
        ),
    )


def _scienceworld_workload(
    root: Path,
    source: ProductionProcessBenchmarkSource,
    rows: tuple[dict[str, object], ...],
    deployment: OfficialScienceWorldProcessFactory,
) -> PrivateBenchmarkWorkload:
    if deployment.runtime.source_revision != source.environment_source_revision:
        raise ValueError("ScienceWorld process revision differs from its declared source")
    _require_pinned_file(
        root,
        source,
        deployment.jar_path,
        label="ScienceWorld JAR",
    )
    cases: list[PrivateScienceWorldCase] = []
    variations: list[tuple[str, int]] = []
    for raw in rows:
        row = _exact_row(raw, fields=_SCIENCEWORLD_FIELDS, benchmark=Benchmark.SCIENCE_WORLD)
        task_name = _text(row["task_name"], field_name="ScienceWorld task_name")
        variation = _non_negative_int(
            row["variation_index"],
            field_name="ScienceWorld variation_index",
        )
        max_steps = _positive_int(row["max_steps"], field_name="ScienceWorld max_steps")
        item = ScienceWorldPublicItem(
            dataset_revision=source.dataset_revision,
            environment_snapshot_id=source.snapshot.snapshot_hash,
            split=source.split,
            task_id=_text(row["task_id"], field_name="ScienceWorld task_id"),
            task_family=_text(row["task_family"], field_name="ScienceWorld task_family"),
            query=_text(row["query"], field_name="ScienceWorld query"),
            public_context={
                "initial_observation": _text(
                    row["initial_observation"],
                    field_name="ScienceWorld initial_observation",
                )
            },
            seed=deployment.seed,
            max_steps=max_steps,
        )
        private_task = OfficialScienceWorldTask(
            task_id=item.task_id,
            environment_id=item.environment_id,
            task_name=task_name,
            variation_index=variation,
            seed=item.seed,
            max_steps=item.max_steps,
            payload={"task_name": task_name, "variation_index": variation},
        )
        variations.append((task_name, variation))
        cases.append(PrivateScienceWorldCase(item, private_task))
    frozen_cases = tuple(cases)
    _validate_task_order(
        tuple(case.public.task_id for case in frozen_cases),
        benchmark=Benchmark.SCIENCE_WORLD,
    )
    if len(set(variations)) != len(variations):
        raise ValueError("ScienceWorld task manifest repeats a task variation")
    return PrivateBenchmarkWorkload(
        Benchmark.SCIENCE_WORLD,
        tuple(case.public.to_rollout_task() for case in frozen_cases),
        PrivateScienceWorldSessionFactory(
            frozen_cases,
            OfficialScienceWorldEpisodeFactory(deployment),
        ),
    )


def _external_process_workload(
    source: ProductionProcessBenchmarkSource,
    rows: tuple[dict[str, object], ...],
    factory: PrivateSessionFactory,
) -> PrivateBenchmarkWorkload:
    """Bind answer-free projections to an isolated official benchmark worker."""

    tasks: list[RolloutTask] = []
    for row in rows:
        if set(row) != {"source_split", "task"}:
            raise ValueError("external process manifest row has an invalid field set")
        if row["source_split"] != source.split:
            raise ValueError("external process task split differs from source identity")
        task = RolloutTask.from_value(row["task"])
        context = task.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != source.benchmark.value:
            raise ValueError("external process task has an incompatible benchmark identity")
        tasks.append(task)
    _validate_task_order(
        tuple(task.task_id for task in tasks),
        benchmark=source.benchmark,
    )
    return PrivateBenchmarkWorkload(source.benchmark, tuple(tasks), factory)


def load_production_process_catalog(
    config: ProductionProcessCatalogConfig,
    dependencies: ProductionProcessCatalogDependencies,
) -> PrivateBenchmarkCatalog:
    """Load all seven process workloads from exact manifests and assets."""

    if not isinstance(config, ProductionProcessCatalogConfig):
        raise TypeError("process catalog requires ProductionProcessCatalogConfig")
    if not isinstance(dependencies, ProductionProcessCatalogDependencies):
        raise TypeError("process catalog requires ProductionProcessCatalogDependencies")
    external_benchmarks = tuple(
        benchmark for benchmark, _factory in dependencies.external_session_factories
    )
    if set(external_benchmarks) != _EXTERNAL_PROCESS_BENCHMARKS:
        raise ValueError("production process catalog requires every external worker")
    if not config.dataset_root.is_dir():
        raise NotADirectoryError(config.dataset_root)
    workloads: list[PrivateBenchmarkWorkload] = []
    for source in config.sources:
        rows = _source_rows(config.dataset_root, source)
        if source.benchmark is Benchmark.WEBSHOP:
            workload = _webshop_workload(config.dataset_root, source, rows, dependencies.webshop)
        elif source.benchmark is Benchmark.ALFWORLD:
            workload = _alfworld_workload(config.dataset_root, source, rows, dependencies.alfworld)
        elif source.benchmark is Benchmark.SCIENCE_WORLD:
            workload = _scienceworld_workload(
                config.dataset_root, source, rows, dependencies.scienceworld
            )
        else:
            workload = _external_process_workload(
                source,
                rows,
                dependencies.external_factory(source.benchmark),
            )
        workloads.append(workload)
    return PrivateBenchmarkCatalog(tuple(workloads))


def load_training_process_catalog(
    config: ProductionProcessCatalogConfig,
    dependencies: ProductionProcessCatalogDependencies,
) -> PrivateBenchmarkCatalog:
    """Load exactly the three interactive workloads present in the B2 schedule."""

    if not isinstance(config, ProductionProcessCatalogConfig):
        raise TypeError("training process catalog requires ProductionProcessCatalogConfig")
    if not isinstance(dependencies, ProductionProcessCatalogDependencies):
        raise TypeError("training process catalog requires ProductionProcessCatalogDependencies")
    external = tuple(benchmark for benchmark, _factory in dependencies.external_session_factories)
    if external != (Benchmark.APPWORLD,):
        raise ValueError("training process catalog requires the AppWorld worker")
    if not config.dataset_root.is_dir():
        raise NotADirectoryError(config.dataset_root)
    workloads: list[PrivateBenchmarkWorkload] = []
    for source in config.sources:
        if source.benchmark not in _TRAINING_PROCESS_BENCHMARKS:
            continue
        rows = _source_rows(config.dataset_root, source)
        if source.benchmark is Benchmark.WEBSHOP:
            workload = _webshop_workload(config.dataset_root, source, rows, dependencies.webshop)
        elif source.benchmark is Benchmark.ALFWORLD:
            workload = _alfworld_workload(config.dataset_root, source, rows, dependencies.alfworld)
        else:
            workload = _external_process_workload(
                source,
                rows,
                dependencies.external_factory(source.benchmark),
            )
        workloads.append(workload)
    if tuple(workload.benchmark for workload in workloads) != (
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.APPWORLD,
    ):
        raise ValueError("training process sources differ from the frozen schedule domains")
    return PrivateBenchmarkCatalog(tuple(workloads))


def compose_complete_production_catalog(
    non_process: PrivateBenchmarkCatalog,
    process: PrivateBenchmarkCatalog,
) -> PrivateBenchmarkCatalog:
    """Compose the independently loaded 13+5 catalogs in protocol order."""

    if not isinstance(non_process, PrivateBenchmarkCatalog):
        raise TypeError("non_process must be PrivateBenchmarkCatalog")
    if not isinstance(process, PrivateBenchmarkCatalog):
        raise TypeError("process must be PrivateBenchmarkCatalog")
    if tuple(item.benchmark for item in non_process.workloads) != _NON_PROCESS_BENCHMARKS:
        raise ValueError("non-process catalog differs from its complete declared order")
    if tuple(item.benchmark for item in process.workloads) != _PROCESS_BENCHMARKS:
        raise ValueError("process catalog differs from its complete declared order")
    by_benchmark = {
        workload.benchmark: workload for workload in (*non_process.workloads, *process.workloads)
    }
    if tuple(by_benchmark) != (*_NON_PROCESS_BENCHMARKS, *_PROCESS_BENCHMARKS):
        raise ValueError("production catalogs contain duplicate benchmark identities")
    return PrivateBenchmarkCatalog(
        tuple(by_benchmark[benchmark] for benchmark in _COMPLETE_BENCHMARK_ORDER)
    )


@dataclass(slots=True)
class LoadedCompleteProductionCatalog:
    """Complete catalog retaining ownership of the shared retrieval index."""

    catalog: PrivateBenchmarkCatalog
    non_process: LoadedProductionCatalog = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if not self._closed:
            self.non_process.close()
            self._closed = True

    def __enter__(self) -> LoadedCompleteProductionCatalog:
        if self._closed:
            raise RuntimeError("complete production catalog is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


def load_complete_production_benchmark_catalog(
    *,
    non_process_config: ProductionCatalogConfig,
    non_process_dependencies: ProductionCatalogDependencies,
    process_config: ProductionProcessCatalogConfig,
    process_dependencies: ProductionProcessCatalogDependencies,
) -> LoadedCompleteProductionCatalog:
    """Load and own the exact complete active production catalog."""

    process = load_production_process_catalog(process_config, process_dependencies)
    with ExitStack() as stack:
        non_process = load_production_benchmark_catalog(
            non_process_config,
            non_process_dependencies,
        )
        stack.callback(non_process.close)
        catalog = compose_complete_production_catalog(non_process.catalog, process)
        stack.pop_all()
    return LoadedCompleteProductionCatalog(catalog, non_process)


__all__ = [
    "PRODUCTION_PROCESS_CATALOG_CONFIG_FORMAT",
    "LoadedCompleteProductionCatalog",
    "ProductionProcessBenchmarkSource",
    "ProductionProcessCatalogConfig",
    "ProductionProcessCatalogDependencies",
    "compose_complete_production_catalog",
    "load_complete_production_benchmark_catalog",
    "load_production_process_catalog",
    "load_training_process_catalog",
]
