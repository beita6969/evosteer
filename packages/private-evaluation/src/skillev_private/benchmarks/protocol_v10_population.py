"""Private Protocol 10 populations and the SkillFlow-style 9 x 512 mix.

Source populations contain unique source tasks.  Repetition happens only when
the frozen training episodes are built, so validation and final-evaluation
populations never inherit duplicated training occurrences.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from skillev.contracts import JsonValue, TerminalReward
from skillev.experiments.protocol_v10 import (
    ACTIVE_BENCHMARKS_V10,
    TRAINING_EPISODES_PER_BENCHMARK,
    ActiveBenchmarkProtocolV10,
    BenchmarkPopulation,
    BenchmarkV10,
    PopulationRole,
)
from skillev.rollout import (
    RolloutSessionBundle,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluator,
)

from .catalog import PrivateSessionFactory

TRAINING_SELECTION_FORMAT = "skillev-private-training-selection@10.1"
TRAINING_SELECTION_ALGORITHM = "skillflow-equal-domain-global-shuffle-seed-0@2"


class PopulationOverlapKind(StrEnum):
    SOURCE_RECORD = "source-record"
    NORMALIZED_PUBLIC_CONTENT = "normalized-public-content"
    CODE_SIGNATURE = "code-signature"
    WORKBOOK_STRUCTURE = "workbook-structure"
    INTERACTIVE_SCENARIO = "interactive-scenario"


ITEM_OVERLAP_KINDS = frozenset(
    {
        PopulationOverlapKind.SOURCE_RECORD,
        PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
        PopulationOverlapKind.CODE_SIGNATURE,
    }
)
GROUP_OVERLAP_KINDS = frozenset(
    {
        PopulationOverlapKind.WORKBOOK_STRUCTURE,
        PopulationOverlapKind.INTERACTIVE_SCENARIO,
    }
)


@dataclass(frozen=True, slots=True)
class PopulationOverlapKey:
    kind: PopulationOverlapKind
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PopulationOverlapKind):
            raise TypeError("population overlap kind must be closed")
        if not self.value.strip():
            raise ValueError("population overlap key must be non-empty")

    def to_value(self) -> dict[str, JsonValue]:
        return {"kind": self.kind.value, "value": self.value}

    @classmethod
    def from_value(cls, value: object) -> PopulationOverlapKey:
        if not isinstance(value, dict) or set(value) != {"kind", "value"}:
            raise ValueError("population overlap key has incompatible fields")
        if type(value["kind"]) is not str or type(value["value"]) is not str:
            raise TypeError("population overlap key fields must be strings")
        return cls(PopulationOverlapKind(value["kind"]), value["value"])


class PopulationOverlapIdentityMaterializer(Protocol):
    """Trusted source-specific reconstruction of declared overlap identities."""

    @property
    def version(self) -> str: ...

    def materialize(
        self,
        *,
        spec: BenchmarkPopulation,
        source_id: str,
        task: RolloutTask,
    ) -> tuple[PopulationOverlapKey, ...]: ...


@dataclass(frozen=True, slots=True)
class PrivatePopulationItem:
    """One private source task with answer-free overlap identities."""

    source_id: str
    task: RolloutTask
    overlap_keys: tuple[PopulationOverlapKey, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("population source ID must be non-empty")
        if not isinstance(self.task, RolloutTask):
            raise TypeError("population item requires a RolloutTask")
        if not self.overlap_keys or any(
            not isinstance(item, PopulationOverlapKey) for item in self.overlap_keys
        ):
            raise TypeError("population item requires overlap keys")
        identities = tuple((item.kind, item.value) for item in self.overlap_keys)
        if len(set(identities)) != len(identities):
            raise ValueError("population item repeats an overlap key")
        kinds = tuple(item.kind for item in self.overlap_keys)
        required = {
            PopulationOverlapKind.SOURCE_RECORD,
            PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
        }
        if not required <= set(kinds):
            raise ValueError("population item requires source-record and public-content identities")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "overlap_keys": [key.to_value() for key in self.overlap_keys],
            "source_id": self.source_id,
            "task": self.task.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivatePopulationItem:
        if not isinstance(value, dict) or set(value) != {"overlap_keys", "source_id", "task"}:
            raise ValueError("private population item has incompatible fields")
        keys = value["overlap_keys"]
        if type(value["source_id"]) is not str or not isinstance(keys, list):
            raise TypeError("private population item fields have incompatible types")
        return cls(
            source_id=value["source_id"],
            task=RolloutTask.from_value(value["task"]),
            overlap_keys=tuple(PopulationOverlapKey.from_value(key) for key in keys),
        )


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkPopulation:
    spec: BenchmarkPopulation
    items: tuple[PrivatePopulationItem, ...]
    identity_materializer_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, BenchmarkPopulation):
            raise TypeError("private population requires its Protocol 10 spec")
        if not self.items or any(
            not isinstance(item, PrivatePopulationItem) for item in self.items
        ):
            raise TypeError("private population requires source items")
        if not self.identity_materializer_version.strip():
            raise ValueError("private population requires a trusted materializer version")
        source_ids = tuple(item.source_id for item in self.items)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source population repeats a source ID")
        item_identities = tuple(
            (key.kind, key.value)
            for item in self.items
            for key in item.overlap_keys
            if key.kind in ITEM_OVERLAP_KINDS
        )
        if len(item_identities) != len(set(item_identities)):
            raise ValueError("source population repeats an item-level identity")
        _require_benchmark_overlap_keys(self.spec.benchmark, self.items)
        for item in self.items:
            context = item.task.public_context
            if not isinstance(context, dict):
                raise ValueError("population task public context must be an object")
            if context.get("benchmark_id") != self.spec.benchmark.value:
                raise ValueError("population task belongs to another benchmark")


@dataclass(frozen=True, slots=True)
class ProtocolV10PopulationManifest:
    """Answer-free population identity bound into a training selection."""

    benchmark: BenchmarkV10
    population_id: str
    role: PopulationRole
    source_version: str
    unique_item_count: int
    evaluator_identity: str
    identity_materializer_version: str

    def __post_init__(self) -> None:
        for value in (
            self.population_id,
            self.source_version,
            self.evaluator_identity,
            self.identity_materializer_version,
        ):
            if not value.strip():
                raise ValueError("population manifest text fields must be non-empty")
        if type(self.unique_item_count) is not int or self.unique_item_count < 1:
            raise ValueError("population manifest requires a positive unique-item count")

    @classmethod
    def from_population(
        cls,
        population: PrivateBenchmarkPopulation,
    ) -> ProtocolV10PopulationManifest:
        return cls(
            benchmark=population.spec.benchmark,
            population_id=population.spec.population_id,
            role=population.spec.role,
            source_version=population.spec.source_version,
            unique_item_count=len(population.items),
            evaluator_identity=population.spec.evaluator_identity,
            identity_materializer_version=population.identity_materializer_version,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "evaluator_identity": self.evaluator_identity,
            "identity_materializer_version": self.identity_materializer_version,
            "population_id": self.population_id,
            "role": self.role.value,
            "source_version": self.source_version,
            "unique_item_count": self.unique_item_count,
        }

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10PopulationManifest:
        fields = {
            "benchmark",
            "evaluator_identity",
            "identity_materializer_version",
            "population_id",
            "role",
            "source_version",
            "unique_item_count",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("population manifest has incompatible fields")
        text_fields = fields - {"unique_item_count"}
        if any(type(value[field]) is not str for field in text_fields):
            raise TypeError("population manifest text fields must be strings")
        if type(value["unique_item_count"]) is not int:
            raise TypeError("population manifest item count must be an integer")
        return cls(
            benchmark=BenchmarkV10(value["benchmark"]),
            population_id=value["population_id"],
            role=PopulationRole(value["role"]),
            source_version=value["source_version"],
            unique_item_count=value["unique_item_count"],
            evaluator_identity=value["evaluator_identity"],
            identity_materializer_version=value["identity_materializer_version"],
        )


def _require_benchmark_overlap_keys(
    benchmark: BenchmarkV10,
    items: tuple[PrivatePopulationItem, ...],
) -> None:
    required = {
        PopulationOverlapKind.SOURCE_RECORD,
        PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
    }
    if benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
        required.add(PopulationOverlapKind.CODE_SIGNATURE)
    if benchmark is BenchmarkV10.SPREADSHEETBENCH:
        required.add(PopulationOverlapKind.WORKBOOK_STRUCTURE)
    if benchmark in {
        BenchmarkV10.WEBSHOP,
        BenchmarkV10.ALFWORLD,
        BenchmarkV10.APPWORLD,
    }:
        required.add(PopulationOverlapKind.INTERACTIVE_SCENARIO)
    for item in items:
        if not required <= {key.kind for key in item.overlap_keys}:
            raise ValueError("population item lacks a benchmark-specific overlap key")


@dataclass(frozen=True, slots=True)
class ProtocolV10PopulationCatalog:
    protocol: ActiveBenchmarkProtocolV10
    populations: tuple[PrivateBenchmarkPopulation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, ActiveBenchmarkProtocolV10):
            raise TypeError("population catalog requires Protocol 10")
        if not self.populations:
            raise ValueError("population catalog cannot be empty")
        keys = tuple(
            (population.spec.benchmark, population.spec.population_id)
            for population in self.populations
        )
        if len(keys) != len(set(keys)):
            raise ValueError("population catalog repeats a population")
        expected = {
            (benchmark.benchmark, population.population_id)
            for benchmark in self.protocol.benchmarks
            for population in benchmark.populations
        }
        if set(keys) != expected:
            raise ValueError("population catalog does not cover Protocol 10")
        self._require_role_isolation()

    def _require_role_isolation(self) -> None:
        source_locations: dict[str, set[tuple[BenchmarkV10, str, PopulationRole]]] = defaultdict(
            set
        )
        overlap_locations: dict[
            tuple[PopulationOverlapKind, str],
            set[tuple[BenchmarkV10, str, PopulationRole]],
        ] = defaultdict(set)
        for population in self.populations:
            location = (
                population.spec.benchmark,
                population.spec.population_id,
                population.spec.role,
            )
            for item in population.items:
                source_locations[item.source_id].add(location)
                for key in item.overlap_keys:
                    overlap_locations[(key.kind, key.value)].add(location)
        if any(len(locations) > 1 for locations in source_locations.values()):
            raise ValueError("source ID crosses Protocol 10 populations")
        if any(len(locations) > 1 for locations in overlap_locations.values()):
            raise ValueError("overlap identity crosses Protocol 10 populations")

    def population(
        self,
        benchmark: BenchmarkV10,
        role: PopulationRole,
    ) -> tuple[PrivateBenchmarkPopulation, ...]:
        return tuple(
            item
            for item in self.populations
            if item.spec.benchmark is benchmark and item.spec.role is role
        )

    def training_population(self, benchmark: BenchmarkV10) -> PrivateBenchmarkPopulation:
        matches = self.population(benchmark, PopulationRole.TRAINING)
        if len(matches) != 1:
            raise ValueError("benchmark lacks one training population")
        return matches[0]

    @property
    def training_manifests(self) -> tuple[ProtocolV10PopulationManifest, ...]:
        return tuple(
            ProtocolV10PopulationManifest.from_population(self.training_population(benchmark))
            for benchmark in ACTIVE_BENCHMARKS_V10
        )


def load_protocol_v10_population_file(
    spec: BenchmarkPopulation,
    path: Path,
    *,
    identity_materializer: PopulationOverlapIdentityMaterializer,
) -> PrivateBenchmarkPopulation:
    """Load one private population and recompute every declared identity."""

    if not identity_materializer.version.strip():
        raise ValueError("population identity materializer version is empty")

    with path.open(encoding="utf-8", newline="\n") as stream:
        items = tuple(
            PrivatePopulationItem.from_value(json.loads(line)) for line in stream if line.strip()
        )
    for item in items:
        recomputed = identity_materializer.materialize(
            spec=spec,
            source_id=item.source_id,
            task=item.task,
        )
        if item.overlap_keys != recomputed:
            raise ValueError("declared population identities differ from trusted materialization")
    return PrivateBenchmarkPopulation(spec, items, identity_materializer.version)


def write_protocol_v10_population_file(
    population: PrivateBenchmarkPopulation,
    path: Path,
) -> None:
    """Publish one private answer-free population with owner-only permissions.

    Population files contain public task text and overlap identities, so they
    stay in the private server data tree and are never suitable Git artifacts.
    The exclusive create and same-directory rename prevent a partial catalog
    from being mistaken for a usable population after interruption.
    """

    if not isinstance(population, PrivateBenchmarkPopulation):
        raise TypeError("population writer requires a private population")
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError("population output must be an absolute file path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = path.with_name(f".{path.name}.staging")
    if path.exists() or staging.exists():
        raise FileExistsError(path if path.exists() else staging)
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for item in population.items:
                stream.write(
                    json.dumps(
                        item.to_value(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def load_protocol_v10_population_catalog(
    protocol: ActiveBenchmarkProtocolV10,
    population_files: Mapping[str, Path],
    identity_materializers: Mapping[str, PopulationOverlapIdentityMaterializer],
) -> ProtocolV10PopulationCatalog:
    """Load every Protocol 10 role from a distinct private population file."""

    expected = tuple(
        population for benchmark in protocol.benchmarks for population in benchmark.populations
    )
    expected_ids = {population.population_id for population in expected}
    if set(population_files) != expected_ids:
        raise ValueError("population files do not cover Protocol 10")
    if set(identity_materializers) != expected_ids:
        raise ValueError("identity materializers do not cover Protocol 10")
    resolved = tuple(population_files[item.population_id].resolve() for item in expected)
    if len(resolved) != len(set(resolved)):
        raise ValueError("Protocol 10 populations must use distinct private files")
    return ProtocolV10PopulationCatalog(
        protocol,
        tuple(
            load_protocol_v10_population_file(
                population,
                population_files[population.population_id],
                identity_materializer=identity_materializers[population.population_id],
            )
            for population in expected
        ),
    )


@dataclass(frozen=True, slots=True)
class ProtocolV10TrainingEpisode:
    benchmark: BenchmarkV10
    population_id: str
    episode_id: str
    source_id: str
    repeat_ordinal: int
    block_position: int
    global_position: int

    def __post_init__(self) -> None:
        for value in (self.population_id, self.episode_id, self.source_id):
            if not value.strip():
                raise ValueError("training episode text fields must be non-empty")
        for position in (self.repeat_ordinal, self.block_position, self.global_position):
            if type(position) is not int or position < 0:
                raise ValueError("training episode positions must be non-negative integers")
        if self.block_position >= TRAINING_EPISODES_PER_BENCHMARK:
            raise ValueError("training episode lies outside its 512-item block")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "block_position": self.block_position,
            "episode_id": self.episode_id,
            "global_position": self.global_position,
            "population_id": self.population_id,
            "repeat_ordinal": self.repeat_ordinal,
            "source_id": self.source_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10TrainingEpisode:
        if not isinstance(value, dict) or set(value) != {
            "benchmark",
            "block_position",
            "episode_id",
            "global_position",
            "population_id",
            "repeat_ordinal",
            "source_id",
        }:
            raise ValueError("training episode has incompatible fields")
        text_fields = ("benchmark", "episode_id", "population_id", "source_id")
        if any(type(value[field]) is not str for field in text_fields):
            raise TypeError("training episode text fields must be strings")
        integer_fields = ("block_position", "global_position", "repeat_ordinal")
        if any(type(value[field]) is not int for field in integer_fields):
            raise TypeError("training episode positions must be integers")
        return cls(
            benchmark=BenchmarkV10(value["benchmark"]),
            population_id=value["population_id"],
            episode_id=value["episode_id"],
            source_id=value["source_id"],
            repeat_ordinal=value["repeat_ordinal"],
            block_position=value["block_position"],
            global_position=value["global_position"],
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10TrainingSelection:
    episodes: tuple[ProtocolV10TrainingEpisode, ...]
    population_manifests: tuple[ProtocolV10PopulationManifest, ...]
    seed: int = 0
    algorithm: str = TRAINING_SELECTION_ALGORITHM
    format: str = TRAINING_SELECTION_FORMAT

    def __post_init__(self) -> None:
        if self.seed != 0 or self.algorithm != TRAINING_SELECTION_ALGORITHM:
            raise ValueError("training selection differs from the frozen shuffle")
        if self.format != TRAINING_SELECTION_FORMAT:
            raise ValueError("training selection format is unsupported")
        if tuple(manifest.benchmark for manifest in self.population_manifests) != (
            ACTIVE_BENCHMARKS_V10
        ):
            raise ValueError("training selection population manifests are incomplete or reordered")
        if any(
            manifest.role is not PopulationRole.TRAINING for manifest in self.population_manifests
        ):
            raise ValueError("training selection may bind only training populations")
        if len(self.episodes) != len(ACTIVE_BENCHMARKS_V10) * 512:
            raise ValueError("training selection must contain 4,608 episodes")
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise ValueError("training episode identities must be unique")
        if tuple(item.global_position for item in self.episodes) != tuple(
            range(len(self.episodes))
        ):
            raise ValueError("training global positions must be contiguous")
        for benchmark in ACTIVE_BENCHMARKS_V10:
            domain = tuple(item for item in self.episodes if item.benchmark is benchmark)
            if len(domain) != TRAINING_EPISODES_PER_BENCHMARK:
                raise ValueError("training mix must contain 512 episodes per benchmark")
            if tuple(sorted(item.block_position for item in domain)) != tuple(
                range(TRAINING_EPISODES_PER_BENCHMARK)
            ):
                raise ValueError("training domain positions must be contiguous")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "algorithm": self.algorithm,
            "episodes": [item.to_value() for item in self.episodes],
            "format": self.format,
            "population_manifests": [manifest.to_value() for manifest in self.population_manifests],
            "seed": self.seed,
        }

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10TrainingSelection:
        if not isinstance(value, dict) or set(value) != {
            "algorithm",
            "episodes",
            "format",
            "population_manifests",
            "seed",
        }:
            raise ValueError("training selection has incompatible fields")
        episodes = value["episodes"]
        manifests = value["population_manifests"]
        if not isinstance(episodes, list) or not isinstance(manifests, list):
            raise TypeError("training selection episodes and manifests must be arrays")
        if type(value["algorithm"]) is not str or type(value["format"]) is not str:
            raise TypeError("training selection identity fields must be strings")
        if type(value["seed"]) is not int:
            raise TypeError("training selection seed must be an integer")
        return cls(
            episodes=tuple(ProtocolV10TrainingEpisode.from_value(item) for item in episodes),
            population_manifests=tuple(
                ProtocolV10PopulationManifest.from_value(item) for item in manifests
            ),
            seed=value["seed"],
            algorithm=value["algorithm"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> ProtocolV10TrainingSelection:
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        """Write only private IDs and routes; task content is never serialized here."""

        encoded = (
            json.dumps(
                self.to_value(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            + b"\n"
        )
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True, slots=True)
class _TrainingEpisodeEvaluator:
    source_evaluator: TerminalEvaluator
    episode_id: str
    source_id: str

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.episode_id:
            raise ValueError("terminal request reached another training episode")
        return await self.source_evaluator.evaluate(replace(request, task_id=self.source_id))


@dataclass(frozen=True, slots=True)
class ProtocolV10TrainingSessionFactory:
    """Route repeated episode identities to their unique source task and evaluator."""

    routes: tuple[tuple[str, RolloutTask, PrivateSessionFactory], ...]

    def __post_init__(self) -> None:
        episode_ids = tuple(episode_id for episode_id, _, _ in self.routes)
        if not episode_ids or len(episode_ids) != len(set(episode_ids)):
            raise ValueError("training session routes must have unique episode IDs")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        matches = tuple(route for route in self.routes if route[0] == task.task_id)
        if len(matches) != 1:
            raise ValueError("training episode has no unique source route")
        episode_id, source_task, source_factory = matches[0]
        source_bundle = source_factory.create(source_task)
        if not isinstance(source_bundle, RolloutSessionBundle):
            raise TypeError("population session factory returned an incompatible bundle")
        return replace(
            source_bundle,
            evaluator=_TrainingEpisodeEvaluator(
                source_bundle.evaluator,
                episode_id,
                source_task.task_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10PopulationSessionRegistry:
    routes: tuple[tuple[str, PrivateSessionFactory], ...]

    def __post_init__(self) -> None:
        population_ids = tuple(population_id for population_id, _ in self.routes)
        if not population_ids or len(population_ids) != len(set(population_ids)):
            raise ValueError("population session routes must have unique population IDs")
        if any(
            not population_id.strip() or not callable(getattr(factory, "create", None))
            for population_id, factory in self.routes
        ):
            raise TypeError("population session registry contains an invalid route")

    def factory(self, population_id: str) -> PrivateSessionFactory:
        matches = tuple(factory for name, factory in self.routes if name == population_id)
        if len(matches) != 1:
            raise ValueError("population has no unique session factory")
        return matches[0]


@dataclass(frozen=True, slots=True)
class ProtocolV10TrainingMix:
    selection: ProtocolV10TrainingSelection
    tasks: tuple[RolloutTask, ...]
    session_factory: ProtocolV10TrainingSessionFactory

    def __post_init__(self) -> None:
        if len(self.tasks) != len(self.selection.episodes):
            raise ValueError("training tasks and selection have different sizes")
        if tuple(task.task_id for task in self.tasks) != tuple(
            episode.episode_id for episode in self.selection.episodes
        ):
            raise ValueError("training tasks differ from the frozen selection order")


def materialize_protocol_v10_training_mix(
    catalog: ProtocolV10PopulationCatalog,
    selection: ProtocolV10TrainingSelection,
    sessions: ProtocolV10PopulationSessionRegistry,
) -> ProtocolV10TrainingMix:
    """Materialize the exact 4,608-episode mix without copying private answers."""

    if selection != build_protocol_v10_training_selection(catalog):
        raise ValueError("training selection differs from deterministic catalog materialization")

    source_routes: dict[
        tuple[BenchmarkV10, str, str], tuple[RolloutTask, PrivateSessionFactory]
    ] = {}
    for benchmark in ACTIVE_BENCHMARKS_V10:
        population = catalog.training_population(benchmark)
        session_factory = sessions.factory(population.spec.population_id)
        for item in population.items:
            source_routes[(benchmark, population.spec.population_id, item.source_id)] = (
                item.task,
                session_factory,
            )

    tasks: list[RolloutTask] = []
    routes: list[tuple[str, RolloutTask, PrivateSessionFactory]] = []
    for episode in selection.episodes:
        try:
            source_task, source_factory = source_routes[
                (episode.benchmark, episode.population_id, episode.source_id)
            ]
        except KeyError as error:
            raise ValueError("selected Protocol 10 source task is absent from catalog") from error
        tasks.append(replace(source_task, task_id=episode.episode_id))
        routes.append((episode.episode_id, source_task, source_factory))
    factory = ProtocolV10TrainingSessionFactory(tuple(routes))
    return ProtocolV10TrainingMix(selection, tuple(tasks), factory)


def _cycle_order(
    source_ids: tuple[str, ...],
    *,
    benchmark_index: int,
    cycle: int,
) -> list[str]:
    ordered = sorted(source_ids)
    # A closed integer seed avoids process-randomized Python object hashes.
    seed = benchmark_index * 1_000_003 + cycle * 10_007
    random.Random(seed).shuffle(ordered)  # noqa: S311 - scientific schedule, not security
    return ordered


def build_protocol_v10_training_selection(
    catalog: ProtocolV10PopulationCatalog,
) -> ProtocolV10TrainingSelection:
    """Build an equal-domain, globally mixed schedule before observing results."""

    episodes: list[ProtocolV10TrainingEpisode] = []
    for benchmark_index, benchmark in enumerate(ACTIVE_BENCHMARKS_V10):
        population = catalog.training_population(benchmark)
        source_ids = tuple(item.source_id for item in population.items)
        selected: list[str] = []
        cycle = 0
        while len(selected) < TRAINING_EPISODES_PER_BENCHMARK:
            selected.extend(_cycle_order(source_ids, benchmark_index=benchmark_index, cycle=cycle))
            cycle += 1
        occurrence: Counter[str] = Counter()
        for block_position, source_id in enumerate(selected[:TRAINING_EPISODES_PER_BENCHMARK]):
            repeat_ordinal = occurrence[source_id]
            occurrence[source_id] += 1
            episodes.append(
                ProtocolV10TrainingEpisode(
                    benchmark=benchmark,
                    population_id=population.spec.population_id,
                    episode_id=f"{benchmark.value}/training/{block_position:04d}",
                    source_id=source_id,
                    repeat_ordinal=repeat_ordinal,
                    block_position=block_position,
                    global_position=len(episodes),
                )
            )
    random.Random(0).shuffle(episodes)  # noqa: S311 - frozen scientific schedule
    return ProtocolV10TrainingSelection(
        tuple(replace(episode, global_position=index) for index, episode in enumerate(episodes)),
        population_manifests=catalog.training_manifests,
    )


__all__ = [
    "TRAINING_SELECTION_ALGORITHM",
    "TRAINING_SELECTION_FORMAT",
    "PopulationOverlapIdentityMaterializer",
    "PopulationOverlapKey",
    "PopulationOverlapKind",
    "PrivateBenchmarkPopulation",
    "PrivatePopulationItem",
    "ProtocolV10PopulationCatalog",
    "ProtocolV10PopulationManifest",
    "ProtocolV10PopulationSessionRegistry",
    "ProtocolV10TrainingEpisode",
    "ProtocolV10TrainingMix",
    "ProtocolV10TrainingSelection",
    "ProtocolV10TrainingSessionFactory",
    "build_protocol_v10_training_selection",
    "load_protocol_v10_population_catalog",
    "load_protocol_v10_population_file",
    "materialize_protocol_v10_training_mix",
    "write_protocol_v10_population_file",
]
