from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.alfworld import PrivateALFWorldSessionFactory
from skillev_private.benchmarks.catalog import (
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
)
from skillev_private.benchmarks.official_process import (
    ALFWorldGameDeployment,
    JsonArrayWebShopDeployment,
    OfficialALFWorldProcessFactory,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    PinnedOfficialProcess,
)
from skillev_private.benchmarks.process_catalog import (
    ProductionProcessBenchmarkSource,
    ProductionProcessCatalogConfig,
    ProductionProcessCatalogDependencies,
    compose_complete_production_catalog,
    load_production_process_catalog,
)
from skillev_private.benchmarks.scienceworld import PrivateScienceWorldSessionFactory
from skillev_private.benchmarks.snapshot import create_private_dataset_snapshot
from skillev_private.benchmarks.webshop import PrivateWebShopSessionFactory

from skillev.contracts import canonical_json
from skillev.experiments import ACTIVE_BENCHMARKS, FIXED_SEED, Benchmark
from skillev.rollout import RolloutTask
from skillev.training import RolloutSessionBundle

_PRIVATE_GOAL = "private-goal-index-seven"
_PRIVATE_GAME = "private-game-seven"
_PRIVATE_SCIENCE_TASK = "private-science-task-seven"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    _write(path, json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_revision(root: Path) -> str:
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "init", "-q", str(root)), check=True
    )
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "config", "user.email", "fixture@example.invalid"),
        check=True,
    )
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "config", "user.name", "Fixture"),
        check=True,
    )
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "add", "."), check=True
    )
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "commit", "-q", "-m", "fixture"),
        check=True,
    )
    return subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True, slots=True)
class _Fixture:
    root: Path
    revision: str
    config: ProductionProcessCatalogConfig
    dependencies: ProductionProcessCatalogDependencies


@dataclass(frozen=True, slots=True)
class _ExternalFactory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        del task
        raise AssertionError("external process session construction is outside catalog loading")


def _source(
    root: Path,
    *,
    benchmark: Benchmark,
    revision: str,
    dataset_revision: str,
    split: str,
    manifest: str,
    files: tuple[str, ...],
) -> ProductionProcessBenchmarkSource:
    frozen_files = tuple(sorted(files))
    return ProductionProcessBenchmarkSource(
        benchmark=benchmark,
        dataset_revision=dataset_revision,
        split=split,
        task_manifest_relative_path=manifest,
        snapshot_relative_files=frozen_files,
        snapshot=create_private_dataset_snapshot(
            name=benchmark.value,
            version=dataset_revision,
            root=root,
            relative_files=frozen_files,
        ),
        environment_source_revision=revision,
    )


@pytest.fixture
def process_fixture(tmp_path: Path) -> _Fixture:
    root = tmp_path / "process-suite"
    webshop_manifest = "tasks/webshop.jsonl"
    alfworld_manifest = "tasks/alfworld.jsonl"
    scienceworld_manifest = "tasks/scienceworld.jsonl"
    external_manifests = {
        benchmark: f"tasks/{benchmark.value}.jsonl"
        for benchmark in (
            Benchmark.APPWORLD,
            Benchmark.BFCL_V3,
        )
    }
    _write_jsonl(
        root / webshop_manifest,
        {
            "goal_id": _PRIVATE_GOAL,
            "goal_index": 7,
            "public_context": {"observation_format": "public-text"},
            "query": "Buy the requested public fixture product.",
            "session_id": "private-session-seven",
            "task_family": "webshop/public-fixture",
            "task_id": "webshop-public-0001",
        },
    )
    _write_jsonl(
        root / alfworld_manifest,
        {
            "admissible_commands": ["look", "finish"],
            "game_id": _PRIVATE_GAME,
            "initial_observation": "A public fixture room.",
            "max_steps": 8,
            "query": "Complete the public fixture household task.",
            "task_family": "alfworld/public-fixture",
            "task_id": "alfworld-public-0001",
        },
    )
    _write_jsonl(
        root / scienceworld_manifest,
        {
            "initial_observation": "A public fixture laboratory.",
            "max_steps": 16,
            "query": "Complete the public fixture science task.",
            "task_family": "scienceworld/public-fixture",
            "task_id": "scienceworld-public-0001",
            "task_name": _PRIVATE_SCIENCE_TASK,
            "variation_index": 7,
        },
    )
    for benchmark, manifest in external_manifests.items():
        task = RolloutTask(
            task_id=f"{benchmark.value}-public-0001",
            environment_id=f"environment:{benchmark.value}:fixture",
            task_family=f"{benchmark.value}/public-fixture",
            context_id=f"{benchmark.value}/public-fixture",
            query=f"Complete the public {benchmark.value} process task.",
            available_tools=(),
            public_context={"benchmark_id": benchmark.value, "public": True},
        )
        _write_jsonl(
            root / manifest,
            {"source_split": "fixture", "task": task.to_value()},
        )

    _write(root / "products.json", "[]\n")
    _write(root / "search_engine" / "indexes_100" / "fixture-index.json", "{}\n")
    _write(root / "alfworld-config.yaml", "env: fixture\n")
    game_root = root / "alfworld-data" / "game-0001"
    _write(game_root / "game.tw-pddl", "fixture game\n")
    _write(game_root / "traj_data.json", "{}\n")
    _write(root / "scienceworld.jar", "fixture jar\n")

    revision = _git_revision(root)
    runtime = PinnedOfficialProcess(
        interpreter_path=Path(sys.executable),
        source_root=root,
        source_revision=revision,
        request_timeout_seconds=1.0,
    )
    webshop = OfficialWebShopProcessFactory(
        JsonArrayWebShopDeployment(
            runtime=runtime,
            products_path=root / "products.json",
            index_path=root / "search_engine" / "indexes_100",
            inventory_size=100,
            seed=FIXED_SEED,
        )
    )
    instruction = "Complete the public fixture household task."
    alfworld = OfficialALFWorldProcessFactory(
        runtime=runtime,
        config_path=root / "alfworld-config.yaml",
        games={
            _PRIVATE_GAME: ALFWorldGameDeployment(
                data_directory=game_root,
                train_eval="train",
                instruction_text=instruction,
            )
        },
        seed=FIXED_SEED,
    )
    scienceworld = OfficialScienceWorldProcessFactory(
        runtime=runtime,
        jar_path=root / "scienceworld.jar",
        simplification="",
        seed=FIXED_SEED,
    )
    dependencies = ProductionProcessCatalogDependencies(
        webshop=webshop,
        alfworld=alfworld,
        scienceworld=scienceworld,
        external_session_factories=tuple(
            (benchmark, _ExternalFactory()) for benchmark in external_manifests
        ),
    )

    sources = (
        _source(
            root,
            benchmark=Benchmark.WEBSHOP,
            revision=revision,
            dataset_revision="webshop-public-fixture@1",
            split="train",
            manifest=webshop_manifest,
            files=(
                webshop_manifest,
                "products.json",
                "search_engine/indexes_100/fixture-index.json",
            ),
        ),
        _source(
            root,
            benchmark=Benchmark.ALFWORLD,
            revision=revision,
            dataset_revision="alfworld-public-fixture@1",
            split="train",
            manifest=alfworld_manifest,
            files=(
                "alfworld-config.yaml",
                "alfworld-data/game-0001/game.tw-pddl",
                "alfworld-data/game-0001/traj_data.json",
                alfworld_manifest,
            ),
        ),
        *(
            _source(
                root,
                benchmark=benchmark,
                revision=revision,
                dataset_revision=f"{benchmark.value}-public-fixture@1",
                split="fixture",
                manifest=external_manifests[benchmark],
                files=(external_manifests[benchmark],),
            )
            for benchmark in (Benchmark.APPWORLD,)
        ),
        _source(
            root,
            benchmark=Benchmark.SCIENCE_WORLD,
            revision=revision,
            dataset_revision="scienceworld-public-fixture@1",
            split="test",
            manifest=scienceworld_manifest,
            files=("scienceworld.jar", scienceworld_manifest),
        ),
        *(
            _source(
                root,
                benchmark=benchmark,
                revision=revision,
                dataset_revision=f"{benchmark.value}-public-fixture@1",
                split="fixture",
                manifest=external_manifests[benchmark],
                files=(external_manifests[benchmark],),
            )
            for benchmark in (Benchmark.BFCL_V3,)
        ),
    )
    return _Fixture(
        root=root,
        revision=revision,
        config=ProductionProcessCatalogConfig(root, sources),
        dependencies=dependencies,
    )


def test_loads_exact_process_workloads_and_wires_official_session_factories(
    process_fixture: _Fixture,
) -> None:
    catalog = load_production_process_catalog(
        process_fixture.config,
        process_fixture.dependencies,
    )

    assert tuple(workload.benchmark for workload in catalog.workloads) == (
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.APPWORLD,
        Benchmark.SCIENCE_WORLD,
        Benchmark.BFCL_V3,
    )
    assert isinstance(
        catalog.workload(Benchmark.WEBSHOP).session_factory, PrivateWebShopSessionFactory
    )
    assert isinstance(
        catalog.workload(Benchmark.ALFWORLD).session_factory,
        PrivateALFWorldSessionFactory,
    )
    assert isinstance(
        catalog.workload(Benchmark.SCIENCE_WORLD).session_factory,
        PrivateScienceWorldSessionFactory,
    )
    public_wire = canonical_json(
        [task.to_value() for workload in catalog.workloads for task in workload.tasks]
    )
    assert _PRIVATE_GOAL not in public_wire
    assert _PRIVATE_GAME not in public_wire
    assert _PRIVATE_SCIENCE_TASK not in public_wire
    for source, workload in zip(
        process_fixture.config.sources,
        catalog.workloads,
        strict=True,
    ):
        if source.benchmark in {
            Benchmark.WEBSHOP,
            Benchmark.ALFWORLD,
            Benchmark.SCIENCE_WORLD,
        }:
            assert source.snapshot.snapshot_hash in workload.tasks[0].environment_id


def test_process_catalog_config_has_a_canonical_round_trip(
    process_fixture: _Fixture,
) -> None:
    restored = ProductionProcessCatalogConfig.from_value(
        json.loads(canonical_json(process_fixture.config.to_value()))
    )

    assert restored == process_fixture.config
    assert restored.content_hash == process_fixture.config.content_hash


def test_changed_process_manifest_bytes_are_rejected(
    process_fixture: _Fixture,
) -> None:
    manifest = process_fixture.root / "tasks" / "webshop.jsonl"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_production_process_catalog(
            process_fixture.config,
            process_fixture.dependencies,
        )


def test_deployment_asset_must_belong_to_declared_snapshot(
    process_fixture: _Fixture,
) -> None:
    old = process_fixture.config.sources[0]
    files = tuple(
        item
        for item in old.snapshot_relative_files
        if item != "search_engine/indexes_100/fixture-index.json"
    )
    replacement = ProductionProcessBenchmarkSource(
        benchmark=old.benchmark,
        dataset_revision=old.dataset_revision,
        split=old.split,
        task_manifest_relative_path=old.task_manifest_relative_path,
        snapshot_relative_files=files,
        snapshot=create_private_dataset_snapshot(
            name=old.benchmark.value,
            version=old.dataset_revision,
            root=process_fixture.root,
            relative_files=files,
        ),
        environment_source_revision=old.environment_source_revision,
    )
    config = replace(
        process_fixture.config,
        sources=(replacement, *process_fixture.config.sources[1:]),
    )

    with pytest.raises(ValueError):
        load_production_process_catalog(config, process_fixture.dependencies)


def test_extra_process_manifest_truth_field_is_rejected(
    process_fixture: _Fixture,
) -> None:
    source = process_fixture.config.sources[2]
    path = process_fixture.root / source.task_manifest_relative_path
    row = json.loads(path.read_text(encoding="utf-8"))
    row["private_answer"] = "fixture-only private value"
    _write_jsonl(path, row)
    replacement = replace(
        source,
        snapshot=create_private_dataset_snapshot(
            name=source.benchmark.value,
            version=source.dataset_revision,
            root=process_fixture.root,
            relative_files=source.snapshot_relative_files,
        ),
    )
    sources = list(process_fixture.config.sources)
    sources[2] = replacement
    config = replace(process_fixture.config, sources=tuple(sources))

    with pytest.raises(ValueError):
        load_production_process_catalog(config, process_fixture.dependencies)


@dataclass(frozen=True, slots=True)
class _UnusedSessionFactory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        del task
        raise AssertionError("fixture session must not be created")


def _non_process_catalog() -> PrivateBenchmarkCatalog:
    process_benchmarks = {
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.APPWORLD,
        Benchmark.SCIENCE_WORLD,
        Benchmark.BFCL_V3,
    }
    non_process = tuple(
        benchmark for benchmark in ACTIVE_BENCHMARKS if benchmark not in process_benchmarks
    )
    workloads = tuple(
        PrivateBenchmarkWorkload(
            benchmark=benchmark,
            tasks=(
                RolloutTask(
                    task_id=f"{benchmark.value}-fixture",
                    environment_id=f"environment:{benchmark.value}",
                    task_family=f"{benchmark.value}/fixture",
                    context_id=f"{benchmark.value}/fixture",
                    query="Public fixture query.",
                    available_tools=(),
                    public_context={"benchmark_id": benchmark.value},
                ),
            ),
            session_factory=_UnusedSessionFactory(),
        )
        for benchmark in non_process
    )
    return PrivateBenchmarkCatalog(workloads)


def test_composes_thirteen_plus_five_in_exact_protocol_order(
    process_fixture: _Fixture,
) -> None:
    process = load_production_process_catalog(
        process_fixture.config,
        process_fixture.dependencies,
    )

    complete = compose_complete_production_catalog(_non_process_catalog(), process)

    assert tuple(workload.benchmark for workload in complete.workloads) == ACTIVE_BENCHMARKS
    assert len(complete.workloads) == 18


def test_process_deployments_reject_a_second_seed(process_fixture: _Fixture) -> None:
    other = OfficialScienceWorldProcessFactory(
        runtime=process_fixture.dependencies.scienceworld.runtime,
        jar_path=process_fixture.dependencies.scienceworld.jar_path,
        simplification=process_fixture.dependencies.scienceworld.simplification,
        seed=FIXED_SEED + 1,
    )

    with pytest.raises(ValueError):
        ProductionProcessCatalogDependencies(
            webshop=process_fixture.dependencies.webshop,
            alfworld=process_fixture.dependencies.alfworld,
            scienceworld=other,
            external_session_factories=process_fixture.dependencies.external_session_factories,
        )
