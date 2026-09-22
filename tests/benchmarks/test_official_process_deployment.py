from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.acquisition import (
    BenchmarkAcquisitionLock,
    load_benchmark_acquisition_lock,
)
from skillev_private.benchmarks.official_process_deployment import (
    ALFWorldGameDeploymentConfig,
    ALFWorldProcessDeploymentConfig,
    ExternalProcessManifestDeploymentConfig,
    FileOfficialProcessPreparationFactory,
    JsonArrayWebShopDeploymentConfig,
    OfficialProcessDeploymentConfig,
    OfficialProcessRuntimeConfig,
    ScienceWorldProcessDeploymentConfig,
    SQLiteWebShopDeploymentConfig,
    load_official_process_deployment_config,
    publish_official_process_deployment_config,
)

from skillev.contracts import canonical_json
from skillev.experiments import FIXED_SEED, Benchmark
from skillev.rollout import RolloutTask


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_revision(root: Path) -> str:
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "init", "-q", str(root)),
        check=True,
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
        ("git", "-C", str(root), "add", "."),
        check=True,
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


def _lock_with_revisions(revisions: dict[Benchmark, str]) -> BenchmarkAcquisitionLock:
    lock = load_benchmark_acquisition_lock(
        (Path(__file__).parents[2] / "benchmark-acquisition-lock.json").resolve()
    )
    return replace(
        lock,
        benchmarks=tuple(
            replace(
                entry,
                source=replace(entry.source, revision=revisions[entry.benchmark]),
            )
            if entry.benchmark in revisions
            else entry
            for entry in lock.benchmarks
        ),
    )


@dataclass(frozen=True, slots=True)
class _DeploymentFixture:
    target_root: Path
    lock: BenchmarkAcquisitionLock
    config: OfficialProcessDeploymentConfig


@pytest.fixture
def deployment_fixture(tmp_path: Path) -> _DeploymentFixture:
    target = (tmp_path / "datasets").resolve()
    webshop_root = target / "webshop" / "repository"
    alfworld_root = target / "alfworld" / "repository"
    scienceworld_root = target / "scienceworld" / "repository"

    _write(webshop_root / "products.json", "[]\n")
    _write(webshop_root / "search_engine" / "indexes_100" / "segments_1", "index\n")
    _write(alfworld_root / "configs" / "base_config.yaml", "dataset: {}\n")
    _write(scienceworld_root / "scienceworld" / "scienceworld.jar", "jar\n")
    revisions = {
        Benchmark.WEBSHOP: _git_revision(webshop_root),
        Benchmark.ALFWORLD: _git_revision(alfworld_root),
        Benchmark.SCIENCE_WORLD: _git_revision(scienceworld_root),
    }

    game_root = target / "alfworld" / "prepared" / "games" / "game-0001"
    _write(game_root / "game.tw-pddl", "game\n")
    _write(
        game_root / "traj_data.json",
        canonical_json({"task_type": "pick_and_place_simple"}) + "\n",
    )

    runtime = OfficialProcessRuntimeConfig(
        interpreter_path=Path(sys.executable).resolve(),
        request_timeout_seconds=30.0,
    )
    external_manifests: list[ExternalProcessManifestDeploymentConfig] = []
    for benchmark, split in (
        (Benchmark.APPWORLD, "train"),
        (Benchmark.BFCL_V3, "test"),
    ):
        manifest = f"{benchmark.value}/prepared/tasks.jsonl"
        asset = f"{benchmark.value}/repository/public.asset"
        task = RolloutTask(
            task_id=f"{benchmark.value}-task-0001",
            environment_id=f"{benchmark.value}-environment",
            task_family=f"{benchmark.value}/fixture",
            context_id=f"{benchmark.value}-context-0001",
            query=f"Complete the public {benchmark.value} fixture task.",
            available_tools=("submit",),
            public_context={"benchmark_id": benchmark.value},
        )
        _write(target / manifest, canonical_json(task.to_value()) + "\n")
        _write(target / asset, "public fixture asset\n")
        external_manifests.append(
            ExternalProcessManifestDeploymentConfig(
                benchmark=benchmark,
                split=split,
                task_manifest_relative_path=manifest,
                asset_relative_files=(asset,),
            )
        )
    config = OfficialProcessDeploymentConfig(
        webshop=JsonArrayWebShopDeploymentConfig(
            runtime=runtime,
            products_relative_path="webshop/repository/products.json",
            search_index_relative_path="webshop/repository/search_engine/indexes_100",
            asset_relative_files=(
                "webshop/repository/products.json",
                "webshop/repository/search_engine/indexes_100/segments_1",
            ),
            inventory_size=100,
        ),
        alfworld=ALFWorldProcessDeploymentConfig(
            runtime=runtime,
            config_relative_path="alfworld/repository/configs/base_config.yaml",
            games=(
                ALFWorldGameDeploymentConfig(
                    game_id="game-0001",
                    data_directory_relative_path=("alfworld/prepared/games/game-0001"),
                    train_eval="train",
                    instruction_text="Put the public object in its destination.",
                ),
            ),
            asset_relative_files=(
                "alfworld/prepared/games/game-0001/game.tw-pddl",
                "alfworld/prepared/games/game-0001/traj_data.json",
                "alfworld/repository/configs/base_config.yaml",
            ),
            max_steps=50,
        ),
        scienceworld=ScienceWorldProcessDeploymentConfig(
            runtime=runtime,
            jar_relative_path="scienceworld/repository/scienceworld/scienceworld.jar",
            asset_relative_files=("scienceworld/repository/scienceworld/scienceworld.jar",),
            simplification="",
            max_steps=100,
        ),
        external_manifests=tuple(external_manifests),
        seed=FIXED_SEED,
    )
    return _DeploymentFixture(
        target_root=target,
        lock=_lock_with_revisions(revisions),
        config=config,
    )


def test_private_deployment_config_has_an_exact_published_round_trip(
    deployment_fixture: _DeploymentFixture,
    tmp_path: Path,
) -> None:
    path = (tmp_path / "official-process-deployment.json").resolve()

    publish_official_process_deployment_config(deployment_fixture.config, path)
    restored = load_official_process_deployment_config(path)

    assert restored == deployment_fixture.config
    assert restored.content_hash == deployment_fixture.config.content_hash
    assert json.loads(path.read_text(encoding="utf-8"))["content_hash"] == (
        deployment_fixture.config.content_hash
    )


def test_disk_backed_webshop_deployment_has_explicit_store_and_goal_assets(
    deployment_fixture: _DeploymentFixture,
) -> None:
    runtime = deployment_fixture.config.webshop.runtime
    config = SQLiteWebShopDeploymentConfig(
        runtime=runtime,
        store_relative_path="webshop/prepared/products.sqlite3",
        goals_relative_path="webshop/prepared/goals.jsonl",
        search_index_relative_path="webshop/prepared/indexes",
        asset_relative_files=(
            "webshop/prepared/goals.jsonl",
            "webshop/prepared/indexes/segments_1",
            "webshop/prepared/products.sqlite3",
        ),
    )

    assert SQLiteWebShopDeploymentConfig.from_value(config.to_value()) == config


def test_file_factory_builds_disk_backed_webshop_dependency(
    deployment_fixture: _DeploymentFixture,
    tmp_path: Path,
) -> None:
    prepared = deployment_fixture.target_root / "webshop" / "prepared"
    _write(prepared / "products.sqlite3", "store\n")
    _write(prepared / "goals.jsonl", "{}\n")
    _write(prepared / "indexes" / "segments_1", "index\n")
    webshop = SQLiteWebShopDeploymentConfig(
        runtime=deployment_fixture.config.webshop.runtime,
        store_relative_path="webshop/prepared/products.sqlite3",
        goals_relative_path="webshop/prepared/goals.jsonl",
        search_index_relative_path="webshop/prepared/indexes",
        asset_relative_files=(
            "webshop/prepared/goals.jsonl",
            "webshop/prepared/indexes/segments_1",
            "webshop/prepared/products.sqlite3",
        ),
    )
    config = replace(deployment_fixture.config, webshop=webshop)
    path = (tmp_path / "disk-official-process-deployment.json").resolve()
    publish_official_process_deployment_config(config, path)

    deployment = FileOfficialProcessPreparationFactory(path).build(
        lock=deployment_fixture.lock,
        target_root=deployment_fixture.target_root,
    )

    assert deployment.dependencies.webshop.deployment_value() == {
        "goals_path": str(prepared / "goals.jsonl"),
        "index_path": str(prepared / "indexes"),
        "kind": "sqlite",
        "seed": FIXED_SEED,
        "store_path": str(prepared / "products.sqlite3"),
    }


def test_file_factory_builds_all_three_pinned_official_dependencies(
    deployment_fixture: _DeploymentFixture,
    tmp_path: Path,
) -> None:
    path = (tmp_path / "official-process-deployment.json").resolve()
    publish_official_process_deployment_config(deployment_fixture.config, path)

    deployment = FileOfficialProcessPreparationFactory(path).build(
        lock=deployment_fixture.lock,
        target_root=deployment_fixture.target_root,
    )

    assert deployment.dependencies.webshop.deployment.seed == FIXED_SEED
    assert deployment.dependencies.alfworld.seed == FIXED_SEED
    assert deployment.dependencies.scienceworld.seed == FIXED_SEED
    assert tuple(deployment.dependencies.alfworld.games) == ("game-0001",)
    assert deployment.alfworld_max_steps == 50
    assert deployment.scienceworld_max_steps == 100
    assert tuple(item.benchmark for item in deployment.external_manifest_inputs) == (
        Benchmark.APPWORLD,
        Benchmark.BFCL_V3,
    )
    for benchmark, runtime in (
        (Benchmark.WEBSHOP, deployment.dependencies.webshop.deployment.runtime),
        (Benchmark.ALFWORLD, deployment.dependencies.alfworld.runtime),
        (Benchmark.SCIENCE_WORLD, deployment.dependencies.scienceworld.runtime),
    ):
        expected = next(
            entry.source.revision
            for entry in deployment_fixture.lock.benchmarks
            if entry.benchmark is benchmark
        )
        assert runtime.source_revision == expected
        assert runtime.source_root == (
            deployment_fixture.target_root / benchmark.value / "repository"
        )


def test_factory_rejects_an_unrecorded_search_index_file(
    deployment_fixture: _DeploymentFixture,
    tmp_path: Path,
) -> None:
    _write(
        deployment_fixture.target_root
        / "webshop"
        / "repository"
        / "search_engine"
        / "indexes_100"
        / "segments_2",
        "unrecorded\n",
    )
    path = (tmp_path / "official-process-deployment.json").resolve()
    publish_official_process_deployment_config(deployment_fixture.config, path)

    with pytest.raises(ValueError):
        FileOfficialProcessPreparationFactory(path).build(
            lock=deployment_fixture.lock,
            target_root=deployment_fixture.target_root,
        )


def test_published_config_rejects_changed_content_hash(
    deployment_fixture: _DeploymentFixture,
    tmp_path: Path,
) -> None:
    path = (tmp_path / "official-process-deployment.json").resolve()
    publish_official_process_deployment_config(deployment_fixture.config, path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["content_hash"] = "sha256:" + "0" * 64
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_official_process_deployment_config(path)
