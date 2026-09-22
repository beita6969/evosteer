from __future__ import annotations

import io
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from skillev_private.benchmarks import (
    acquisition_cli,
    official_process,
)
from skillev_private.benchmarks import (
    official_process_preparation as process_preparation,
)
from skillev_private.benchmarks.acquisition import (
    AcquiredArtifactReceipt,
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
    LockedBenchmarkAcquisition,
)
from skillev_private.benchmarks.acquisition_git import (
    AcquiredRepositoryReceipt,
    RepositoryAcquisitionReceipt,
)
from skillev_private.benchmarks.archive_preparation import (
    LockedArchiveBatchReceipt,
    PreparedLockedArchive,
    locked_archive_preparations,
)
from skillev_private.benchmarks.deadline_pipe import DeadlinePipe
from skillev_private.benchmarks.official_process import (
    ALFWorldGameDeployment,
    JsonArrayWebShopDeployment,
    OfficialALFWorldProcessFactory,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    OfficialWorkerClient,
    PinnedOfficialProcess,
)
from skillev_private.benchmarks.official_process_preparation import (
    OfficialProcessPreparationDeployment,
    build_official_process_manifest_inputs,
    enumerate_official_alfworld_tasks,
    enumerate_official_scienceworld_tasks,
    enumerate_official_webshop_tasks,
    load_official_alfworld_task_families,
)
from skillev_private.benchmarks.process_catalog import (
    ProductionProcessCatalogDependencies,
)
from skillev_private.benchmarks.process_preparation import (
    ExternalProcessTaskRecord,
    ProcessTaskManifestInput,
    load_locked_process_preparation_receipt,
)

from skillev.experiments import FIXED_SEED, Benchmark
from skillev.rollout import RolloutTask

_GIT_SOURCE_KINDS = {
    "pinned-atlas-preparation",
    "pinned-git-dataset",
    "pinned-git-environment",
    "pinned-git-file",
    "pinned-huggingface-environment",
}


def test_official_worker_has_one_explicit_success_terminal_path() -> None:
    for removed in ("abort", "_abort", "__enter__", "__exit__", "__del__"):
        assert not hasattr(OfficialWorkerClient, removed)
    assert hasattr(OfficialWorkerClient, "close_successfully")


def test_official_worker_does_not_inherit_parent_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    runtime = cast(
        PinnedOfficialProcess,
        SimpleNamespace(
            interpreter_path=Path(sys.executable),
            worker_script_path=worker,
            worker_stderr_path=None,
            source_root=source,
        ),
    )
    captured: dict[str, object] = {}

    class _Process:
        request_reader, request_writer = os.pipe()
        stdin = os.fdopen(request_writer, "wb")

    def popen(*args: object, **kwargs: object) -> _Process:
        del args
        captured.update(kwargs)
        return _Process()

    monkeypatch.setenv("PYTHONPATH", "/parent/runtime")
    monkeypatch.setenv("JVM_PATH", "/pinned/libjvm.so")
    monkeypatch.setattr(official_process.subprocess, "Popen", popen)

    client = OfficialWorkerClient(runtime)
    environment = cast(dict[str, str], captured["env"])
    assert "PYTHONPATH" not in environment
    assert environment["JVM_PATH"] == "/pinned/libjvm.so"
    os.close(client._response_fd)
    _Process.stdin.close()
    os.close(_Process.request_reader)


class _ScriptedProcess:
    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.returncode: int | None = None
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        assert timeout > 0.0
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.kill_calls += 1

    def terminate(self) -> None:
        self.returncode = -15


def _scripted_worker_client() -> tuple[OfficialWorkerClient, _ScriptedProcess, int]:
    process = _ScriptedProcess()
    response_fd, response_writer = os.pipe()
    client = object.__new__(OfficialWorkerClient)
    client.runtime = cast(
        PinnedOfficialProcess,
        SimpleNamespace(request_timeout_seconds=0.01),
    )
    client._process = cast(subprocess.Popen[bytes], process)
    client._response_fd = response_fd
    client._request_id = 0
    client._state = official_process.WorkerLifecycleState.OPEN
    return client, process, response_writer


def test_official_worker_close_is_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, response_writer = _scripted_worker_client()
    monkeypatch.setattr(
        OfficialWorkerClient,
        "request",
        lambda self, operation, payload: {"closed": True},
    )
    os.close(response_writer)

    client.close_successfully()
    with pytest.raises(RuntimeError):
        client.close_successfully()


def test_official_worker_timeout_discards_transport_and_cleanup_preserves_failure() -> None:
    client, process, response_writer = _scripted_worker_client()

    class TimedOutChannel:
        def write(self, request: bytes, *, deadline: float) -> None:
            del request, deadline

        def read(self, *, deadline: float) -> bytes:
            del deadline
            raise TimeoutError("incomplete response")

    client._channel = cast(DeadlinePipe, TimedOutChannel())

    try:
        with pytest.raises(official_process.OfficialEnvironmentInfrastructureError):
            client.request("step", {})
        assert process.poll() is not None
        assert (
            client._state is official_process.WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE
        )
        client.close_successfully()
        with pytest.raises(RuntimeError):
            client.request("step", {})
    finally:
        os.close(response_writer)


def test_official_worker_failed_initialization_is_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, response_writer = _scripted_worker_client()

    def fail_initialization(*args: object) -> None:
        del args
        raise official_process.OfficialEnvironmentInfrastructureError("fixture failure")

    monkeypatch.setattr(official_process, "_initialize", fail_initialization)
    try:
        with pytest.raises(official_process.OfficialEnvironmentInfrastructureError):
            official_process._initialize_or_discard(client, {})
        assert process.poll() is not None
        assert process.stdin.closed
        assert client._state is (
            official_process.WorkerLifecycleState.DISCARDED_AFTER_INITIALIZATION_FAILURE
        )
        with pytest.raises(OSError):
            os.fstat(client._response_fd)
    finally:
        os.close(response_writer)


_WEBSHOP_MODULE = """
class _Server:
    def __init__(self):
        self.goals = [
            {
                "category": "home",
                "instruction_text": "Find the public home product.",
            },
            {
                "category": "office",
                "instruction_text": "Find the public office product.",
            },
        ]


class WebAgentTextEnv:
    def __init__(self, observation_mode, file_path, num_products, human_goals):
        self.server = _Server()

    def reset(self, session=None):
        return "public shop landing", None

    def get_instruction_text(self):
        return self.server.goals[0]["instruction_text"]

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": ["buy now"]}

    def step(self, action):
        return "public page", 1.0, True, None

    def close(self):
        pass
"""

_ALFWORLD_MODULE = """
import os


class _BatchEnv:
    def __init__(self, game_files, batch_size):
        self.gamefiles = list(game_files)
        self.batch_size = batch_size

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        game_files = self.gamefiles[: self.batch_size]
        return ["public room" for _ in game_files], {
            "admissible_commands": [["look", "finish"] for _ in game_files],
            "extra.gamefile": game_files,
        }

    def step(self, actions):
        return ["public state"], [0.0], [False], {
            "admissible_commands": [["finish"]],
            "won": [False],
        }

    def close(self):
        pass


class _Builder:
    def __init__(self, config, train_eval):
        key = {
            "train": "data_path",
            "eval_in_distribution": "eval_id_data_path",
            "eval_out_of_distribution": "eval_ood_data_path",
        }[train_eval]
        root = config["dataset"][key]
        self.game_files = []
        for directory, _children, files in os.walk(root):
            if "game.tw-pddl" in files:
                self.game_files.append(os.path.join(directory, "game.tw-pddl"))

    def init_env(self, batch_size):
        return _BatchEnv(self.game_files, batch_size)


def get_environment(name):
    return _Builder
"""

_SCIENCEWORLD_MODULE = """
class ScienceWorldEnv:
    def __init__(self, serverPath, envStepLimit):
        self.loaded = None

    def get_task_names(self):
        return ["task-a", "task-b"]

    def get_variations_test(self):
        return [1, 2] if self.loaded[0] == "task-a" else [3]

    def load(self, task_name, variation_index, simplification, generate_gold):
        self.loaded = (task_name, variation_index)

    def reset(self):
        return "public lab " + self.loaded[0] + ":" + str(self.loaded[1]), {}

    def get_task_description(self):
        return "public instruction " + self.loaded[0] + ":" + str(self.loaded[1])

    def step(self, action):
        return "public lab", 0, False, {"score": 0}

    def close(self):
        pass
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(*arguments: str) -> str:
    return subprocess.run(  # noqa: S603 - isolated fixture repository and fixed git CLI
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_revision(root: Path) -> str:
    _git("init", "-q", str(root))
    _git("-C", str(root), "config", "user.email", "fixture@example.invalid")
    _git("-C", str(root), "config", "user.name", "Fixture")
    _git("-C", str(root), "add", ".")
    _git("-C", str(root), "commit", "-q", "-m", "fixture")
    return _git("-C", str(root), "rev-parse", "HEAD")


def _lock_with_process_revision(revision: str) -> BenchmarkAcquisitionLock:
    path = Path(__file__).parents[2] / "benchmark-acquisition-lock.json"
    base = acquisition_cli.load_benchmark_acquisition_lock(path)
    process_benchmarks = {
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.APPWORLD,
        Benchmark.SCIENCE_WORLD,
        Benchmark.BFCL_V3,
    }
    entries = tuple(
        LockedBenchmarkAcquisition(
            benchmark=entry.benchmark,
            source=replace(entry.source, revision=revision),
        )
        if entry.benchmark in process_benchmarks
        else entry
        for entry in base.benchmarks
    )
    return BenchmarkAcquisitionLock(
        benchmarks=entries,
        shared_infrastructure=base.shared_infrastructure,
    )


def _receipt_chain(
    lock: BenchmarkAcquisitionLock,
    root: Path,
) -> tuple[
    BenchmarkAcquisitionReceipt,
    LockedArchiveBatchReceipt,
    RepositoryAcquisitionReceipt,
]:
    acquired = BenchmarkAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=root,
        artifacts=tuple(
            AcquiredArtifactReceipt(
                benchmark=entry.benchmark,
                locator=artifact.locator,
                relative_path=artifact.relative_path,
                size_bytes=artifact.expected_size or 1,
                sha256=artifact.expected_sha256 or "a" * 64,
            )
            for entry in lock.benchmarks
            for artifact in entry.source.artifacts
        ),
    )
    archives = LockedArchiveBatchReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquired.content_hash,
        archives=tuple(
            PreparedLockedArchive(
                benchmark=plan.benchmark,
                artifact_relative_path=plan.artifact_relative_path,
                output_relative_path=plan.output_relative_path,
                preparation_content_hash=f"sha256:{'b' * 64}",
            )
            for plan in locked_archive_preparations(lock)
        ),
    )
    repositories = RepositoryAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=root,
        repositories=tuple(
            AcquiredRepositoryReceipt(
                benchmark=entry.benchmark,
                repository=entry.source.repository,
                revision=entry.source.revision,
                relative_checkout_path=f"{entry.benchmark.value}/repository",
            )
            for entry in lock.benchmarks
            if entry.source.kind in _GIT_SOURCE_KINDS
        ),
    )
    return acquired, archives, repositories


@pytest.fixture
def process_dependencies(
    tmp_path: Path,
) -> tuple[Path, ProductionProcessCatalogDependencies]:
    root = tmp_path / "official-source"
    _write(root / "web_agent_site/__init__.py", "")
    _write(root / "web_agent_site/envs/__init__.py", "")
    _write(
        root / "web_agent_site/envs/web_agent_text_env.py",
        _WEBSHOP_MODULE,
    )
    _write(root / "alfworld/__init__.py", "")
    _write(root / "alfworld/agents/__init__.py", "")
    _write(root / "alfworld/agents/environment/__init__.py", _ALFWORLD_MODULE)
    _write(root / "scienceworld/__init__.py", _SCIENCEWORLD_MODULE)
    _write(root / "products.json", "[]\n")
    (root / "search_engine/indexes_100").mkdir(parents=True)
    _write(root / "scienceworld.jar", "fixture\n")
    _write(
        root / "alfworld-config.yaml",
        (
            "dataset:\n"
            "  data_path: unused\n"
            "  eval_id_data_path: unused\n"
            "  eval_ood_data_path: unused\n"
            "general:\n"
            "  random_seed: 0\n"
            "rl:\n"
            "  training:\n"
            "    max_nb_steps_per_episode: 1\n"
            "dagger:\n"
            "  training:\n"
            "    max_nb_steps_per_episode: 1\n"
        ),
    )
    game = root / "alfworld-data/game-1"
    _write(game / "game.tw-pddl", "fixture\n")
    _write(
        game / "traj_data.json",
        '{"task_type": "pick_and_place_simple"}\n',
    )
    revision = _git_revision(root)
    runtime = PinnedOfficialProcess(
        interpreter_path=Path(sys.executable),
        source_root=root,
        source_revision=revision,
        request_timeout_seconds=5.0,
    )
    dependencies = ProductionProcessCatalogDependencies(
        webshop=OfficialWebShopProcessFactory(
            JsonArrayWebShopDeployment(
                runtime=runtime,
                products_path=root / "products.json",
                index_path=root / "search_engine/indexes_100",
                inventory_size=100,
                seed=FIXED_SEED,
            )
        ),
        alfworld=OfficialALFWorldProcessFactory(
            runtime=runtime,
            config_path=root / "alfworld-config.yaml",
            games={
                "game-1": ALFWorldGameDeployment(
                    data_directory=game,
                    train_eval="train",
                    instruction_text="Complete the public household task.",
                )
            },
            seed=FIXED_SEED,
        ),
        scienceworld=OfficialScienceWorldProcessFactory(
            runtime=runtime,
            jar_path=root / "scienceworld.jar",
            simplification="",
            seed=FIXED_SEED,
        ),
    )
    return root, dependencies


def test_enumerates_exact_answer_free_official_process_records(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
) -> None:
    _, dependencies = process_dependencies

    webshop = enumerate_official_webshop_tasks(dependencies.webshop)
    alfworld = enumerate_official_alfworld_tasks(
        dependencies.alfworld,
        task_families={"game-1": "alfworld/pick-and-place"},
        max_steps=50,
    )
    scienceworld = enumerate_official_scienceworld_tasks(
        dependencies.scienceworld,
        max_steps=100,
    )

    assert {record.goal_index for record in webshop} == {0, 1}
    assert {record.task_family for record in webshop} == {
        "webshop/home",
        "webshop/office",
    }
    assert alfworld[0].initial_observation == "public room"
    assert alfworld[0].admissible_commands == ("look", "finish")
    assert alfworld[0].query == "Complete the public household task."
    assert {(record.task_name, record.variation_index) for record in scienceworld} == {
        ("task-a", 1),
        ("task-a", 2),
        ("task-b", 3),
    }
    assert all(record.query.startswith("public instruction") for record in scienceworld)
    wire = repr(
        [
            *(record.to_value() for record in webshop),
            *(record.to_value() for record in alfworld),
            *(record.to_value() for record in scienceworld),
        ]
    )
    assert "reward" not in wire
    assert "answer" not in wire
    assert "gold" not in wire


def test_alfworld_manifest_reuses_one_pinned_worker_for_explicit_games(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dependencies = process_dependencies
    original = dependencies.alfworld
    deployment = original.games["game-1"]
    game_ids = tuple(f"game-{index}" for index in range(1, 10))
    factory = OfficialALFWorldProcessFactory(
        runtime=original.runtime,
        config_path=original.config_path,
        games=dict.fromkeys(game_ids, deployment),
        seed=original.seed,
    )
    worker_count = 0
    progress: list[tuple[int, int]] = []
    worker_client = process_preparation.OfficialWorkerClient

    def counted_worker(runtime: PinnedOfficialProcess) -> object:
        nonlocal worker_count
        worker_count += 1
        return worker_client(runtime)

    monkeypatch.setattr(process_preparation, "OfficialWorkerClient", counted_worker)

    records = enumerate_official_alfworld_tasks(
        factory,
        task_families=dict.fromkeys(game_ids, "alfworld/pick-and-place"),
        max_steps=50,
        progress_observer=lambda completed, total: progress.append((completed, total)),
    )

    assert worker_count == 1
    assert progress == [(8, 9), (9, 9)]
    assert {record.game_id for record in records} == set(game_ids)
    assert all(record.initial_observation == "public room" for record in records)
    assert all(record.admissible_commands == ("look", "finish") for record in records)


@pytest.mark.parametrize(
    "train_eval",
    ["eval_in_distribution", "eval_out_of_distribution"],
)
def test_alfworld_manifest_supports_official_validation_splits(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
    train_eval: str,
) -> None:
    _, dependencies = process_dependencies
    original = dependencies.alfworld
    deployment = original.games["game-1"]
    factory = OfficialALFWorldProcessFactory(
        runtime=original.runtime,
        config_path=original.config_path,
        games={
            "game-1": ALFWorldGameDeployment(
                data_directory=deployment.data_directory,
                train_eval=train_eval,
                instruction_text=deployment.instruction_text,
            )
        },
        seed=original.seed,
    )

    records = enumerate_official_alfworld_tasks(
        factory,
        task_families={"game-1": "alfworld/pick-and-place"},
        max_steps=50,
    )

    assert len(records) == 1
    assert records[0].game_id == "game-1"


def test_builds_three_manifest_inputs_in_production_order(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
) -> None:
    _, dependencies = process_dependencies

    inputs = build_official_process_manifest_inputs(
        dependencies,
        webshop_asset_relative_files=("webshop/environment.asset",),
        alfworld_asset_relative_files=("alfworld/environment.asset",),
        scienceworld_asset_relative_files=("scienceworld/environment.asset",),
        alfworld_max_steps=50,
        scienceworld_max_steps=100,
    )

    assert tuple(item.benchmark for item in inputs) == (
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.SCIENCE_WORLD,
    )
    assert tuple(len(item.records) for item in inputs) == (2, 1, 3)
    assert all(
        item.records == tuple(sorted(item.records, key=lambda row: row.task_id)) for item in inputs
    )
    assert inputs[1].records[0].task_family == "alfworld/pick_and_place_simple"


def test_explicit_preparation_deployment_uses_the_shared_official_builder(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
) -> None:
    _, dependencies = process_dependencies
    deployment = OfficialProcessPreparationDeployment(
        dependencies=dependencies,
        webshop_asset_relative_files=("webshop/environment.asset",),
        alfworld_asset_relative_files=("alfworld/environment.asset",),
        scienceworld_asset_relative_files=("scienceworld/environment.asset",),
        alfworld_max_steps=50,
        scienceworld_max_steps=100,
    )

    inputs = deployment.manifest_inputs()

    assert tuple(item.benchmark for item in inputs) == (
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.SCIENCE_WORLD,
    )
    assert tuple(len(item.records) for item in inputs) == (2, 1, 3)


def test_prepare_process_command_materializes_official_manifests_and_receipt(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dependencies = process_dependencies
    revision = dependencies.webshop.deployment.runtime.source_revision
    lock = _lock_with_process_revision(revision)
    acquired, archives, repositories = _receipt_chain(lock, root)
    external_inputs: list[ProcessTaskManifestInput] = []
    for benchmark, split in (
        (Benchmark.APPWORLD, "train"),
        (Benchmark.BFCL_V3, "test"),
    ):
        asset = f"{benchmark.value}/environment.asset"
        _write(root / asset, "public fixture asset\n")
        task = RolloutTask(
            task_id=f"{benchmark.value}-task-0001",
            environment_id=f"{benchmark.value}-environment",
            task_family=f"{benchmark.value}/fixture",
            context_id=f"{benchmark.value}-context-0001",
            query=f"Complete the public {benchmark.value} task.",
            available_tools=("submit",),
            public_context={"benchmark_id": benchmark.value},
        )
        external_inputs.append(
            ProcessTaskManifestInput(
                benchmark=benchmark,
                dataset_revision=revision,
                split=split,
                task_manifest_relative_path=f"_derived/process/{benchmark.value}.jsonl",
                asset_relative_files=(asset,),
                environment_source_revision=revision,
                records=(
                    ExternalProcessTaskRecord(
                        benchmark=benchmark,
                        source_split=split,
                        task=task,
                    ),
                ),
            )
        )
    deployment = OfficialProcessPreparationDeployment(
        dependencies=dependencies,
        webshop_asset_relative_files=("products.json",),
        alfworld_asset_relative_files=(
            "alfworld-config.yaml",
            "alfworld-data/game-1/game.tw-pddl",
            "alfworld-data/game-1/traj_data.json",
        ),
        scienceworld_asset_relative_files=("scienceworld.jar",),
        alfworld_max_steps=50,
        scienceworld_max_steps=100,
        external_manifest_inputs=tuple(external_inputs),
    )

    class Factory:
        @staticmethod
        def build(
            *,
            lock: BenchmarkAcquisitionLock,
            target_root: Path,
        ) -> OfficialProcessPreparationDeployment:
            assert target_root == root
            assert lock == lock_for_factory
            return deployment

    lock_for_factory = lock
    monkeypatch.setattr(
        acquisition_cli,
        "load_benchmark_acquisition_lock",
        lambda path: lock,
    )
    monkeypatch.setattr(
        acquisition_cli,
        "load_benchmark_acquisition_receipt",
        lambda path: acquired,
    )
    monkeypatch.setattr(
        acquisition_cli,
        "load_locked_archive_batch_receipt",
        lambda path: archives,
    )
    monkeypatch.setattr(
        acquisition_cli,
        "load_repository_acquisition_receipt",
        lambda path: repositories,
    )
    output = (tmp_path / "process-preparation.json").resolve()

    result = acquisition_cli.prepare_process(
        lock_path=tmp_path / "lock.json",
        target_root=root,
        acquisition_receipt_path=tmp_path / "acquired.json",
        archive_receipt_path=tmp_path / "archives.json",
        repository_receipt_path=tmp_path / "repositories.json",
        receipt_path=output,
        factory=Factory(),
    )

    restored = load_locked_process_preparation_receipt(output)
    assert result["operation"] == "prepare-process"
    assert result["benchmark_count"] == 5
    assert result["task_count"] == 8
    assert restored.target_root == root
    assert tuple(source.task_count for source in restored.sources) == (2, 1, 1, 3, 1)


def test_loads_alfworld_family_from_pinned_official_metadata(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
) -> None:
    _, dependencies = process_dependencies

    assert load_official_alfworld_task_families(dependencies.alfworld) == {
        "game-1": "alfworld/pick_and_place_simple"
    }


def test_alfworld_requires_an_explicit_family_for_every_game(
    process_dependencies: tuple[Path, ProductionProcessCatalogDependencies],
) -> None:
    _, dependencies = process_dependencies

    with pytest.raises(ValueError):
        enumerate_official_alfworld_tasks(
            dependencies.alfworld,
            task_families={},
            max_steps=50,
        )


@pytest.mark.parametrize("failure", ["request-id", "protocol", "partial-eof", "invalid-json"])
def test_worker_bad_response_is_infrastructure_failure_and_reaped(failure):
    import json

    client, process, writer = _scripted_worker_client()
    request_reader, request_writer = os.pipe()
    process.stdin = os.fdopen(request_writer, "wb", buffering=0)
    client._channel = DeadlinePipe(request_writer, client._response_fd, 4096)
    response = {
        "protocol_version": official_process._PROTOCOL_VERSION,
        "request_id": 2 if failure == "request-id" else 1,
        "result": {"ok": True},
    }
    if failure == "protocol":
        response["protocol_version"] = "unsupported"
    payload = json.dumps(response).encode() + b"\n"
    if failure == "partial-eof":
        payload = b'{"result":'
    if failure == "invalid-json":
        payload = b"not-json\n"
    os.write(writer, payload)
    os.close(writer)
    try:
        with pytest.raises(official_process.OfficialEnvironmentInfrastructureError):
            client.request("step", {"action": "look"})
        assert process.poll() is not None
        assert (
            client._state is official_process.WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE
        )
        client.close_successfully()
    finally:
        os.close(request_reader)
