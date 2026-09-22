from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from skillev_private.benchmarks.alfworld_official import OfficialALFWorldTask
from skillev_private.benchmarks.official_process import (
    ALFWorldGameDeployment,
    JsonArrayWebShopDeployment,
    OfficialALFWorldProcessFactory,
    OfficialEnvironmentInfrastructureError,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    PinnedOfficialProcess,
    SQLiteWebShopDeployment,
)
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldTask
from skillev_private.benchmarks.webshop_official import OfficialWebShopGoal


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


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


_WEBSHOP_MODULE = """
import os
import time


class SimServer:
    pass


class WebAgentTextEnv:
    def __init__(
        self,
        observation_mode,
        file_path,
        num_products=None,
        human_goals=False,
        server=None,
    ):
        self.initialization = (observation_mode, file_path, num_products, human_goals)
        self.server = server
        self.goal_index = None
        self.closed = False

    def reset(self, session=None):
        self.goal_index = session
        return "public shop landing for index " + str(session), None

    def get_instruction_text(self):
        return "Instruction: public shopping instruction"

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": ["item-1", "buy now"]}

    def step(self, action):
        if action == "crash":
            raise RuntimeError("private official crash detail")
        if action == "hang":
            time.sleep(2.0)
        terminal = action == "click[buy now]"
        return "public page after " + action, 1.0 if terminal else 0.25, terminal, None

    def close(self):
        self.closed = True
"""


_PYSERINI_MODULE = """
class LuceneSearcher:
    def __init__(self, index_path):
        self.index_path = index_path
"""


_ALFWORLD_MODULE = """
import os


class _BatchEnv:
    def __init__(self, game_file):
        self.game_file = game_file

    def reset(self):
        return ["public room"], {
            "admissible_commands": [["look", "finish"]],
            "extra.gamefile": [self.game_file],
        }

    def step(self, actions):
        action = actions[0]
        terminal = action == "finish"
        return ["public state after " + action], [1.0 if terminal else 0.0], [terminal], {
            "admissible_commands": [[] if terminal else ["finish"]],
            "won": [terminal],
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
        assert batch_size == 1
        return _BatchEnv(self.game_files[0])


def get_environment(name):
    assert name == "AlfredTWEnv"
    return _Builder
"""


_SCIENCEWORLD_MODULE = """
class ScienceWorldEnv:
    def __init__(self, serverPath, envStepLimit):
        self.server_path = serverPath
        self.limit = envStepLimit
        self.loaded = None

    def load(self, task_name, variation_index, simplification, generate_gold):
        assert generate_gold is False
        self.loaded = (task_name, variation_index, simplification)

    def reset(self):
        return "public laboratory for " + self.loaded[0] + ":" + str(self.loaded[1]), {"score": 0}

    def get_task_description(self):
        return "public science instruction"

    def step(self, action):
        terminal = action == "finish"
        return "public lab after " + action, 100 if terminal else 10, terminal, {
            "score": 100 if terminal else 10,
        }

    def close(self):
        pass
"""


@pytest.fixture
def official_source(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "official-source"
    _write(root / "web_agent_site" / "__init__.py", "")
    _write(root / "web_agent_site" / "envs" / "__init__.py", "")
    _write(
        root / "web_agent_site" / "envs" / "web_agent_text_env.py",
        _WEBSHOP_MODULE,
    )
    _write(root / "pyserini" / "__init__.py", "")
    _write(root / "pyserini" / "search" / "__init__.py", "")
    _write(root / "pyserini" / "search" / "lucene.py", _PYSERINI_MODULE)
    _write(root / "alfworld" / "__init__.py", "")
    _write(root / "alfworld" / "agents" / "__init__.py", "")
    _write(root / "alfworld" / "agents" / "environment" / "__init__.py", _ALFWORLD_MODULE)
    _write(root / "scienceworld" / "__init__.py", _SCIENCEWORLD_MODULE)
    _write(root / "products.json", "[]")
    (root / "search_engine" / "indexes_100").mkdir(parents=True)
    _write(root / "scienceworld.jar", "fixture")
    revision = _git_revision(root)
    return root, revision


def _runtime(
    source: tuple[Path, str],
    *,
    timeout: float = 2.0,
) -> PinnedOfficialProcess:
    root, revision = source
    return PinnedOfficialProcess(
        interpreter_path=Path(sys.executable),
        source_root=root,
        source_revision=revision,
        request_timeout_seconds=timeout,
    )


def test_pinned_process_preserves_virtualenv_interpreter_symlink(
    official_source: tuple[Path, str],
    tmp_path: Path,
) -> None:
    root, revision = official_source
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable))

    runtime = PinnedOfficialProcess(
        interpreter_path=interpreter,
        source_root=root,
        source_revision=revision,
    )

    assert runtime.interpreter_path == interpreter.absolute()
    assert runtime.interpreter_path.is_symlink()


def test_webshop_factory_binds_numeric_goal_and_projects_actions_reward(
    official_source: tuple[Path, str],
) -> None:
    root, _ = official_source
    factory = OfficialWebShopProcessFactory(
        JsonArrayWebShopDeployment(
            runtime=_runtime(official_source),
            products_path=root / "products.json",
            index_path=root / "search_engine" / "indexes_100",
            inventory_size=100,
            seed=19,
        )
    )
    goal = OfficialWebShopGoal(
        task_id="shop-task",
        environment_id="shop-environment",
        goal_id="goal-7",
        session_id="session-7",
        payload=7,
    )
    environment = factory.create(goal)

    assert environment.reset("session-7") == "public shop landing for index 7"
    assert environment.instruction_text == "public shopping instruction"
    assert environment.get_available_actions() == (
        "search",
        "click[item-1]",
        "click[buy now]",
    )
    first = environment.step("search[fixture]")
    terminal = environment.step("click[buy now]")
    assert first.reward == 0.25
    assert not first.terminal
    assert terminal.reward == 1.0
    assert terminal.terminal
    asyncio.run(environment.close())

    abandoned = factory.create(goal)
    abandoned.reset("session-7")
    assert not abandoned.step("search[fixture]").terminal
    process = abandoned.client._process
    asyncio.run(abandoned.close())
    asyncio.run(abandoned.close())
    assert process.poll() == 0


def test_webshop_factory_uses_disk_backed_official_server(
    official_source: tuple[Path, str],
) -> None:
    root, _ = official_source
    store_path = root / "prepared" / "products.sqlite3"
    store_path.parent.mkdir()
    connection = sqlite3.connect(store_path)
    try:
        connection.executescript(
            """
            CREATE TABLE build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO build_meta(key, value) VALUES ('phase', 'complete');
            INSERT INTO build_meta(key, value) VALUES ('accepted', '1');
            CREATE TABLE products (
                source_id INTEGER PRIMARY KEY,
                asin TEXT NOT NULL UNIQUE,
                product_json TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT NOT NULL,
                query TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO products(
                source_id, asin, product_json, price, category, query
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                0,
                "ASIN000001",
                json.dumps({"asin": "ASIN000001", "Title": "Fixture"}),
                12.5,
                "fixture",
                "fixture",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    goals_path = root / "prepared" / "goals.jsonl"
    goals_path.write_text(
        json.dumps(
            {
                "category": "fixture",
                "instruction_text": "public shopping instruction",
                "weight": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = root / "prepared" / "indexes"
    index_path.mkdir()
    revision = _git_revision_after_change(root)
    factory = OfficialWebShopProcessFactory(
        SQLiteWebShopDeployment(
            runtime=PinnedOfficialProcess(Path(sys.executable), root, revision),
            store_path=store_path,
            goals_path=goals_path,
            index_path=index_path,
            seed=19,
        )
    )
    environment = factory.create(
        OfficialWebShopGoal(
            task_id="shop-task",
            environment_id="shop-environment",
            goal_id="goal-0",
            session_id="session-0",
            payload=0,
        )
    )

    assert environment.reset("session-0") == "public shop landing for index 0"
    assert environment.instruction_text == "public shopping instruction"
    assert environment.step("click[buy now]").reward == 1.0


def test_alfworld_factory_binds_single_game_without_passing_seed_to_reset(
    official_source: tuple[Path, str],
) -> None:
    root, _ = official_source
    game_directory = root / "case-game" / "game"
    _write(game_directory / "traj_data.json", "{}")
    _write(game_directory / "game.tw-pddl", "{}")
    config_path = root / "alfworld-config.yaml"
    _write(
        config_path,
        """
dataset:
  {data_path: '', eval_id_data_path: '', eval_ood_data_path: '',
   num_train_games: -1, num_eval_games: -1}
general: {random_seed: 0, training_method: dagger}
rl: {training: {max_nb_steps_per_episode: 50}}
dagger: {training: {max_nb_steps_per_episode: 50}}
""".strip(),
    )
    # The checkout identity changes when its explicit deployment data changes.
    revision = _git_revision_after_change(root)
    runtime = PinnedOfficialProcess(Path(sys.executable), root, revision, 2.0)
    factory = OfficialALFWorldProcessFactory(
        runtime=runtime,
        config_path=config_path,
        games={
            "game-1": ALFWorldGameDeployment(
                game_directory.parent,
                "train",
                "public ALFWorld instruction",
            )
        },
        seed=23,
    )
    task = OfficialALFWorldTask(
        task_id="alf-task",
        environment_id="alf-environment",
        game_id="game-1",
        seed=23,
        max_steps=2,
        payload={"private": "not projected"},
    )
    environment = factory.create(task)

    reset = environment.reset(23)
    first = environment.step("look")
    terminal = environment.step("finish")
    assert reset.observation_text == "public room"
    assert reset.instruction_text == "public ALFWorld instruction"
    assert reset.admissible_commands == ("look", "finish")
    assert first.success is None
    assert not first.terminal
    assert terminal.success is True
    assert terminal.terminal
    asyncio.run(environment.close())

    abandoned = factory.create(task)
    abandoned.reset(23)
    assert not abandoned.step("look").terminal
    process = abandoned.client._process
    asyncio.run(abandoned.close())
    asyncio.run(abandoned.close())
    assert process.poll() == 0


def _git_revision_after_change(root: Path) -> str:
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "add", "."), check=True
    )
    subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "commit", "-q", "-m", "deployment"), check=True
    )
    return subprocess.run(  # noqa: S603 - isolated fixture repository
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_scienceworld_factory_binds_jar_task_variation_and_native_score(
    official_source: tuple[Path, str],
) -> None:
    root, _ = official_source
    factory = OfficialScienceWorldProcessFactory(
        runtime=_runtime(official_source),
        jar_path=root / "scienceworld.jar",
        simplification="",
        seed=29,
    )
    task = OfficialScienceWorldTask(
        task_id="science-task",
        environment_id="science-environment",
        task_name="public-science-task",
        variation_index=4,
        seed=29,
        max_steps=3,
        payload={"private": "not projected"},
    )
    environment = factory.create(task)

    assert (
        environment.reset(
            task_name=task.task_name,
            variation_index=4,
            seed=29,
            max_steps=3,
        )
        == "public laboratory for public-science-task:4"
    )
    assert environment.task_description == "public science instruction"
    first = environment.step("inspect")
    terminal = environment.step("finish")
    assert first.score == 10.0
    assert not first.terminal
    assert terminal.score == 100.0
    assert terminal.terminal


def test_scienceworld_close_does_not_block_other_episodes(
    official_source: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = official_source
    environment = OfficialScienceWorldProcessFactory(
        runtime=_runtime(official_source),
        jar_path=root / "scienceworld.jar",
        simplification="",
        seed=29,
    ).create(
        OfficialScienceWorldTask(
            task_id="science-close",
            environment_id="science-close-environment",
            task_name="public-science-task",
            variation_index=4,
            seed=29,
            max_steps=3,
            payload={},
        )
    )
    started, release = threading.Event(), threading.Event()
    client_type = type(environment.client)
    original_close = client_type.close_successfully
    progress: list[str] = []

    def slow_close(client):
        started.set()
        try:
            assert release.wait(2), "another episode must progress while close waits"
        finally:
            original_close(client)
        progress.append("closed")

    monkeypatch.setattr(client_type, "close_successfully", slow_close)

    async def another_episode() -> None:
        assert await asyncio.to_thread(started.wait, 2)
        progress.append("other-episode")
        release.set()

    async def run() -> None:
        await asyncio.gather(environment.close(), another_episode())
        await environment.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert progress == ["other-episode", "closed"]
    assert environment.client._process.poll() == 0


@pytest.mark.parametrize("action", ["crash", "hang"])
def test_worker_crash_and_timeout_are_infrastructure_failures(
    official_source: tuple[Path, str],
    action: str,
) -> None:
    root, _ = official_source
    # Leave enough time for a contended interpreter startup while remaining
    # strictly below the fixture's two-second hung step.
    timeout = 1.0 if action == "hang" else 2.0
    factory = OfficialWebShopProcessFactory(
        JsonArrayWebShopDeployment(
            runtime=_runtime(official_source, timeout=timeout),
            products_path=root / "products.json",
            index_path=root / "search_engine" / "indexes_100",
            inventory_size=100,
            seed=31,
        )
    )
    environment = factory.create(
        OfficialWebShopGoal(
            task_id="failure-task",
            environment_id="failure-environment",
            goal_id="failure-goal",
            session_id="failure-session",
            payload=0,
        )
    )
    environment.reset("failure-session")

    with pytest.raises(OfficialEnvironmentInfrastructureError):
        environment.step(action)


def test_source_revision_and_numeric_goal_are_strict(
    official_source: tuple[Path, str],
) -> None:
    root, revision = official_source
    with pytest.raises(ValueError):
        PinnedOfficialProcess(Path(sys.executable), root, "0" * 40)

    factory = OfficialWebShopProcessFactory(
        JsonArrayWebShopDeployment(
            runtime=PinnedOfficialProcess(Path(sys.executable), root, revision),
            products_path=root / "products.json",
            index_path=root / "search_engine" / "indexes_100",
            inventory_size=100,
            seed=37,
        )
    )
    with pytest.raises(ValueError):
        factory.create(
            OfficialWebShopGoal(
                task_id="invalid-task",
                environment_id="invalid-environment",
                goal_id="invalid-goal",
                session_id="invalid-session",
                payload={"goal_index": 1},
            )
        )
