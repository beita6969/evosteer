"""Synthetic public reset strings only; no native dataset or model calls."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks import protocol_v13_training_sessions as sessions
from skillev_private.benchmarks.alfworld_official import (
    OfficialALFWorldEpisodeFactory,
    OfficialALFWorldResetResult,
    bind_reset_public_item,
)
from skillev_private.benchmarks.alfworld_public_goal import (
    LEGACY_GOAL_BINDING,
    RESET_GOAL_BINDING,
    reset_public_goal,
)
from skillev_private.benchmarks.evaluation_episode import (
    EvaluationEpisodeRecord,
    EvaluationSource,
    EvaluationTarget,
)
from skillev_private.benchmarks.official_environment_worker import reset_alfworld_env
from skillev_private.benchmarks.protocol_v13_session_deployments import (
    Protocol13TrainingDeployments,
)
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingOutput

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from tests.benchmarks.test_alfworld_official_bridge import _case, _ScriptedOfficialFactory
from tests.benchmarks.test_protocol_v13_training_sessions import _record

OBSERVATION = "You see a public workroom.\nYour task is to: move the blue block to the tray."
GOAL = "move the blue block to the tray."


@pytest.mark.parametrize(
    "observation",
    [
        "public room",
        "Your task is to:",
        "Your task is to: first\nYour task is to: second",
        "inline Your task is to: unknown",
    ],
)
def test_missing_ambiguous_or_unsupported_goal_is_preparation_error(observation):
    with pytest.raises(ValueError):
        reset_public_goal(observation)


def test_worker_uses_only_real_public_goal_and_keeps_legacy_explicitly():
    class Env:
        def reset(self):
            return [OBSERVATION], {
                "admissible_commands": [["look"]],
                "extra.gamefile": ["/synthetic-game"],
            }

    kwargs = {"expected_game": "/synthetic-game", "instruction_text": "old catalog annotation"}
    modern = reset_alfworld_env(Env(), **kwargs, goal_binding=RESET_GOAL_BINDING)
    assert modern["instruction_text"] == GOAL
    assert modern["observation_text"] == OBSERVATION
    assert reset_alfworld_env(Env(), **kwargs)["instruction_text"] == "old catalog annotation"


def test_frozen_public_goal_and_every_actual_reset_must_match(monkeypatch):
    from tests.benchmarks.test_alfworld_official_bridge import _ScriptedOfficialEnv

    monkeypatch.setattr(
        _ScriptedOfficialEnv,
        "close_after_preparation_failure",
        lambda env: setattr(env, "close_calls", env.close_calls + 1),
        raising=False,
    )
    old = _case()
    reset = OfficialALFWorldResetResult(OBSERVATION, GOAL, ("look",))
    public = bind_reset_public_item(old.public, reset, RESET_GOAL_BINDING)
    assert public.query == GOAL
    assert old.public.query != GOAL  # Original source object is untouched.
    case = replace(old, public=public)
    envs = _ScriptedOfficialFactory(
        public, reset_observation_override=OBSERVATION, reset_commands_override=("look",)
    )
    factory = OfficialALFWorldEpisodeFactory(envs, RESET_GOAL_BINDING)
    for _ in range(2):
        session = factory.create(case)
        assert json.loads(session.observed_reset_json)["instruction_text"] == GOAL
        asyncio.run(session.cleanup())
    envs.reset_observation_override = OBSERVATION.replace("blue", "red")
    # A fake echoed query cannot pass the new actual-observation comparison.
    with pytest.raises(ValueError):
        factory.create(case)
    assert envs.envs[-1].close_calls == 1
    envs.reset_observation_override = OBSERVATION + "\nDifferent public room detail."
    with pytest.raises(ValueError):
        factory.create(case)
    envs.reset_observation_override = OBSERVATION
    envs.reset_commands_override = ("look", "different command")
    with pytest.raises(ValueError):
        factory.create(case)


@pytest.mark.parametrize("record_kind", ["training", "readonly-development"])
def test_shared_hydration_replaces_annotation_not_public_goal(tmp_path, monkeypatch, record_kind):
    dataset = tmp_path / "json_2.1.1" / "synthetic"
    dataset.mkdir(parents=True)
    game = dataset / "game.tw-pddl"
    game.write_text("synthetic")
    (dataset / "traj_data.json").write_text("{}")
    config = tmp_path / "environment.yaml"
    config.write_text("synthetic")
    original = _record()
    task = replace(
        original.input,
        query="old annotation targets another object",
        task_family="alfworld/pick-and-place",
        public_context={
            "benchmark_id": "alfworld",
            "dataset_revision": "synthetic@1",
            "split": "train",
            "input_profile": "synthetic-profile",
        },
    )
    target = {
        "environment_route": {
            "game_file": str(game),
            "config_file": str(config),
            "seed": 0,
            "max_steps": 5,
            "mode": "train",
        }
    }
    record = replace(
        original,
        episode=replace(original.episode, benchmark=Protocol13Benchmark.ALF_WORLD),
        input=task,
        output=Protocol13TrainingOutput("alfworld-success", target),
    )
    if record_kind != "training":
        record = EvaluationEpisodeRecord(
            EvaluationSource(
                Protocol13Benchmark.ALF_WORLD,
                original.episode.population_id,
                original.episode.source_id,
                task.task_id,
            ),
            task,
            EvaluationTarget(target),
        )
    deployments = Protocol13TrainingDeployments(
        SimpleNamespace(
            dataset_root=tmp_path,
            config_path=config,
            interpreter=tmp_path,
            source_root=tmp_path,
            source_revision="synthetic",
            timeout_seconds=1,
        ),
        None,
        alfworld_goal_binding=RESET_GOAL_BINDING,
    )
    created = []

    class Factory:
        observation = OBSERVATION

        def __init__(self, runtime, config_path, games, seed, *, simulator_max_steps, goal_binding):
            assert goal_binding == RESET_GOAL_BINDING
            assert games[record.episode.source_id].instruction_text == task.query

        def create(self, official_task):
            from tests.benchmarks.test_alfworld_official_bridge import _ScriptedOfficialEnv

            env = _ScriptedOfficialEnv(official_task, GOAL, self.observation, ("look",))
            created.append(env)
            return env

    monkeypatch.setattr(sessions, "OfficialALFWorldProcessFactory", Factory)
    monkeypatch.setattr(sessions, "PinnedOfficialProcess", lambda *args: None)
    hydrated, create = asyncio.run(sessions._alfworld_route(record, deployments))
    assert hydrated.query == GOAL
    assert task.query not in json.dumps(hydrated.to_value())
    assert record.input.query == task.query
    assert created[0].close_calls == 1
    bundle = create()
    assert bundle.evaluator.public.query == GOAL
    asyncio.run(bundle.cleanup())
    Factory.observation = "A public room with no goal marker."
    with pytest.raises(ValueError):
        asyncio.run(sessions._alfworld_route(record, deployments))
    assert created[-1].close_calls == 1


def test_goal_binding_configuration_is_explicit_and_legacy_default():
    original = Protocol13TrainingDeployments(None, None)
    assert original.alfworld_goal_binding == LEGACY_GOAL_BINDING
    assert (
        replace(original, alfworld_goal_binding=RESET_GOAL_BINDING).alfworld_goal_binding
        == RESET_GOAL_BINDING
    )
    with pytest.raises(ValueError):
        replace(original, alfworld_goal_binding="guess-from-game-id")


@pytest.mark.parametrize("binding", [LEGACY_GOAL_BINDING, RESET_GOAL_BINDING])
def test_worker_deployment_dispatch_preserves_explicit_goal_contract(monkeypatch, binding):
    from skillev_private.benchmarks import official_environment_worker as worker

    class Env:
        def reset(self):
            return [OBSERVATION], {
                "admissible_commands": [["look"]],
                "extra.gamefile": ["/synthetic-game"],
            }

    monkeypatch.setattr(worker, "load_alfworld_config", lambda _: {})
    monkeypatch.setattr(
        worker, "create_alfworld_env", lambda *args, **kwargs: (Env(), "/synthetic-game")
    )
    deployment = {
        "config_path": "synthetic",
        "data_directory": "synthetic",
        "instruction_text": "catalog annotation",
        "seed": 0,
        "train_eval": "train",
    }
    if binding == RESET_GOAL_BINDING:
        deployment["goal_binding"] = binding
    bound = worker.ALFWorldWorker(
        "synthetic",
        deployment,
        {"game_id": "not-used-for-goal", "max_steps": 2, "simulator_max_steps": 2},
    )
    assert bound.reset()["instruction_text"] == (
        GOAL if binding == RESET_GOAL_BINDING else "catalog annotation"
    )
    factory = sessions.Protocol13TrainingSessionFactory(
        {}, SimpleNamespace(to_value=dict), alfworld_goal_binding=binding
    )
    controls = json.loads(factory.terminal_evaluation_conditions_json)
    if binding == RESET_GOAL_BINDING:
        assert controls["alfworld_goal_binding"] == binding
    else:
        assert "alfworld_goal_binding" not in controls  # Historical projection unchanged.


def test_real_process_env_closes_owned_worker_on_preparation_failure():
    from skillev_private.benchmarks.official_process import (
        WorkerLifecycleState,
        _ALFWorldProcessEnv,
    )

    closed = []
    client = SimpleNamespace(
        _state=WorkerLifecycleState.OPEN, close_successfully=lambda: closed.append(True)
    )
    env = _ALFWorldProcessEnv(_case().private_task, client)
    env.close_after_preparation_failure()
    assert closed == [True]
    client._state = WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE
    env.close_after_preparation_failure()
    assert closed == [True]
