"""Native simulator construction without prompt assets or symbolic policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillev.evaluation.direct_baseline.interactive_tasks import NativeInteractiveEnvironment
from skillev.evaluation.scienceworld_commands import LEGACY_COMMAND_PROFILE
from skillev_private.benchmarks.alfworld_official import OfficialALFWorldTask
from skillev_private.benchmarks.official_process import (
    ALFWorldGameDeployment,
    OfficialALFWorldProcessFactory,
    OfficialWebShopProcessFactory,
    PinnedOfficialProcess,
    SQLiteWebShopDeployment,
)
from skillev_private.benchmarks.webshop_official import OfficialWebShopGoal
from skillev_private.direct_reference.environments import (
    DirectALFWorldEnvironment,
    DirectWebShopEnvironment,
)


def create_native_environment(
    specification: dict[str, Any],
    *,
    seed: int,
    stderr_path: Path | None = None,
    maximum_steps: int | None = None,
    webshop_observation_mode: str = "text",
    scienceworld_observation_profile: str = "text-only@1",
    scienceworld_command_profile: str = LEGACY_COMMAND_PROFILE,
) -> NativeInteractiveEnvironment:
    if specification["case"]["benchmark"] == "scienceworld":
        from .ood_scienceworld import create_scienceworld_environment

        return create_scienceworld_environment(
            specification,
            seed=seed,
            stderr_path=stderr_path,
            maximum_steps=maximum_steps,
            observation_profile=scienceworld_observation_profile,
            command_profile=scienceworld_command_profile,
        )
    case, manifest = specification["case"], specification["manifest"]
    deployment = manifest["deployments"][case["deployment"]]
    raw = manifest["runtimes"][deployment["runtime"]]
    runtime = PinnedOfficialProcess(
        Path(raw["interpreter_path"]),
        Path(raw["source_root"]),
        raw["source_revision"],
        float(raw.get("request_timeout_seconds", 120)),
        worker_stderr_path=stderr_path,
    )
    payload, task_id, task = case["payload"], case["task_id"], case["task"]
    if case["benchmark"] == "webshop":
        factory = OfficialWebShopProcessFactory(
            SQLiteWebShopDeployment(
                runtime,
                Path(deployment["store_path"]),
                Path(deployment["goals_path"]),
                Path(deployment["index_path"]),
                seed,
            ),
            observation_mode=webshop_observation_mode,
        )
        goal = OfficialWebShopGoal(
            task_id,
            payload["environment_id"],
            payload["goal_id"],
            payload["session_id"],
            int(payload["goal_index"]),
        )
        return DirectWebShopEnvironment(factory.create(goal), goal.session_id, task)
    if case["benchmark"] == "alfworld":
        step_limit = int(case["max_steps"]) if maximum_steps is None else maximum_steps
        if type(step_limit) is not int or step_limit < 1:
            raise ValueError("native environment action budget must be a positive integer")
        game_id = payload["game_id"]
        game = deployment["games"][game_id]
        game_deployment = ALFWorldGameDeployment(
            Path(game["data_directory"]), game["train_eval"], game["instruction_text"]
        )
        alfworld_factory = OfficialALFWorldProcessFactory(
            runtime,
            Path(deployment["config_path"]),
            {game_id: game_deployment},
            seed,
            # A newly declared outer horizon must reach the actual simulator.
            # Otherwise a longer actor loop still dies at the old native cap.
            max(step_limit, int(deployment.get("simulator_max_steps") or step_limit)),
        )
        alfworld_task = OfficialALFWorldTask(
            task_id, payload["environment_id"], game_id, seed, step_limit, dict(payload)
        )
        return DirectALFWorldEnvironment(alfworld_factory.create(alfworld_task), seed, task)
    raise ValueError("unsupported IID simulator")
