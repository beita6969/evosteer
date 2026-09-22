"""Real ALFWorld text-environment bindings for EvoSteer curves.

This module deliberately keeps ALFWorld optional.  Importing it is cheap; the
official ``alfworld`` and ``textworld`` packages are loaded only when a session
is created.  A session owns one resettable TextWorld episode.  The model sees
the reset observation and admissible commands, while ``won`` and score remain
inside the terminal evaluator.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillev.contracts.evosteer import EvoTask
from skillev.evosteer_application import ResetReceipt, SessionRequest, TaskBinding, TaskSession
from skillev.orchestration.graph import NodeExecutionRequest, NodeExecutionResult
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy
from skillev.runtime import BudgetVector

ALFWORLD_FAMILY = "alfworld"
ALFWORLD_CONFIG_ENV = "ALFWORLD_CONFIG_FILE"
ALFWORLD_DATA_ENV = "ALFWORLD_DATA"


def _patch_textworld_python313() -> None:
    """Patch the upstream EvalSymbol implementation on Python 3.13."""
    try:
        from textworld.envs.pddl import textgen
    except ImportError:
        return
    if getattr(textgen.EvalSymbol, "_skillev_python313_compatible", False):
        return

    def derive(symbol: Any, context: Any = None) -> list[Any]:
        active = context or symbol.context
        value = eval(symbol.expression, vars(textgen), dict(active["variables"]))  # noqa: S307
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = derive
    textgen.EvalSymbol._skillev_python313_compatible = True


def _load_alfworld_env() -> Any:
    """Return the repository's official TextWorld wrapper."""
    _patch_textworld_python313()
    try:
        from ragen_adapter import ALFWorldEnv, AlfredEnvConfig
    except ImportError:
        from skillev.ragen_adapter import ALFWorldEnv, AlfredEnvConfig  # type: ignore
    return ALFWorldEnv, AlfredEnvConfig


def _configured_path() -> str:
    path = os.environ.get(ALFWORLD_CONFIG_ENV, "").strip()
    if path:
        return path
    # RAGEN's config is also the upstream default used by existing deployments.
    root = os.environ.get("RAGEN_ROOT", "")
    if root:
        candidate = Path(root) / "ragen/env/alfworld/alfworld_config.yaml"
        if candidate.exists():
            return str(candidate)
    try:
        package = importlib.import_module("alfworld")
        candidate = Path(str(getattr(package, "__file__", ""))).parent / "configs/base_config.yaml"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return ""


def _catalog(*, mode: str = "train") -> tuple[tuple[str, ...], str]:
    """Discover official game files without starting an episode."""
    ALFWorldEnv, AlfredEnvConfig = _load_alfworld_env()
    config = AlfredEnvConfig(config_file=_configured_path())
    catalog = ALFWorldEnv(config=config, mode=mode)
    files = tuple(str(item) for item in catalog.game_files)
    if not files:
        raise RuntimeError("ALFWorld has no game files; set ALFWORLD_DATA and run provision script")
    return files, str(config.config_file)


def training_split_manifest(*, mode: str = "train") -> dict[str, Any]:
    """Return exact game paths and count for the configured official split."""
    files, config = _catalog(mode=mode)
    return {
        "benchmark": ALFWORLD_FAMILY,
        "split": mode,
        "count": len(files),
        "config_file": config,
        "data_root": os.environ.get(ALFWORLD_DATA_ENV, ""),
        "game_files": list(files),
    }


class _ALFWorldEpisode:
    def __init__(self, *, request: SessionRequest, mode: str, max_steps: int, selection_seed: int | None = None) -> None:
        ALFWorldEnv, AlfredEnvConfig = _load_alfworld_env()
        config = AlfredEnvConfig(config_file=_configured_path())
        self.env = ALFWorldEnv(config=config, mode=mode)
        self.request = request
        self.max_steps = max(1, int(max_steps))
        self.steps = 0
        self.done = False
        self.won = False
        self.last_observation = ""
        self.admissible: tuple[str, ...] = ()
        self.game_file = ""
        self.reset(selection_seed=selection_seed)

    def reset(self, *, selection_seed: int | None = None) -> None:
        # ALFWorldEnv maps a seed to a game-file index.  Bindings pass their
        # catalog index here so each task covers exactly one official game.
        self.last_observation = str(self.env.reset(seed=self.request.seed if selection_seed is None else selection_seed))
        self.admissible = tuple(str(x) for x in getattr(self.env, "_admissible_commands", ()))
        self.game_file = str(getattr(self.env, "current_game_file", ""))

    def step(self, action: str) -> tuple[str, bool]:
        if self.done:
            return self.last_observation, self.won
        obs, _reward, done, info = self.env.step(action)
        self.steps += 1
        self.last_observation = str(obs)
        self.admissible = tuple(str(x) for x in info.get("available_actions", ()))
        self.won = bool(info.get("won", False))
        self.done = bool(done) or self.steps >= self.max_steps
        return self.last_observation, self.won

    def close(self) -> None:
        close = getattr(self.env, "alfred_env", None)
        if close is not None and hasattr(close, "close"):
            close.close()

    def public_state(self) -> dict[str, Any]:
        return {
            "environment": "alfworld-textworld",
            "steps": self.steps,
            "done": self.done,
            "admissible_commands": list(self.admissible),
            "observation": self.last_observation,
        }


def _select_action(text: str, admissible: tuple[str, ...]) -> str:
    """Extract a single command while never inventing an unavailable command."""
    for command in admissible:
        if text.strip() == command or command in text.splitlines():
            return command
    candidate = next((line.strip() for line in text.splitlines() if line.strip()), "look")
    return candidate if candidate in admissible else (admissible[0] if admissible else candidate)


class ALFWorldTextExecutor:
    """Frozen policy executor whose node calls are real environment actions."""

    def __init__(self, policy: Any, episode: _ALFWorldEpisode, *, temperature: float = 0.3) -> None:
        self.policy = policy
        self.episode = episode
        self.temperature = float(temperature)
        self._identity = "alfworld-text-executor@1:" + hashlib.sha256(
            (str(getattr(policy, "configuration_id", "policy")) + os.environ.get(ALFWORLD_DATA_ENV, "")).encode()
        ).hexdigest()[:16]

    @property
    def frozen_identity(self) -> str:
        return self._identity

    def public_environment_state(self) -> dict[str, Any]:
        return self.episode.public_state()

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        prompt = json.dumps(
            {
                "role": request.role.instruction,
                "task": request.task_prompt,
                "observation": self.episode.last_observation,
                "admissible_commands": list(self.episode.admissible),
                "messages": list(request.messages),
                "previous_output": request.previous_output,
                "instruction": "Return exactly one admissible ALFWorld command.",
            },
            sort_keys=True,
        )
        started = time.monotonic()
        output, inputs, outputs = self.policy.frozen_text(
            prompt,
            max_new_tokens=request.role.model_maximum.output_tokens,
            input_limit=request.role.model_maximum.input_tokens,
            temperature=self.temperature,
            seed=request.seed,
        )
        action = _select_action(str(output), self.episode.admissible)
        observation, _won = self.episode.step(action)
        public = json.dumps(
            {"action": action, "observation": observation, "admissible_commands": list(self.episode.admissible)},
            sort_keys=True,
        )
        return NodeExecutionResult(
            public,
            BudgetVector(
                input_tokens=int(inputs),
                output_tokens=int(outputs),
                model_calls=1,
                agent_turns=1,
                wall_time_milliseconds=max(0, int((time.monotonic() - started) * 1000)),
            ),
            {"environment_step": {"action": action, "done": self.episode.done}},
        )


def make_bindings(*, policy: Any, config: Any, mode: str = "train", count: int | None = None) -> tuple[TaskBinding, ...]:
    """Build EvoSteer bindings over the official ALFWorld training game split."""
    files, config_file = _catalog(mode=mode)
    selected = files[: int(count)] if count is not None else files
    bindings: list[TaskBinding] = []
    # The instruction is supplied by each real reset observation.  Game paths,
    # goals, and success labels are intentionally absent from the public task.
    for index, _game_file in enumerate(selected):
        task = EvoTask(
            task_id=f"alfworld/{mode}/{index:06d}",
            family=ALFWORLD_FAMILY,
            prompt="Complete the ALFWorld instruction shown by the reset observation.",
            reset_id=f"reset-{index}",
            environment_config_id=f"alfworld-textworld:{mode}:{config_file}",
        )
        executor_id = "alfworld-text-executor@1:" + hashlib.sha256(
            (str(getattr(policy, "configuration_id", "policy")) + os.environ.get(ALFWORLD_DATA_ENV, "")).encode()
        ).hexdigest()[:16]

        def session(request: SessionRequest, *, _task=task, _index=index) -> TaskSession:
            episode = _ALFWorldEpisode(request=request, mode=mode, max_steps=int(os.environ.get("ALFWORLD_MAX_STEPS", "50")), selection_seed=_index)
            executor = ALFWorldTextExecutor(policy, episode)

            def evaluate(_output: str) -> float:
                return 1.0 if episode.done and episode.won else 0.0

            receipt = ResetReceipt(
                request.task.identity,
                request.task.reset_id,
                request.task.environment_config_id,
                str(hashlib.sha256(episode.last_observation.encode()).hexdigest()),
                str(uuid.uuid4()),
                request.seed,
            )
            risk = ExecutionRiskPolicy(
                executor.frozen_identity,
                request.task.environment_config_id,
                scope="interactive_text",
                capability_id="alfworld-textworld@1",
            )
            return TaskSession(executor, evaluate, close=episode.close, reset_receipt=receipt, risk_assessor=risk)

        bindings.append(TaskBinding(task, executor_id, session, replay_safe=False))
    return tuple(bindings)


def alfworld_bindings(*, policy: Any, config: Any) -> tuple[TaskBinding, ...]:
    return make_bindings(policy=policy, config=config)


__all__ = ["ALFWORLD_FAMILY", "alfworld_bindings", "make_bindings", "training_split_manifest"]
