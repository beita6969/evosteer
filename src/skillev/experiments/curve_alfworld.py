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
    """Return the official TextWorld wrapper (``skillev.alfworld_env``).

    A RAGEN ``ragen_adapter`` on the path is still accepted as a fallback so
    older deployments keep working.
    """
    _patch_textworld_python313()
    try:
        from skillev.alfworld_env import ALFWorldEnv, AlfredEnvConfig
    except ImportError:
        from ragen_adapter import ALFWorldEnv, AlfredEnvConfig  # type: ignore
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
    # The repository ships the upstream base config (the pip package does not).
    repo_config = Path(__file__).resolve().parents[3] / "configs/alfworld/base_config.yaml"
    if repo_config.exists():
        return str(repo_config)
    try:
        package = importlib.import_module("alfworld")
        candidate = Path(str(getattr(package, "__file__", ""))).parent / "configs/base_config.yaml"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return ""


def _max_episode_steps() -> int:
    return max(1, int(os.environ.get("ALFWORLD_MAX_STEPS", "50")))


def _steps_per_node() -> int:
    return max(1, int(os.environ.get("ALFWORLD_STEPS_PER_NODE", "20")))


def _catalog(*, mode: str = "train") -> tuple[tuple[str, ...], str]:
    """Discover official game files without starting an episode."""
    ALFWorldEnv, AlfredEnvConfig = _load_alfworld_env()
    config = AlfredEnvConfig(config_file=_configured_path(), max_episode_steps=_max_episode_steps())
    catalog = ALFWorldEnv(config=config, mode=mode)
    files = tuple(str(item) for item in catalog.game_files)
    if not files:
        raise RuntimeError("ALFWorld has no game files; set ALFWORLD_DATA and run scripts/provision_alfworld.sh")
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
        config = AlfredEnvConfig(config_file=_configured_path(), max_episode_steps=max(1, int(max_steps)))
        self.env = ALFWorldEnv(config=config, mode=mode)
        self.history: list[tuple[str, str]] = []
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
        self.initial_observation = self.last_observation
        self.history = []
        self.steps = 0
        self.done = False
        self.won = False
        self.admissible = tuple(str(x) for x in getattr(self.env, "_admissible_commands", ()))
        self.game_file = str(getattr(self.env, "current_game_file", ""))

    def step(self, action: str) -> tuple[str, bool]:
        if self.done:
            return self.last_observation, self.won
        obs, _reward, done, info = self.env.step(action)
        self.steps += 1
        self.last_observation = str(obs)
        self.history.append((action, self.last_observation))
        self.admissible = tuple(str(x) for x in info.get("available_actions", ()))
        self.won = bool(info.get("won", False))
        self.done = bool(done) or self.won or self.steps >= self.max_steps
        return self.last_observation, self.won

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
            return
        inner = getattr(self.env, "alfred_env", None)
        if inner is not None and hasattr(inner, "close"):
            inner.close()

    def public_state(self) -> dict[str, Any]:
        return {
            "environment": "alfworld-textworld",
            "steps": self.steps,
            "done": self.done,
            "admissible_commands": list(self.admissible),
            "observation": self.last_observation,
        }


def _normalise(text: str) -> str:
    return " ".join(str(text).strip().strip("`'\".").lower().split())


def _select_action(text: str, admissible: tuple[str, ...]) -> str:
    """Map model text to one admissible command; never invent an unavailable one.

    Order: an exact line match (last line wins, so "Thought ... \\n go to desk 1"
    works), then the longest admissible command contained in the text, then
    ``look`` if it is admissible, else the first admissible command.
    """
    if not admissible:
        return next((line.strip() for line in str(text).splitlines() if line.strip()), "look")
    by_norm = {_normalise(command): command for command in admissible}
    lines = [_normalise(line.split(":", 1)[-1] if line.lower().startswith(("action", "command")) else line)
             for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        if line in by_norm:
            return by_norm[line]
    flat = _normalise(text)
    contained = [command for norm, command in by_norm.items() if norm and norm in flat]
    if contained:
        return max(contained, key=len)
    return by_norm.get("look", admissible[0])


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

    def _step_prompt(self, request: NodeExecutionRequest, goal: str) -> str:
        recent = self.episode.history[-8:]
        return json.dumps(
            {
                "role": request.role.instruction,
                "task": request.task_prompt,
                "goal_and_room": goal,
                "team_messages": list(request.messages),
                "previous_output": request.previous_output,
                "recent_history": [{"action": a, "observation": o} for a, o in recent],
                "observation": self.episode.last_observation,
                "admissible_commands": list(self.episode.admissible),
                "instruction": "Reply with exactly one command copied from admissible_commands and nothing else.",
            },
            sort_keys=True,
        )

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        """One agent turn = up to ALFWORLD_STEPS_PER_NODE real environment steps.

        The episode is shared by every node of the team, so a later node (for
        example a RERUN) continues from the current environment state.
        """
        started = time.monotonic()
        goal = self.episode.initial_observation
        per_step_output = max(16, min(64, int(request.role.model_maximum.output_tokens)))
        inputs_total = outputs_total = calls = 0
        taken: list[str] = []
        if self.episode.done:
            # The game already ended (won or step limit).  A node still has to be
            # a real model call, so it reports the final state instead of acting.
            output, inputs, outputs = self.policy.frozen_text(
                json.dumps(
                    {
                        "role": request.role.instruction,
                        "final_observation": self.episode.last_observation,
                        "instruction": "The episode has ended. Summarise the final observation in one line.",
                    },
                    sort_keys=True,
                ),
                max_new_tokens=per_step_output,
                input_limit=request.role.model_maximum.input_tokens,
                temperature=self.temperature,
                seed=int(request.seed),
            )
            inputs_total, outputs_total, calls = int(inputs), int(outputs), 1
        for offset in range(_steps_per_node()):
            if self.episode.done:
                break
            output, inputs, outputs = self.policy.frozen_text(
                self._step_prompt(request, goal),
                max_new_tokens=per_step_output,
                input_limit=request.role.model_maximum.input_tokens,
                temperature=self.temperature,
                seed=int(request.seed) + offset,
            )
            inputs_total += int(inputs)
            outputs_total += int(outputs)
            calls += 1
            action = _select_action(str(output), self.episode.admissible)
            self.episode.step(action)
            taken.append(action)
        public = json.dumps(
            {
                "actions": taken,
                "observation": self.episode.last_observation,
                "episode_done": self.episode.done,
                "environment_steps_total": self.episode.steps,
                "admissible_commands": list(self.episode.admissible),
            },
            sort_keys=True,
        )
        return NodeExecutionResult(
            public,
            BudgetVector(
                input_tokens=inputs_total,
                output_tokens=outputs_total,
                model_calls=calls,
                agent_turns=1,
                wall_time_milliseconds=max(0, int((time.monotonic() - started) * 1000)),
            ),
            {"environment_step": {"actions": taken, "done": self.episode.done, "steps": self.episode.steps}},
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
            episode = _ALFWorldEpisode(request=request, mode=mode, max_steps=_max_episode_steps(), selection_seed=_index)
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

        # Every session registers a fresh single-game TextWorld env for the same
        # game file, so two sessions of one task start from identical, isolated
        # states; the paired candidate/control trials of the full method need
        # exactly that (verified by tests/experiments/test_curve_alfworld_real.py).
        bindings.append(TaskBinding(task, executor_id, session, replay_safe=True))
    return tuple(bindings)


def alfworld_bindings(*, policy: Any, config: Any) -> tuple[TaskBinding, ...]:
    return make_bindings(policy=policy, config=config)


def _count_from_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def task_factory(*, policy: Any, config: Any) -> tuple[TaskBinding, ...]:
    """ALFWorld-only EvoSteer task factory.

    Use with ``--task-factory skillev.experiments.curve_alfworld:task_factory``.
    Node calls go to the same frozen executor as the text benchmarks (the
    SGLang server when ``EVOSTEER_EXECUTOR_URL`` is set).  Environment:
    ``EVOSTEER_ALFWORLD_SPLIT`` (default ``train``), ``EVOSTEER_ALFWORLD_COUNT``
    (first N games of the sorted split; default all), ``ALFWORLD_MAX_STEPS``
    (default 50), ``ALFWORLD_STEPS_PER_NODE`` (default 20).
    """
    from skillev.experiments.curve_benchmarks import _executor_model

    return make_bindings(
        policy=_executor_model(policy),
        config=config,
        mode=os.environ.get("EVOSTEER_ALFWORLD_SPLIT", "train"),
        count=_count_from_env("EVOSTEER_ALFWORLD_COUNT"),
    )


__all__ = ["ALFWORLD_FAMILY", "alfworld_bindings", "make_bindings", "task_factory", "training_split_manifest"]
