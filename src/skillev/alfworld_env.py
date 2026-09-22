"""Official ALFWorld text environment (TextWorld) for EvoSteer.

This replaces the RAGEN adapter that ``skillev.experiments.curve_alfworld`` used
to import.  It talks to the upstream ``alfworld`` package directly:

* ``AlfredTWEnv`` enumerates the official game files of a split
  (``train``, ``valid_seen``/``eval_in_distribution``,
  ``valid_unseen``/``eval_out_of_distribution``) from the upstream YAML config.
* Every ``reset(seed)`` registers ONE game (``game_files[seed % n]``) as a
  single-environment TextWorld gym env with the same wrappers ALFWorld uses
  (``AlfredDemangler`` + ``AlfredInfos``), so a task id always maps to exactly
  one official game.

Nothing here reads the PDDL goal or the expert plan: the model only ever sees
the observation text and the admissible commands.  ``won`` stays in ``info``
for the terminal evaluator.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

_SPLITS = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "eval_in_distribution": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
    "eval_out_of_distribution": "eval_out_of_distribution",
}
_WELCOME = "-= Welcome to TextWorld, ALFRED! =-"
# (config file, split, $ALFWORLD_DATA) -> sorted game files; scanning a split walks
# ~9k directories, so it is done once per process instead of once per episode.
_CATALOG: dict[tuple[str, str, str], tuple[str, ...]] = {}
_CATALOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class AlfredEnvConfig:
    config_file: str
    max_episode_steps: int = 50


def _read_config(path: str) -> dict[str, Any]:
    import yaml

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"ALFWorld config not found: {path!r}; set ALFWORLD_CONFIG_FILE or install alfworld"
        )
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _clean(observation: str) -> str:
    text = str(observation)
    if text.startswith(_WELCOME):
        text = text[len(_WELCOME):]
    return text.strip()


def _first(value: Any) -> Any:
    """Unbatch a batch_size=1 TextWorld value."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


class ALFWorldEnv:
    """One resettable single-game ALFWorld text episode."""

    def __init__(self, config: AlfredEnvConfig, mode: str = "train") -> None:
        if mode not in _SPLITS:
            raise ValueError(f"unknown ALFWorld split {mode!r}; use one of {sorted(_SPLITS)}")
        self.config = config
        self.mode = mode
        key = (str(config.config_file), mode, os.environ.get("ALFWORLD_DATA", ""))
        with _CATALOG_LOCK:
            if key not in _CATALOG:
                from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

                alfred = AlfredTWEnv(_read_config(config.config_file), train_eval=_SPLITS[mode])
                # Sorted so that task index -> game file is stable across processes.
                _CATALOG[key] = tuple(sorted(str(item) for item in alfred.game_files))
        self.game_files = _CATALOG[key]
        self.current_game_file = ""
        self._admissible_commands: tuple[str, ...] = ()
        self._env: Any = None

    def _make_single_game_env(self, game_file: str) -> Any:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=int(self.config.max_episode_steps),
            wrappers=[AlfredDemangler(), AlfredInfos],
        )
        return textworld.gym.make(env_id)

    def reset(self, seed: int = 0) -> str:
        if not self.game_files:
            raise RuntimeError("ALFWorld split has no game files; run scripts/provision_alfworld.sh")
        self.close_episode()
        self.current_game_file = self.game_files[int(seed) % len(self.game_files)]
        self._env = self._make_single_game_env(self.current_game_file)
        observation, infos = self._env.reset()
        self._admissible_commands = tuple(str(x) for x in _first(infos.get("admissible_commands", ())) or ())
        return _clean(_first(observation))

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        observation, scores, dones, infos = self._env.step([str(action)])
        admissible = tuple(str(x) for x in _first(infos.get("admissible_commands", ())) or ())
        self._admissible_commands = admissible
        won = bool(_first(infos.get("won", False)))
        info = {"available_actions": admissible, "won": won, "gamefile": self.current_game_file}
        return _clean(_first(observation)), float(_first(scores) or 0.0), bool(_first(dones)), info

    def close_episode(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None

    def close(self) -> None:
        self.close_episode()


__all__ = ["ALFWorldEnv", "AlfredEnvConfig"]
