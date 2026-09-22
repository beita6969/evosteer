"""Real ALFWorld checks; skipped unless alfworld and its data are installed."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("alfworld")
pytest.importorskip("textworld")
if not os.environ.get("ALFWORLD_DATA"):
    pytest.skip("ALFWORLD_DATA is not set", allow_module_level=True)

from skillev.experiments import curve_alfworld as ca  # noqa: E402


def _episode(index: int) -> ca._ALFWorldEpisode:
    return ca._ALFWorldEpisode(request=SimpleNamespace(seed=0), mode="train", max_steps=50, selection_seed=index)


def test_official_train_split_is_available():
    files, config = ca._catalog(mode="train")
    assert len(files) > 1000  # the official train split has 3,553 text games
    assert config


def test_two_sessions_of_one_task_start_identically():
    first, second = _episode(3), _episode(3)
    try:
        assert first.game_file == second.game_file
        assert first.initial_observation == second.initial_observation
        assert first.admissible == second.admissible
        assert "Your task is to:" in first.initial_observation
    finally:
        first.close()
        second.close()


def test_an_admissible_command_advances_the_game():
    episode = _episode(0)
    try:
        command = next(c for c in episode.admissible if c.startswith("go to"))
        observation, won = episode.step(command)
        assert observation and not won and episode.steps == 1
        assert episode.admissible
    finally:
        episode.close()
