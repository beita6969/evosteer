"""ALFWorld adapter tests that run without the alfworld package (fake env)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from skillev.experiments import curve_alfworld as ca


class FakeConfig:
    def __init__(self, config_file: str = "fake.yaml", max_episode_steps: int = 50) -> None:
        self.config_file = config_file
        self.max_episode_steps = max_episode_steps


class FakeEnv:
    """Two-step game: 'go to desk 1' then 'take key 1 from desk 1' wins."""

    game_files = ("games/b/game.tw-pddl", "games/a/game.tw-pddl")

    def __init__(self, config: FakeConfig, mode: str = "train") -> None:
        self.config = config
        self.mode = mode
        self.closed = False
        self._admissible_commands: tuple[str, ...] = ()
        self.current_game_file = ""
        self._progress = 0

    def reset(self, seed: int = 0) -> str:
        self.current_game_file = self.game_files[seed % len(self.game_files)]
        self._progress = 0
        self._admissible_commands = ("look", "go to desk 1", "go to bed 1")
        return "You are in a room. Your task is to: take the key."

    def step(self, action: str):
        won = False
        if self._progress == 0 and action == "go to desk 1":
            self._progress = 1
            self._admissible_commands = ("look", "take key 1 from desk 1")
            obs = "On the desk 1, you see a key 1."
        elif self._progress == 1 and action == "take key 1 from desk 1":
            won = True
            obs = "You pick up the key 1 from the desk 1."
        else:
            obs = "Nothing happens."
        return obs, 0.0, won, {"available_actions": self._admissible_commands, "won": won}

    def close(self) -> None:
        self.closed = True


class ScriptedPolicy:
    """Frozen-text stand-in that answers with a scripted command sequence."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def frozen_text(self, prompt, *, max_new_tokens, input_limit=None, temperature, seed):
        self.prompts.append(prompt)
        assert max_new_tokens <= 64
        reply = self.replies.pop(0) if self.replies else "look"
        return reply, 100, 5


@pytest.fixture(autouse=True)
def fake_alfworld(monkeypatch):
    monkeypatch.setattr(ca, "_load_alfworld_env", lambda: (FakeEnv, FakeConfig))
    monkeypatch.setattr(ca, "_configured_path", lambda: "fake.yaml")
    monkeypatch.setenv("ALFWORLD_STEPS_PER_NODE", "5")


def _request(seed: int = 7, model_calls: int = 21):
    role = SimpleNamespace(instruction="act", model_maximum=SimpleNamespace(
        output_tokens=8192, input_tokens=20000, model_calls=model_calls, wall_time_milliseconds=600000))
    return SimpleNamespace(role=role, task_prompt="Complete the task.", messages=(), previous_output=None, seed=seed)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("go to desk 1", "go to desk 1"),
        ("Thought: the key may be on the desk.\nAction: go to desk 1", "go to desk 1"),
        ("I will GO TO DESK 1.", "go to desk 1"),
        ("something unrelated", "look"),
    ],
)
def test_select_action_never_invents_commands(text, expected):
    assert ca._select_action(text, ("look", "go to desk 1", "go to bed 1")) == expected


def test_select_action_prefers_longest_contained_command():
    admissible = ("take key 1", "take key 1 from desk 1")
    assert ca._select_action("take key 1 from desk 1 now", admissible) == "take key 1 from desk 1"


def test_one_node_takes_multiple_environment_steps_until_won():
    episode = ca._ALFWorldEpisode(request=SimpleNamespace(seed=0), mode="train", max_steps=50, selection_seed=0)
    policy = ScriptedPolicy(["go to desk 1", "take key 1 from desk 1", "look"])
    result = asyncio.run(ca.ALFWorldTextExecutor(policy, episode).execute(_request()))
    assert episode.won and episode.done
    assert episode.steps == 2  # stopped as soon as the game was won
    assert result.usage.model_calls == 2
    assert result.usage.input_tokens == 200 and result.usage.output_tokens == 10
    assert "Your task is to: take the key." in policy.prompts[0]


def test_later_node_continues_the_same_episode():
    episode = ca._ALFWorldEpisode(request=SimpleNamespace(seed=0), mode="train", max_steps=50, selection_seed=0)
    executor = ca.ALFWorldTextExecutor(ScriptedPolicy(["go to desk 1"] + ["look"] * 4), episode)
    asyncio.run(executor.execute(_request()))
    assert not episode.won and episode.steps == 5
    executor.policy = ScriptedPolicy(["take key 1 from desk 1"])
    asyncio.run(executor.execute(_request(seed=8)))
    assert episode.won


def test_node_after_episode_end_is_a_real_report_call():
    episode = ca._ALFWorldEpisode(request=SimpleNamespace(seed=0), mode="train", max_steps=50, selection_seed=0)
    executor = ca.ALFWorldTextExecutor(ScriptedPolicy(["go to desk 1", "take key 1 from desk 1"]), episode)
    asyncio.run(executor.execute(_request()))
    assert episode.won
    result = asyncio.run(executor.execute(_request(seed=9)))
    assert result.usage.model_calls == 1 and episode.steps == 2  # no environment step after the end


def test_episode_step_limit_is_enforced():
    episode = ca._ALFWorldEpisode(request=SimpleNamespace(seed=0), mode="train", max_steps=3, selection_seed=0)
    asyncio.run(ca.ALFWorldTextExecutor(ScriptedPolicy(["look"] * 10), episode).execute(_request()))
    assert episode.done and not episode.won and episode.steps == 3


def test_bindings_map_task_index_to_one_game():
    bindings = ca.make_bindings(policy=ScriptedPolicy([]), config=None, mode="train", count=2)
    assert [b.task.task_id for b in bindings] == ["alfworld/train/000000", "alfworld/train/000001"]
    assert all(b.task.family == ca.ALFWORLD_FAMILY for b in bindings)
    assert all("game" not in b.task.prompt for b in bindings)  # no game path or goal leaks into the task


def test_node_never_exceeds_the_role_reservation():
    episode = ca._ALFWorldEpisode(request=SimpleNamespace(seed=0), mode="train", max_steps=50, selection_seed=0)
    result = asyncio.run(ca.ALFWorldTextExecutor(ScriptedPolicy(["look"] * 10), episode).execute(_request(model_calls=3)))
    assert result.usage.model_calls == 3 and episode.steps == 3
