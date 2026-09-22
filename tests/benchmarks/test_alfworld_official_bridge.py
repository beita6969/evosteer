from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast

import pytest
from skillev_private.benchmarks import (
    OfficialALFWorldEpisodeFactory,
    OfficialALFWorldResetResult,
    OfficialALFWorldStepResult,
    OfficialALFWorldTask,
    PrivateALFWorldCase,
    PrivateALFWorldSessionFactory,
)

from skillev.benchmarks import (
    ALFWORLD_RESOURCE_ID,
    ALFWorldEnvironment,
    ALFWorldPublicItem,
)
from skillev.contracts import JsonValue, canonical_json, stable_hash
from skillev.rollout import (
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, StructuredAction

PRIVATE_CANARY = "PRIVATE-OFFICIAL-ALFWORLD-GOAL-CANARY"
INITIAL_OBSERVATION = "You are in a public kitchen."
INITIAL_COMMANDS = ("look", "take apple from counter")
TERMINAL_ACTION = "put apple in fridge"


def _item(*, max_steps: int = 3) -> ALFWorldPublicItem:
    return ALFWorldPublicItem(
        dataset_revision="alfworld-official-fixture@1",
        environment_snapshot_id="official-textworld-fixture@1",
        split="train",
        task_id="alfworld/official-task-1",
        task_family="alfworld/pick-and-place",
        query="Put the apple in the fridge.",
        public_context={
            "admissible_commands": list(INITIAL_COMMANDS),
            "initial_observation": INITIAL_OBSERVATION,
        },
        seed=1729,
        max_steps=max_steps,
    )


def _private_task(item: ALFWorldPublicItem) -> OfficialALFWorldTask:
    return OfficialALFWorldTask(
        task_id=item.task_id,
        environment_id=item.environment_id,
        game_id="official-game-0001",
        seed=item.seed,
        max_steps=item.max_steps,
        payload={"goal_predicate": PRIVATE_CANARY},
    )


def _case(*, max_steps: int = 3) -> PrivateALFWorldCase:
    item = _item(max_steps=max_steps)
    return PrivateALFWorldCase(item, _private_task(item))


def _action(command: str) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name="act",
        arguments={"command": command},
        resource_id=ALFWORLD_RESOURCE_ID,
    )


def _request(item: ALFWorldPublicItem, submission: JsonValue) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="official-alfworld-trajectory",
        task_id=item.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(submission),
        public_transcript_hash=stable_hash({"public": "official ALFWorld transcript"}),
    )


@dataclass(slots=True)
class _ScriptedOfficialEnv:
    task: OfficialALFWorldTask
    instruction_text: str
    reset_observation: str = INITIAL_OBSERVATION
    reset_commands: tuple[str, ...] = INITIAL_COMMANDS
    terminal_success: bool = True
    drift: bool = False
    fail_reset: bool = False
    fail_step: bool = False
    game_id_override: str | None = None
    seed_override: int | None = None
    max_steps_override: int | None = None
    reset_calls: list[int] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    close_calls: int = 0

    @property
    def game_id(self) -> str:
        return self.game_id_override or self.task.game_id

    @property
    def seed(self) -> int:
        return self.task.seed if self.seed_override is None else self.seed_override

    @property
    def max_steps(self) -> int:
        return self.task.max_steps if self.max_steps_override is None else self.max_steps_override

    def reset(self, seed: int) -> OfficialALFWorldResetResult:
        if self.fail_reset:
            raise AssertionError(PRIVATE_CANARY)
        self.reset_calls.append(seed)
        self.actions = []
        return OfficialALFWorldResetResult(
            observation_text=self.reset_observation,
            instruction_text=self.instruction_text,
            admissible_commands=self.reset_commands,
        )

    def step(self, action: str) -> OfficialALFWorldStepResult:
        if self.fail_step:
            raise AssertionError(PRIVATE_CANARY)
        self.actions.append(action)
        terminal = action == TERMINAL_ACTION
        observation = f"Official observation after {action}."
        if self.drift:
            observation = f"DRIFTED {observation}"
        if terminal:
            commands: tuple[str, ...] = ()
        elif action == "look":
            commands = ("take apple from counter",)
        else:
            commands = (TERMINAL_ACTION,)
        return OfficialALFWorldStepResult(
            observation_text=observation,
            admissible_commands=commands,
            terminal=terminal,
            success=self.terminal_success if terminal else None,
        )

    async def close(self) -> None:
        self.close_calls += 1


@dataclass(slots=True)
class _ScriptedOfficialFactory:
    public: ALFWorldPublicItem
    terminal_success: bool = True
    drift_indices: set[int] = field(default_factory=set)
    fail_reset_indices: set[int] = field(default_factory=set)
    fail_step_indices: set[int] = field(default_factory=set)
    instruction_override: str | None = None
    reset_observation_override: str | None = None
    reset_commands_override: tuple[str, ...] | None = None
    game_id_override: str | None = None
    seed_override: int | None = None
    max_steps_override: int | None = None
    envs: list[_ScriptedOfficialEnv] = field(default_factory=list)

    def create(self, task: OfficialALFWorldTask) -> _ScriptedOfficialEnv:
        index = len(self.envs) + 1
        env = _ScriptedOfficialEnv(
            task=task,
            instruction_text=self.instruction_override or self.public.query,
            reset_observation=self.reset_observation_override or INITIAL_OBSERVATION,
            reset_commands=self.reset_commands_override or INITIAL_COMMANDS,
            terminal_success=self.terminal_success,
            drift=index in self.drift_indices,
            fail_reset=index in self.fail_reset_indices,
            fail_step=index in self.fail_step_indices,
            game_id_override=self.game_id_override,
            seed_override=self.seed_override,
            max_steps_override=self.max_steps_override,
        )
        self.envs.append(env)
        return env


def _sessions(
    case: PrivateALFWorldCase,
    official: _ScriptedOfficialFactory,
) -> PrivateALFWorldSessionFactory:
    return PrivateALFWorldSessionFactory(
        (case,),
        OfficialALFWorldEpisodeFactory(official),
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_official_actions_public_projection_and_terminal_signal_are_separated() -> None:
    case = _case()
    official = _ScriptedOfficialFactory(case.public)
    bundle = _sessions(case, official).create(case.public.to_rollout_task())
    environment = cast(ALFWorldEnvironment, bundle.environment)

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(bundle.evaluator.evaluate(_request(case.public, None)))

    observations = []
    for index, command in enumerate(("look", "take apple from counter", TERMINAL_ACTION), 1):
        observations.append(
            asyncio.run(
                environment.execute(
                    _action(command),
                    step_index=index,
                )
            )
        )
    reward = asyncio.run(
        bundle.evaluator.evaluate(_request(case.public, observations[-1].terminal_submission))
    )

    assert official.envs[0].reset_calls == [case.public.seed]
    assert official.envs[0].actions == ["look", "take apple from counter", TERMINAL_ACTION]
    assert all(observation.observation_status == "success" for observation in observations)
    assert all(observation.budget_usage.tool_calls == 1 for observation in observations)
    assert all(observation.budget_usage.wall_time_milliseconds >= 0 for observation in observations)
    public = cast(dict[str, JsonValue], observations[0].public_value)
    assert set(public) == {"admissible_commands", "terminal", "text"}
    assert observations[-1].terminal
    assert reward.value == 1.0
    assert reward.success

    public_bundle = {
        "observations": [observation.to_value() for observation in observations],
        "reward": reward.to_value(),
        "task": case.public.to_rollout_task().to_value(),
    }
    assert PRIVATE_CANARY not in canonical_json(public_bundle)
    assert {"goal_predicate", "reward", "won"}.isdisjoint(
        _all_keys([observation.to_value() for observation in observations])
    )


def test_official_false_terminal_signal_produces_binary_failure() -> None:
    case = _case()
    official = _ScriptedOfficialFactory(case.public, terminal_success=False)
    bundle = _sessions(case, official).create(case.public.to_rollout_task())
    environment = cast(ALFWorldEnvironment, bundle.environment)

    terminal = asyncio.run(
        environment.execute(
            _action(TERMINAL_ACTION),
            step_index=1,
        )
    )
    reward = asyncio.run(
        bundle.evaluator.evaluate(_request(case.public, terminal.terminal_submission))
    )

    assert terminal.terminal
    assert reward.value == 0.0
    assert not reward.success


def test_official_session_exposes_episode_cleanup() -> None:
    case = _case()
    official = _ScriptedOfficialFactory(case.public)
    bundle = _sessions(case, official).create(case.public.to_rollout_task())

    assert bundle.cleanup is not None
    asyncio.run(bundle.cleanup())
    assert official.envs[0].close_calls == 1


@pytest.mark.parametrize(
    ("factory_kwargs", "message"),
    [
        ({"instruction_override": "another instruction"}, "instruction differs"),
        ({"reset_observation_override": "another room"}, "reset differs"),
        ({"reset_commands_override": ("inventory",)}, "reset differs"),
        ({"game_id_override": "another-game"}, "another game"),
        ({"seed_override": 1730}, "another seed"),
        ({"max_steps_override": 4}, "another step limit"),
    ],
)
def test_official_reset_and_episode_identity_are_pinned(
    factory_kwargs: dict[str, object],
    message: str,
) -> None:
    case = _case()
    official = _ScriptedOfficialFactory(case.public, **factory_kwargs)

    with pytest.raises(ValueError, match=message):
        OfficialALFWorldEpisodeFactory(official).create(case)


@pytest.mark.parametrize("failure", ["reset", "step"])
def test_unknown_official_exception_propagates_without_observation_or_reward(
    failure: str,
) -> None:
    case = _case()
    official = _ScriptedOfficialFactory(
        case.public,
        fail_reset_indices={1} if failure == "reset" else set(),
        fail_step_indices={1} if failure == "step" else set(),
    )
    sessions = _sessions(case, official)

    if failure == "reset":
        with pytest.raises(AssertionError, match=PRIVATE_CANARY):
            sessions.create(case.public.to_rollout_task())
        return

    bundle = sessions.create(case.public.to_rollout_task())
    environment = cast(ALFWorldEnvironment, bundle.environment)
    with pytest.raises(AssertionError, match=PRIVATE_CANARY):
        asyncio.run(
            environment.execute(
                _action("look"),
                step_index=1,
            )
        )
    assert official.envs[0].actions == []
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(bundle.evaluator.evaluate(_request(case.public, None)))


def test_official_step_limit_mismatch_is_rejected_not_fabricated_as_failure() -> None:
    case = _case(max_steps=1)
    official = _ScriptedOfficialFactory(case.public)
    bundle = _sessions(case, official).create(case.public.to_rollout_task())
    environment = cast(ALFWorldEnvironment, bundle.environment)

    with pytest.raises(ValueError, match="did not terminate"):
        asyncio.run(
            environment.execute(
                _action("look"),
                step_index=1,
            )
        )
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(bundle.evaluator.evaluate(_request(case.public, None)))


def test_official_step_result_requires_terminal_signal_only_at_terminal() -> None:
    with pytest.raises(TypeError):
        OfficialALFWorldStepResult(
            observation_text="terminal observation",
            admissible_commands=(),
            terminal=True,
            success=None,
        )
    with pytest.raises(ValueError):
        OfficialALFWorldStepResult(
            observation_text="continuing observation",
            admissible_commands=("look",),
            terminal=False,
            success=False,
        )
