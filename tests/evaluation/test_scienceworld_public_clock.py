"""The native clock is a public budget, not a task-progress or stopping oracle."""

import asyncio
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldStepResult
from skillev_private.evaluation.ood_scienceworld import OODScienceWorldEnvironment

from skillev.evaluation.scienceworld_commands import PERSISTENT_FOCUS_COMMAND_PROFILE
from skillev.evaluation.scienceworld_observation import (
    CLOCK_OBSERVATION_PROFILE,
    public_simulator_clock,
)

PUBLIC = {
    "current_look": "a public room",
    "inventory": "an instrument",
    "task_description": "a public goal",
}


@pytest.mark.parametrize("profile", ["public-state@1", CLOCK_OBSERVATION_PROFILE])
def test_clock_uses_same_reset_and_step_without_extra_action_or_private_feedback(profile):
    commands = []
    results = iter(
        [
            OfficialScienceWorldStepResult(
                "Time passes", 13, False, public_state=PUBLIC, native_moves=11
            ),
            OfficialScienceWorldStepResult(
                "No known action matches that input.",
                13,
                False,
                public_state=PUBLIC,
                native_moves=11,
            ),
            OfficialScienceWorldStepResult(
                "Native feedback",
                -100,
                True,
                public_state=PUBLIC,
                native_moves=12,
                private_diagnostics={"branch": "secret-failure-reason"},
            ),
        ]
    )
    native = SimpleNamespace(
        reset=lambda **kwargs: "Initial feedback",
        step=lambda command: commands.append(command) or next(results),
        task_description=PUBLIC["task_description"],
        public_state=PUBLIC,
        reset_score=0,
        native_moves=0,
    )
    env = OODScienceWorldEnvironment(
        native,
        "synthetic",
        0,
        0,
        200,
        observation_profile=profile,
        command_profile=PERSISTENT_FOCUS_COMMAND_PROFILE,
    )

    async def run():
        initial = await env.reset()
        assert commands == []
        assert (public_simulator_clock(0, 200) in initial.observation_text) == (
            profile == CLOCK_OBSERVATION_PROFILE
        )
        for command, moves in [("wait", 11), ("unknown command", 11), ("focus on vessel", 12)]:
            result = await env.step(command)
            assert PUBLIC["current_look"] in result.observation
            assert (public_simulator_clock(moves, 200) in result.observation) == (
                profile == CLOCK_OBSERVATION_PROFILE
            )
            assert "-100" not in result.observation
            assert "secret-failure-reason" not in result.observation
        assert commands == ["wait", "unknown command", "focus on vessel"]
        assert env.last_transition.native_diagnostics["branch"] == "secret-failure-reason"
        outcome = await env.outcome()
        assert outcome.native_final_score == -100
        assert outcome.simulator_moves == 12
        assert outcome.native_steps == 3
        assert outcome.terminal_reached
        assert not outcome.success

    asyncio.run(run())


@pytest.mark.parametrize(("moves", "terminal"), [(200, False), (201, True)])
def test_clock_does_not_rewrite_native_horizon_or_partial_score(moves, terminal):
    native = SimpleNamespace(
        step=lambda command: OfficialScienceWorldStepResult(
            "Native observation", 35, terminal, public_state=PUBLIC, native_moves=moves
        )
    )
    env = OODScienceWorldEnvironment(
        native, "synthetic", 0, 0, 200, observation_profile=CLOCK_OBSERVATION_PROFILE
    )
    result = asyncio.run(env.step("wait1"))
    assert result.terminal is terminal
    assert public_simulator_clock(moves, 200) in result.observation
    outcome = asyncio.run(env.outcome())
    assert outcome.native_final_score == 35
    assert outcome.simulator_moves == moves
    assert outcome.native_steps == 1
    assert outcome.terminated_by_horizon is terminal
    assert not outcome.success


def test_missing_clock_is_not_fabricated_as_zero_or_replaced_with_owner_calls():
    env = OODScienceWorldEnvironment(
        None, "synthetic", 0, 0, 200, observation_profile=CLOCK_OBSERVATION_PROFILE
    )
    env._public_state = PUBLIC
    env._steps = 8
    with pytest.raises(RuntimeError):
        env.public_observation("Native observation")
    env.observation_profile = "public-state@1"
    assert "simulator_clock" not in env.public_observation("Native observation")
