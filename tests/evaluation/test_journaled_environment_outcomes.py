"""Step-limit evidence comes from native acknowledgements without changing reward."""

import asyncio

import pytest

from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.journaled_environment import JournaledEnvironment
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_horizon_flag_uses_native_step_count_and_preserves_native_score(
    tmp_path, terminal, success
):
    class Environment:
        async def reset(self):
            return NativePublicState("Synthetic observation.", ("look",))

        async def step(self, action):
            return NativeEnvironmentStep(
                "Native observation.", terminal, float(success), success, True
            )

        async def outcome(self):
            return NativeEnvironmentOutcome(float(success), success, terminal)

        async def close(self):
            pass

    journal = CandidateJournal(tmp_path / "native.sqlite")
    scope = ("synthetic", "A2", "case")
    environment = JournaledEnvironment(
        Environment(), journal, scope, owner="owner", maximum_steps=1
    )

    async def run():
        await environment.reset()
        await environment.step("look")
        return await environment.outcome()

    outcome = asyncio.run(run())
    assert outcome.reward == float(success)
    assert outcome.success is success
    assert outcome.terminal_reached is terminal
    assert outcome.terminated_by_horizon is (not success and not terminal)
    stored = journal.traces(scope, "native-outcome", origin=EventOrigin.ENVIRONMENT)[0]
    assert stored["terminated_by_horizon"] == outcome.terminated_by_horizon
    journal.close()
