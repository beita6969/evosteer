from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from skillev_private.benchmarks import (
    ALFWorldOutcomeUnavailableError,
    PrivateALFWorldCase,
    PrivateALFWorldEpisodeSession,
    PrivateALFWorldTerminalEvaluator,
    PrivateScienceWorldCase,
    PrivateScienceWorldEpisodeSession,
    PrivateScienceWorldSessionFactory,
    PrivateScienceWorldTerminalEvaluator,
    ScienceWorldOutcomeUnavailableError,
)

from skillev.benchmarks import (
    ALFWORLD_RESOURCE_ID,
    SCIENCEWORLD_RESOURCE_ID,
    ALFWorldCommand,
    ALFWorldEnvironment,
    ALFWorldPublicItem,
    ALFWorldPublicStep,
    OrderedALFWorldTaskProvider,
    OrderedScienceWorldTaskProvider,
    ScienceWorldCommand,
    ScienceWorldEnvironment,
    ScienceWorldPublicItem,
    ScienceWorldPublicStep,
)
from skillev.contracts import JsonValue, canonical_json, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, BudgetVector, StructuredAction

PRIVATE_CANARY = "PRIVATE-EMBODIED-GOAL-CANARY"


class SyntheticSimulatorError(RuntimeError):
    pass


@dataclass(slots=True)
class _SyntheticEpisode:
    item: ALFWorldPublicItem | ScienceWorldPublicItem
    private_canary: str = field(repr=False)
    commands: list[str] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return self.item.task_id

    @property
    def environment_id(self) -> str:
        return self.item.environment_id

    @property
    def seed(self) -> int:
        return self.item.seed

    @property
    def max_steps(self) -> int:
        return self.item.max_steps

    @property
    def success(self) -> bool:
        return "finish" in self.commands

    @property
    def science_score(self) -> float:
        if self.success:
            return 100.0
        return 25.0 if "inspect" in self.commands else 0.0

    async def execute(
        self,
        command: ALFWorldCommand | ScienceWorldCommand,
        *,
        step_index: int,
    ) -> ALFWorldPublicStep | ScienceWorldPublicStep:
        if command.text == "explode":
            raise SyntheticSimulatorError("private simulator failed")
        if command.text == "bad-budget":
            return ALFWorldPublicStep(
                public_observation={"message": "invalid measured budget"},
                terminal=False,
                budget_usage=BudgetVector(),
            )
        self.commands.append(command.text)
        return ALFWorldPublicStep(
            public_observation={
                "command_count": len(self.commands),
                "message": f"public deterministic observation seed={self.seed}",
                "step_index": step_index,
            },
            terminal=self.success,
            budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=3),
        )


@dataclass(frozen=True, slots=True)
class _SyntheticALFOutcome:
    episode: _SyntheticEpisode
    error: Exception | None = field(default=None, repr=False)

    @property
    def task_id(self) -> str:
        return self.episode.task_id

    @property
    def environment_id(self) -> str:
        return self.episode.environment_id

    @property
    def seed(self) -> int:
        return self.episode.seed

    def final_success(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.episode.success


@dataclass(frozen=True, slots=True)
class _SyntheticScienceOutcome:
    episode: _SyntheticEpisode
    error: Exception | None = field(default=None, repr=False)

    @property
    def task_id(self) -> str:
        return self.episode.task_id

    @property
    def environment_id(self) -> str:
        return self.episode.environment_id

    @property
    def seed(self) -> int:
        return self.episode.seed

    def final_score(self) -> float:
        if self.error is not None:
            raise self.error
        return self.episode.science_score


@dataclass(slots=True)
class _SyntheticALFFactory:
    episodes: list[_SyntheticEpisode] = field(default_factory=list)

    def create(self, case: PrivateALFWorldCase) -> PrivateALFWorldEpisodeSession:
        episode = _SyntheticEpisode(case.public, str(case.private_task))
        self.episodes.append(episode)
        return PrivateALFWorldEpisodeSession(episode, _SyntheticALFOutcome(episode))


@dataclass(slots=True)
class _SyntheticScienceFactory:
    episodes: list[_SyntheticEpisode] = field(default_factory=list)

    def create(self, case: PrivateScienceWorldCase) -> PrivateScienceWorldEpisodeSession:
        episode = _SyntheticEpisode(case.public, str(case.private_task))
        self.episodes.append(episode)
        return PrivateScienceWorldEpisodeSession(episode, _SyntheticScienceOutcome(episode))


def _alf_item(*, max_steps: int = 3) -> ALFWorldPublicItem:
    return ALFWorldPublicItem(
        dataset_revision="alfworld-fixture@1",
        environment_snapshot_id="textworld-fixture@1",
        split="train",
        task_id="alfworld/synthetic-1",
        task_family="alfworld/pick-and-place",
        query="Put the public object in the public receptacle.",
        public_context={"initial_observation": "A public room description."},
        seed=1729,
        max_steps=max_steps,
    )


def _science_item(*, max_steps: int = 3) -> ScienceWorldPublicItem:
    return ScienceWorldPublicItem(
        dataset_revision="scienceworld-fixture@1",
        environment_snapshot_id="jvm-fixture@1",
        split="test",
        task_id="scienceworld/synthetic-1",
        task_family="scienceworld/chemistry",
        query="Complete the public synthetic science experiment.",
        public_context={"initial_observation": "A public laboratory description."},
        seed=1729,
        max_steps=max_steps,
    )


def _action(resource_id: str, command: str) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name="act",
        arguments={"command": command},
        resource_id=resource_id,
    )


def _request(task_id: str, submission: JsonValue) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="synthetic-trajectory",
        task_id=task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(submission),
        public_transcript_hash=stable_hash({"public": "transcript"}),
    )


def _no_submission_request(task_id: str) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="synthetic-no-submission",
        task_id=task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )


def test_embodied_no_submission_is_zero_without_reading_private_outcome() -> None:
    alf_item = _alf_item()
    science_item = _science_item()
    alf_episode = _SyntheticEpisode(alf_item, PRIVATE_CANARY)
    science_episode = _SyntheticEpisode(science_item, PRIVATE_CANARY)
    unexpected = AssertionError("private outcome must not be read")
    evaluators = (
        PrivateALFWorldTerminalEvaluator(
            alf_item,
            _SyntheticALFOutcome(alf_episode, unexpected),
        ),
        PrivateScienceWorldTerminalEvaluator(
            science_item,
            _SyntheticScienceOutcome(science_episode, unexpected),
        ),
    )

    rewards = tuple(
        asyncio.run(evaluator.evaluate(_no_submission_request(item.task_id)))
        for evaluator, item in zip(evaluators, (alf_item, science_item), strict=True)
    )

    assert tuple(reward.value for reward in rewards) == (0.0, 0.0)
    assert all(reward.success is False for reward in rewards)


def test_scienceworld_normalizes_native_score_and_applies_step_limit() -> None:
    item = _science_item(max_steps=1)
    case = PrivateScienceWorldCase(item, {"rubric": PRIVATE_CANARY})
    factory = _SyntheticScienceFactory()
    session = PrivateScienceWorldSessionFactory((case,), factory).create(item.to_rollout_task())

    observation = asyncio.run(
        session.environment.execute(
            _action(SCIENCEWORLD_RESOURCE_ID, "inspect"),
            step_index=1,
        )
    )
    reward = asyncio.run(
        session.evaluator.evaluate(_request(item.task_id, observation.terminal_submission))
    )

    assert observation.terminal
    assert observation.terminal_submission == {
        "benchmark_id": "scienceworld",
        "episode_terminal": True,
        "reason": "step-limit",
        "task_id": item.task_id,
    }
    assert reward.value == 0.25
    assert not reward.success
    assert reward.native_payload["native_score"] == 25.0
    assert reward.native_payload["public_metrics"] == {"scienceworld-native-score": 25.0}
    assert PRIVATE_CANARY not in canonical_json(
        {
            "observation": observation.to_value(),
            "reward_payload": reward.native_payload,
            "task": item.to_rollout_task().to_value(),
        }
    )
    with pytest.raises(ValueError):
        asyncio.run(
            session.environment.execute(
                _action(SCIENCEWORLD_RESOURCE_ID, "finish"),
                step_index=2,
            )
        )


def test_embodied_environment_accepts_a_valid_tool_after_an_invalid_rollout_turn() -> None:
    item = _alf_item()
    episode = _SyntheticEpisode(item, PRIVATE_CANARY)
    environment = ALFWorldEnvironment(item, episode)

    observation = asyncio.run(
        environment.execute(
            _action(ALFWORLD_RESOURCE_ID, "inspect"),
            step_index=2,
        )
    )

    assert observation.observation_status == "success"
    assert episode.commands == ["inspect"]


@pytest.mark.parametrize(
    ("item", "environment_type", "resource_id"),
    [
        (_alf_item(), ALFWorldEnvironment, ALFWORLD_RESOURCE_ID),
        (_science_item(), ScienceWorldEnvironment, SCIENCEWORLD_RESOURCE_ID),
    ],
)
def test_embodied_backend_failures_and_unmeasured_usage_fail_fast(
    item: ALFWorldPublicItem | ScienceWorldPublicItem,
    environment_type: type[ALFWorldEnvironment] | type[ScienceWorldEnvironment],
    resource_id: str,
) -> None:
    episode = _SyntheticEpisode(item, PRIVATE_CANARY)
    environment = environment_type(item, episode)

    with pytest.raises(SyntheticSimulatorError):
        asyncio.run(
            environment.execute(
                _action(resource_id, "explode"),
                step_index=1,
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            environment.execute(
                _action(resource_id, "bad-budget"),
                step_index=1,
            )
        )


@pytest.mark.parametrize(
    ("item", "provider_type"),
    [
        (_alf_item(), OrderedALFWorldTaskProvider),
        (_science_item(), OrderedScienceWorldTaskProvider),
    ],
)
def test_embodied_task_provider_and_seed_identity_are_deterministic(
    item: ALFWorldPublicItem | ScienceWorldPublicItem,
    provider_type: type[OrderedALFWorldTaskProvider] | type[OrderedScienceWorldTaskProvider],
) -> None:
    provider = provider_type((item,))

    assert provider.next_task() == item.to_rollout_task()
    assert f"seed:{item.seed}" in item.environment_id
    with pytest.raises(RuntimeError):
        provider.next_task()


def test_unsupported_agent_action_is_measured_data_not_backend_failure() -> None:
    item = _alf_item()
    episode = _SyntheticEpisode(item, PRIVATE_CANARY)
    environment = ALFWorldEnvironment(item, episode)
    unsupported = StructuredAction(
        kind=ActionKind.TOOL,
        name="unknown",
        arguments={},
        resource_id=ALFWORLD_RESOURCE_ID,
    )

    observation = asyncio.run(environment.execute(unsupported, step_index=1))

    assert observation.observation_status == "tool_error"
    assert observation.budget_usage.tool_calls == 1
    assert episode.commands == []


def test_environment_rejects_backend_seed_or_limit_mismatch() -> None:
    item = _science_item()
    wrong_seed = ScienceWorldPublicItem(
        dataset_revision=item.dataset_revision,
        environment_snapshot_id=item.environment_snapshot_id,
        split=item.split,
        task_id=item.task_id,
        task_family=item.task_family,
        query=item.query,
        public_context=item.public_context,
        seed=item.seed + 1,
        max_steps=item.max_steps,
    )
    episode = _SyntheticEpisode(wrong_seed, PRIVATE_CANARY)

    with pytest.raises(ValueError):
        ScienceWorldEnvironment(item, episode)


@pytest.mark.parametrize("benchmark", ["alfworld", "scienceworld"])
def test_declared_outcome_unavailable_error_becomes_rejection(benchmark: str) -> None:
    if benchmark == "alfworld":
        item = _alf_item()
        episode = _SyntheticEpisode(item, PRIVATE_CANARY)
        evaluator = PrivateALFWorldTerminalEvaluator(
            item,
            _SyntheticALFOutcome(
                episode,
                ALFWorldOutcomeUnavailableError(PRIVATE_CANARY),
            ),
        )
    else:
        item = _science_item()
        episode = _SyntheticEpisode(item, PRIVATE_CANARY)
        evaluator = PrivateScienceWorldTerminalEvaluator(
            item,
            _SyntheticScienceOutcome(
                episode,
                ScienceWorldOutcomeUnavailableError(PRIVATE_CANARY),
            ),
        )

    with pytest.raises(TerminalEvaluatorError) as captured:
        asyncio.run(evaluator.evaluate(_request(item.task_id, None)))
    assert PRIVATE_CANARY not in str(captured.value)


@pytest.mark.parametrize("benchmark", ["alfworld", "scienceworld"])
def test_unknown_outcome_assertion_propagates_without_reward(benchmark: str) -> None:
    if benchmark == "alfworld":
        item = _alf_item()
        episode = _SyntheticEpisode(item, PRIVATE_CANARY)
        evaluator = PrivateALFWorldTerminalEvaluator(
            item,
            _SyntheticALFOutcome(episode, AssertionError(PRIVATE_CANARY)),
        )
    else:
        item = _science_item()
        episode = _SyntheticEpisode(item, PRIVATE_CANARY)
        evaluator = PrivateScienceWorldTerminalEvaluator(
            item,
            _SyntheticScienceOutcome(episode, AssertionError(PRIVATE_CANARY)),
        )

    with pytest.raises(AssertionError, match=PRIVATE_CANARY):
        asyncio.run(evaluator.evaluate(_request(item.task_id, None)))
