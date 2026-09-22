from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import cast

import pytest
from skillev_private.benchmarks import (
    PrivateWebShopCase,
    PrivateWebShopEpisodeSession,
    PrivateWebShopSessionFactory,
    PrivateWebShopTerminalEvaluator,
    WebShopRewardUnavailableError,
)

from skillev.benchmarks import (
    WEBSHOP_RESOURCE_ID,
    OrderedWebShopTaskProvider,
    WebShopCommand,
    WebShopCommandKind,
    WebShopEnvironment,
    WebShopPublicItem,
    WebShopPublicStep,
)
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, BudgetVector, StructuredAction

PRIVATE_GOAL_CANARY = "PRIVATE-WEBSHOP-GOAL-CANARY"


def _item(*, task_id: str = "webshop-task-001") -> WebShopPublicItem:
    return WebShopPublicItem(
        dataset_revision="webshop-fixture@1",
        environment_snapshot_id="products-index-fixture@1",
        split="train",
        task_id=task_id,
        task_family="webshop/home-office",
        query="Find and purchase the requested public shopping item.",
        public_context={"observation_format": "structured-page"},
    )


def _action(name: str, arguments: JsonValue) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=name,
        arguments=normalize_json(arguments),
        resource_id=WEBSHOP_RESOURCE_ID,
    )


def _request(item: WebShopPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-webshop-001",
        task_id=item.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue({"webshop_episode_terminal": True}),
        public_transcript_hash=stable_hash({"public": "webshop-transcript"}),
    )


def _no_submission_request(item: WebShopPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-webshop-no-submission",
        task_id=item.task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )


def test_no_submission_scores_zero_without_reading_webshop_reward() -> None:
    item = _item()
    view = _RewardView(item.task_id, item.environment_id, reward=0.75)
    evaluator = PrivateWebShopTerminalEvaluator(item, view)

    reward = asyncio.run(evaluator.evaluate(_no_submission_request(item)))

    assert reward.value == 0.0
    assert reward.success is False


@dataclass(slots=True)
class _RewardView:
    task_id: str
    environment_id: str
    reward: float = 0.0

    def final_reward(self) -> float:
        return self.reward


class _ScriptedEpisode:
    def __init__(
        self,
        case: PrivateWebShopCase,
        reward_view: _RewardView,
        *,
        fail_execute: bool,
        reported_tool_calls: int,
    ) -> None:
        goal = cast(dict[str, object], case.private_goal)
        self.task_id = case.public.task_id
        self.environment_id = case.public.environment_id
        self.reward_view = reward_view
        raw_purchase_reward = goal["purchase_reward"]
        if isinstance(raw_purchase_reward, bool) or not isinstance(
            raw_purchase_reward, int | float
        ):
            raise TypeError("scripted purchase reward must be numeric")
        self.purchase_reward = float(raw_purchase_reward)
        self.fail_execute = fail_execute
        self.reported_tool_calls = reported_tool_calls
        self.commands: list[WebShopCommand] = []
        self.execute_calls = 0
        self.cursor = 0
        self.purchased = False

    async def execute(
        self,
        command: WebShopCommand,
        *,
        step_index: int,
    ) -> WebShopPublicStep:
        self.execute_calls += 1
        if self.fail_execute:
            raise RuntimeError("official WebShop service unavailable")
        self.commands.append(command)
        self.cursor = step_index
        self.purchased = command.kind is WebShopCommandKind.PURCHASE
        if self.purchased:
            self.reward_view.reward = self.purchase_reward
        return WebShopPublicStep(
            public_observation={
                "command": command.kind.value,
                "page": "purchase-confirmation" if self.purchased else "catalog",
            },
            terminal=self.purchased,
            budget_usage=BudgetVector(
                tool_calls=self.reported_tool_calls,
                wall_time_milliseconds=7,
            ),
        )


@dataclass(slots=True)
class _EpisodeFactory:
    fail_execute: bool = False
    reported_tool_calls: int = 1
    sessions: list[PrivateWebShopEpisodeSession] | None = None

    def __post_init__(self) -> None:
        self.sessions = []

    def create(self, case: PrivateWebShopCase) -> PrivateWebShopEpisodeSession:
        reward = _RewardView(case.public.task_id, case.public.environment_id)
        episode = _ScriptedEpisode(
            case,
            reward,
            fail_execute=self.fail_execute,
            reported_tool_calls=self.reported_tool_calls,
        )
        session = PrivateWebShopEpisodeSession(episode=episode, reward_view=reward)
        cast(list[PrivateWebShopEpisodeSession], self.sessions).append(session)
        return session


def _private_factory(
    *,
    reward: float = 1.0,
    fail_execute: bool = False,
    reported_tool_calls: int = 1,
) -> tuple[WebShopPublicItem, _EpisodeFactory, PrivateWebShopSessionFactory]:
    item = _item()
    case = PrivateWebShopCase(
        public=item,
        private_goal={"canary": PRIVATE_GOAL_CANARY, "purchase_reward": reward},
    )
    episode_factory = _EpisodeFactory(
        fail_execute=fail_execute,
        reported_tool_calls=reported_tool_calls,
    )
    return (
        item,
        episode_factory,
        PrivateWebShopSessionFactory((case,), episode_factory),
    )


def _last_episode(factory: _EpisodeFactory) -> _ScriptedEpisode:
    session = cast(list[PrivateWebShopEpisodeSession], factory.sessions)[-1]
    return cast(_ScriptedEpisode, session.episode)


def test_public_task_provider_projects_only_answer_free_webshop_identity() -> None:
    item, _, _ = _private_factory()
    provider = OrderedWebShopTaskProvider((item,))

    task = provider.next_task()

    assert task.environment_id == item.environment_id
    assert task.task_family == "webshop/home-office"
    tools = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], task.public_context)["tools"],
    )
    assert set(tools) == {
        "click",
        "purchase",
        "search",
    }
    assert PRIVATE_GOAL_CANARY not in canonical_json(task.to_value())
    with pytest.raises(RuntimeError):
        provider.next_task()


def test_action_json_maps_search_click_purchase_and_preserves_measured_usage() -> None:
    item, episode_factory, session_factory = _private_factory(reward=0.65)
    bundle = session_factory.create(item.to_rollout_task())
    environment = cast(WebShopEnvironment, bundle.environment)
    actions = (
        _action("search", {"query": "ergonomic blue chair"}),
        _action("click", {"target": "item-42"}),
        _action("purchase", {}),
    )

    observations = tuple(
        asyncio.run(
            environment.execute(
                action,
                step_index=index,
            )
        )
        for index, action in enumerate(actions, start=1)
    )

    episode = _last_episode(episode_factory)
    assert episode.commands == [
        WebShopCommand(WebShopCommandKind.SEARCH, "ergonomic blue chair"),
        WebShopCommand(WebShopCommandKind.CLICK, "item-42"),
        WebShopCommand(WebShopCommandKind.PURCHASE, None),
    ]
    assert all(
        observation.budget_usage == BudgetVector(tool_calls=1, wall_time_milliseconds=7)
        for observation in observations
    )
    assert not observations[0].terminal
    assert observations[-1].terminal
    assert observations[-1].terminal_submission == {"webshop_episode_terminal": True}
    reward = asyncio.run(bundle.evaluator.evaluate(_request(item)))
    assert reward.value == pytest.approx(0.65)
    assert not reward.success


def test_webshop_accepts_a_valid_tool_after_an_invalid_rollout_turn() -> None:
    item, episode_factory, session_factory = _private_factory()
    environment = cast(
        WebShopEnvironment,
        session_factory.create(item.to_rollout_task()).environment,
    )

    observation = asyncio.run(
        environment.execute(
            _action("search", {"query": "public item"}),
            step_index=2,
        )
    )

    assert observation.observation_status == "success"
    assert _last_episode(episode_factory).cursor == 2


@pytest.mark.parametrize(("score", "success"), [(0.0, False), (0.55, False), (1.0, True)])
def test_terminal_evaluator_uses_only_shared_graded_reward(
    score: float,
    success: bool,
) -> None:
    item, _, session_factory = _private_factory(reward=score)
    bundle = session_factory.create(item.to_rollout_task())
    environment = cast(WebShopEnvironment, bundle.environment)
    asyncio.run(
        environment.execute(
            _action("purchase", {}),
            step_index=1,
        )
    )

    reward = asyncio.run(bundle.evaluator.evaluate(_request(item)))

    assert reward.value == score
    assert reward.success is success
    assert reward.success is (reward.value == 1.0)
    assert reward.native_payload["public_metrics"] == {"webshop-success": 1.0 if success else 0.0}
    assert PRIVATE_GOAL_CANARY not in canonical_json(reward.native_payload)


def test_environment_infrastructure_error_propagates_without_observation_fallback() -> None:
    item, _, session_factory = _private_factory(fail_execute=True)
    environment = cast(
        WebShopEnvironment,
        session_factory.create(item.to_rollout_task()).environment,
    )

    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(
            environment.execute(
                _action("search", {"query": "public query"}),
                step_index=1,
            )
        )


def test_agent_schema_error_is_public_but_never_calls_official_episode() -> None:
    item, episode_factory, session_factory = _private_factory()
    environment = cast(
        WebShopEnvironment,
        session_factory.create(item.to_rollout_task()).environment,
    )

    observation = asyncio.run(
        environment.execute(
            _action("search", {"wrong": "shape"}),
            step_index=1,
        )
    )

    assert observation.observation_status == "tool_error"
    assert observation.public_value == {"error": "unsupported_webshop_action"}
    assert observation.budget_usage == BudgetVector(tool_calls=1)
    assert _last_episode(episode_factory).execute_calls == 0


def test_environment_rejects_unmeasured_tool_usage_instead_of_coercing_it() -> None:
    item, _, session_factory = _private_factory(reported_tool_calls=0)
    environment = cast(
        WebShopEnvironment,
        session_factory.create(item.to_rollout_task()).environment,
    )

    with pytest.raises(ValueError, match="exactly one measured tool call"):
        asyncio.run(
            environment.execute(
                _action("click", {"target": "item-42"}),
                step_index=1,
            )
        )


def test_private_factory_and_evaluator_enforce_exact_task_identity() -> None:
    item, _, session_factory = _private_factory()
    with pytest.raises(ValueError, match="projection differs"):
        session_factory.create(replace(item.to_rollout_task(), query="tampered public query"))

    bundle = session_factory.create(item.to_rollout_task())
    with pytest.raises(TerminalEvaluatorError, match="different WebShop task"):
        asyncio.run(bundle.evaluator.evaluate(replace(_request(item), task_id="another-task")))


class _FailingRewardView:
    def __init__(self, item: WebShopPublicItem, error: Exception) -> None:
        self.task_id = item.task_id
        self.environment_id = item.environment_id
        self.error = error

    def final_reward(self) -> float:
        raise self.error


def test_evaluator_only_translates_declared_reward_availability_failure() -> None:
    item = _item()
    unavailable = PrivateWebShopTerminalEvaluator(
        item,
        _FailingRewardView(item, WebShopRewardUnavailableError("backend unavailable")),
    )
    unexpected = PrivateWebShopTerminalEvaluator(
        item,
        _FailingRewardView(item, AssertionError("unexpected official backend defect")),
    )

    with pytest.raises(TerminalEvaluatorError, match="final reward is unavailable"):
        asyncio.run(unavailable.evaluate(_request(item)))
    with pytest.raises(AssertionError, match="unexpected official backend defect"):
        asyncio.run(unexpected.evaluate(_request(item)))
