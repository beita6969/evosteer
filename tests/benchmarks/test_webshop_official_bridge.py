from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast

import pytest
from skillev_private.benchmarks.webshop import (
    PrivateWebShopCase,
    PrivateWebShopSessionFactory,
    WebShopRewardUnavailableError,
)
from skillev_private.benchmarks.webshop_official import (
    OfficialWebShopEpisodeFactory,
    OfficialWebShopGoal,
    OfficialWebShopStepResult,
)

from skillev.benchmarks.webshop import (
    WEBSHOP_RESOURCE_ID,
    WebShopEnvironment,
    WebShopPublicItem,
)
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.rollout import (
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, StructuredAction

PRIVATE_CANARY = "PRIVATE-OFFICIAL-WEBSHOP-GOAL-CANARY"


def _item() -> WebShopPublicItem:
    return WebShopPublicItem(
        dataset_revision="webshop-official-fixture@1",
        environment_snapshot_id="official-products-fixture@1",
        split="train",
        task_id="official-webshop-task-1",
        task_family="webshop/official",
        query="Buy the public requested product.",
        public_context={
            "initial_available_actions": ["click[item-42]", "click[buy now]"],
            "initial_observation": "Official public landing page.",
            "observation_format": "official-text",
        },
    )


def _goal(item: WebShopPublicItem) -> OfficialWebShopGoal:
    return OfficialWebShopGoal(
        task_id=item.task_id,
        environment_id=item.environment_id,
        goal_id="goal-0001",
        session_id="session-0001",
        payload={"private_goal": PRIVATE_CANARY},
    )


def _case() -> PrivateWebShopCase:
    item = _item()
    return PrivateWebShopCase(item, _goal(item))


def _action(name: str, arguments: JsonValue) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=name,
        arguments=normalize_json(arguments),
        resource_id=WEBSHOP_RESOURCE_ID,
    )


def _request(item: WebShopPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="official-webshop-trajectory",
        task_id=item.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue({"webshop_episode_terminal": True}),
        public_transcript_hash=stable_hash({"public": "official webshop transcript"}),
    )


@dataclass(slots=True)
class _FakeOfficialEnv:
    goal: OfficialWebShopGoal
    instruction: str
    drift: bool = False
    fail_step: bool = False
    goal_id_override: str | None = None
    session_id_override: str | None = None
    reset_calls: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    terminal: bool = False
    close_calls: int = 0

    @property
    def goal_id(self) -> str:
        return self.goal_id_override or self.goal.goal_id

    @property
    def session_id(self) -> str:
        return self.session_id_override or self.goal.session_id

    @property
    def instruction_text(self) -> str:
        return self.instruction

    def reset(self, session_id: str) -> str:
        self.reset_calls.append(session_id)
        self.actions = []
        self.terminal = False
        return "Official public landing page."

    def step(self, action: str) -> OfficialWebShopStepResult:
        if self.fail_step:
            raise AssertionError("unexpected official environment defect")
        self.actions.append(action)
        self.terminal = action == "click[buy now]"
        text = f"Official page after {action}."
        if self.drift:
            text = f"DRIFTED {text}"
        return OfficialWebShopStepResult(
            observation_text=text,
            reward=1.0 if self.terminal else 0.0,
            terminal=self.terminal,
        )

    def get_available_actions(self) -> tuple[str, ...]:
        if self.terminal:
            return ()
        return ("click[item-42]", "click[buy now]")

    async def close(self) -> None:
        self.close_calls += 1


@dataclass(slots=True)
class _FakeOfficialFactory:
    public: WebShopPublicItem
    drift_indices: set[int] = field(default_factory=set)
    fail_step_indices: set[int] = field(default_factory=set)
    instruction_override: str | None = None
    goal_id_override: str | None = None
    session_id_override: str | None = None
    envs: list[_FakeOfficialEnv] = field(default_factory=list)

    def create(self, goal: OfficialWebShopGoal) -> _FakeOfficialEnv:
        index = len(self.envs) + 1
        env = _FakeOfficialEnv(
            goal=goal,
            instruction=self.instruction_override or self.public.query,
            drift=index in self.drift_indices,
            fail_step=index in self.fail_step_indices,
            goal_id_override=self.goal_id_override,
            session_id_override=self.session_id_override,
        )
        self.envs.append(env)
        return env


def _session_factory(
    case: PrivateWebShopCase,
    official: _FakeOfficialFactory,
) -> PrivateWebShopSessionFactory:
    return PrivateWebShopSessionFactory(
        (case,),
        OfficialWebShopEpisodeFactory(official),
    )


def test_official_command_mapping_public_projection_and_private_reward() -> None:
    case = _case()
    official = _FakeOfficialFactory(case.public)
    bundle = _session_factory(case, official).create(case.public.to_rollout_task())
    environment = cast(WebShopEnvironment, bundle.environment)

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    search = asyncio.run(
        environment.execute(
            _action("search", {"query": "public blue chair"}),
            step_index=1,
        )
    )
    click = asyncio.run(
        environment.execute(
            _action("click", {"target": "item-42"}),
            step_index=2,
        )
    )
    purchase = asyncio.run(
        environment.execute(
            _action("purchase", {}),
            step_index=3,
        )
    )
    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    env = official.envs[0]
    assert env.reset_calls == [_goal(case.public).session_id]
    assert env.actions == [
        "search[public blue chair]",
        "click[item-42]",
        "click[buy now]",
    ]
    assert set(cast(dict[str, JsonValue], search.public_value)) == {
        "available_actions",
        "terminal",
        "text",
    }
    assert "reward" not in canonical_json(search.public_value).lower()
    assert click.budget_usage.tool_calls == purchase.budget_usage.tool_calls == 1
    assert search.budget_usage.wall_time_milliseconds >= 0
    assert purchase.terminal
    assert reward.value == 1.0
    assert PRIVATE_CANARY not in canonical_json(
        {
            "observations": [search.to_value(), click.to_value(), purchase.to_value()],
            "reward": reward.to_value(),
            "task": case.public.to_rollout_task().to_value(),
        }
    )


def test_official_reward_view_is_unavailable_before_terminal() -> None:
    case = _case()
    official = _FakeOfficialFactory(case.public)
    session = OfficialWebShopEpisodeFactory(official).create(case)

    with pytest.raises(WebShopRewardUnavailableError):
        session.reward_view.final_reward()


def test_official_session_exposes_episode_cleanup() -> None:
    case = _case()
    official = _FakeOfficialFactory(case.public)
    bundle = _session_factory(case, official).create(case.public.to_rollout_task())

    assert bundle.cleanup is not None
    asyncio.run(bundle.cleanup())
    assert official.envs[0].close_calls == 1


@pytest.mark.parametrize(
    (
        "instruction_override",
        "goal_id_override",
        "session_id_override",
        "message",
    ),
    [
        ("another instruction", None, None, "instruction differs"),
        (None, "another-goal", None, "another goal"),
        (None, None, "another-session", "another session"),
    ],
)
def test_official_instruction_and_goal_session_identity_are_pinned(
    instruction_override: str | None,
    goal_id_override: str | None,
    session_id_override: str | None,
    message: str,
) -> None:
    case = _case()
    official = _FakeOfficialFactory(
        case.public,
        instruction_override=instruction_override,
        goal_id_override=goal_id_override,
        session_id_override=session_id_override,
    )

    with pytest.raises(ValueError, match=message):
        OfficialWebShopEpisodeFactory(official).create(case)


def test_unknown_official_exception_propagates_without_public_observation() -> None:
    case = _case()
    official = _FakeOfficialFactory(case.public, fail_step_indices={1})
    environment = cast(
        WebShopEnvironment,
        _session_factory(case, official).create(case.public.to_rollout_task()).environment,
    )

    with pytest.raises(AssertionError, match="unexpected official environment defect"):
        asyncio.run(
            environment.execute(
                _action("search", {"query": "public query"}),
                step_index=1,
            )
        )
    assert official.envs[0].actions == []
