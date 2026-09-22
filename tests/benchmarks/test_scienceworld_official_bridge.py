from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast

import pytest
from skillev_private.benchmarks.scienceworld import (
    PrivateScienceWorldCase,
    PrivateScienceWorldSessionFactory,
    ScienceWorldOutcomeUnavailableError,
)
from skillev_private.benchmarks.scienceworld_official import (
    OfficialScienceWorldEpisodeFactory,
    OfficialScienceWorldStepResult,
    OfficialScienceWorldTask,
)

from skillev.benchmarks.scienceworld import (
    SCIENCEWORLD_RESOURCE_ID,
    ScienceWorldCommand,
    ScienceWorldEnvironment,
    ScienceWorldPublicItem,
)
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.rollout import (
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, StructuredAction

PRIVATE_CANARY = "PRIVATE-OFFICIAL-SCIENCEWORLD-TRUTH-CANARY"


def _item(*, max_steps: int = 3) -> ScienceWorldPublicItem:
    return ScienceWorldPublicItem(
        dataset_revision="scienceworld-official-fixture@1",
        environment_snapshot_id="official-jvm-fixture@1",
        split="test",
        task_id="scienceworld-official-task-1",
        task_family="scienceworld/official-chemistry",
        query="Complete the public chemistry experiment.",
        public_context={"initial_observation": "Official public laboratory."},
        seed=1729,
        max_steps=max_steps,
    )


def _task(item: ScienceWorldPublicItem) -> OfficialScienceWorldTask:
    return OfficialScienceWorldTask(
        task_id=item.task_id,
        environment_id=item.environment_id,
        task_name="official-chemistry-task",
        variation_index=7,
        seed=item.seed,
        max_steps=item.max_steps,
        payload={"gold_path": PRIVATE_CANARY},
    )


def _case(*, max_steps: int = 3) -> PrivateScienceWorldCase:
    item = _item(max_steps=max_steps)
    return PrivateScienceWorldCase(item, _task(item))


def _action(command: str) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name="act",
        arguments=normalize_json({"command": command}),
        resource_id=SCIENCEWORLD_RESOURCE_ID,
    )


def _request(item: ScienceWorldPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="official-scienceworld-trajectory",
        task_id=item.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(
            {
                "benchmark_id": "scienceworld",
                "episode_terminal": True,
                "reason": "simulator",
                "task_id": item.task_id,
            }
        ),
        public_transcript_hash=stable_hash({"public": "scienceworld transcript"}),
    )


@dataclass(slots=True)
class _FakeOfficialScienceEnv:
    task: OfficialScienceWorldTask
    description: str
    initial_observation: str
    drift: bool = False
    fail_step: bool = False
    task_name_override: str | None = None
    variation_override: int | None = None
    seed_override: int | None = None
    max_steps_override: int | None = None
    reset_calls: list[tuple[str, int, int, int]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    @property
    def task_name(self) -> str:
        return self.task_name_override or self.task.task_name

    @property
    def variation_index(self) -> int:
        return (
            self.variation_override
            if self.variation_override is not None
            else self.task.variation_index
        )

    @property
    def seed(self) -> int:
        return self.seed_override if self.seed_override is not None else self.task.seed

    @property
    def max_steps(self) -> int:
        return (
            self.max_steps_override if self.max_steps_override is not None else self.task.max_steps
        )

    @property
    def task_description(self) -> str:
        return self.description

    def reset(
        self,
        *,
        task_name: str,
        variation_index: int,
        seed: int,
        max_steps: int,
    ) -> str:
        self.reset_calls.append((task_name, variation_index, seed, max_steps))
        self.actions = []
        return self.initial_observation

    def step(self, action: str) -> OfficialScienceWorldStepResult:
        if self.fail_step:
            raise AssertionError("unexpected official ScienceWorld defect")
        self.actions.append(action)
        terminal = action == "finish"
        score = 100.0 if terminal else 25.0 if action == "inspect" else 0.0
        text = f"Official public observation after {action}."
        if self.drift:
            text = f"DRIFTED {text}"
        return OfficialScienceWorldStepResult(text, score, terminal)


@dataclass(slots=True)
class _FakeOfficialScienceFactory:
    public: ScienceWorldPublicItem
    drift_indices: set[int] = field(default_factory=set)
    fail_step_indices: set[int] = field(default_factory=set)
    description_override: str | None = None
    initial_override: str | None = None
    task_name_override: str | None = None
    variation_override: int | None = None
    seed_override: int | None = None
    max_steps_override: int | None = None
    envs: list[_FakeOfficialScienceEnv] = field(default_factory=list)

    def create(self, task: OfficialScienceWorldTask) -> _FakeOfficialScienceEnv:
        index = len(self.envs) + 1
        public_context = cast(dict[str, JsonValue], self.public.public_context)
        env = _FakeOfficialScienceEnv(
            task=task,
            description=self.description_override or self.public.query,
            initial_observation=(
                self.initial_override or cast(str, public_context["initial_observation"])
            ),
            drift=index in self.drift_indices,
            fail_step=index in self.fail_step_indices,
            task_name_override=self.task_name_override,
            variation_override=self.variation_override,
            seed_override=self.seed_override,
            max_steps_override=self.max_steps_override,
        )
        self.envs.append(env)
        return env


def _sessions(
    case: PrivateScienceWorldCase,
    official: _FakeOfficialScienceFactory,
) -> PrivateScienceWorldSessionFactory:
    return PrivateScienceWorldSessionFactory(
        (case,),
        OfficialScienceWorldEpisodeFactory(official),
    )


def test_official_reset_action_projection_budget_and_score_isolation() -> None:
    case = _case()
    official = _FakeOfficialScienceFactory(case.public)
    bundle = _sessions(case, official).create(case.public.to_rollout_task())
    environment = cast(ScienceWorldEnvironment, bundle.environment)

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    inspect = asyncio.run(
        environment.execute(
            _action("inspect"),
            step_index=1,
        )
    )
    finish = asyncio.run(
        environment.execute(
            _action("finish"),
            step_index=2,
        )
    )
    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    env = official.envs[0]
    task = cast(OfficialScienceWorldTask, case.private_task)
    assert env.reset_calls == [(task.task_name, task.variation_index, task.seed, task.max_steps)]
    assert env.actions == ["inspect", "finish"]
    assert inspect.public_value == {
        "terminal": False,
        "text": "Official public observation after inspect.",
    }
    assert set(cast(dict[str, JsonValue], finish.public_value)) == {"terminal", "text"}
    assert "score" not in canonical_json(finish.public_value).lower()
    assert inspect.budget_usage.tool_calls == finish.budget_usage.tool_calls == 1
    assert inspect.budget_usage.wall_time_milliseconds >= 0
    assert reward.value == 1.0
    assert PRIVATE_CANARY not in canonical_json(
        {
            "observations": [inspect.to_value(), finish.to_value()],
            "reward": reward.to_value(),
            "task": case.public.to_rollout_task().to_value(),
        }
    )


def test_final_score_is_available_at_pinned_step_limit_only_after_step() -> None:
    case = _case(max_steps=1)
    official = _FakeOfficialScienceFactory(case.public)
    session = OfficialScienceWorldEpisodeFactory(official).create(case)
    with pytest.raises(ScienceWorldOutcomeUnavailableError):
        session.outcome_view.final_score()

    step = asyncio.run(
        session.episode.execute(
            ScienceWorldCommand("inspect"),
            step_index=1,
        )
    )

    assert step.terminal
    assert session.outcome_view.final_score() == 25.0


@pytest.mark.parametrize(
    (
        "description_override",
        "initial_override",
        "task_name_override",
        "variation_override",
        "seed_override",
        "max_steps_override",
        "message",
    ),
    [
        ("another description", None, None, None, None, None, "description differs"),
        (None, "another initial state", None, None, None, None, "reset observation differs"),
        (None, None, "another-task", None, None, None, "another task name"),
        (None, None, None, 8, None, None, "another task variation"),
        (None, None, None, None, 1730, None, "another seed"),
        (None, None, None, None, None, 99, "another step limit"),
    ],
)
def test_official_task_variation_seed_limit_and_public_reset_are_pinned(
    description_override: str | None,
    initial_override: str | None,
    task_name_override: str | None,
    variation_override: int | None,
    seed_override: int | None,
    max_steps_override: int | None,
    message: str,
) -> None:
    case = _case()
    official = _FakeOfficialScienceFactory(
        case.public,
        description_override=description_override,
        initial_override=initial_override,
        task_name_override=task_name_override,
        variation_override=variation_override,
        seed_override=seed_override,
        max_steps_override=max_steps_override,
    )

    with pytest.raises(ValueError, match=message):
        OfficialScienceWorldEpisodeFactory(official).create(case)


def test_unknown_official_exception_propagates_without_public_observation() -> None:
    case = _case()
    official = _FakeOfficialScienceFactory(case.public, fail_step_indices={1})
    environment = cast(
        ScienceWorldEnvironment,
        _sessions(case, official).create(case.public.to_rollout_task()).environment,
    )

    with pytest.raises(AssertionError, match="unexpected official ScienceWorld defect"):
        asyncio.run(
            environment.execute(
                _action("inspect"),
                step_index=1,
            )
        )
    assert official.envs[0].actions == []
