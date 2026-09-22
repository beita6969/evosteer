from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast

import pytest
from skillev_private.benchmarks.mind2web_official import (
    MIND2WEB_BROWSER_ACTION_NAME,
    MIND2WEB_OFFICIAL_RESOURCE_ID,
    OfficialMind2WebAction,
    OfficialMind2WebCase,
    OfficialMind2WebElement,
    OfficialMind2WebEnvironment,
    OfficialMind2WebPage,
    OfficialMind2WebSessionFactory,
    OfficialMind2WebStepResult,
    OfficialMind2WebTask,
)

from skillev.benchmarks.static import BenchmarkPublicItem
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, StructuredAction

PRIVATE_CANARY = "PRIVATE-OFFICIAL-MIND2WEB-GOLD-CANARY"
TARGET_NODE = "node-target"


def _page(label: str) -> OfficialMind2WebPage:
    return OfficialMind2WebPage(
        url=f"https://public.example/{label}",
        text=f"Public browser page {label}.",
        elements=(
            OfficialMind2WebElement(
                backend_node_id=TARGET_NODE,
                tag="input",
                text="Public destination",
                attributes={"role": "textbox"},
            ),
            OfficialMind2WebElement(
                backend_node_id="node-other",
                tag="button",
                text="Public alternative",
                attributes={"role": "button"},
            ),
        ),
    )


def _public(*, max_steps: int = 3) -> BenchmarkPublicItem:
    initial = _page("initial")
    return BenchmarkPublicItem(
        benchmark_id="mind2web",
        dataset_revision="mind2web-official-browser-fixture@1",
        split="test_website",
        task_id="mind2web/official-task-1",
        task_family="mind2web/test_website/travel",
        query="Complete the public browser task.",
        public_context={
            "browser_snapshot_id": "browser-snapshot-fixture@1",
            "initial_page": initial.to_value(),
            "max_steps": max_steps,
            "seed": 1729,
            "tools": {
                MIND2WEB_BROWSER_ACTION_NAME: {
                    "arguments": ["operation", "backend_node_id", "value"]
                }
            },
            "website": "public.example",
        },
    )


def _case(*, max_steps: int = 3) -> OfficialMind2WebCase:
    public = _public(max_steps=max_steps)
    return OfficialMind2WebCase(
        public,
        OfficialMind2WebTask(
            task_id=public.task_id,
            environment_id=public.environment_id,
            website="public.example",
            browser_snapshot_id="browser-snapshot-fixture@1",
            seed=1729,
            max_steps=max_steps,
            initial_page=_page("initial"),
            payload={"gold_action": PRIVATE_CANARY},
        ),
    )


def _action(
    *,
    operation: str = "TYPE",
    backend_node_id: str = TARGET_NODE,
    value: str = "public value",
) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=MIND2WEB_BROWSER_ACTION_NAME,
        arguments=normalize_json(
            {
                "backend_node_id": backend_node_id,
                "operation": operation,
                "value": value,
            }
        ),
        resource_id=MIND2WEB_OFFICIAL_RESOURCE_ID,
    )


def _request(public: BenchmarkPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="official-mind2web-trajectory",
        task_id=public.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue({"episode_terminal": True}),
        public_transcript_hash=stable_hash({"public": "mind2web browser transcript"}),
    )


def _no_submission_request(public: BenchmarkPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="official-mind2web-no-submission",
        task_id=public.task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )


@dataclass(slots=True)
class _FakeBrowserEnv:
    task: OfficialMind2WebTask
    description: str
    terminal_after: int
    drift: bool = False
    fail_step: bool = False
    task_id_override: str | None = None
    website_override: str | None = None
    snapshot_override: str | None = None
    seed_override: int | None = None
    max_steps_override: int | None = None
    initial_override: OfficialMind2WebPage | None = None
    reset_calls: list[tuple[str, str, str, int, int]] = field(default_factory=list)
    actions: list[OfficialMind2WebAction] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return self.task_id_override or self.task.task_id

    @property
    def website(self) -> str:
        return self.website_override or self.task.website

    @property
    def browser_snapshot_id(self) -> str:
        return self.snapshot_override or self.task.browser_snapshot_id

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
        task_id: str,
        website: str,
        browser_snapshot_id: str,
        seed: int,
        max_steps: int,
    ) -> OfficialMind2WebPage:
        self.reset_calls.append((task_id, website, browser_snapshot_id, seed, max_steps))
        self.actions = []
        return self.initial_override or self.task.initial_page

    def step(self, action: OfficialMind2WebAction) -> OfficialMind2WebStepResult:
        if self.fail_step:
            raise AssertionError("unexpected official browser defect")
        self.actions.append(action)
        terminal = len(self.actions) >= self.terminal_after
        success = action.backend_node_id == TARGET_NODE and action.operation == "TYPE"
        page = _page(f"step-{len(self.actions)}")
        if self.drift:
            page = OfficialMind2WebPage(page.url, f"DRIFTED {page.text}", page.elements)
        return OfficialMind2WebStepResult(page, success, terminal)


@dataclass(slots=True)
class _FakeBrowserFactory:
    public: BenchmarkPublicItem
    terminal_after: int = 1
    drift_indices: set[int] = field(default_factory=set)
    fail_step_indices: set[int] = field(default_factory=set)
    description_override: str | None = None
    task_id_override: str | None = None
    website_override: str | None = None
    snapshot_override: str | None = None
    seed_override: int | None = None
    max_steps_override: int | None = None
    initial_override: OfficialMind2WebPage | None = None
    envs: list[_FakeBrowserEnv] = field(default_factory=list)

    def create(self, task: OfficialMind2WebTask) -> _FakeBrowserEnv:
        index = len(self.envs) + 1
        env = _FakeBrowserEnv(
            task=task,
            description=self.description_override or self.public.query,
            terminal_after=self.terminal_after,
            drift=index in self.drift_indices,
            fail_step=index in self.fail_step_indices,
            task_id_override=self.task_id_override,
            website_override=self.website_override,
            snapshot_override=self.snapshot_override,
            seed_override=self.seed_override,
            max_steps_override=self.max_steps_override,
            initial_override=self.initial_override,
        )
        self.envs.append(env)
        return env


def test_official_no_submission_is_zero_without_reading_browser_outcome() -> None:
    case = _case()
    official = _FakeBrowserFactory(case.public, terminal_after=2)
    bundle = OfficialMind2WebSessionFactory((case,), official).create(case.public.to_rollout_task())

    reward = asyncio.run(bundle.evaluator.evaluate(_no_submission_request(case.public)))

    assert reward.value == 0.0
    assert reward.success is False
    assert official.envs[0].actions == []


def test_official_action_mapping_page_projection_budget_and_success_isolation() -> None:
    case = _case()
    official = _FakeBrowserFactory(case.public)
    bundle = OfficialMind2WebSessionFactory((case,), official).create(case.public.to_rollout_task())

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    environment = cast(OfficialMind2WebEnvironment, bundle.environment)
    observation = asyncio.run(
        environment.execute(
            _action(),
            step_index=1,
        )
    )
    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    assert official.envs[0].reset_calls == [
        (
            case.task.task_id,
            case.task.website,
            case.task.browser_snapshot_id,
            case.task.seed,
            case.task.max_steps,
        )
    ]
    assert official.envs[0].actions == [OfficialMind2WebAction("TYPE", TARGET_NODE, "public value")]
    assert set(cast(dict[str, JsonValue], observation.public_value)) == {
        "elements",
        "terminal",
        "text",
        "url",
    }
    assert "success" not in canonical_json(observation.public_value).lower()
    assert observation.budget_usage.tool_calls == 1
    assert observation.budget_usage.wall_time_milliseconds >= 0
    assert observation.terminal
    assert reward.value == 1.0
    assert PRIVATE_CANARY not in canonical_json(
        {
            "observation": observation.to_value(),
            "reward": reward.to_value(),
            "task": case.public.to_rollout_task().to_value(),
        }
    )


def test_official_unsuccessful_terminal_action_is_real_zero_reward() -> None:
    case = _case()
    official = _FakeBrowserFactory(case.public)
    bundle = OfficialMind2WebSessionFactory((case,), official).create(case.public.to_rollout_task())
    environment = cast(OfficialMind2WebEnvironment, bundle.environment)
    asyncio.run(
        environment.execute(
            _action(operation="CLICK", backend_node_id="node-other", value=""),
            step_index=1,
        )
    )

    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public)))

    assert reward.value == 0.0
    assert not reward.success


@pytest.mark.parametrize(
    (
        "description_override",
        "task_id_override",
        "website_override",
        "snapshot_override",
        "seed_override",
        "max_steps_override",
        "initial_override",
        "message",
    ),
    [
        ("another task", None, None, None, None, None, None, "description differs"),
        (None, "other-task", None, None, None, None, None, "another task identity"),
        (None, None, "other.example", None, None, None, None, "another website"),
        (None, None, None, "other-snapshot", None, None, None, "another browser snapshot"),
        (None, None, None, None, 1730, None, None, "another seed"),
        (None, None, None, None, None, 99, None, "another step limit"),
        (None, None, None, None, None, None, _page("other"), "reset page differs"),
    ],
)
def test_official_website_task_snapshot_seed_and_initial_page_are_pinned(
    description_override: str | None,
    task_id_override: str | None,
    website_override: str | None,
    snapshot_override: str | None,
    seed_override: int | None,
    max_steps_override: int | None,
    initial_override: OfficialMind2WebPage | None,
    message: str,
) -> None:
    case = _case()
    official = _FakeBrowserFactory(
        case.public,
        description_override=description_override,
        task_id_override=task_id_override,
        website_override=website_override,
        snapshot_override=snapshot_override,
        seed_override=seed_override,
        max_steps_override=max_steps_override,
        initial_override=initial_override,
    )

    with pytest.raises(ValueError, match=message):
        OfficialMind2WebEnvironment(case, official)


def test_unknown_official_exception_propagates_without_public_observation() -> None:
    case = _case()
    official = _FakeBrowserFactory(case.public, fail_step_indices={1})
    environment = OfficialMind2WebEnvironment(case, official)

    with pytest.raises(AssertionError, match="unexpected official browser defect"):
        asyncio.run(
            environment.execute(
                _action(),
                step_index=1,
            )
        )
    assert official.envs[0].actions == []
