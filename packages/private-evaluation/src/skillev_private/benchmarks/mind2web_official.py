"""Private bridge to an injected official Mind2Web browser environment.

The official browser stack is dependency-injected and never imported here.
This module maps the existing public ``operation/backend_node_id/value``
schema to official steps, exposes only model-visible page projections, and
keeps terminal success behind a private evaluator view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.mind2web import MIND2WEB_BENCHMARK_ID, MIND2WEB_OPERATIONS
from skillev.benchmarks.static import BenchmarkPublicItem
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentObservation,
    StructuredAction,
)
from skillev.training import RolloutSessionBundle

from .terminal_inputs import no_submission_reward

MIND2WEB_OFFICIAL_RESOURCE_ID = "mind2web"
MIND2WEB_BROWSER_ACTION_NAME = "browser_action"


def _text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} has invalid text")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class OfficialMind2WebElement:
    backend_node_id: str
    tag: str
    text: str
    attributes: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.backend_node_id, field_name="backend_node_id")
        _text(self.tag, field_name="element tag")
        _text(self.text, field_name="element text", allow_empty=True)
        attributes = normalize_json(self.attributes)
        if not isinstance(attributes, dict) or attributes != self.attributes:
            raise TypeError("Mind2Web element attributes must be a JSON object")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attributes": self.attributes,
            "backend_node_id": self.backend_node_id,
            "tag": self.tag,
            "text": self.text,
        }

    @classmethod
    def from_value(cls, value: object) -> OfficialMind2WebElement:
        data = _object(
            value,
            fields={"attributes", "backend_node_id", "tag", "text"},
            label="Mind2Web element",
        )
        attributes = data["attributes"]
        if not isinstance(attributes, dict):
            raise TypeError("Mind2Web element attributes must be an object")
        return cls(
            backend_node_id=_text(data["backend_node_id"], field_name="backend_node_id"),
            tag=_text(data["tag"], field_name="element tag"),
            text=_text(data["text"], field_name="element text", allow_empty=True),
            attributes=attributes,
        )


@dataclass(frozen=True, slots=True)
class OfficialMind2WebPage:
    url: str
    text: str
    elements: tuple[OfficialMind2WebElement, ...]

    def __post_init__(self) -> None:
        _text(self.url, field_name="page URL")
        _text(self.text, field_name="page text")
        if not isinstance(self.elements, tuple):
            raise TypeError("Mind2Web page elements must be a tuple")
        if any(not isinstance(element, OfficialMind2WebElement) for element in self.elements):
            raise TypeError("Mind2Web page contains an incompatible element")
        identities = tuple(element.backend_node_id for element in self.elements)
        if len(set(identities)) != len(identities):
            raise ValueError("Mind2Web page element identities must be unique")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "elements": [element.to_value() for element in self.elements],
            "text": self.text,
            "url": self.url,
        }

    @classmethod
    def from_value(cls, value: object) -> OfficialMind2WebPage:
        data = _object(value, fields={"elements", "text", "url"}, label="Mind2Web page")
        elements = data["elements"]
        if not isinstance(elements, list):
            raise TypeError("Mind2Web page elements must be an array")
        return cls(
            url=_text(data["url"], field_name="page URL"),
            text=_text(data["text"], field_name="page text"),
            elements=tuple(OfficialMind2WebElement.from_value(item) for item in elements),
        )


@dataclass(frozen=True, slots=True)
class OfficialMind2WebAction:
    operation: str
    backend_node_id: str
    value: str

    def __post_init__(self) -> None:
        if self.operation not in MIND2WEB_OPERATIONS:
            raise ValueError("Mind2Web operation is unsupported")
        _text(self.backend_node_id, field_name="backend_node_id")
        _text(self.value, field_name="operation value", allow_empty=True)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "backend_node_id": self.backend_node_id,
            "operation": self.operation,
            "value": self.value,
        }

    @classmethod
    def from_value(cls, value: object) -> OfficialMind2WebAction:
        data = _object(
            value,
            fields={"backend_node_id", "operation", "value"},
            label="Mind2Web browser action",
        )
        return cls(
            operation=_text(data["operation"], field_name="operation"),
            backend_node_id=_text(data["backend_node_id"], field_name="backend_node_id"),
            value=_text(data["value"], field_name="operation value", allow_empty=True),
        )


@dataclass(frozen=True, slots=True)
class OfficialMind2WebTask:
    """Pinned browser task and private deployment payload."""

    task_id: str
    environment_id: str
    website: str
    browser_snapshot_id: str
    seed: int
    max_steps: int
    initial_page: OfficialMind2WebPage
    payload: object = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in ("task_id", "environment_id", "website", "browser_snapshot_id"):
            _text(getattr(self, field_name), field_name=field_name)
        _non_negative_int(self.seed, field_name="seed")
        _positive_int(self.max_steps, field_name="max_steps")
        if not isinstance(self.initial_page, OfficialMind2WebPage):
            raise TypeError("official Mind2Web task requires an initial page")
        if self.payload is None:
            raise ValueError("official Mind2Web private task payload cannot be absent")


@dataclass(frozen=True, slots=True)
class OfficialMind2WebStepResult:
    page: OfficialMind2WebPage
    success: bool
    terminal: bool

    def __post_init__(self) -> None:
        if not isinstance(self.page, OfficialMind2WebPage):
            raise TypeError("official Mind2Web step requires a page projection")
        if type(self.success) is not bool or type(self.terminal) is not bool:
            raise TypeError("official Mind2Web success and terminal flags must be boolean")


class OfficialMind2WebBrowserEnv(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def website(self) -> str: ...

    @property
    def browser_snapshot_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def max_steps(self) -> int: ...

    @property
    def task_description(self) -> str: ...

    def reset(
        self,
        *,
        task_id: str,
        website: str,
        browser_snapshot_id: str,
        seed: int,
        max_steps: int,
    ) -> OfficialMind2WebPage: ...

    def step(self, action: OfficialMind2WebAction) -> OfficialMind2WebStepResult: ...


class OfficialMind2WebBrowserEnvFactory(Protocol):
    def create(self, task: OfficialMind2WebTask) -> OfficialMind2WebBrowserEnv: ...


@dataclass(frozen=True, slots=True)
class OfficialMind2WebCase:
    public: BenchmarkPublicItem
    task: OfficialMind2WebTask = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.public, BenchmarkPublicItem):
            raise TypeError("official Mind2Web case requires BenchmarkPublicItem")
        if self.public.benchmark_id != MIND2WEB_BENCHMARK_ID:
            raise ValueError("official Mind2Web case belongs to another benchmark")
        if not isinstance(self.task, OfficialMind2WebTask):
            raise TypeError("official Mind2Web case requires OfficialMind2WebTask")
        if self.task.task_id != self.public.task_id:
            raise ValueError("official Mind2Web task belongs to another public task")
        if self.task.environment_id != self.public.environment_id:
            raise ValueError("official Mind2Web task belongs to another environment")
        context = self.public.public_context
        if not isinstance(context, dict):
            raise TypeError("official Mind2Web public context must be an object")
        if context.get("website") != self.task.website:
            raise ValueError("official Mind2Web website differs from public context")
        if context.get("browser_snapshot_id") != self.task.browser_snapshot_id:
            raise ValueError("official Mind2Web snapshot differs from public context")
        if context.get("seed") != self.task.seed:
            raise ValueError("official Mind2Web seed differs from public context")
        if context.get("max_steps") != self.task.max_steps:
            raise ValueError("official Mind2Web step limit differs from public context")
        if context.get("initial_page") != self.task.initial_page.to_value():
            raise ValueError("official Mind2Web initial page differs from public context")


@dataclass(frozen=True, slots=True)
class _Record:
    step_index: int
    action: StructuredAction
    official_action: OfficialMind2WebAction | None
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        _positive_int(self.step_index, field_name="step_index")


def _page_observation(page: OfficialMind2WebPage, *, terminal: bool) -> dict[str, JsonValue]:
    return {**page.to_value(), "terminal": terminal}


def _terminal_submission(task_id: str) -> dict[str, JsonValue]:
    return {
        "benchmark_id": MIND2WEB_BENCHMARK_ID,
        "episode_terminal": True,
        "task_id": task_id,
    }


def _official_action(action: StructuredAction) -> OfficialMind2WebAction | None:
    if (
        action.kind is not ActionKind.TOOL
        or action.resource_id != MIND2WEB_OFFICIAL_RESOURCE_ID
        or action.name != MIND2WEB_BROWSER_ACTION_NAME
        or not isinstance(action.arguments, dict)
    ):
        return None
    try:
        return OfficialMind2WebAction.from_value(action.arguments)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class OfficialMind2WebEnvironment:
    case: OfficialMind2WebCase
    env_factory: OfficialMind2WebBrowserEnvFactory
    _env: OfficialMind2WebBrowserEnv = field(init=False, repr=False)
    _records: list[_Record] = field(default_factory=list, init=False, repr=False)
    _terminal_success: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.env_factory, "create", None)):
            raise TypeError("official Mind2Web browser factory must implement create")
        self._env = _create_pinned_env(self.case, self.env_factory)

    @property
    def environment_id(self) -> str:
        return self.case.public.environment_id

    @property
    def task_family(self) -> str:
        return self.case.public.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if not isinstance(action, StructuredAction):
            raise TypeError("official Mind2Web action must be StructuredAction")
        _positive_int(step_index, field_name="step_index")
        if step_index != len(self._records) + 1:
            raise ValueError("official Mind2Web steps must be contiguous")
        if step_index > self.case.task.max_steps:
            raise ValueError("official Mind2Web exceeded its pinned step limit")
        if self._terminal_success is not None:
            raise ValueError("official Mind2Web episode already terminated")

        projected = _official_action(action)
        if action.kind is ActionKind.SKILL and action.skill_id is not None:
            observation = EnvironmentObservation(
                public_value={"status": "skill-invoked"},
                observation_status="success",
                invoked_skill_ids=(action.skill_id,),
                budget_usage=BudgetVector(tool_calls=1),
            )
        elif projected is None:
            observation = EnvironmentObservation(
                public_value={"error": "unsupported_mind2web_action"},
                observation_status="tool_error",
                budget_usage=BudgetVector(tool_calls=1),
            )
        else:
            started_ns = time.perf_counter_ns()
            result = self._env.step(projected)
            elapsed_ns = time.perf_counter_ns() - started_ns
            if not isinstance(result, OfficialMind2WebStepResult):
                raise TypeError("official Mind2Web env returned an incompatible step")
            terminal = result.terminal or step_index == self.case.task.max_steps
            observation = EnvironmentObservation(
                public_value=_page_observation(result.page, terminal=terminal),
                observation_status="success",
                terminal_submission=(
                    _terminal_submission(self.case.public.task_id) if terminal else None
                ),
                terminal=terminal,
                budget_usage=BudgetVector(
                    tool_calls=1,
                    wall_time_milliseconds=(elapsed_ns + 999_999) // 1_000_000,
                ),
            )
            if terminal:
                self._terminal_success = result.success

        self._records.append(_Record(step_index, action, projected, observation))
        return observation

    def validate_completion(self, submission: JsonValue) -> bool:
        del submission
        return False


def _create_pinned_env(
    case: OfficialMind2WebCase,
    env_factory: OfficialMind2WebBrowserEnvFactory,
) -> OfficialMind2WebBrowserEnv:
    task = case.task
    env = env_factory.create(task)
    initial_page = env.reset(
        task_id=task.task_id,
        website=task.website,
        browser_snapshot_id=task.browser_snapshot_id,
        seed=task.seed,
        max_steps=task.max_steps,
    )
    if not isinstance(initial_page, OfficialMind2WebPage):
        raise TypeError("official Mind2Web reset returned an incompatible page")
    if env.task_id != task.task_id:
        raise ValueError("official Mind2Web env has another task identity")
    if env.website != task.website:
        raise ValueError("official Mind2Web env has another website")
    if env.browser_snapshot_id != task.browser_snapshot_id:
        raise ValueError("official Mind2Web env has another browser snapshot")
    if env.seed != task.seed:
        raise ValueError("official Mind2Web env has another seed")
    if env.max_steps != task.max_steps:
        raise ValueError("official Mind2Web env has another step limit")
    if _text(env.task_description, field_name="official task description") != case.public.query:
        raise ValueError("official Mind2Web task description differs from public query")
    if initial_page != task.initial_page:
        raise ValueError("official Mind2Web reset page differs from pinned public page")
    return env


class Mind2WebOutcomeUnavailableError(RuntimeError):
    """The official browser has not reached a terminal success outcome."""


@dataclass(frozen=True, slots=True)
class _OfficialMind2WebOutcomeView:
    environment: OfficialMind2WebEnvironment = field(repr=False)

    def final_success(self) -> bool:
        if self.environment._terminal_success is None:
            raise Mind2WebOutcomeUnavailableError(
                "official Mind2Web success is unavailable before terminal state"
            )
        return self.environment._terminal_success


@dataclass(slots=True)
class OfficialMind2WebTerminalEvaluator:
    case: OfficialMind2WebCase
    outcome_view: _OfficialMind2WebOutcomeView = field(repr=False)

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.case.public.task_id:
            raise TerminalEvaluatorError("terminal request reached another Mind2Web task")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="mind2web-official-success",
                native_payload={
                    "benchmark_id": MIND2WEB_BENCHMARK_ID,
                    "browser_snapshot_id": self.case.task.browser_snapshot_id,
                    "split": self.case.public.split,
                    "website": self.case.task.website,
                },
                environment_id=self.case.public.environment_id,
                verifier_version="mind2web-official-browser-evaluator@1",
            )
        try:
            success = self.outcome_view.final_success()
        except Mind2WebOutcomeUnavailableError as error:
            raise TerminalEvaluatorError("Mind2Web final outcome is unavailable") from error
        if type(success) is not bool:
            raise TerminalEvaluatorError("Mind2Web final success must be boolean")
        reward = float(success)
        return TerminalReward(
            value=reward,
            success=success,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="mind2web-official-success",
            native_payload={
                "benchmark_id": MIND2WEB_BENCHMARK_ID,
                "browser_snapshot_id": self.case.task.browser_snapshot_id,
                "split": self.case.public.split,
                "website": self.case.task.website,
            },
            environment_id=self.case.public.environment_id,
            verifier_version="mind2web-official-browser-evaluator@1",
        )


@dataclass(slots=True)
class OfficialMind2WebSessionFactory:
    cases: tuple[OfficialMind2WebCase, ...]
    env_factory: OfficialMind2WebBrowserEnvFactory

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("official Mind2Web session factory requires cases")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("official Mind2Web cases must have unique task identities")
        if not callable(getattr(self.env_factory, "create", None)):
            raise TypeError("official Mind2Web browser factory must implement create")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("official Mind2Web session creation requires RolloutTask")
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique official Mind2Web case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public Mind2Web task projection differs from official case")
        environment = OfficialMind2WebEnvironment(case, self.env_factory)
        outcome_view = _OfficialMind2WebOutcomeView(environment)
        return RolloutSessionBundle(
            environment=environment,
            evaluator=OfficialMind2WebTerminalEvaluator(case, outcome_view),
            retrieved_skills=(),
        )


__all__ = [
    "MIND2WEB_BROWSER_ACTION_NAME",
    "MIND2WEB_OFFICIAL_RESOURCE_ID",
    "Mind2WebOutcomeUnavailableError",
    "OfficialMind2WebAction",
    "OfficialMind2WebBrowserEnv",
    "OfficialMind2WebBrowserEnvFactory",
    "OfficialMind2WebCase",
    "OfficialMind2WebElement",
    "OfficialMind2WebEnvironment",
    "OfficialMind2WebPage",
    "OfficialMind2WebSessionFactory",
    "OfficialMind2WebStepResult",
    "OfficialMind2WebTask",
    "OfficialMind2WebTerminalEvaluator",
]
