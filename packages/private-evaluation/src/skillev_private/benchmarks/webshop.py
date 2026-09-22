"""Private WebShop session composition and graded terminal evaluation.

The official environment deployment supplies ``PrivateWebShopEpisodeFactory``
at runtime.  Its public episode and private reward view are separate objects:
only the former is given to the rollout environment, while only the latter is
given to the terminal evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.webshop import (
    WebShopEnvironment,
    WebShopEpisode,
    WebShopPublicItem,
)
from skillev.contracts import JsonValue, SuccessRule, TerminalReward
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.training import RolloutSessionBundle

from .terminal_inputs import no_submission_reward


class WebShopRewardUnavailableError(RuntimeError):
    """The official episode explicitly cannot provide its measured reward."""


class PrivateWebShopRewardView(Protocol):
    """Read-only private reward channel shared with one live episode."""

    @property
    def task_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    def final_reward(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PrivateWebShopCase:
    """Public projection paired with deployment-only goal-matching truth."""

    public: WebShopPublicItem
    private_goal: object = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.public, WebShopPublicItem):
            raise TypeError("private WebShop case requires WebShopPublicItem")
        if self.private_goal is None:
            raise ValueError("private WebShop goal cannot be absent")


@dataclass(frozen=True, slots=True)
class PrivateWebShopEpisodeSession:
    """Two non-interchangeable views over one official episode state."""

    episode: WebShopEpisode
    reward_view: PrivateWebShopRewardView
    cleanup: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for label, candidate in (
            ("episode", self.episode),
            ("reward view", self.reward_view),
        ):
            if not getattr(candidate, "task_id", ""):
                raise ValueError(f"WebShop {label} task identity is empty")
            if not getattr(candidate, "environment_id", ""):
                raise ValueError(f"WebShop {label} environment identity is empty")
        if not callable(getattr(self.episode, "execute", None)):
            raise TypeError("WebShop episode must implement execute")
        if not callable(getattr(self.reward_view, "final_reward", None)):
            raise TypeError("WebShop reward view must implement final_reward")
        if self.episode.task_id != self.reward_view.task_id:
            raise ValueError("WebShop public and private views disagree on task identity")
        if self.episode.environment_id != self.reward_view.environment_id:
            raise ValueError("WebShop public and private views disagree on environment identity")


class PrivateWebShopEpisodeFactory(Protocol):
    """Deployment injection point; implementations may import official WebShop."""

    def create(self, case: PrivateWebShopCase) -> PrivateWebShopEpisodeSession: ...


@dataclass(slots=True)
class PrivateWebShopTerminalEvaluator:
    """Read the shared episode reward without inspecting public transcript text."""

    public: WebShopPublicItem
    reward_view: PrivateWebShopRewardView

    def __post_init__(self) -> None:
        if self.reward_view.task_id != self.public.task_id:
            raise ValueError("WebShop evaluator reward belongs to another task")
        if self.reward_view.environment_id != self.public.environment_id:
            raise ValueError("WebShop evaluator reward belongs to another environment")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.public.task_id:
            raise TerminalEvaluatorError("terminal request reached a different WebShop task")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="webshop-score",
                native_payload={
                    "benchmark_id": "webshop",
                    "dataset_revision": self.public.dataset_revision,
                    "environment_snapshot_id": self.public.environment_snapshot_id,
                    "public_metrics": {"webshop-success": 0.0},
                    "split": self.public.split,
                },
                environment_id=self.public.environment_id,
                verifier_version="webshop-terminal-evaluator@1",
            )
        try:
            raw_reward = self.reward_view.final_reward()
        except WebShopRewardUnavailableError as error:
            raise TerminalEvaluatorError("WebShop final reward is unavailable") from error
        if isinstance(raw_reward, bool) or not isinstance(raw_reward, int | float):
            raise TerminalEvaluatorError("WebShop final reward must be numeric")
        reward = float(raw_reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise TerminalEvaluatorError("WebShop final reward must lie in [0, 1]")
        native_payload: dict[str, JsonValue] = {
            "benchmark_id": "webshop",
            "dataset_revision": self.public.dataset_revision,
            "environment_snapshot_id": self.public.environment_snapshot_id,
            "public_metrics": {"webshop-success": 1.0 if reward == 1.0 else 0.0},
            "split": self.public.split,
        }
        return TerminalReward(
            value=reward,
            success=reward == 1.0,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="webshop-score",
            native_payload=native_payload,
            environment_id=self.public.environment_id,
            verifier_version="webshop-terminal-evaluator@1",
        )


@dataclass(slots=True)
class PrivateWebShopSessionFactory:
    """Pair exact public tasks with isolated official episodes and private rewards."""

    cases: tuple[PrivateWebShopCase, ...]
    episode_factory: PrivateWebShopEpisodeFactory

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private WebShop session factory requires cases")
        if any(not isinstance(case, PrivateWebShopCase) for case in self.cases):
            raise TypeError("private WebShop session factory cases are invalid")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private WebShop cases must have unique task identities")
        if not callable(getattr(self.episode_factory, "create", None)):
            raise TypeError("private WebShop episode factory must implement create")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("WebShop session creation requires RolloutTask")
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private WebShop case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public WebShop task projection differs from its private case")
        session = self.episode_factory.create(case)
        if not isinstance(session, PrivateWebShopEpisodeSession):
            raise TypeError("private WebShop episode factory returned an incompatible session")
        if session.episode.task_id != case.public.task_id:
            raise ValueError("WebShop episode task identity differs from its private case")
        if session.episode.environment_id != case.public.environment_id:
            raise ValueError("WebShop episode environment identity differs from its private case")
        return RolloutSessionBundle(
            environment=WebShopEnvironment(case.public, session.episode),
            evaluator=PrivateWebShopTerminalEvaluator(case.public, session.reward_view),
            retrieved_skills=(),
            cleanup=session.cleanup,
        )


__all__ = [
    "PrivateWebShopCase",
    "PrivateWebShopEpisodeFactory",
    "PrivateWebShopEpisodeSession",
    "PrivateWebShopRewardView",
    "PrivateWebShopSessionFactory",
    "PrivateWebShopTerminalEvaluator",
    "WebShopRewardUnavailableError",
]
