"""Private ScienceWorld composition and native 0--100 evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.scienceworld import (
    ScienceWorldEnvironment,
    ScienceWorldEpisode,
    ScienceWorldPublicItem,
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


class ScienceWorldOutcomeUnavailableError(RuntimeError):
    """The official simulator explicitly could not expose its final score."""


class PrivateScienceWorldOutcomeView(Protocol):
    """Private native-score channel paired with one live public episode."""

    @property
    def task_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    def final_score(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PrivateScienceWorldCase:
    public: ScienceWorldPublicItem
    private_task: object = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.public, ScienceWorldPublicItem):
            raise TypeError("private ScienceWorld case requires ScienceWorldPublicItem")
        if self.private_task is None:
            raise ValueError("private ScienceWorld task cannot be absent")


@dataclass(frozen=True, slots=True)
class PrivateScienceWorldEpisodeSession:
    episode: ScienceWorldEpisode
    outcome_view: PrivateScienceWorldOutcomeView

    def __post_init__(self) -> None:
        if self.episode.task_id != self.outcome_view.task_id:
            raise ValueError("ScienceWorld public and private views disagree on task identity")
        if self.episode.environment_id != self.outcome_view.environment_id:
            raise ValueError(
                "ScienceWorld public and private views disagree on environment identity"
            )
        if self.episode.seed != self.outcome_view.seed:
            raise ValueError("ScienceWorld public and private views disagree on seed")


class PrivateScienceWorldEpisodeFactory(Protocol):
    """Deployment injection point allowed to import official ScienceWorld/JVM."""

    def create(self, case: PrivateScienceWorldCase) -> PrivateScienceWorldEpisodeSession: ...


@dataclass(slots=True)
class PrivateScienceWorldTerminalEvaluator:
    public: ScienceWorldPublicItem
    outcome_view: PrivateScienceWorldOutcomeView

    def __post_init__(self) -> None:
        if self.outcome_view.task_id != self.public.task_id:
            raise ValueError("ScienceWorld evaluator belongs to another task")
        if self.outcome_view.environment_id != self.public.environment_id:
            raise ValueError("ScienceWorld evaluator belongs to another environment")
        if self.outcome_view.seed != self.public.seed:
            raise ValueError("ScienceWorld evaluator uses a different seed")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.public.task_id:
            raise TerminalEvaluatorError("terminal request reached a different ScienceWorld task")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="scienceworld-score",
                native_payload={
                    "benchmark_id": "scienceworld",
                    "dataset_revision": self.public.dataset_revision,
                    "environment_snapshot_id": self.public.environment_snapshot_id,
                    "native_score": 0.0,
                    "public_metrics": {"scienceworld-native-score": 0.0},
                    "split": self.public.split,
                },
                environment_id=self.public.environment_id,
                verifier_version="scienceworld-terminal-evaluator@1",
            )
        try:
            raw_score = self.outcome_view.final_score()
        except ScienceWorldOutcomeUnavailableError as error:
            raise TerminalEvaluatorError("ScienceWorld final score is unavailable") from error
        if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
            raise TerminalEvaluatorError("ScienceWorld final score must be numeric")
        native_score = float(raw_score)
        if not math.isfinite(native_score) or not 0.0 <= native_score <= 100.0:
            raise TerminalEvaluatorError("ScienceWorld final score must lie in [0, 100]")
        reward = native_score / 100.0
        native_payload: dict[str, JsonValue] = {
            "benchmark_id": "scienceworld",
            "dataset_revision": self.public.dataset_revision,
            "environment_snapshot_id": self.public.environment_snapshot_id,
            "native_score": native_score,
            "public_metrics": {"scienceworld-native-score": native_score},
            "split": self.public.split,
        }
        return TerminalReward(
            value=reward,
            success=native_score == 100.0,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="scienceworld-score",
            native_payload=native_payload,
            environment_id=self.public.environment_id,
            verifier_version="scienceworld-terminal-evaluator@1",
        )


@dataclass(slots=True)
class PrivateScienceWorldSessionFactory:
    cases: tuple[PrivateScienceWorldCase, ...]
    episode_factory: PrivateScienceWorldEpisodeFactory

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private ScienceWorld session factory requires cases")
        if any(not isinstance(case, PrivateScienceWorldCase) for case in self.cases):
            raise TypeError("private ScienceWorld session factory cases are invalid")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private ScienceWorld cases must have unique task identities")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("ScienceWorld session creation requires RolloutTask")
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private ScienceWorld case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public ScienceWorld task projection differs from its private case")
        session = self.episode_factory.create(case)
        if not isinstance(session, PrivateScienceWorldEpisodeSession):
            raise TypeError("ScienceWorld episode factory returned an incompatible session")
        if session.episode.task_id != case.public.task_id:
            raise ValueError("ScienceWorld episode task identity differs from its private case")
        if session.episode.environment_id != case.public.environment_id:
            raise ValueError("ScienceWorld episode environment differs from its private case")
        if session.episode.seed != case.public.seed:
            raise ValueError("ScienceWorld episode seed differs from its private case")
        if session.episode.max_steps != case.public.max_steps:
            raise ValueError("ScienceWorld episode step limit differs from its private case")
        return RolloutSessionBundle(
            environment=ScienceWorldEnvironment(case.public, session.episode),
            evaluator=PrivateScienceWorldTerminalEvaluator(case.public, session.outcome_view),
            retrieved_skills=(),
        )


__all__ = [
    "PrivateScienceWorldCase",
    "PrivateScienceWorldEpisodeFactory",
    "PrivateScienceWorldEpisodeSession",
    "PrivateScienceWorldOutcomeView",
    "PrivateScienceWorldSessionFactory",
    "PrivateScienceWorldTerminalEvaluator",
    "ScienceWorldOutcomeUnavailableError",
]
