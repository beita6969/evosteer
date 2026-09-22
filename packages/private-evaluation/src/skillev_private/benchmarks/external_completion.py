"""Narrow private worker boundary for official completion-style benchmarks.

Public task manifests contain only :class:`RolloutTask` values.  Gold SQL,
unit tests, table answers, and official evaluator internals remain owned by an
injected out-of-process worker keyed by the frozen task identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.rollout import (
    EnvironmentObservation,
    NoTerminalSubmission,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    StructuredAction,
)
from skillev.training import RolloutSessionBundle

from .terminal_inputs import submitted_value


class ExternalCompletionInfrastructureError(RuntimeError):
    """A pinned completion worker could not produce an official outcome."""


@dataclass(frozen=True, slots=True)
class ExternalCompletionResult:
    reward_value: float
    success: bool
    native_metric_name: str
    public_metrics: dict[str, JsonValue]
    verifier_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.reward_value <= 1.0:
            raise ValueError("external completion reward must be in [0, 1]")
        if type(self.success) is not bool:
            raise TypeError("external completion success must be boolean")
        for field_name in ("native_metric_name", "verifier_version"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"external completion {field_name} must be non-empty text")
        normalized = normalize_json(self.public_metrics)
        if not isinstance(normalized, dict):
            raise TypeError("external completion public_metrics must be an object")


class ExternalCompletionWorker(Protocol):
    """Trusted isolated worker implementing the pinned official evaluator."""

    @property
    def benchmark_id(self) -> str: ...

    def no_submission_result(self, *, task_id: str) -> ExternalCompletionResult: ...

    async def evaluate(
        self,
        *,
        task_id: str,
        submission: str,
    ) -> ExternalCompletionResult: ...


@dataclass(slots=True)
class ExternalCompletionEnvironment:
    """Completion environment constructed directly from an answer-free RolloutTask."""

    task: RolloutTask
    _last_environment_step_index: int = field(default=0, init=False, repr=False)

    @property
    def environment_id(self) -> str:
        return self.task.environment_id

    @property
    def task_family(self) -> str:
        return self.task.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        # Invalid turns are consumed by BoundedAgent without calling the
        # environment, so environment-visible step indices can legitimately skip.
        # Only replay or out-of-order execution is invalid here.
        if step_index <= self._last_environment_step_index:
            raise RuntimeError("external completion requires an increasing step")
        self._last_environment_step_index = step_index
        if action.kind is not ActionKind.SKILL or action.skill_id is None:
            raise EnvironmentMethodFailedError(
                budget_usage=BudgetVector(tool_calls=1),
                public_error_code="unsupported_benchmark_action",
            )
        return EnvironmentObservation(
            public_value={"status": "skill-invoked"},
            observation_status="success",
            invoked_skill_ids=(action.skill_id,),
            budget_usage=BudgetVector(tool_calls=1),
        )

    def validate_completion(self, submission: JsonValue) -> bool:
        normalized = normalize_json(submission)
        return (
            isinstance(normalized, dict)
            and set(normalized) == {"answer"}
            and type(normalized["answer"]) is str
            and bool(normalized["answer"].strip())
        )


@dataclass(slots=True)
class ExternalCompletionTerminalEvaluator:
    task: RolloutTask
    worker: ExternalCompletionWorker

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("external completion reached a different task")
        try:
            if isinstance(request.evaluation_input, NoTerminalSubmission):
                result = self.worker.no_submission_result(task_id=self.task.task_id)
            else:
                submission = normalize_json(submitted_value(request))
                if not isinstance(submission, dict) or set(submission) != {"answer"}:
                    raise TerminalEvaluatorError("external completion submission is incompatible")
                answer = submission["answer"]
                if type(answer) is not str or not answer.strip():
                    raise TerminalEvaluatorError(
                        "external completion answer must be non-empty text"
                    )
                result = await self.worker.evaluate(
                    task_id=self.task.task_id,
                    submission=answer,
                )
        except ExternalCompletionInfrastructureError as error:
            raise TerminalEvaluatorError("external completion worker failed") from error
        if not isinstance(result, ExternalCompletionResult):
            raise TerminalEvaluatorError(
                "external completion worker returned an incompatible result"
            )
        native_payload: dict[str, JsonValue] = {"public_metrics": result.public_metrics}
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            native_payload["no_submission_reason"] = request.evaluation_input.reason.value
        return TerminalReward(
            value=result.reward_value,
            success=result.success,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name=result.native_metric_name,
            native_payload=native_payload,
            environment_id=self.task.environment_id,
            verifier_version=result.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class ExternalCompletionSessionFactory:
    """Route exact public tasks to a sealed official evaluation worker."""

    tasks: tuple[RolloutTask, ...]
    worker: ExternalCompletionWorker

    def __post_init__(self) -> None:
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("external completion tasks must be non-empty and unique")
        benchmark_ids = {
            task.public_context.get("benchmark_id")
            for task in self.tasks
            if isinstance(task.public_context, dict)
        }
        if benchmark_ids != {self.worker.benchmark_id}:
            raise ValueError("external completion worker benchmark differs from tasks")
        if not callable(getattr(self.worker, "evaluate", None)):
            raise TypeError("external completion worker must implement evaluate")
        if not callable(getattr(self.worker, "no_submission_result", None)):
            raise TypeError("external completion worker must implement no_submission_result")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        matches = tuple(candidate for candidate in self.tasks if candidate.task_id == task.task_id)
        if len(matches) != 1 or matches[0] != task:
            raise ValueError("external completion task differs from its frozen projection")
        return RolloutSessionBundle(
            environment=ExternalCompletionEnvironment(task),
            evaluator=ExternalCompletionTerminalEvaluator(task, self.worker),
            retrieved_skills=(),
        )


__all__ = [
    "ExternalCompletionEnvironment",
    "ExternalCompletionInfrastructureError",
    "ExternalCompletionResult",
    "ExternalCompletionSessionFactory",
    "ExternalCompletionTerminalEvaluator",
    "ExternalCompletionWorker",
]
