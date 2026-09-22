from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from skillev_private.benchmarks import (
    ExternalCompletionEnvironment,
    ExternalCompletionInfrastructureError,
    ExternalCompletionResult,
    ExternalCompletionTerminalEvaluator,
)

from skillev.contracts import stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTask,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import (
    ActionKind,
    AttemptFailureCode,
    AttemptFailureStage,
    StructuredAction,
)


def _task() -> RolloutTask:
    return RolloutTask(
        task_id="external-completion-task",
        environment_id="external-completion-environment",
        task_family="bird-sql/database/query-generation",
        context_id="bird-sql/database",
        query="Return a SQL query.",
        available_tools=(),
        public_context={"benchmark_id": "bird-sql"},
    )


def test_external_completion_environment_allows_unobserved_invalid_turns() -> None:
    environment = ExternalCompletionEnvironment(_task())
    action = StructuredAction(
        kind=ActionKind.SKILL,
        name="apply-skill",
        arguments={},
        resource_id="skill-runtime",
        skill_id="skill-alpha",
    )

    observation = asyncio.run(environment.execute(action, step_index=2))

    assert observation.invoked_skill_ids == ("skill-alpha",)
    with pytest.raises(RuntimeError):
        asyncio.run(environment.execute(action, step_index=1))


def _request(*, answer: str | None) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="external-completion-trajectory",
        task_id=_task().task_id,
        termination=(
            RolloutTermination.HORIZON_EXHAUSTED if answer is None else RolloutTermination.COMPLETED
        ),
        evaluation_input=(
            NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED)
            if answer is None
            else SubmittedTerminalValue({"answer": answer})
        ),
        public_transcript_hash=stable_hash({"public": "terminal transcript"}),
    )


@dataclass(slots=True)
class _Worker:
    fail: bool = False
    evaluate_calls: list[str] = field(default_factory=list)
    no_submission_calls: int = 0

    @property
    def benchmark_id(self) -> str:
        return "bird-sql"

    def no_submission_result(self, *, task_id: str) -> ExternalCompletionResult:
        self.no_submission_calls += 1
        if self.fail:
            raise ExternalCompletionInfrastructureError("private case missing")
        assert task_id == _task().task_id
        return ExternalCompletionResult(
            reward_value=0.0,
            success=False,
            native_metric_name="execution_accuracy",
            public_metrics={"execution_accuracy": 0.0},
            verifier_version="external-completion-fixture@1",
        )

    async def evaluate(self, *, task_id: str, submission: str) -> ExternalCompletionResult:
        if self.fail:
            raise ExternalCompletionInfrastructureError("private worker crashed")
        assert task_id == _task().task_id
        self.evaluate_calls.append(submission)
        return ExternalCompletionResult(
            reward_value=0.0,
            success=False,
            native_metric_name="execution_accuracy",
            public_metrics={"execution_accuracy": 0.0},
            verifier_version="external-completion-fixture@1",
        )


def test_no_submission_and_wrong_submission_are_agent_outcomes() -> None:
    worker = _Worker()
    evaluator = ExternalCompletionTerminalEvaluator(_task(), worker)

    no_submission = asyncio.run(evaluator.evaluate(_request(answer=None)))
    wrong_submission = asyncio.run(evaluator.evaluate(_request(answer="SELECT wrong")))

    assert no_submission.value == wrong_submission.value == 0.0
    assert worker.no_submission_calls == 1
    assert worker.evaluate_calls == ["SELECT wrong"]


@pytest.mark.parametrize("answer", [None, "SELECT answer"])
def test_worker_resource_or_case_failure_is_typed_fatal(answer: str | None) -> None:
    evaluator = ExternalCompletionTerminalEvaluator(_task(), _Worker(fail=True))

    with pytest.raises(TerminalEvaluatorError) as captured:
        asyncio.run(evaluator.evaluate(_request(answer=answer)))

    assert captured.value.code is AttemptFailureCode.TERMINAL_EVALUATOR_FAILED
    assert captured.value.stage is AttemptFailureStage.TERMINAL_EVALUATION
