"""Trusted HumanEval and MATH-Hard terminal-evaluation adapters.

The public rollout environment only admits an ``{"answer": <text>}``
submission.  Executable tests, reference answers, and equivalence logic remain
behind the private evaluator boundary.  In particular, this module defines
only narrow backend protocols; it never executes candidate code in-process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from skillev.benchmarks import CompletionBenchmarkEnvironment
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.rollout import (
    NoTerminalSubmission,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.training import RolloutSessionBundle

from .public_code_context import HUMANEVAL_VERIFIER
from .source_cases import PrivateHumanEvalCase, PrivateMathCase
from .terminal_inputs import no_submission_reward, submitted_value

_CaseT = TypeVar("_CaseT", PrivateHumanEvalCase, PrivateMathCase)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _submission_answer(request: TerminalEvaluationRequest, *, task_id: str) -> str:
    if request.task_id != task_id:
        raise TerminalEvaluatorError("terminal request reached a different private case")
    submission = normalize_json(submitted_value(request))
    if not isinstance(submission, dict) or set(submission) != {"answer"}:
        raise TerminalEvaluatorError("admitted completion has incompatible fields")
    answer = submission["answer"]
    if type(answer) is not str or not answer.strip():
        raise TerminalEvaluatorError("admitted completion answer must be non-empty text")
    return answer


class CodeExecutionStatus(StrEnum):
    """Explicit isolated-execution outcome; non-passing statuses score zero."""

    PASSED = "passed"
    SYNTAX_ERROR = "syntax-error"
    RUNTIME_ERROR = "runtime-error"
    TEST_FAILURE = "test-failure"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource-limit"


@dataclass(frozen=True, slots=True)
class CodeExecutionRequest:
    """Private request sent to an out-of-process HumanEval executor."""

    task_id: str
    prompt: str
    completion: str
    test_source: str
    entry_point: str

    def __post_init__(self) -> None:
        for field in ("task_id", "completion", "test_source", "entry_point"):
            _text(getattr(self, field), field=f"code execution {field}")
        # A syntax-only full-module submission needs no completion prefix.
        if not isinstance(self.prompt, str) or "\x00" in self.prompt:
            raise ValueError("code execution prompt must be text without NUL")


@dataclass(frozen=True, slots=True)
class CodeExecutionResult:
    """Content-free result returned by an isolated code-execution backend."""

    status: CodeExecutionStatus
    native_result: str | None = None
    exception_type: str | None = None
    stage: str | None = None

    def diagnostics(self) -> dict[str, JsonValue]:
        return {
            "format": "humaneval-native-result@2",
            "passed": self.passed,
            "execution_status": self.status.value,
            "native_result": self.native_result,
            "exception_type": self.exception_type,
            "stage": self.stage,
        }

    def __post_init__(self) -> None:
        if not isinstance(self.status, CodeExecutionStatus):
            raise TypeError("code execution status has an incompatible type")

    @property
    def passed(self) -> bool:
        return self.status is CodeExecutionStatus.PASSED


class CodeExecutionBackend(Protocol):
    """Isolated executor boundary; implementations must not run in this process."""

    async def run(self, request: CodeExecutionRequest) -> CodeExecutionResult: ...


class CodeExecutionInfrastructureError(RuntimeError):
    """The isolated executor explicitly could not produce an outcome."""


@dataclass(slots=True)
class HumanEvalTerminalEvaluator:
    """Score pass@1 through an injected isolated execution backend."""

    case: PrivateHumanEvalCase
    backend: CodeExecutionBackend

    def __post_init__(self) -> None:
        if self.case.public.benchmark_id != "humaneval":
            raise ValueError("HumanEval evaluator requires a HumanEval public identity")
        if not callable(getattr(self.backend, "run", None)):
            raise TypeError("code execution backend must implement run")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.case.target.task_id:
            raise TerminalEvaluatorError("terminal request reached a different private case")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="pass@1",
                native_payload={
                    "benchmark_id": self.case.public.benchmark_id,
                    "execution_status": "no-submission",
                    "split": self.case.public.split,
                },
                environment_id=self.case.public.environment_id,
                verifier_version=HUMANEVAL_VERIFIER,
            )
        completion = _submission_answer(request, task_id=self.case.target.task_id)
        from .public_code_context import CodeSubmission, PublicCodeContext, compose_code_candidate

        assembled = compose_code_candidate(
            PublicCodeContext(self.case.public.query, self.case.target.entry_point),
            CodeSubmission(completion),
        )
        execution_request = CodeExecutionRequest(
            task_id=self.case.target.task_id,
            prompt=assembled.prefix,
            completion=assembled.completion,
            test_source=self.case.target.test,
            entry_point=self.case.target.entry_point,
        )
        try:
            result = await self.backend.run(execution_request)
        except CodeExecutionInfrastructureError as error:
            raise TerminalEvaluatorError("isolated code execution backend failed") from error
        if not isinstance(result, CodeExecutionResult):
            raise TerminalEvaluatorError("isolated code execution backend returned invalid output")
        value = float(result.passed)
        return TerminalReward(
            value=value,
            success=result.passed,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="pass@1",
            native_payload={
                "benchmark_id": self.case.public.benchmark_id,
                **result.diagnostics(),
                **assembled.diagnostics(),
                "split": self.case.public.split,
            },
            environment_id=self.case.public.environment_id,
            verifier_version=HUMANEVAL_VERIFIER,
        )


@dataclass(slots=True)
class HumanEvalSessionFactory:
    """Bind exact public HumanEval projections to private cases and executor."""

    cases: tuple[PrivateHumanEvalCase, ...]
    backend: CodeExecutionBackend

    def __post_init__(self) -> None:
        _validate_cases(self.cases, benchmark_id="humaneval")
        if not callable(getattr(self.backend, "run", None)):
            raise TypeError("code execution backend must implement run")

    def create(self, task: object) -> RolloutSessionBundle:
        case = _matching_case(self.cases, task)
        return RolloutSessionBundle(
            environment=CompletionBenchmarkEnvironment(case.public),
            evaluator=HumanEvalTerminalEvaluator(case, self.backend),
            retrieved_skills=(),
        )


class MathEquivalenceKind(StrEnum):
    """Verifier decision supporting exact and symbolic equivalence."""

    EXACT = "exact"
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not-equivalent"


@dataclass(frozen=True, slots=True)
class MathEquivalenceRequest:
    """Private candidate/reference pair sent to the equivalence verifier."""

    task_id: str
    prediction: str
    reference_answer: str

    def __post_init__(self) -> None:
        for field in ("task_id", "prediction", "reference_answer"):
            _text(getattr(self, field), field=f"math equivalence {field}")


@dataclass(frozen=True, slots=True)
class MathEquivalenceResult:
    """Content-free exact/equivalence verdict returned by the verifier."""

    kind: MathEquivalenceKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MathEquivalenceKind):
            raise TypeError("math equivalence kind has an incompatible type")

    @property
    def equivalent(self) -> bool:
        return self.kind is not MathEquivalenceKind.NOT_EQUIVALENT


class MathEquivalenceBackend(Protocol):
    """Trusted exact/symbolic equivalence checker boundary."""

    async def check(self, request: MathEquivalenceRequest) -> MathEquivalenceResult: ...


class MathEquivalenceInfrastructureError(RuntimeError):
    """The trusted equivalence service explicitly could not produce a verdict."""


@dataclass(slots=True)
class MathHardTerminalEvaluator:
    """Score MATH-Hard accuracy through an injected private verifier."""

    case: PrivateMathCase
    backend: MathEquivalenceBackend

    def __post_init__(self) -> None:
        if self.case.public.benchmark_id != "math-hard":
            raise ValueError("MATH-Hard evaluator requires a MATH-Hard public identity")
        if not callable(getattr(self.backend, "check", None)):
            raise TypeError("math equivalence backend must implement check")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.case.target.task_id:
            raise TerminalEvaluatorError("terminal request reached a different private case")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="accuracy",
                native_payload={
                    "benchmark_id": self.case.public.benchmark_id,
                    "match_kind": "no-submission",
                    "split": self.case.public.split,
                },
                environment_id=self.case.public.environment_id,
                verifier_version="math-hard-equivalence-evaluator@1",
            )
        prediction = _submission_answer(request, task_id=self.case.target.task_id)
        equivalence_request = MathEquivalenceRequest(
            task_id=self.case.target.task_id,
            prediction=prediction,
            reference_answer=self.case.target.boxed_answer,
        )
        try:
            result = await self.backend.check(equivalence_request)
        except MathEquivalenceInfrastructureError as error:
            raise TerminalEvaluatorError("math equivalence backend failed") from error
        if not isinstance(result, MathEquivalenceResult):
            raise TerminalEvaluatorError("math equivalence backend returned invalid output")
        value = float(result.equivalent)
        return TerminalReward(
            value=value,
            success=result.equivalent,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="accuracy",
            native_payload={
                "benchmark_id": self.case.public.benchmark_id,
                "match_kind": result.kind.value,
                "split": self.case.public.split,
            },
            environment_id=self.case.public.environment_id,
            verifier_version="math-hard-equivalence-evaluator@1",
        )


@dataclass(slots=True)
class MathHardSessionFactory:
    """Bind exact public MATH-Hard projections to private verifier cases."""

    cases: tuple[PrivateMathCase, ...]
    backend: MathEquivalenceBackend

    def __post_init__(self) -> None:
        _validate_cases(self.cases, benchmark_id="math-hard")
        if not callable(getattr(self.backend, "check", None)):
            raise TypeError("math equivalence backend must implement check")

    def create(self, task: object) -> RolloutSessionBundle:
        case = _matching_case(self.cases, task)
        return RolloutSessionBundle(
            environment=CompletionBenchmarkEnvironment(case.public),
            evaluator=MathHardTerminalEvaluator(case, self.backend),
            retrieved_skills=(),
        )


def _validate_cases(
    cases: tuple[_CaseT, ...],
    *,
    benchmark_id: str,
) -> None:
    if not cases:
        raise ValueError("private benchmark session factory requires cases")
    task_ids = tuple(case.public.task_id for case in cases)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("private benchmark cases must have unique task identities")
    if any(case.public.benchmark_id != benchmark_id for case in cases):
        raise ValueError("private benchmark cases have incompatible benchmark identities")


def _matching_case(
    cases: tuple[_CaseT, ...],
    task: object,
) -> _CaseT:
    task_id = getattr(task, "task_id", None)
    matches = tuple(case for case in cases if case.public.task_id == task_id)
    if len(matches) != 1:
        raise ValueError("public task has no unique private benchmark case")
    case = matches[0]
    if task != case.public.to_rollout_task():
        raise ValueError("public task projection differs from its private case")
    return case


__all__ = [
    "CodeExecutionBackend",
    "CodeExecutionInfrastructureError",
    "CodeExecutionRequest",
    "CodeExecutionResult",
    "CodeExecutionStatus",
    "HumanEvalSessionFactory",
    "HumanEvalTerminalEvaluator",
    "MathEquivalenceBackend",
    "MathEquivalenceInfrastructureError",
    "MathEquivalenceKind",
    "MathEquivalenceRequest",
    "MathEquivalenceResult",
    "MathHardSessionFactory",
    "MathHardTerminalEvaluator",
]
