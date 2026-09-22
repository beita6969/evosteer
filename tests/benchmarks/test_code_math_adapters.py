from __future__ import annotations

import ast
import asyncio
import inspect
from dataclasses import dataclass, field

import pytest
from skillev_private.benchmarks import (
    CodeExecutionInfrastructureError,
    CodeExecutionRequest,
    CodeExecutionResult,
    CodeExecutionStatus,
    HumanEvalSessionFactory,
    HumanEvalTerminalEvaluator,
    MathEquivalenceInfrastructureError,
    MathEquivalenceKind,
    MathEquivalenceRequest,
    MathEquivalenceResult,
    MathHardSessionFactory,
    MathHardTerminalEvaluator,
    PrivateHumanEvalCase,
    PrivateHumanEvalTarget,
    PrivateMathCase,
    PrivateMathTarget,
    code_math,
)

from skillev.benchmarks import BenchmarkPublicItem, CompletionBenchmarkEnvironment
from skillev.contracts import canonical_json, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluator,
    TerminalEvaluatorError,
)

PRIVATE_CANARY = "PRIVATE-VERIFIER-CANARY-NEVER-PUBLIC"


def _human_case() -> PrivateHumanEvalCase:
    public = BenchmarkPublicItem(
        benchmark_id="humaneval",
        dataset_revision="fixture@1",
        split="test",
        task_id="HumanEval/synthetic-1",
        task_family="humaneval/python",
        query="def public_function(value: int) -> int:\n",
        public_context={
            "answer_format": "python-completion",
            "entry_point": "public_function",
        },
    )
    return PrivateHumanEvalCase(
        public,
        PrivateHumanEvalTarget(
            task_id=public.task_id,
            canonical_solution=f"    return value  # {PRIVATE_CANARY}\n",
            test=f"def check(candidate): assert candidate(1) == 1  # {PRIVATE_CANARY}",
            entry_point="public_function",
        ),
    )


def _math_case() -> PrivateMathCase:
    public = BenchmarkPublicItem(
        benchmark_id="math-hard",
        dataset_revision="fixture@1",
        split="test",
        task_id="math-hard/synthetic-1",
        task_family="math-hard/Algebra",
        query="Compute the public expression.",
        public_context={"answer_format": "latex-expression", "level": "Level 5"},
    )
    return PrivateMathCase(
        public,
        PrivateMathTarget(
            task_id=public.task_id,
            solution=f"Private reasoning {PRIVATE_CANARY}; final answer follows.",
            boxed_answer=r"\frac{2}{3}",
        ),
    )


def _request(item: BenchmarkPublicItem, answer: str) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-001",
        task_id=item.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue({"answer": answer}),
        public_transcript_hash=stable_hash({"public": "transcript"}),
    )


def _no_submission_request(item: BenchmarkPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-no-submission",
        task_id=item.task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )


@dataclass(slots=True)
class _ScriptedCodeBackend:
    result: CodeExecutionResult
    requests: list[CodeExecutionRequest] = field(default_factory=list)

    async def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        self.requests.append(request)
        return self.result


@dataclass(slots=True)
class _ScriptedMathBackend:
    result: MathEquivalenceResult
    requests: list[MathEquivalenceRequest] = field(default_factory=list)

    async def check(self, request: MathEquivalenceRequest) -> MathEquivalenceResult:
        self.requests.append(request)
        return self.result


def test_no_submission_scores_zero_without_calling_code_or_math_backend() -> None:
    human_case = _human_case()
    math_case = _math_case()
    code_backend = _ScriptedCodeBackend(CodeExecutionResult(CodeExecutionStatus.PASSED))
    math_backend = _ScriptedMathBackend(MathEquivalenceResult(MathEquivalenceKind.EQUIVALENT))

    code_reward = asyncio.run(
        HumanEvalTerminalEvaluator(human_case, code_backend).evaluate(
            _no_submission_request(human_case.public)
        )
    )
    math_reward = asyncio.run(
        MathHardTerminalEvaluator(math_case, math_backend).evaluate(
            _no_submission_request(math_case.public)
        )
    )

    assert code_reward.value == math_reward.value == 0.0
    assert code_backend.requests == []
    assert math_backend.requests == []


@pytest.mark.parametrize(
    "status",
    [
        CodeExecutionStatus.SYNTAX_ERROR,
        CodeExecutionStatus.RUNTIME_ERROR,
        CodeExecutionStatus.TEST_FAILURE,
        CodeExecutionStatus.TIMEOUT,
        CodeExecutionStatus.RESOURCE_LIMIT,
    ],
)
def test_humaneval_explicit_candidate_failures_score_zero(status: CodeExecutionStatus) -> None:
    case = _human_case()
    backend = _ScriptedCodeBackend(CodeExecutionResult(status))
    bundle = HumanEvalSessionFactory((case,), backend).create(case.public.to_rollout_task())

    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public, "    return value\n")))

    assert reward.value == 0.0
    assert not reward.success
    assert reward.native_payload["execution_status"] == status.value
    assert backend.requests == [
        CodeExecutionRequest(
            task_id=case.public.task_id,
            prompt=case.public.query,
            completion="    return value\n",
            test_source=case.target.test,
            entry_point=case.target.entry_point,
        )
    ]
    assert PRIVATE_CANARY not in canonical_json(reward.to_value())


def test_humaneval_passes_only_backend_pass_and_keeps_tests_private() -> None:
    case = _human_case()
    backend = _ScriptedCodeBackend(CodeExecutionResult(CodeExecutionStatus.PASSED))
    bundle = HumanEvalSessionFactory((case,), backend).create(case.public.to_rollout_task())

    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public, "    return value\n")))

    assert reward.value == 1.0
    assert reward.success
    assert isinstance(bundle.environment, CompletionBenchmarkEnvironment)
    assert PRIVATE_CANARY not in canonical_json(case.public.to_rollout_task().to_value())
    assert PRIVATE_CANARY not in canonical_json(reward.to_value())


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (MathEquivalenceKind.EXACT, 1.0),
        (MathEquivalenceKind.EQUIVALENT, 1.0),
        (MathEquivalenceKind.NOT_EQUIVALENT, 0.0),
    ],
)
def test_math_hard_uses_binary_exact_or_equivalence_reward(
    kind: MathEquivalenceKind,
    expected: float,
) -> None:
    case = _math_case()
    backend = _ScriptedMathBackend(MathEquivalenceResult(kind))
    bundle = MathHardSessionFactory((case,), backend).create(case.public.to_rollout_task())

    reward = asyncio.run(bundle.evaluator.evaluate(_request(case.public, "2/3")))

    assert reward.value == expected
    assert reward.success is (expected == 1.0)
    assert reward.native_payload["match_kind"] == kind.value
    assert backend.requests == [
        MathEquivalenceRequest(
            task_id=case.public.task_id,
            prediction="2/3",
            reference_answer=case.target.boxed_answer,
        )
    ]
    assert PRIVATE_CANARY not in canonical_json(reward.to_value())


@dataclass(slots=True)
class _InfrastructureFailingCodeBackend:
    async def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        del request
        raise CodeExecutionInfrastructureError(PRIVATE_CANARY)


@dataclass(slots=True)
class _InfrastructureFailingMathBackend:
    async def check(self, request: MathEquivalenceRequest) -> MathEquivalenceResult:
        del request
        raise MathEquivalenceInfrastructureError(PRIVATE_CANARY)


@dataclass(slots=True)
class _AssertingCodeBackend:
    async def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        del request
        raise AssertionError(PRIVATE_CANARY)


@dataclass(slots=True)
class _AssertingMathBackend:
    async def check(self, request: MathEquivalenceRequest) -> MathEquivalenceResult:
        del request
        raise AssertionError(PRIVATE_CANARY)


@pytest.mark.parametrize("benchmark", ["humaneval", "math-hard"])
def test_declared_backend_infrastructure_failure_becomes_rejection(benchmark: str) -> None:
    evaluator: TerminalEvaluator
    if benchmark == "humaneval":
        human_case = _human_case()
        evaluator = (
            HumanEvalSessionFactory((human_case,), _InfrastructureFailingCodeBackend())
            .create(human_case.public.to_rollout_task())
            .evaluator
        )
        request = _request(human_case.public, "public candidate")
    else:
        math_case = _math_case()
        evaluator = (
            MathHardSessionFactory((math_case,), _InfrastructureFailingMathBackend())
            .create(math_case.public.to_rollout_task())
            .evaluator
        )
        request = _request(math_case.public, "public candidate")

    with pytest.raises(TerminalEvaluatorError) as captured:
        asyncio.run(evaluator.evaluate(request))
    assert PRIVATE_CANARY not in str(captured.value)


@pytest.mark.parametrize("benchmark", ["humaneval", "math-hard"])
def test_unknown_backend_assertion_propagates_without_reward(benchmark: str) -> None:
    evaluator: TerminalEvaluator
    if benchmark == "humaneval":
        human_case = _human_case()
        evaluator = (
            HumanEvalSessionFactory((human_case,), _AssertingCodeBackend())
            .create(human_case.public.to_rollout_task())
            .evaluator
        )
        request = _request(human_case.public, "public candidate")
    else:
        math_case = _math_case()
        evaluator = (
            MathHardSessionFactory((math_case,), _AssertingMathBackend())
            .create(math_case.public.to_rollout_task())
            .evaluator
        )
        request = _request(math_case.public, "public candidate")

    with pytest.raises(AssertionError, match=PRIVATE_CANARY):
        asyncio.run(evaluator.evaluate(request))


def test_factories_bind_the_exact_public_identity() -> None:
    human = _human_case()
    other = BenchmarkPublicItem(
        benchmark_id=human.public.benchmark_id,
        dataset_revision=human.public.dataset_revision,
        split=human.public.split,
        task_id=human.public.task_id,
        task_family=human.public.task_family,
        query="tampered public prompt",
        public_context=human.public.public_context,
    )
    factory = HumanEvalSessionFactory(
        (human,),
        _ScriptedCodeBackend(CodeExecutionResult(CodeExecutionStatus.PASSED)),
    )

    with pytest.raises(ValueError):
        factory.create(other.to_rollout_task())


def test_private_adapter_contains_no_in_process_exec_or_eval() -> None:
    tree = ast.parse(inspect.getsource(code_math))
    forbidden = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"exec", "eval"}
    }
    assert forbidden == set()
