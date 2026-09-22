"""Private native scorers with definitive candidate/infra separation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.parsing import ParsedResponse
from skillev_private.benchmarks.code_math import (
    CodeExecutionRequest,
    CodeExecutionStatus,
    MathEquivalenceRequest,
)
from skillev_private.benchmarks.humaneval_official import IsolatedHumanEvalExecutionBackend
from skillev_private.benchmarks.math_hard_official import SympyMathEquivalenceBackend
from skillev_private.benchmarks.mind2web import score_mind2web_submission
from skillev_private.benchmarks.qa_metrics import (
    best_alias_metrics,
    normalize_musique_answer,
    normalize_nq_open_answer,
    normalize_triviaqa_answer,
    score_hotpotqa_answers,
)
from skillev_private.benchmarks.source_cases import PrivateMind2WebStepTarget

from .populations import (
    PrivateDirectCase,
    PrivateHumanEvalTarget,
    PrivateMathTarget,
    PrivateQATarget,
)


class ScorerVerdictKind(StrEnum):
    SCORED = "scored"
    CANDIDATE_INVALID = "candidate-invalid"


@dataclass(frozen=True, slots=True)
class DirectScore:
    metrics: dict[str, float]
    verdict_kind: ScorerVerdictKind
    scorer_invoked: bool
    verifier_version: str
    diagnostics: dict[str, str]

    def __post_init__(self) -> None:
        if not self.metrics or any(not 0 <= value <= 1 for value in self.metrics.values()):
            raise ValueError("definitive direct score requires metrics in [0, 1]")
        if not self.verifier_version.strip():
            raise ValueError("verifier_version must be non-empty")

    @property
    def scorer_reached(self) -> bool:
        """Compatibility telemetry; never use this as definitive coverage."""

        return self.scorer_invoked


def _accepted(case: PrivateDirectCase) -> tuple[str, ...]:
    target = case.target
    if isinstance(target, PrivateQATarget):
        return target.accepted_answers
    if type(target) is str:
        return (target,)
    raise TypeError("QA target must contain accepted answers")


def _invalid(metrics: tuple[str, ...], parsed: ParsedResponse, verifier: str) -> DirectScore:
    return DirectScore(
        metrics=dict.fromkeys(metrics, 0.0),
        verdict_kind=ScorerVerdictKind.CANDIDATE_INVALID,
        scorer_invoked=False,
        verifier_version=verifier,
        diagnostics={"parse_reason": parsed.reason.value},
    )


async def _score_qa(
    case: PrivateDirectCase,
    parsed: ParsedResponse,
    *,
    normalizer: Callable[[str], str],
    verifier: str,
) -> DirectScore:
    if parsed.value is None:
        return _invalid(("em", "f1"), parsed, verifier)
    metrics = best_alias_metrics(parsed.value, _accepted(case), normalize=normalizer)
    return DirectScore(
        {"em": metrics.em, "f1": metrics.f1},
        ScorerVerdictKind.SCORED,
        True,
        verifier,
        {},
    )


async def score_hotpotqa(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    if parsed.value is None:
        return _invalid(("em", "f1"), parsed, "hotpotqa-answer-scorer@2")
    metrics = score_hotpotqa_answers(parsed.value, _accepted(case))
    return DirectScore(
        {"em": metrics.em, "f1": metrics.f1},
        ScorerVerdictKind.SCORED,
        True,
        "hotpotqa-answer-scorer@2",
        {},
    )


async def score_triviaqa(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    return await _score_qa(
        case,
        parsed,
        normalizer=normalize_triviaqa_answer,
        verifier="triviaqa-alias-scorer@2",
    )


async def score_musique(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    return await _score_qa(
        case,
        parsed,
        normalizer=normalize_musique_answer,
        verifier="musique-answer-scorer@1",
    )


async def score_nq_open(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    return await _score_qa(
        case,
        parsed,
        normalizer=normalize_nq_open_answer,
        verifier="nq-open-answer-scorer@1",
    )


async def score_aime(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    return _score_text_target(case, parsed, "aime-integer-scorer@2")


async def score_option(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    return _score_text_target(case, parsed, f"{case.public_task.benchmark.value}-option@1")


def _score_text_target(
    case: PrivateDirectCase, parsed: ParsedResponse, verifier: str
) -> DirectScore:
    if parsed.value is None:
        return _invalid(("accuracy",), parsed, verifier)
    if type(case.target) is not str:
        raise TypeError("choice/integer target must be text")
    return DirectScore(
        {"accuracy": float(parsed.value.strip() == case.target.strip())},
        ScorerVerdictKind.SCORED,
        True,
        verifier,
        {},
    )


async def score_math_hard(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    if parsed.value is None:
        return _invalid(("accuracy",), parsed, "math-hard-equivalence@2")
    if not isinstance(case.target, PrivateMathTarget):
        raise TypeError("MATH-Hard target has incompatible type")
    result = await SympyMathEquivalenceBackend().check(
        MathEquivalenceRequest(
            task_id=case.public_task.task_id,
            prediction=parsed.value,
            reference_answer=case.target.boxed_answer,
        )
    )
    return DirectScore(
        {"accuracy": float(result.equivalent)},
        ScorerVerdictKind.SCORED,
        True,
        "math-hard-equivalence@2",
        {},
    )


async def score_humaneval(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    if parsed.value is None:
        return _invalid(("pass_at_1",), parsed, "humaneval-original-pass1@3")
    if not isinstance(case.target, PrivateHumanEvalTarget):
        raise TypeError("HumanEval target has incompatible type")
    completion = parsed.value
    if completion.startswith(case.target.prompt):
        completion = completion[len(case.target.prompt) :]
    result = await IsolatedHumanEvalExecutionBackend().run(
        CodeExecutionRequest(
            task_id=case.public_task.task_id,
            prompt=case.target.prompt,
            completion=completion,
            test_source=case.target.test,
            entry_point=case.target.entry_point,
        )
    )
    return DirectScore(
        {"pass_at_1": float(result.status is CodeExecutionStatus.PASSED)},
        ScorerVerdictKind.SCORED,
        True,
        "humaneval-original-pass1@3",
        {},
    )


async def score_mind2web(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    metric_ids = (
        "element_accuracy",
        "operation_accuracy",
        "value_f1",
        "action_f1",
        "step_accuracy",
    )
    if parsed.value is None:
        return _invalid(metric_ids, parsed, "mind2web-official-step@2")
    if not isinstance(case.target, PrivateMind2WebStepTarget):
        raise TypeError("Mind2Web target has incompatible type")
    parts = parsed.value.split(maxsplit=2)
    if len(parts) < 2:
        return _invalid(metric_ids, parsed, "mind2web-official-step@2")
    operation, backend_node_id = parts[:2]
    value = parts[2] if len(parts) == 3 else ""
    try:
        metrics = score_mind2web_submission(
            case.target,
            operation=operation.upper(),
            backend_node_id=backend_node_id,
            value=value,
        )
    except ValueError:
        return _invalid(metric_ids, parsed, "mind2web-official-step@2")
    return DirectScore(
        {
            "element_accuracy": metrics.element_accuracy,
            "operation_accuracy": metrics.operation_accuracy,
            "value_f1": metrics.value_f1,
            "action_f1": metrics.action_f1,
            "step_accuracy": metrics.step_success,
        },
        ScorerVerdictKind.SCORED,
        True,
        "mind2web-official-step@2",
        {},
    )


DirectScorer = Callable[[PrivateDirectCase, ParsedResponse], Awaitable[DirectScore]]
SCORER_REGISTRY: dict[str, DirectScorer] = {
    "hotpotqa-official-em-f1@1": score_hotpotqa,
    "hotpotqa-official-em-f1@2": score_hotpotqa,
    "triviaqa-official-alias-em-f1@1": score_triviaqa,
    "triviaqa-official-alias-em-f1@2": score_triviaqa,
    "musique-official-em-f1@1": score_musique,
    "nq-open-official-em-f1@1": score_nq_open,
    "aime-official-integer@1": score_aime,
    "aime-official-integer@2": score_aime,
    "medqa-option@1": score_option,
    "gpqa-diamond-option@1": score_option,
    "math-hard-official-equivalence@2": score_math_hard,
    "humaneval-official-pass1@1": score_humaneval,
    "humaneval-official-pass1@2": score_humaneval,
    "humaneval-original-pass1@3": score_humaneval,
    "mind2web-official-step@2": score_mind2web,
}


def require_protocol13_scorer_profile(profile_id: str) -> DirectScorer:
    required = {
        "hotpotqa-official-em-f1@2": score_hotpotqa,
        "triviaqa-official-alias-em-f1@2": score_triviaqa,
        "aime-official-integer@2": score_aime,
        "humaneval-official-pass1@2": score_humaneval,
    }
    expected = required.get(profile_id)
    if expected is None or SCORER_REGISTRY.get(profile_id) is not expected:
        raise ValueError("Protocol 13 static scorer registration differs")
    return expected


async def score_static_case(case: PrivateDirectCase, parsed: ParsedResponse) -> DirectScore:
    profile = {
        DirectBenchmark.HOTPOT_QA: "hotpotqa-official-em-f1@1",
        DirectBenchmark.TRIVIA_QA: "triviaqa-official-alias-em-f1@1",
        DirectBenchmark.MUSIQUE: "musique-official-em-f1@1",
        DirectBenchmark.NQ_OPEN: "nq-open-official-em-f1@1",
        DirectBenchmark.AIME_2026: "aime-official-integer@1",
        DirectBenchmark.MED_QA: "medqa-option@1",
        DirectBenchmark.GPQA_DIAMOND: "gpqa-diamond-option@1",
        DirectBenchmark.MATH_HARD: "math-hard-official-equivalence@2",
        DirectBenchmark.HUMAN_EVAL: "humaneval-official-pass1@1",
        DirectBenchmark.MIND2WEB: "mind2web-official-step@2",
    }.get(case.public_task.benchmark)
    if profile is None:
        raise ValueError(f"{case.public_task.benchmark.value} requires a non-static scorer")
    return await SCORER_REGISTRY[profile](case, parsed)
