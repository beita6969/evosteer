from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from skillev_private.benchmarks.protocol_v10_evaluator import ProtocolV10TerminalEvaluator
from skillev_private.benchmarks.protocol_v10_official import (
    AppWorldNativeBackend,
    AppWorldScenarioAccumulator,
    AppWorldTaskGrade,
    HealthBenchGrade,
    HealthBenchNativeBackend,
    ProtocolV10StaticBackend,
    SpreadsheetBenchGrade,
    SpreadsheetBenchNativeBackend,
)
from skillev_private.benchmarks.static import (
    PrivateStaticBenchmarkCase,
    PrivateStaticTarget,
    StaticScoringRule,
    normalize_hotpotqa_answer,
    normalize_triviaqa_answer,
    parse_aime_answer,
)

from skillev.benchmarks import BenchmarkPublicItem
from skillev.contracts import SuccessRule, stable_hash
from skillev.experiments.protocol_v10 import BenchmarkV10, load_active_protocol_v10
from skillev.rollout import (
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)

ROOT = Path(__file__).parents[2]
PROTOCOL = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")


def _request(task_id: str, value: object) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-v10",
        task_id=task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(value),
        public_transcript_hash=stable_hash({"transcript": "public-only"}),
    )


def _qa_case(benchmark: BenchmarkV10) -> PrivateStaticBenchmarkCase:
    public = BenchmarkPublicItem(
        benchmark_id=benchmark.value,
        dataset_revision="private-fixture",
        split="training",
        task_id=f"{benchmark.value}/fixture",
        task_family=f"{benchmark.value}/qa",
        query="Provide a concise answer.",
        public_context={"answer_format": "short-text"},
    )
    return PrivateStaticBenchmarkCase(
        public=public,
        target=PrivateStaticTarget(
            task_id=public.task_id,
            scoring_rule=StaticScoringRule.TOKEN_F1,
            accepted_answers=("alpha beta gamma",),
        ),
    )


def test_qa_uses_f1_for_reward_and_exact_match_for_posterior() -> None:
    case = _qa_case(BenchmarkV10.HOTPOT_QA)
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.HOTPOT_QA),
        backend=ProtocolV10StaticBackend(
            case,
            BenchmarkV10.HOTPOT_QA,
            "official-hotpotqa-fixture",
        ),
    )

    reward = asyncio.run(
        evaluator.evaluate(_request(case.public.task_id, {"answer": "alpha gamma"}))
    )

    assert reward.value == pytest.approx(0.8)
    assert reward.success is False
    assert reward.success_rule is SuccessRule.TRUSTED_NATIVE_PROJECTION
    assert reward.native_payload["native_fields"] == {
        "answer-exact-match": 0.0,
        "answer-f1": pytest.approx(0.8),
    }
    assert "alpha beta gamma" not in str(reward.to_value())


def test_hotpot_and_trivia_normalization_conformance_cases() -> None:
    assert normalize_hotpotqa_answer(" The, Eiffel Tower! ") == "eiffel tower"
    assert normalize_triviaqa_answer("NEW_YORK") == "new york"
    trivia = PrivateStaticTarget(
        task_id="triviaqa/fixture",
        scoring_rule=StaticScoringRule.TOKEN_F1,
        accepted_answers=("New York City", "NYC"),
    )
    assert trivia.score("new_york_city") == 1.0
    assert trivia.exact_match("nyc!") == 1.0
    assert 0.0 < trivia.score("New York") < 1.0
    assert trivia.score("London") == 0.0


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("42", "42"),
        ("0", "0"),
        ("999", "999"),
        ("042", "42"),
        (r"\boxed{42}", None),
        ("answer is 42", None),
        ("42.0", None),
        ("1000", None),
        ("", None),
    ],
)
def test_aime_candidate_parser_is_one_canonical_integer(
    candidate: str,
    expected: str | None,
) -> None:
    assert parse_aime_answer(candidate) == expected


class _HealthGrader:
    verifier_version = "healthbench-fixture"

    def __init__(self, grade: HealthBenchGrade | Exception) -> None:
        self.grade_value = grade

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        del task_id, candidate_answer
        if isinstance(self.grade_value, Exception):
            raise self.grade_value
        return self.grade_value


def test_healthbench_requires_score_and_no_negative_rubric() -> None:
    task_id = "healthbench/fixture"
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.HEALTHBENCH),
        backend=HealthBenchNativeBackend(
            task_id,
            "healthbench/private",
            _HealthGrader(HealthBenchGrade(0.91, 1)),
        ),
    )

    reward = asyncio.run(evaluator.evaluate(_request(task_id, {"answer": "candidate"})))

    assert reward.value == pytest.approx(0.91)
    assert reward.success is False


def test_healthbench_negative_official_score_is_preserved_and_reward_is_clipped() -> None:
    task_id = "healthbench/fixture"
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.HEALTHBENCH),
        backend=HealthBenchNativeBackend(
            task_id,
            "healthbench/private",
            _HealthGrader(HealthBenchGrade(-0.25, 1)),
        ),
    )

    reward = asyncio.run(evaluator.evaluate(_request(task_id, {"answer": "candidate"})))

    assert reward.value == 0.0
    assert reward.success is False
    assert reward.native_payload["native_fields"]["official-rubric-score"] == -0.25


def test_healthbench_infrastructure_failure_aborts_instead_of_becoming_zero() -> None:
    task_id = "healthbench/fixture"
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.HEALTHBENCH),
        backend=HealthBenchNativeBackend(
            task_id,
            "healthbench/private",
            _HealthGrader(RuntimeError("private grader failure")),
        ),
    )

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(evaluator.evaluate(_request(task_id, {"answer": "candidate"})))


class _SpreadsheetOJ:
    verifier_version = "spreadsheetbench-oj-fixture"

    async def grade(self, task_id: str, submitted_workbook: object) -> SpreadsheetBenchGrade:
        del task_id, submitted_workbook
        return SpreadsheetBenchGrade(8, 9)


def test_spreadsheet_partial_case_pass_is_zero_and_not_success() -> None:
    task_id = "spreadsheetbench/fixture"
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.SPREADSHEETBENCH),
        backend=SpreadsheetBenchNativeBackend(
            task_id,
            "spreadsheetbench/private",
            _SpreadsheetOJ(),
        ),
    )

    reward = asyncio.run(evaluator.evaluate(_request(task_id, {"workbook": "private-handle"})))

    assert reward.value == 0.0
    assert reward.success is False


def test_appworld_sgc_rejects_incomplete_scenario_group() -> None:
    accumulator = AppWorldScenarioAccumulator("scenario-a", ("task-a", "task-b"))
    accumulator.record("task-a", AppWorldTaskGrade(1.0, True, "scenario-a"))

    with pytest.raises(RuntimeError):
        accumulator.aggregate()

    accumulator.record("task-b", AppWorldTaskGrade(0.5, False, "scenario-a"))
    aggregate = accumulator.aggregate()
    assert aggregate.average_task_goal_completion == pytest.approx(0.75)
    assert aggregate.scenario_goal_completion == 0.0


class _AppWorldOutcome:
    verifier_version = "appworld-fixture@1"

    def __init__(self, grade: AppWorldTaskGrade) -> None:
        self._grade = grade

    async def grade(self, task_id: str) -> AppWorldTaskGrade:
        del task_id
        return self._grade


def test_appworld_native_reward_preserves_scenario_for_group_aggregation() -> None:
    task_id = "appworld/task-a"
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.APPWORLD),
        backend=AppWorldNativeBackend(
            task_id=task_id,
            scenario_id="scenario-a",
            environment_id="appworld/private",
            outcome=_AppWorldOutcome(AppWorldTaskGrade(0.75, True, "scenario-a")),
        ),
    )

    reward = asyncio.run(evaluator.evaluate(_request(task_id, {"submitted": True})))

    assert reward.value == pytest.approx(0.75)
    assert reward.success is True
    assert reward.native_payload["native_fields"]["scenario-id"] == "scenario-a"


def test_appworld_rejects_a_grade_from_another_scenario() -> None:
    task_id = "appworld/task-a"
    evaluator = ProtocolV10TerminalEvaluator(
        protocol=PROTOCOL.benchmark(BenchmarkV10.APPWORLD),
        backend=AppWorldNativeBackend(
            task_id=task_id,
            scenario_id="scenario-a",
            environment_id="appworld/private",
            outcome=_AppWorldOutcome(AppWorldTaskGrade(1.0, True, "scenario-b")),
        ),
    )

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(evaluator.evaluate(_request(task_id, {"submitted": True})))
