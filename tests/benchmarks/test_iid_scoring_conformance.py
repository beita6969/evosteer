"""Synthetic evaluator regressions; no licensed examples or model calls."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.qa_metrics import (
    best_alias_metrics,
    normalize_hotpotqa_answer,
    normalize_triviaqa_answer,
    score_hotpotqa_answers,
)
from skillev_private.evaluation.integrity_scoring import SealedHealthScorer, SealedQAScorer
from skillev_private.evaluation.result_contracts import (
    NativeMetricAggregate,
    NativeMetricValue,
    public_native_metric_values,
)

from skillev.evaluation.sealed_candidates import CandidateJournal, FinalCandidate
from skillev.rollout import TerminalEvaluatorError
from tests.benchmarks.test_protocol_v10_evaluators import HealthBenchGrade, _HealthGrader
from tests.benchmarks.test_protocol_v13_training_sessions import _record


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("North-East", "north east"),
        ("The_Alpha/Beta", "alpha beta"),
        ("D\u2019Alpha", "d alpha"),
        ("\u2018alpha\u2019\u00b4beta`", "alpha beta"),
    ],
)
def test_trivia_punctuation_is_a_separator_not_a_deleted_character(
    answer: str, expected: str
) -> None:
    assert normalize_triviaqa_answer(answer) == expected
    score = best_alias_metrics(answer, (expected,), normalize=normalize_triviaqa_answer)
    assert score.em == score.f1 == 1.0


def test_hotpot_and_trivia_keep_their_distinct_official_normalizers() -> None:
    assert normalize_hotpotqa_answer("north-east") == "northeast"
    assert normalize_triviaqa_answer("north-east") == "north east"
    assert score_hotpotqa_answers("yes", ("yes indeed",)).f1 == 0.0
    for normalizer in (normalize_triviaqa_answer, normalize_hotpotqa_answer):
        empty = best_alias_metrics("the", ("a",), normalize=normalizer)
        assert empty.em == 1.0
        assert empty.f1 == 0.0


def test_native_qa_projection_retains_em_and_f1_separately(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "qa.sqlite")
    journal.seal(FinalCandidate("run", "arm", "qa", "qwen", "final", "answer", "qa", 1, 1))
    scorer = SealedQAScorer(journal, "hotpotqa", ("private answer",), "fixture:qa")
    reward = asyncio.run(scorer.score("run", "arm", "qa"))
    metrics = {item.metric_name: item.value for item in public_native_metric_values(reward)}
    assert metrics["answer-exact-match"] == 0.0
    assert 0 < metrics["answer-f1"] < 1
    assert reward.success is False
    journal.close()


@pytest.mark.parametrize(
    ("raw", "negative", "success"), [(-0.5, 1, False), (0.9, 1, False), (0.9, 0, True)]
)
def test_health_raw_score_is_not_replaced_by_training_reward(
    raw: float, negative: int, success: bool, tmp_path: Path
) -> None:
    record = _record()
    task = replace(record.input, public_context={"benchmark_id": "healthbench"})
    journal = CandidateJournal(tmp_path / "health.sqlite")
    journal.seal(
        FinalCandidate(
            "run",
            "arm",
            task.task_id,
            "qwen-policy",
            "final",
            "Synthetic candidate",
            "natural-language-explicit-owner-v2",
            10,
            2,
        )
    )
    evaluator = SealedHealthScorer(
        journal, _HealthGrader(HealthBenchGrade(raw, negative)), task.environment_id
    )
    reward = asyncio.run(evaluator.score("run", "arm", task.task_id))
    metrics = public_native_metric_values(reward)
    assert {item.metric_name: item.value for item in metrics}["qwen-local-rubric-score"] == raw
    assert reward.value == max(0, min(1, raw))
    assert reward.success is success
    assert NativeMetricValue.from_value(metrics[0].to_value()) == metrics[0]
    journal.close()


def test_negative_native_moments_round_trip_without_clipping_each_record() -> None:
    aggregate = NativeMetricAggregate("qwen-local-rubric-score", 2, -0.5, 1.25)
    assert NativeMetricAggregate.from_value(aggregate.to_value()) == aggregate
    assert aggregate.value_sum == -0.5


def test_health_grader_failure_stays_an_infrastructure_failure(tmp_path: Path) -> None:
    task = _record().input
    journal = CandidateJournal(tmp_path / "health.sqlite")
    journal.seal(
        FinalCandidate(
            "run",
            "arm",
            task.task_id,
            "qwen-policy",
            "final",
            "Synthetic candidate",
            "natural-language-explicit-owner-v2",
            10,
            2,
        )
    )
    evaluator = SealedHealthScorer(
        journal, _HealthGrader(RuntimeError("synthetic transport failure")), task.environment_id
    )
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(evaluator.score("run", "arm", task.task_id))
    journal.close()
