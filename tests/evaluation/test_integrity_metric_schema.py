"""Synthetic native-score contracts; no licensed task or evaluator data."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from skillev_private.benchmarks.code_math import (
    CodeExecutionInfrastructureError,
    CodeExecutionResult,
    CodeExecutionStatus,
)
from skillev_private.evaluation import integrity_native_scoring

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.input_metric_contracts import CONTRACTS
from skillev.evaluation.integrity_metric_schema import (
    NATIVE_VERIFIER_VERSIONS,
    aggregate_secondary_metrics,
)
from skillev.evaluation.integrity_results import EvaluationStatus, NativeScore, exact_panel_join
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin, FinalCandidate


def score(
    task_id: str,
    benchmark: str,
    value: float,
    secondary_metrics: dict[str, float],
    *,
    grader_used: bool | None = None,
) -> NativeScore:
    primary = {
        "hotpotqa": "answer-f1",
        "triviaqa": "answer-f1",
        "aime-2026": "accuracy",
        "healthbench": "qwen-local-rubric-score",
        "webshop": "native-reward",
        "alfworld": "success",
        "mbpp-plus": "base-plus-pass-at-1",
        "humaneval": "pass-at-1",
    }[benchmark]
    return NativeScore(
        task_id,
        benchmark,
        primary,
        value,
        secondary_metrics=secondary_metrics,
        verifier_version=NATIVE_VERIFIER_VERSIONS[benchmark],
        grader_used=grader_used,
    )


def test_qa_primary_is_f1_and_em_is_required_binary_evidence() -> None:
    with pytest.raises(ValueError):
        score("qa", "hotpotqa", 0.5, {"answer-exact-match": 0.5, "answer-f1": 0.5})
    with pytest.raises(ValueError):
        score("qa", "triviaqa", 0.5, {"answer-exact-match": 0.0})
    accepted = score(
        "qa",
        "hotpotqa",
        0.5,
        {"answer-exact-match": 0.0, "answer-f1": 0.5},
    )
    assert accepted.metric == "answer-f1"


def test_interactive_and_code_metric_relations_cannot_be_substituted() -> None:
    with pytest.raises(ValueError):
        score("web", "webshop", 0.6, {"success": 1.0})
    with pytest.raises(ValueError):
        score("alf", "alfworld", 1.0, {"success": 0.0})
    with pytest.raises(ValueError):
        score("mbpp", "mbpp-plus", 1.0, {"base-pass": 1.0, "plus-pass": 0.0})

    assert score("web", "webshop", 1.0, {"success": 1.0}).value == 1.0
    assert score("alf", "alfworld", 0.0, {"success": 0.0}).value == 0.0
    assert score("mbpp", "mbpp-plus", 1.0, {"base-pass": 1.0, "plus-pass": 1.0}).value == 1.0


def test_health_keeps_negative_records_and_clips_only_the_complete_mean() -> None:
    scores = (
        score(
            "negative",
            "healthbench",
            -0.5,
            {"triggered-negative-rubric-count": 1.0},
            grader_used=True,
        ),
        score(
            "positive",
            "healthbench",
            1.0,
            {"triggered-negative-rubric-count": 0.0},
            grader_used=True,
        ),
    )
    joined = exact_panel_join(
        ("negative", "positive"),
        scores,
        expected_verifier=NATIVE_VERIFIER_VERSIONS["healthbench"],
    )
    assert sum(item.value for item in joined) / len(joined) == 0.25


def test_blank_health_candidate_is_not_sent_to_the_grader(tmp_path: Path, monkeypatch) -> None:
    journal = CandidateJournal(tmp_path / "candidates.sqlite")
    scope = ("run", "A2", "health")
    journal.seal(
        FinalCandidate(
            *scope,
            "policy",
            "final",
            " \n",
            CONTRACTS["healthbench"].parser,
            0,
            0,
        )
    )

    def unexpected_grader(*args: object, **kwargs: object) -> object:
        raise AssertionError("blank HealthBench candidates must not invoke the grader")

    monkeypatch.setattr(integrity_native_scoring, "_grade_health", unexpected_grader)
    result = asyncio.run(
        integrity_native_scoring.score_native(
            journal,
            scope,
            "healthbench",
            {},
            settings={},
            sandbox=cast(ActorSandbox, None),
            mbpp_sandbox=cast(ActorSandbox, None),
        )
    )
    assert result.status is EvaluationStatus.CANDIDATE_FAILURE
    assert result.secondary_metrics == {"triggered-negative-rubric-count": 0.0}
    assert result.grader_used is False
    journal.close()


def test_interactive_native_score_reads_only_environment_owned_outcomes(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "candidates.sqlite")
    scope = ("run", "A2", "web")
    journal.seal(
        FinalCandidate(
            *scope,
            "policy",
            "final",
            "submit",
            CONTRACTS["webshop"].parser,
            1,
            1,
        )
    )
    outcome = {"reward": 0.5, "success": False, "terminal_reached": True}
    journal.record(scope, "native-outcome", outcome, origin=EventOrigin.ENVIRONMENT)
    result = asyncio.run(
        integrity_native_scoring.score_native(
            journal,
            scope,
            "webshop",
            {},
            settings={},
            sandbox=cast(ActorSandbox, None),
            mbpp_sandbox=cast(ActorSandbox, None),
        )
    )
    assert result.value == 0.5
    assert result.secondary_metrics == {"success": 0.0}
    journal.close()


def test_originless_interactive_outcome_is_not_native_evidence(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "candidates.sqlite")
    scope = ("run", "A2", "web")
    journal.seal(
        FinalCandidate(
            *scope,
            "policy",
            "final",
            "submit",
            CONTRACTS["webshop"].parser,
            1,
            1,
        )
    )
    journal.record(scope, "native-outcome", {"reward": 1.0, "success": True})
    with pytest.raises(RuntimeError):
        asyncio.run(
            integrity_native_scoring.score_native(
                journal,
                scope,
                "webshop",
                {},
                settings={},
                sandbox=cast(ActorSandbox, None),
                mbpp_sandbox=cast(ActorSandbox, None),
            )
        )
    journal.close()


def test_exact_join_requires_one_concrete_frozen_verifier_and_full_columns() -> None:
    first = score("one", "webshop", 0.25, {"success": 0.0})
    second = score("two", "webshop", 1.0, {"success": 1.0})
    with pytest.raises(ValueError):
        exact_panel_join(("one", "two"), (first, second), expected_verifier="unspecified")
    with pytest.raises(ValueError):
        exact_panel_join(
            ("one", "two"),
            (first, second),
            expected_verifier=NATIVE_VERIFIER_VERSIONS["alfworld"],
        )

    second.secondary_metrics.pop("success")
    with pytest.raises(ValueError):
        aggregate_secondary_metrics((first, second))


def code_candidate(tmp_path, benchmark):
    journal = CandidateJournal(tmp_path / "code.sqlite")
    scope = ("synthetic", "A2", "code")
    journal.seal(
        FinalCandidate(
            *scope, "policy", "final", "def f():\n    return 1\n", CONTRACTS[benchmark].parser, 1, 1
        )
    )
    return journal, scope


@pytest.mark.parametrize("status", list(CodeExecutionStatus))
def test_humaneval_retains_native_candidate_failure_category(tmp_path, monkeypatch, status):
    journal, scope = code_candidate(tmp_path, "humaneval")

    class Backend:
        def __init__(self, **kwargs):
            pass

        async def run(self, request):
            return CodeExecutionResult(status)

    monkeypatch.setattr(integrity_native_scoring, "IsolatedHumanEvalExecutionBackend", Backend)
    result = asyncio.run(
        integrity_native_scoring.score_native(
            journal,
            scope,
            "humaneval",
            {"prompt": "def f():\n    pass\n", "test": "synthetic test", "entry_point": "f"},
            settings={},
            sandbox=SimpleNamespace(command=lambda: ("fixture",)),
            mbpp_sandbox=None,
        )
    )
    assert result.value == float(status is CodeExecutionStatus.PASSED)
    assert result.failure_kind == (None if status is CodeExecutionStatus.PASSED else status.value)
    assert NativeScore.from_value(asdict(result)) == result
    journal.close()


def test_humaneval_infrastructure_failure_produces_no_native_score(tmp_path, monkeypatch):
    journal, scope = code_candidate(tmp_path, "humaneval")

    class Backend:
        def __init__(self, **kwargs):
            pass

        async def run(self, request):
            raise CodeExecutionInfrastructureError("synthetic unavailable executor")

    monkeypatch.setattr(integrity_native_scoring, "IsolatedHumanEvalExecutionBackend", Backend)
    with pytest.raises(CodeExecutionInfrastructureError):
        asyncio.run(
            integrity_native_scoring.score_native(
                journal,
                scope,
                "humaneval",
                {"prompt": "def f():\n    pass\n", "test": "synthetic test", "entry_point": "f"},
                settings={},
                sandbox=SimpleNamespace(command=lambda: ("fixture",)),
                mbpp_sandbox=None,
            )
        )
    journal.close()


@pytest.mark.parametrize(("base", "plus"), [("true", "false"), (1, 0), (True, None)])
def test_evalplus_nonboolean_verdict_is_infrastructure_not_a_passing_candidate(
    tmp_path, monkeypatch, base, plus
):
    journal, scope = code_candidate(tmp_path, "mbpp-plus")
    monkeypatch.setattr(
        integrity_native_scoring,
        "_grade_mbpp",
        lambda *args: {"base_passed": base, "plus_passed": plus},
    )
    with pytest.raises(TypeError):
        asyncio.run(
            integrity_native_scoring.score_native(
                journal,
                scope,
                "mbpp-plus",
                {},
                settings={"mbpp-plus": {}},
                sandbox=None,
                mbpp_sandbox=None,
            )
        )
    journal.close()
