"""Synthetic official-metric and private native-evidence regression cases."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.musique_answer import AnswerMetric
from skillev_private.benchmarks.qa_metrics import (
    normalize_nq_open_answer,
    score_musique_answers,
    score_qa_answers,
)
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldStepResult
from skillev_private.evaluation.ood_luna_judge import export_judge_batch, render_judge_prompt
from skillev_private.evaluation.ood_scienceworld import OODScienceWorldEnvironment
from skillev_private.evaluation.ood_scoring import code_failure_kind, grade_code

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.integrity_results import scienceworld_terminal_metrics


def test_science_report_never_confuses_native_negative_scores_with_clipped_rewards():
    report = scienceworld_terminal_metrics(
        [{"native_final_score": value} for value in [-100, 0, 70, 100]]
    )
    assert report["native_final_score_mean"] == 17.5
    assert report["clipped_reward_mean"] == 0.425
    assert report["success_100"] == 0.25
    assert report["success_70"] == 0.5
    assert report["negative_final_scores"] == 1
    report = scienceworld_terminal_metrics([{"native_final_score": 100}, {}])
    assert report["count"] == 2
    assert report["missing_native_scores"] == 1
    assert report["native_final_score_mean"] is None


@pytest.mark.parametrize(
    ("prediction", "answers", "em", "f1"),
    [
        ("The BLUE-bird!", ("bluebird",), 1.0, 1.0),
        ("blue bird bird", ("blue bird",), 0.0, 0.8),
        ("blue", ("azure bird", "blue"), 1.0, 1.0),
        ("the", ("a",), 1.0, 1.0),
        ("", ("",), 1.0, 1.0),
        ("", ("blue",), 0.0, 0.0),
    ],
)
def test_musique_official_alias_and_empty_token_behavior(prediction, answers, em, f1):
    result = score_musique_answers(prediction, answers)
    assert (result.em, result.f1) == pytest.approx((em, f1))


def test_musique_accumulator_keeps_unrounded_question_mean_and_resets():
    metric = AnswerMetric()
    metric("blue", ["blue bird"])
    metric("wrong", ["blue bird"])
    assert metric.get_metric(reset=True) == pytest.approx((0, 1 / 3))
    assert metric.get_metric() == (0, 0)


def test_nq_em_does_not_use_substring_or_change_iid_empty_f1():
    result = score_qa_answers(
        "I think the blue bird is correct", ("blue bird",), normalizer=normalize_nq_open_answer
    )
    assert result.em == 0
    assert result.f1 > 0
    assert score_qa_answers("a", ("the",), normalizer=normalize_nq_open_answer).f1 == 0
    assert normalize_nq_open_answer("the\u0301 bird") == "the\u0301 bird"


@pytest.mark.parametrize(
    ("benchmark", "verdicts", "metadata", "passed", "kind"),
    [
        ("apps-introductory", [True, 2], {}, True, None),
        ("apps-introductory", [-1], {}, False, "runtime-error-or-timeout"),
        ("apps-introductory", [-2], {}, False, "compile-or-load-error"),
        ("apps-introductory", [True, False], {}, False, "wrong-answer"),
        ("livecodebench", [-2], {"error_code": -2}, False, "wrong-answer"),
        ("livecodebench", [-1], {"error_code": -3}, False, "time-limit-exceeded"),
        ("livecodebench", [-4], {"error_code": -4}, False, "runtime-or-load-error"),
    ],
)
def test_code_uses_positive_verdicts_and_benchmark_specific_error_codes(
    tmp_path, monkeypatch, benchmark, verdicts, metadata, passed, kind
):
    checker = tmp_path / "checker.py"
    checker.write_text("# synthetic checker\n")
    observed = []

    def run(command, **kwargs):
        observed.append(json.loads(kwargs["input"]))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"results": verdicts, "metadata": metadata}),
            stderr="",
        )

    monkeypatch.setattr("skillev_private.evaluation.ood_scoring.subprocess.run", run)
    result = grade_code(
        benchmark,
        "print(1)",
        {"input_output": {"inputs": [""] * len(verdicts), "outputs": ["1"] * len(verdicts)}},
        {"checker_path": str(checker), "test_timeout_seconds": 2},
        ActorSandbox.current(tmp_path),
    )
    assert result["passed"] is passed
    assert code_failure_kind(benchmark, result) == kind
    assert observed[0]["test_timeout"] == result["test_timeout_seconds"] == 2


@pytest.mark.parametrize("verdicts", [[], None, [True]])
def test_missing_native_test_results_are_infrastructure_failures(tmp_path, monkeypatch, verdicts):
    checker = tmp_path / "checker.py"
    checker.write_text("# synthetic\n")
    monkeypatch.setattr(
        "skillev_private.evaluation.ood_scoring.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({"results": verdicts}), stderr=""
        ),
    )
    with pytest.raises(RuntimeError):
        grade_code(
            "apps-introductory",
            "print(1)",
            {"input_output": {"inputs": ["", ""], "outputs": ["1", "1"]}},
            {"checker_path": str(checker)},
            ActorSandbox.current(tmp_path),
        )


@pytest.mark.parametrize(
    ("score", "terminal", "steps", "reward", "reason"),
    [
        (-10, True, 1, 0, "native-negative-score"),
        (100, True, 1, 1, "score-100"),
        (45, True, 1, 0.45, "environment-terminal-unspecified"),
        (45, False, 2, 0.45, "runner-horizon"),
        (45, False, 1, 0.45, "owner-stopped-or-budget-exhausted"),
    ],
)
def test_science_raw_final_score_is_private_and_not_done_or_peak(
    score, terminal, steps, reward, reason
):
    native = SimpleNamespace(
        step=lambda action: OfficialScienceWorldStepResult("Public observation", score, terminal)
    )
    env = OODScienceWorldEnvironment(native, "synthetic", 0, 0, 2)
    # A previous high score must not replace a lower final score.
    env._raw_score, env._score = 90, 0.9
    for _ in range(steps):
        public_step = asyncio.run(env.step("look around"))
    outcome = asyncio.run(env.outcome())
    assert outcome.native_final_score == score
    assert outcome.reward == reward
    assert outcome.success is (score == 100)
    assert outcome.termination_reason == reason
    assert outcome.terminated_by_horizon is (reason == "runner-horizon")
    assert "native_final_score" not in asdict(public_step)
    assert asdict(outcome)["native_final_score"] == score


def test_science_without_native_step_reports_unknown_raw_score():
    env = OODScienceWorldEnvironment(SimpleNamespace(), "synthetic", 0, 0, 2)
    outcome = asyncio.run(env.outcome())
    assert outcome.native_final_score is None
    assert outcome.reward == 0
    assert not outcome.terminated_by_horizon


def test_omni_export_retains_exact_template_and_one_pass_rendering():
    template = "Question {{Problem}}\nReference {{Reference Answer}}\nStudent {{Solution}}"
    rendered = render_judge_prompt(
        template, problem="literal {{Solution}}", answer="4", candidate="final 4"
    )
    assert "literal {{Solution}}" in rendered
    source = SimpleNamespace(
        panel=SimpleNamespace(
            entries=[SimpleNamespace(benchmark="omni-math", task_id="synthetic")]
        ),
        targets={"synthetic": {"problem": "two plus two", "answer": "4"}},
    )
    reader = SimpleNamespace(get=lambda *args: SimpleNamespace(owner_call_id="c", text="final 4"))
    exported = export_judge_batch(reader, source, "run", "arm", template=template)
    assert exported["template_text"] == template
    assert exported["planned_count"] == 1
    assert "Student final 4" in exported["records"][0]["judge_prompt"]


def test_omni_incomplete_template_cannot_silently_drop_reference():
    with pytest.raises(ValueError):
        render_judge_prompt("{{Problem}}", problem="two plus two", answer="4", candidate="4")
