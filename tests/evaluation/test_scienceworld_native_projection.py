"""Raw environment scores must survive every generic native-result boundary."""

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from skillev_private.evaluation.ood_scoring import score_ood

from scripts.reproject_scienceworld_scores import reproject
from skillev.evaluation.integrity_results import EvaluationStatus, NativeScore


def reader_for(outcomes):
    return SimpleNamespace(
        get=lambda *scope: SimpleNamespace(episode_id=scope[-1]),
        traces=lambda scope, stage, **kwargs: (outcomes[scope[-1]],),
    )


@pytest.mark.parametrize("raw", [-100, -10, 0, 45, 100])
@pytest.mark.parametrize(
    "reason", ["environment-terminal-unspecified", "owner-stopped-or-budget-exhausted"]
)
def test_native_score_and_learning_reward_are_distinct(raw, reason):
    outcome = {"native_final_score": raw, "termination_reason": reason, "reward": max(0, raw) / 100}
    score = asyncio.run(
        score_ood(
            reader_for({"case": outcome}),
            ("run", "arm", "case"),
            "scienceworld",
            {},
            settings={},
            sandbox=None,
            diagnostics=lambda value: None,
        )
    )
    assert score.value == raw / 100
    assert score.metric == "native-final-score"
    assert score.status is EvaluationStatus.SCORED
    assert score.secondary_metrics["zero-clipped-learning-reward"] == max(0, raw) / 100
    assert score.secondary_metrics["success"] == float(raw == 100)
    assert NativeScore.from_value(asdict(score)) == score


@pytest.mark.parametrize("raw", [None, "0", True, float("nan"), float("inf")])
def test_missing_or_invalid_native_evidence_cannot_be_filled_from_reward(raw):
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(
            score_ood(
                reader_for(
                    {
                        "case": {
                            "native_final_score": raw,
                            "reward": 0,
                            "termination_reason": "budget",
                        }
                    }
                ),
                ("run", "arm", "case"),
                "scienceworld",
                {},
                settings={},
                sandbox=None,
                diagnostics=lambda value: None,
            )
        )


def test_raw_negative_pair_aggregates_to_zero_without_replacing_history():
    outcomes = {
        str(i): {"native_final_score": score, "termination_reason": "environment-terminal"}
        for i, score in enumerate([100, -100])
    }
    report = reproject(reader_for(outcomes), ("0", "1"), "old-run", "old-arm")
    assert report["value"] == 0
    assert report["terminal_metrics"]["native_final_score_mean"] == 0
    assert report["terminal_metrics"]["clipped_reward_mean"] == 0.5
    assert report["terminal_metrics"]["success_100"] == 0.5
    assert report["source_records_modified"] is False
    assert report["new_model_calls"] == report["new_environment_steps"] == 0


def test_legacy_clipped_score_is_readable_but_not_relabelled():
    old = NativeScore(
        "case",
        "scienceworld",
        "native-reward",
        0,
        secondary_metrics={"success": 0},
        verifier_version="scienceworld-official-final-score-with-raw@ood2",
    )
    assert NativeScore.from_value(asdict(old)).metric == "native-reward"
    assert NativeScore.from_value(asdict(old)).value == 0
