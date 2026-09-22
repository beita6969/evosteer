from __future__ import annotations

from dataclasses import fields

import pytest

from skillev.evaluation import (
    EvaluationStatus,
    TerminalRewardAdapter,
    TerminalRewardSignal,
    TrustedEvaluatorOutcome,
)


def test_reward_adapter_writes_only_scalar_side_channel() -> None:
    outcome = TrustedEvaluatorOutcome(
        trajectory_id="trajectory-1",
        status=EvaluationStatus.COMPLETED,
        scientific_success=True,
        score=0.75,
    )
    side_channel: dict[str, float] = {}
    signal = TerminalRewardAdapter().write_side_channel(outcome, side_channel)

    assert signal.value == 0.75
    assert side_channel[signal.trajectory_id] == signal.value
    assert all(
        fragment not in field.name
        for field in fields(TerminalRewardSignal)
        for fragment in ("answer", "diagnostic", "history", "prompt", "truth")
    )


def test_noncompleted_evaluation_receives_no_scientific_reward() -> None:
    outcome = TrustedEvaluatorOutcome(
        trajectory_id="trajectory-2",
        status=EvaluationStatus.INVALID_SUBMISSION,
        scientific_success=False,
        score=0.0,
    )
    assert TerminalRewardAdapter().adapt(outcome).value == 0


def test_side_channel_is_append_only_per_trajectory() -> None:
    outcome = TrustedEvaluatorOutcome(
        trajectory_id="trajectory-3",
        status=EvaluationStatus.COMPLETED,
        scientific_success=False,
        score=0.0,
    )
    adapter = TerminalRewardAdapter()
    destination: dict[str, float] = {}
    adapter.write_side_channel(outcome, destination)
    with pytest.raises(KeyError):
        adapter.write_side_channel(outcome, destination)


def test_evaluator_error_aborts_without_inserting_a_zero_reward() -> None:
    outcome = TrustedEvaluatorOutcome(
        "failed-evaluation", EvaluationStatus.EVALUATOR_ERROR, False, 0.0
    )
    destination = {"completed-trajectory": 0.5}
    with pytest.raises(RuntimeError):
        TerminalRewardAdapter().write_side_channel(outcome, destination)
    assert destination == {"completed-trajectory": 0.5}


@pytest.mark.parametrize("reward", [0.1, 1.0])
def test_invalid_submission_reward_is_not_a_configurable_score_boost(reward) -> None:
    with pytest.raises(ValueError):
        TerminalRewardAdapter(invalid_reward=reward)
