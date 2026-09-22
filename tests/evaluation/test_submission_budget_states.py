"""One owner, one total ledger, explicit carriers and no repair after submission."""

import asyncio
from dataclasses import replace

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.submission_outcome import (
    SubmissionBudget,
    SubmissionOutcome,
    SubmissionState,
)
from tests.evaluation.test_integrity_broker_boundary import runtime


@pytest.mark.parametrize(
    ("has_final", "finish", "tokens", "calls", "state"),
    [
        (False, "length", 1000, 1, SubmissionState.LENGTH_WITH_BALANCE),
        (False, "length", 0, 1, SubmissionState.BUDGET_EXHAUSTED),
        (False, "stop", 1000, 0, SubmissionState.BUDGET_EXHAUSTED),
        (False, "stop", 1000, 1, SubmissionState.NO_FINAL_CARRIER),
        (True, "length", 0, 0, SubmissionState.UNIQUE_FINAL),
        (True, "stop", 500, 1, SubmissionState.UNIQUE_FINAL),
    ],
)
def test_submission_state_does_not_claim_compilable_or_correct_code(
    has_final, finish, tokens, calls, state
):
    outcome = SubmissionOutcome.after_response(
        has_final=has_final,
        finish_reason=finish,
        remaining_tokens=tokens,
        remaining_calls=calls,
    )
    assert outcome.state is state
    sealed = outcome.seal()
    assert sealed.preseal_state is state
    assert sealed.state is SubmissionState.SEALED
    with pytest.raises(ValueError):
        sealed.seal()


@pytest.mark.parametrize("reserve", [0, 1000])
@pytest.mark.parametrize("needs_clarification", [False, True])
def test_real_actor_reserve_is_inside_total_and_unused_after_valid_submission(
    tmp_path, reserve, needs_clarification
):
    entry = PublicTaskView.from_record(
        "code", "livecodebench", {"prompt": "Print the integer two."}
    )
    malformed = "<tool_call><function=submit_answer><parameter=answer>print(2)"
    complete = "```python\nprint(2)\n```"
    outputs = [malformed, complete] if needs_clarification else [complete]
    instance = runtime(tmp_path, entry, outputs, calls=2)
    limits = instance.config["budgets"][entry.benchmark]
    limits.update(total_output_tokens=12000, finalization_reserve_tokens=reserve)
    instance.config["decoding"][entry.benchmark]["max_new_tokens"] = 12000
    if needs_clarification:
        original = instance.generators[0].generate_evaluation

        async def generate(request, **kwargs):
            result = await original(request, **kwargs)
            return replace(
                result,
                finish_reason="length" if len(instance.generators[0].profiles) == 1 else "stop",
            )

        instance.generators[0].generate_evaluation = generate
    try:
        final = asyncio.run(instance.generate(entry, InferenceArm("single-owner"), "synthetic"))
        assert final.text == "print(2)"
        profiles = instance.generators[0].profiles
        assert len(profiles) == len(outputs)
        assert profiles[0].max_new_tokens == 12000 - reserve
        if needs_clarification:
            assert profiles[1].max_new_tokens == 12000 - len(malformed)
        assert final.completion_tokens == sum(map(len, outputs))
        assert final.intervention_counts["peer_model_calls"] == 0
        saved = instance.journal.model_outputs(("synthetic", "single-owner", "code"))
        assert all(row.participant == "owner" for row in saved)
        assert all(row.continuation_of_call_id is None for row in saved)
        assert saved[-1].submission_outcome["final_carrier_present"]
        instance.validate_candidate(
            instance.journal, entry, InferenceArm("single-owner"), "synthetic"
        )
    finally:
        instance.journal.close()


def test_original_uninterrupted_profile_does_not_silently_reserve_tokens():
    budget = SubmissionBudget(12000)
    assert budget.allowance(0, 0) == 12000
    assert budget.allowance(12000, 1) == 0
    assert budget.profile_id != SubmissionBudget(12000, 1000).profile_id
