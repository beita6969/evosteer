"""An actor cannot replace the executed owner's answer or its frozen identity."""

from dataclasses import asdict, replace

import pytest

from skillev.evaluation.integrity_final_validation import (
    validate_final_identity,
    validate_owner_projection,
)
from skillev.evaluation.owner_final import project_owner_final
from skillev.evaluation.sealed_candidates import FinalCandidate
from skillev.evaluation.step0_completion import StepZeroTerminalMode


def candidate():
    submission = project_owner_final(
        StepZeroTerminalMode.AIME_INTEGER,
        "Final answer: 42",
        owner_id="owner",
        message_id="case:owner-final",
    )
    return FinalCandidate(
        "run",
        "A2",
        "case",
        "frozen-policy",
        "case:owner-final",
        submission.payload,
        "frozen-parser",
        1,
        1,
        submission=asdict(submission),
    )


@pytest.mark.parametrize(
    "change",
    [{"policy_id": "other-policy"}, {"parser_id": "other-parser"}, {"arm_id": "other-arm"}],
)
def test_final_must_match_the_executed_identity(change):
    with pytest.raises(ValueError):
        validate_final_identity(
            replace(candidate(), **change),
            expected_scope=("run", "A2", "case"),
            expected_policy="frozen-policy",
            expected_parser="frozen-parser",
            expected_message_id="case:owner-final",
        )


def test_projection_must_match_actual_owner_generation_not_actor_claimed_text():
    validate_owner_projection(
        candidate(),
        raw_response="Final answer: 42",
        participant="owner",
        mode=StepZeroTerminalMode.AIME_INTEGER,
    )
    with pytest.raises(ValueError):
        validate_owner_projection(
            candidate(),
            raw_response="Final answer: 12",
            participant="owner",
            mode=StepZeroTerminalMode.AIME_INTEGER,
        )
    with pytest.raises(ValueError):
        validate_owner_projection(
            candidate(),
            raw_response="Final answer: 42",
            participant="solver",
            mode=StepZeroTerminalMode.AIME_INTEGER,
        )
