import pytest

from skillev.evaluation.current_iid.protocol14.code_eval import (
    CodeExtractionStatus,
    CodeOutcomeKind,
    CodeTerminalOutcome,
)


def test_invalid_model_output_is_definitive_zero() -> None:
    outcome = CodeTerminalOutcome(
        CodeOutcomeKind.CANDIDATE_INVALID,
        CodeExtractionStatus.EMPTY_FINAL,
        None,
        None,
    )
    assert outcome.definitive
    assert not outcome.success


def test_infrastructure_is_not_a_definitive_failure() -> None:
    outcome = CodeTerminalOutcome(
        CodeOutcomeKind.SCORER_INFRASTRUCTURE,
        None,
        None,
        None,
    )
    assert not outcome.definitive
    with pytest.raises(ValueError):
        CodeTerminalOutcome(
            CodeOutcomeKind.SCORER_INFRASTRUCTURE,
            CodeExtractionStatus.SYNTAX_ERROR,
            False,
            False,
        )
