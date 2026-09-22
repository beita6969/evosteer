"""Worker publication tests for closed, terminal domain failure classes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from skillev.contracts import stable_hash
from skillev.evolution import AuthoringFailedError
from skillev.rollout import TerminalEvaluatorError
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptFailed,
    AttemptFailureCode,
    AttemptFailureStage,
    AttemptRequest,
    EventAppendFailedError,
    EvolutionMutationFailedError,
    LibraryApplyFailedError,
    PartitionResetFailedError,
    UnpublishedAttemptBundle,
    read_attempt_outcome,
)
from skillev.runtime.attempt_builders import BuiltAttempt
from skillev.runtime.attempt_worker import execute_attempt_request
from skillev.scoring import ScoringDirection, TrajectoryScoringMemoryError


def _request(tmp_path: Path, *, attempt_id: str) -> AttemptRequest:
    bundle = UnpublishedAttemptBundle.create(tmp_path / attempt_id)
    return AttemptRequest(
        run_id="typed-failure-test-run",
        attempt_id=attempt_id,
        builder_kind=AttemptBuilderKind.FULL,
        exact_input_path=bundle.directory / "exact-input.json",
        exact_input_sha256=stable_hash({"attempt": attempt_id}),
        private_bundle_directory=bundle.directory,
    )


def _raising_builder(error: Exception) -> Callable[[AttemptRequest], BuiltAttempt]:
    def build(_: AttemptRequest) -> NoReturn:
        raise error

    return build


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (AuthoringFailedError("private authoring detail"), AttemptFailureCode.AUTHORING_FAILED),
        (
            EvolutionMutationFailedError("private mutation detail"),
            AttemptFailureCode.MUTATION_FAILED,
        ),
        (
            LibraryApplyFailedError("private library detail"),
            AttemptFailureCode.LIBRARY_APPLY_FAILED,
        ),
        (
            PartitionResetFailedError("private reset detail"),
            AttemptFailureCode.PARTITION_RESET_FAILED,
        ),
        (EventAppendFailedError("private event detail"), AttemptFailureCode.EVENT_APPEND_FAILED),
    ],
)
def test_known_domain_failure_publishes_only_its_typed_public_code(
    tmp_path: Path,
    error: Exception,
    code: AttemptFailureCode,
) -> None:
    request = _request(tmp_path, attempt_id=f"attempt-{code.value}")

    with pytest.raises(type(error)):
        asyncio.run(execute_attempt_request(request, build=_raising_builder(error)))

    outcome = read_attempt_outcome(request.private_bundle_directory / "outcome.json")
    assert isinstance(outcome, AttemptFailed)
    assert outcome.code is code
    assert outcome.stage is AttemptFailureStage.EXECUTION
    assert "private" not in outcome.public_message
    assert "private" not in outcome.exception_type


def test_terminal_evaluator_failure_has_dedicated_public_code_and_stage(tmp_path: Path) -> None:
    error = TerminalEvaluatorError("private evaluator path and cause")
    request = _request(tmp_path, attempt_id="attempt-terminal-evaluator")

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(execute_attempt_request(request, build=_raising_builder(error)))

    outcome = read_attempt_outcome(request.private_bundle_directory / "outcome.json")
    assert isinstance(outcome, AttemptFailed)
    assert outcome.code is AttemptFailureCode.TERMINAL_EVALUATOR_FAILED
    assert outcome.stage is AttemptFailureStage.TERMINAL_EVALUATION
    assert "path" not in outcome.public_message


def test_scoring_memory_failure_has_dedicated_public_code_and_stage(tmp_path: Path) -> None:
    error = TrajectoryScoringMemoryError(
        trajectory_id="private-trajectory-id",
        step_index=7,
        direction=ScoringDirection.FORWARD,
        prefix_token_count=12_345,
        action_token_count=512,
    )
    request = _request(tmp_path, attempt_id="attempt-scoring-memory")

    with pytest.raises(TrajectoryScoringMemoryError):
        asyncio.run(execute_attempt_request(request, build=_raising_builder(error)))

    outcome = read_attempt_outcome(request.private_bundle_directory / "outcome.json")
    assert isinstance(outcome, AttemptFailed)
    assert outcome.code is AttemptFailureCode.TRAINING_MEMORY_EXHAUSTED
    assert outcome.stage is AttemptFailureStage.TTB_SCORING
    assert "trajectory" not in outcome.public_message
