"""Content-agnostic exact-match evaluator with a private answer-key object.

No benchmark instances or keys are bundled with this source module.
"""

from __future__ import annotations

from dataclasses import dataclass

from skillev.evaluation.outcomes import EvaluationStatus, TrustedEvaluatorOutcome

from .result import PrivateEvaluationResult


@dataclass(frozen=True, slots=True)
class PrivateAnswerEntry:
    instance_id: str
    valid_options: frozenset[str]
    correct_option: str

    def __post_init__(self) -> None:
        if (
            not self.instance_id
            or not self.valid_options
            or self.correct_option not in self.valid_options
        ):
            raise ValueError("private answer entry is inconsistent")


@dataclass(frozen=True, slots=True)
class PrivateAnswerKey:
    entries: tuple[PrivateAnswerEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("private answer key cannot be empty")
        identifiers = [item.instance_id for item in self.entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("private answer entries must have unique identities")


def evaluate_exact_match(
    *,
    trajectory_id: str,
    instance_id: str,
    selected_option: str,
    key: PrivateAnswerKey,
) -> PrivateEvaluationResult:
    entries = {item.instance_id: item for item in key.entries}
    entry = entries.get(instance_id)
    if entry is None or selected_option not in entry.valid_options:
        return PrivateEvaluationResult(
            TrustedEvaluatorOutcome(
                trajectory_id,
                EvaluationStatus.INVALID_SUBMISSION,
                False,
                0.0,
            ),
            (("submission_valid", False),),
        )
    success = selected_option == entry.correct_option
    return PrivateEvaluationResult(
        TrustedEvaluatorOutcome(
            trajectory_id,
            EvaluationStatus.COMPLETED,
            success,
            float(success),
        ),
        (("exact_match", success), ("submission_valid", True)),
    )


__all__ = [
    "PrivateAnswerEntry",
    "PrivateAnswerKey",
    "evaluate_exact_match",
]
