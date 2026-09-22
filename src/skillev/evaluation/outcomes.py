"""Answer-free values permitted to cross the evaluator/trainer boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class EvaluationStatus(StrEnum):
    COMPLETED = "COMPLETED"
    INVALID_SUBMISSION = "INVALID_SUBMISSION"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"


@dataclass(frozen=True, slots=True)
class TrustedEvaluatorOutcome:
    """Minimal evaluator projection consumed outside the private process.

    It intentionally carries no task text, answer, protected parameter,
    diagnostic, or model-visible feedback.
    """

    trajectory_id: str
    status: EvaluationStatus
    scientific_success: bool
    score: float

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory identity is required")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("evaluator score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("evaluator score must lie in [0, 1]")
        if self.status is not EvaluationStatus.COMPLETED and (
            self.scientific_success or score != 0
        ):
            raise ValueError("non-completed evaluation cannot claim scientific credit")


@dataclass(frozen=True, slots=True)
class TerminalRewardSignal:
    """Scalar side-channel value keyed by trajectory, never a history event."""

    trajectory_id: str
    value: float

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory identity is required")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError("terminal reward must be numeric")
        reward = float(self.value)
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("terminal reward must lie in [0, 1]")


__all__ = [
    "EvaluationStatus",
    "TerminalRewardSignal",
    "TrustedEvaluatorOutcome",
]
