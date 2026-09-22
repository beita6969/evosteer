"""Typed task outcomes for answer-isolated TriviaQA retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class TriviaTermination(StrEnum):
    COMPLETED = "completed"
    CANDIDATE_INVALID = "candidate-invalid"
    BUDGET_EXHAUSTED = "budget-exhausted"
    GENERATION_INFRASTRUCTURE = "generation-infrastructure"
    RETRIEVAL_INFRASTRUCTURE = "retrieval-infrastructure"


@dataclass(frozen=True, slots=True)
class TriviaTaskResult:
    task_id: str
    termination: TriviaTermination
    answer: str | None
    searches: int
    reads: int
    rejected_actions: int
    em: Decimal | None
    f1: Decimal | None
    finish_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("Trivia task result requires an ID")
        if min(self.searches, self.reads, self.rejected_actions) < 0:
            raise ValueError("Trivia task counters must be non-negative")
        if self.termination is TriviaTermination.COMPLETED:
            if not self.answer or self.em is None or self.f1 is None:
                raise ValueError("completed Trivia task requires answer and metrics")
        elif self.em is not None or self.f1 is not None:
            raise ValueError("non-completed Trivia task cannot carry scored metrics")


__all__ = ["TriviaTaskResult", "TriviaTermination"]
