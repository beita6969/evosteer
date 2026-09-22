"""Submission state is not transport completion, Python validity or task success."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class SubmissionState(StrEnum):
    GENERATING = "generating"
    LENGTH_WITH_BALANCE = "length-stop-with-balance"
    NO_FINAL_CARRIER = "response-without-final-carrier"
    UNIQUE_FINAL = "unique-final-carrier"
    BUDGET_EXHAUSTED = "budget-exhausted"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class SubmissionOutcome:
    state: SubmissionState
    response_received: bool
    final_carrier_present: bool
    finish_reason: str
    remaining_output_tokens: int
    remaining_model_calls: int
    preseal_state: SubmissionState | None = None

    @classmethod
    def after_response(
        cls, *, has_final: bool, finish_reason: str, remaining_tokens: int, remaining_calls: int
    ) -> SubmissionOutcome:
        state = (
            SubmissionState.UNIQUE_FINAL
            if has_final
            else SubmissionState.BUDGET_EXHAUSTED
            if remaining_tokens <= 0 or remaining_calls <= 0
            else SubmissionState.LENGTH_WITH_BALANCE
            if finish_reason == "length"
            else SubmissionState.NO_FINAL_CARRIER
        )
        return cls(
            state, True, has_final, finish_reason, max(0, remaining_tokens), max(0, remaining_calls)
        )

    def seal(self) -> SubmissionOutcome:
        if self.state in {SubmissionState.GENERATING, SubmissionState.SEALED}:
            raise ValueError("submission state cannot be sealed again or before a response")
        return replace(self, state=SubmissionState.SEALED, preseal_state=self.state)


@dataclass(frozen=True, slots=True)
class SubmissionBudget:
    total_tokens: int
    finalization_reserve_tokens: int = 0

    def __post_init__(self) -> None:
        if type(self.total_tokens) is not int or type(self.finalization_reserve_tokens) is not int:
            raise TypeError("submission budgets use whole tokens")
        if not 0 <= self.finalization_reserve_tokens < self.total_tokens:
            raise ValueError("finalization reserve must leave a positive first response")

    @property
    def profile_id(self) -> str:
        return (
            "first-response-finalization-reserve@1"
            if self.finalization_reserve_tokens
            else "uninterrupted-first-response-full-episode@1"
        )

    def allowance(self, used_tokens: int, calls_started: int) -> int:
        remaining = self.total_tokens - used_tokens
        return max(
            0, remaining - self.finalization_reserve_tokens if calls_started == 0 else remaining
        )
