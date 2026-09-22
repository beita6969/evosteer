"""Non-executable compatibility API for withdrawn candidate selection.

Public examples and rubric-free preferences do not authorize picking between
policies. Active and historical imports both reject selection and fallback;
the evaluated owner must submit its own final. Historical data types remain
readable without retaining the old executable selection behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn


class CandidateSelectionForbidden(RuntimeError):  # noqa: N818 -- explicit policy boundary
    """The project forbids choosing or falling back between candidate answers."""


class CandidateAuthority(StrEnum):
    FROZEN_REFERENCE = "frozen-reference"
    LEARNED_FORWARD = "learned-forward"


class PublicPairwiseDecision(StrEnum):
    """One rubric-free comparison made from model-visible evidence only."""

    FROZEN_REFERENCE = "frozen-reference"
    LEARNED_FORWARD = "learned-forward"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class PublicValidationResult:
    """Aggregate result of executing only model-visible validation checks."""

    passed: int
    total: int

    def __post_init__(self) -> None:
        if type(self.passed) is not int or type(self.total) is not int:
            raise TypeError("public validation counts must be integers")
        if self.total <= 0 or not 0 <= self.passed <= self.total:
            raise ValueError("public validation counts are inconsistent")

    @property
    def complete(self) -> bool:
        return self.passed == self.total


def choose_publicly_dominant_candidate(
    *,
    frozen_reference: PublicValidationResult,
    learned_forward: PublicValidationResult,
) -> NoReturn:
    """Reject selection regardless of either candidate's public-check results."""

    del frozen_reference, learned_forward
    raise CandidateSelectionForbidden(
        "Candidate selection is withdrawn: evaluate the policy's own single final answer."
    )


def choose_order_invariant_pairwise_candidate(
    decisions: tuple[PublicPairwiseDecision, ...],
) -> NoReturn:
    """Reject label-swapped selection, including unanimous votes and ties."""

    del decisions
    raise CandidateSelectionForbidden(
        "Label-swapped voting and baseline fallback are forbidden, including legacy entry points."
    )


__all__ = [
    "CandidateAuthority",
    "CandidateSelectionForbidden",
    "PublicPairwiseDecision",
    "PublicValidationResult",
    "choose_order_invariant_pairwise_candidate",
    "choose_publicly_dominant_candidate",
]
