"""Withdrawn import path: historical results are readable, selectors cannot run.

The executable voting/fallback implementation is retained only in Git history.
There is no legacy-mode or environment-variable exemption to the owner-final rule.
"""

from .public_validation_guard import (
    CandidateAuthority,
    CandidateSelectionForbidden,
    PublicPairwiseDecision,
    PublicValidationResult,
    choose_order_invariant_pairwise_candidate,
    choose_publicly_dominant_candidate,
)

__all__ = [
    "CandidateAuthority",
    "CandidateSelectionForbidden",
    "PublicPairwiseDecision",
    "PublicValidationResult",
    "choose_order_invariant_pairwise_candidate",
    "choose_publicly_dominant_candidate",
]
