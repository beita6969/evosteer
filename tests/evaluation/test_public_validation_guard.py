"""Both import paths reject selection, even when public checks favor the LoRA."""

from itertools import product

import pytest

from skillev.evaluation import legacy_candidate_selection, public_validation_guard
from skillev.evaluation.public_validation_guard import (
    CandidateSelectionForbidden,
    PublicPairwiseDecision,
    PublicValidationResult,
)


@pytest.mark.parametrize("module", [public_validation_guard, legacy_candidate_selection])
@pytest.mark.parametrize("passed", list(product(range(4), repeat=2)))
def test_no_public_example_result_can_select_or_fall_back(module, passed):
    with pytest.raises(CandidateSelectionForbidden):
        module.choose_publicly_dominant_candidate(
            frozen_reference=PublicValidationResult(passed=passed[0], total=3),
            learned_forward=PublicValidationResult(passed=passed[1], total=3),
        )


@pytest.mark.parametrize("module", [public_validation_guard, legacy_candidate_selection])
@pytest.mark.parametrize("decisions", [(), *product(PublicPairwiseDecision, repeat=2)])
def test_no_label_order_preferences_or_ties_can_select_or_fall_back(module, decisions):
    with pytest.raises(CandidateSelectionForbidden):
        module.choose_order_invariant_pairwise_candidate(decisions)


def test_historical_validation_records_remain_readable_without_selection():
    assert PublicValidationResult(passed=2, total=2).complete
    assert not PublicValidationResult(passed=1, total=2).complete
    with pytest.raises(ValueError):
        PublicValidationResult(passed=2, total=1)
    with pytest.raises(CandidateSelectionForbidden):
        legacy_candidate_selection.choose_publicly_dominant_candidate(
            frozen_reference=PublicValidationResult(passed=1, total=1),
            learned_forward=PublicValidationResult(passed=1, total=2),
        )
