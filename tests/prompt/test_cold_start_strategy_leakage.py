import pytest

from skillev.rollout.prompt_profiles import (
    InitialContextProfile,
    PromptFragmentKind,
    PublicPromptFragment,
    validate_initial_context_profile,
    validate_prompt_fragments,
)


def _fragment(kind: PromptFragmentKind) -> PublicPromptFragment:
    return PublicPromptFragment(kind, "source", "content")


def test_cold_start_allows_contract_but_not_strategy_or_demo() -> None:
    validate_prompt_fragments(
        (_fragment(PromptFragmentKind.TASK), _fragment(PromptFragmentKind.ACTION_SCHEMA)),
        profile=InitialContextProfile.SKILLEV_COLD_START,
    )
    for kind in (PromptFragmentKind.STRATEGY, PromptFragmentKind.PUBLIC_DEMONSTRATION):
        with pytest.raises(ValueError):
            validate_prompt_fragments(
                (_fragment(kind),), profile=InitialContextProfile.SKILLEV_COLD_START
            )


def test_gold_is_forbidden_everywhere_and_cold_start_has_no_skills() -> None:
    with pytest.raises(ValueError):
        validate_prompt_fragments(
            (_fragment(PromptFragmentKind.GOLD_DERIVED),),
            profile=InitialContextProfile.TRAINED_SKILLEV,
        )
    with pytest.raises(ValueError):
        validate_initial_context_profile(
            profile=InitialContextProfile.SKILLEV_COLD_START,
            active_skill_ids=("skill",),
            retrieved_skill_ids=(),
        )
