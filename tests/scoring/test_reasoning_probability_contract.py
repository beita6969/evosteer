from dataclasses import replace

import pytest

from skillev.evaluation.direct_baseline import DirectDecodingProfile
from skillev.evaluation.step0_conditions import (
    ReasoningAuthority,
    ReasoningAuthorityMode,
    matched_step_zero_condition,
)


def _direct_profile() -> DirectDecodingProfile:
    return DirectDecodingProfile(
        profile_id="benchmark@1",
        enable_thinking=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_new_tokens=256,
        seed=42,
    )


def test_unscored_reasoner_is_frozen_adapter_free_and_deterministic() -> None:
    condition = matched_step_zero_condition(
        _direct_profile(),
        condition_id="step0@1",
        reasoning_tokens=1024,
    )
    assert condition.reasoning.sampling_mode == "greedy"
    assert condition.reasoning.temperature == 0.0
    assert condition.reasoning.top_k == 1
    assert condition.reasoning_authority.adapter_active is False
    assert condition.reasoning_authority.probability_in_forward_edge is False


def test_unscored_reasoner_contract_rejects_a_trainable_authority() -> None:
    authority = ReasoningAuthority(
        ReasoningAuthorityMode.FROZEN_DETERMINISTIC,
        "qwen-frozen",
        "greedy@1",
    )
    with pytest.raises(ValueError):
        replace(authority, adapter_active=True)


def test_benchmark_matched_reasoner_copies_the_public_thinking_profile() -> None:
    direct = replace(
        _direct_profile(),
        enable_thinking=True,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_new_tokens=81_920,
    )
    condition = matched_step_zero_condition(
        direct,
        condition_id="aime-step0@1",
        reasoning_tokens=81_920,
        reasoning_authority_mode=ReasoningAuthorityMode.FROZEN_BENCHMARK_MATCHED,
        deterministic_action=True,
    )

    assert condition.reasoning.enable_thinking is True
    assert condition.reasoning.sampling_mode == direct.sampling_mode
    assert condition.reasoning.temperature == direct.temperature
    assert condition.reasoning.top_p == direct.top_p
    assert condition.reasoning.top_k == direct.top_k
    assert condition.reasoning.presence_penalty == direct.presence_penalty
    assert condition.reasoning.seed == direct.seed
    assert condition.reasoning_authority.adapter_active is False
    assert condition.reasoning_authority.probability_in_forward_edge is False
    assert condition.action.sampling_mode == "greedy"
    assert condition.action.temperature == 0.0
    assert condition.action.enable_thinking is False


def test_benchmark_matched_reasoner_rejects_a_non_thinking_profile() -> None:
    with pytest.raises(ValueError):
        matched_step_zero_condition(
            _direct_profile(),
            condition_id="aime-step0@1",
            reasoning_tokens=81_920,
            reasoning_authority_mode=ReasoningAuthorityMode.FROZEN_BENCHMARK_MATCHED,
        )
