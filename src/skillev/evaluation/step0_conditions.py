"""Frozen inference conditions for the seeded SkillFlow Step-0 lane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillev.evaluation.direct_baseline import DirectDecodingProfile
from skillev.evolution import (
    SkillApplicabilityV2,
    SkillExposureMode,
    StepZeroRetrievalPolicy,
)
from skillev.rollout.evaluation_sglang import EvaluationGenerationProfile


class ReasoningAuthorityMode(StrEnum):
    """Authorities allowed for reasoning that is outside the TTB edge score."""

    FROZEN_DETERMINISTIC = "frozen-deterministic"
    FROZEN_BENCHMARK_MATCHED = "frozen-benchmark-matched"


@dataclass(frozen=True, slots=True)
class ReasoningAuthority:
    mode: ReasoningAuthorityMode
    model_revision: str
    decoding_profile: str
    adapter_active: bool = False
    probability_in_forward_edge: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReasoningAuthorityMode):
            raise TypeError("unscored Step-0 reasoning authority must be typed")
        if not self.model_revision.strip() or not self.decoding_profile.strip():
            raise ValueError("reasoning authority identity is incomplete")
        if self.adapter_active or self.probability_in_forward_edge:
            raise ValueError("unscored reasoning cannot use an adapter or enter the action edge")


@dataclass(frozen=True, slots=True)
class StepZeroGenerationCondition:
    """Separate reasoning and terminal/action decoding identities."""

    condition_id: str
    reasoning: EvaluationGenerationProfile
    action: EvaluationGenerationProfile
    reasoning_authority: ReasoningAuthority

    def __post_init__(self) -> None:
        if not self.condition_id.strip():
            raise ValueError("Step-0 generation condition identity is required")
        if self.reasoning.profile_id != self.reasoning_authority.decoding_profile:
            raise ValueError("reasoning profile and authority identities differ")
        if self.reasoning_authority.mode is ReasoningAuthorityMode.FROZEN_DETERMINISTIC:
            if self.reasoning.sampling_mode != "greedy" or self.reasoning.temperature != 0.0:
                raise ValueError("deterministic Step-0 reasoning requires greedy decoding")
        elif not self.reasoning.enable_thinking:
            raise ValueError("benchmark-matched Step-0 reasoning requires Qwen thinking")


def matched_step_zero_condition(
    direct: DirectDecodingProfile,
    *,
    condition_id: str,
    reasoning_tokens: int,
    action_tokens: int | None = None,
    reasoning_authority_mode: ReasoningAuthorityMode = (
        ReasoningAuthorityMode.FROZEN_DETERMINISTIC
    ),
    deterministic_action: bool = False,
) -> StepZeroGenerationCondition:
    """Derive a frozen Step-0 condition without the training raw-softmax sampler.

    Static benchmarks may opt into their public backbone decoding contract for
    unscored reasoning.  The realized reasoning remains adapter-free and stays
    outside the TTB action edge; only its frozen generation distribution changes.
    """

    if not isinstance(direct, DirectDecodingProfile):
        raise TypeError("a benchmark DirectDecodingProfile is required")
    terminal_tokens = direct.max_new_tokens if action_tokens is None else action_tokens
    if terminal_tokens < 1:
        raise ValueError("Step-0 action token budget must be positive")
    if reasoning_authority_mode is ReasoningAuthorityMode.FROZEN_DETERMINISTIC:
        reasoning = EvaluationGenerationProfile.frozen_reasoner(
            max_new_tokens=reasoning_tokens,
            seed=direct.seed,
        )
    elif reasoning_authority_mode is ReasoningAuthorityMode.FROZEN_BENCHMARK_MATCHED:
        if not direct.enable_thinking:
            raise ValueError("benchmark-matched reasoning requires a thinking profile")
        reasoning = EvaluationGenerationProfile(
            profile_id=f"{direct.profile_id}:step0-reasoning-matched@1",
            enable_thinking=True,
            temperature=direct.temperature,
            top_p=direct.top_p,
            top_k=direct.top_k,
            min_p=direct.min_p,
            presence_penalty=direct.presence_penalty,
            repetition_penalty=direct.repetition_penalty,
            max_new_tokens=reasoning_tokens,
            seed=direct.seed,
            stop=direct.stop,
            sampling_mode=direct.sampling_mode,
        )
    else:
        raise TypeError("reasoning_authority_mode must be ReasoningAuthorityMode")
    if type(deterministic_action) is not bool:
        raise TypeError("deterministic_action must be boolean")
    if deterministic_action:
        action = EvaluationGenerationProfile(
            profile_id=f"{direct.profile_id}:step0-terminal-transcriber@1",
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=terminal_tokens,
            seed=direct.seed,
            stop=direct.stop,
            sampling_mode="greedy",
        )
    else:
        action = EvaluationGenerationProfile(
            profile_id=f"{direct.profile_id}:step0-action@1",
            enable_thinking=False,
            temperature=direct.temperature,
            top_p=direct.top_p,
            top_k=direct.top_k,
            min_p=direct.min_p,
            presence_penalty=direct.presence_penalty,
            repetition_penalty=direct.repetition_penalty,
            max_new_tokens=terminal_tokens,
            seed=direct.seed,
            stop=direct.stop,
            sampling_mode=direct.sampling_mode,
        )
    return StepZeroGenerationCondition(
        condition_id=condition_id,
        reasoning=reasoning,
        action=action,
        reasoning_authority=ReasoningAuthority(
            mode=reasoning_authority_mode,
            model_revision="Qwen3.5-9B-frozen-base",
            decoding_profile=reasoning.profile_id,
        ),
    )


def default_step_zero_retrieval_policy() -> StepZeroRetrievalPolicy:
    """Return the frozen exact-eight seed routing table."""

    routes = (
        SkillApplicabilityV2(
            "skill-aime-integer-verification",
            ("aime-2026",),
            (),
        ),
        SkillApplicabilityV2(
            "skill-alfworld-subgoal-machine",
            ("alfworld",),
            (),
            required_tools_all=("act",),
        ),
        SkillApplicabilityV2(
            "skill-health-comprehensive-response",
            ("healthbench",),
            (),
        ),
        SkillApplicabilityV2(
            "skill-hotpot-evidence-chain",
            ("hotpotqa",),
            (),
        ),
        SkillApplicabilityV2(
            "skill-python-function-completion",
            ("humaneval", "mbpp-plus"),
            (),
        ),
        SkillApplicabilityV2(
            "skill-trivia-search-synthesis",
            ("triviaqa",),
            (),
        ),
        SkillApplicabilityV2(
            "skill-webshop-constraint-ledger",
            ("webshop",),
            (),
            required_tools_all=("click", "search"),
        ),
    )
    return StepZeroRetrievalPolicy(
        policy_id="step0-exact-eight-deterministic-top1@1",
        routes=tuple(sorted(routes, key=lambda item: item.skill_id)),
        top_k=1,
        minimum_score=1000,
        exposure_mode=SkillExposureMode.FULL_INLINE_NO_INVOKE,
    )


__all__ = [
    "ReasoningAuthority",
    "ReasoningAuthorityMode",
    "StepZeroGenerationCondition",
    "default_step_zero_retrieval_policy",
    "matched_step_zero_condition",
]
