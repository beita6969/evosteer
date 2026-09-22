"""Structured, condition-aware prompt-fragment admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class InitialContextProfile(StrEnum):
    SKILLEV_COLD_START = "skillev-cold-start-neutral@1"
    BENCHMARK_NATIVE_NO_SKILL = "benchmark-native-no-skill@1"
    SEEDED_STEP_ZERO = "seeded-step-zero@1"
    TRAINED_SKILLEV = "trained-skillev@1"


class PromptFragmentKind(StrEnum):
    TASK = "task"
    PUBLIC_OBSERVATION = "public-observation"
    ACTION_SCHEMA = "action-schema"
    BUDGET = "budget"
    PUBLIC_DEMONSTRATION = "public-demonstration"
    STRATEGY = "strategy"
    GOLD_DERIVED = "gold-derived"


@dataclass(frozen=True, slots=True)
class PublicPromptFragment:
    kind: PromptFragmentKind
    source_id: str
    content: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.content.strip():
            raise ValueError("prompt fragment identity and content must be non-empty")


@dataclass(frozen=True, slots=True)
class ColdStartPromptContract:
    allow_retrieved_skills: bool = False
    allow_demonstrations: bool = False
    allow_strategy_fragments: bool = False
    allow_benchmark_native_public_examples: bool = False


COLD_START_CONTRACT = ColdStartPromptContract()


def validate_prompt_fragments(
    fragments: tuple[PublicPromptFragment, ...], *, profile: InitialContextProfile
) -> None:
    if any(item.kind is PromptFragmentKind.GOLD_DERIVED for item in fragments):
        raise ValueError("gold-derived prompt fragments are forbidden in every condition")
    if profile is InitialContextProfile.SKILLEV_COLD_START:
        allowed = {
            PromptFragmentKind.TASK,
            PromptFragmentKind.PUBLIC_OBSERVATION,
            PromptFragmentKind.ACTION_SCHEMA,
            PromptFragmentKind.BUDGET,
        }
        if any(item.kind not in allowed for item in fragments):
            raise ValueError("cold-start prompt contains strategy or demonstration content")


def validate_initial_context_profile(
    *,
    profile: InitialContextProfile,
    active_skill_ids: tuple[str, ...],
    retrieved_skill_ids: tuple[str, ...],
) -> None:
    if profile in {
        InitialContextProfile.SKILLEV_COLD_START,
        InitialContextProfile.BENCHMARK_NATIVE_NO_SKILL,
    } and (active_skill_ids or retrieved_skill_ids):
        raise ValueError("no-skill H0 cannot contain active or retrieved skills")


__all__ = [
    "COLD_START_CONTRACT",
    "ColdStartPromptContract",
    "InitialContextProfile",
    "PromptFragmentKind",
    "PublicPromptFragment",
    "validate_initial_context_profile",
    "validate_prompt_fragments",
]
