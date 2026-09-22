"""Evaluation-condition identities for backbone-only comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CurrentIIDBenchmark,
    expected_final_count,
)


class EvaluationConditionKind(StrEnum):
    DIRECT_BACKBONE = "direct-backbone"
    RETRIEVAL_AUGMENTED_BACKBONE = "retrieval-augmented-backbone"
    STRUCTURED_NO_SKILL = "structured-no-skill"
    REPOSITORY_AGENT_NO_SKILL = "repository-agent-no-skill"
    LOCAL_JUDGE_DIAGNOSTIC = "local-judge-diagnostic"
    TRAINED_SKILLEV = "trained-skillev"


@dataclass(frozen=True, slots=True)
class EvaluationCondition:
    condition_id: str
    kind: EvaluationConditionKind
    actor_model: str
    actor_route: str
    actor_service_profile: str
    context_length: int
    adapter_policy: str
    prompt_profile: str
    decoding_profile: str
    tool_surface: tuple[str, ...]
    environment_profile: str | None
    scorer_profile: str
    grader_profile: str | None
    auxiliary_models: tuple[str, ...] = ()
    retrieval_corpus: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.condition_id,
            self.actor_model,
            self.actor_route,
            self.actor_service_profile,
            self.prompt_profile,
            self.decoding_profile,
            self.scorer_profile,
        )
        if any(not item.strip() for item in required):
            raise ValueError("condition identity fields must be non-empty")
        if self.context_length <= 0:
            raise ValueError("condition context must be positive")
        if self.adapter_policy != "forbidden":
            raise ValueError("backbone-only condition must forbid adapters")
        if len(set(self.tool_surface)) != len(self.tool_surface):
            raise ValueError("condition tool surface must be unique")
        if self.kind is EvaluationConditionKind.DIRECT_BACKBONE:
            if self.auxiliary_models:
                raise ValueError("direct backbone condition cannot use auxiliary models")
            if self.tool_surface or self.retrieval_corpus is not None:
                raise ValueError("direct backbone condition cannot expose retrieval or tools")
        if self.kind is EvaluationConditionKind.RETRIEVAL_AUGMENTED_BACKBONE:
            if not self.retrieval_corpus or not self.tool_surface:
                raise ValueError("retrieval condition requires a corpus and tool surface")
        if self.kind is EvaluationConditionKind.LOCAL_JUDGE_DIAGNOSTIC:
            if not self.grader_profile:
                raise ValueError("local-judge diagnostic requires a grader profile")


class ExecutionLane(StrEnum):
    REFERENCE_BACKBONE = "reference-backbone"
    BENCHMARK_NATIVE_SCAFFOLDED = "benchmark-native-scaffolded"
    ASSISTED_DIAGNOSTIC = "assisted-diagnostic"
    TRAINED_SKILLEV = "trained-skillev"


class ReferenceEligibility(StrEnum):
    FORMAL = "formal"
    DIAGNOSTIC_ONLY = "diagnostic-only"


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    """Complete executable identity for one Protocol 12 benchmark lane."""

    benchmark: CurrentIIDBenchmark
    condition_id: str
    lane: ExecutionLane
    reference_eligibility: ReferenceEligibility
    actor_model: str
    actor_route: str
    actor_service_profile: str
    context_length: int
    adapter_policy: str
    population_id: str
    dataset_revision: str
    selection_rule: str
    expected_count: int
    prompt_profile: str
    decoding_profile: str
    parser_profile: str
    tool_surface: tuple[str, ...]
    completion_profile: str | None
    environment_profile: str | None
    scorer_profile: str
    grader_profile: str | None
    metric_profile: str
    seed_aggregation: str

    def __post_init__(self) -> None:
        required = (
            self.condition_id,
            self.actor_model,
            self.actor_route,
            self.actor_service_profile,
            self.population_id,
            self.dataset_revision,
            self.selection_rule,
            self.prompt_profile,
            self.decoding_profile,
            self.parser_profile,
            self.scorer_profile,
            self.metric_profile,
            self.seed_aggregation,
        )
        if any(not value.strip() for value in required):
            raise ValueError("execution contract identity is incomplete")
        if self.benchmark not in ACTIVE_CURRENT_IID_BENCHMARKS:
            raise ValueError("inactive benchmark cannot enter an execution contract")
        if self.expected_count != expected_final_count(self.benchmark):
            raise ValueError("execution count differs from the authoritative panel")
        if self.context_length <= 0:
            raise ValueError("execution context length must be positive")
        if self.adapter_policy != "forbidden" and self.lane is not ExecutionLane.TRAINED_SKILLEV:
            raise ValueError("backbone execution must forbid adapters")
        if len(self.tool_surface) != len(set(self.tool_surface)):
            raise ValueError("execution tool surface must be unique")
        if self.lane is ExecutionLane.REFERENCE_BACKBONE:
            if self.tool_surface or self.environment_profile is not None:
                raise ValueError("direct reference lane cannot expose tools or an environment")
        if self.reference_eligibility is ReferenceEligibility.FORMAL and (
            self.lane is ExecutionLane.ASSISTED_DIAGNOSTIC
        ):
            raise ValueError("assisted diagnostic cannot claim formal eligibility")
        if self.benchmark is CurrentIIDBenchmark.HEALTHBENCH:
            official = self.grader_profile == "gpt-4.1-2025-04-14-simple-evals@1"
            if self.reference_eligibility is ReferenceEligibility.FORMAL and not official:
                raise ValueError("formal HealthBench requires the reference grader identity")
            if not official and self.lane is not ExecutionLane.ASSISTED_DIAGNOSTIC:
                raise ValueError("local HealthBench grading must be diagnostic")


@dataclass(frozen=True, slots=True)
class RuntimeProfileRegistry:
    prompts: frozenset[str]
    decodings: frozenset[str]
    parsers: frozenset[str]
    environments: frozenset[str]
    scorers: frozenset[str]
    graders: frozenset[str]
    completions: frozenset[str]

    def admit(self, contract: ExecutionContract) -> None:
        required = (
            ("prompt", contract.prompt_profile, self.prompts),
            ("decoding", contract.decoding_profile, self.decodings),
            ("parser", contract.parser_profile, self.parsers),
            ("scorer", contract.scorer_profile, self.scorers),
        )
        for label, profile, registry in required:
            if profile not in registry:
                raise ValueError(f"unknown {label} runtime profile: {profile}")
        optional: tuple[tuple[str, str | None, frozenset[str]], ...] = (
            ("environment", contract.environment_profile, self.environments),
            ("grader", contract.grader_profile, self.graders),
            ("completion", contract.completion_profile, self.completions),
        )
        for optional_label, optional_profile, optional_registry in optional:
            if optional_profile is not None and optional_profile not in optional_registry:
                raise ValueError(f"unknown {optional_label} runtime profile: {optional_profile}")


__all__ = [
    "EvaluationCondition",
    "EvaluationConditionKind",
    "ExecutionContract",
    "ExecutionLane",
    "ReferenceEligibility",
    "RuntimeProfileRegistry",
]
