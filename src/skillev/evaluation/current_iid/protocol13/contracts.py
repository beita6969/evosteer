"""Complete execution identities for Protocol 13."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .catalog import Protocol13Benchmark, expected_final_count


class ExecutionLane(StrEnum):
    REFERENCE_BACKBONE = "reference-backbone"
    BENCHMARK_NATIVE_SCAFFOLDED = "benchmark-native-scaffolded"
    ASSISTED_DIAGNOSTIC = "assisted-diagnostic"


class ReferenceEligibility(StrEnum):
    FORMAL = "formal"
    DIAGNOSTIC_ONLY = "diagnostic-only"


@dataclass(frozen=True, slots=True)
class InteractiveExecutionExtension:
    horizon_policy: str
    max_steps_cap: int
    required_max_steps: int | None
    history_window_steps: int | None
    history_maximum_characters: int | None
    include_reasoning_in_history: bool
    invalid_candidate_policy: str
    invalid_environment_action_policy: str
    prompt_asset_id: str
    prompt_asset_source_revision: str

    def __post_init__(self) -> None:
        texts = (
            self.horizon_policy,
            self.invalid_candidate_policy,
            self.invalid_environment_action_policy,
            self.prompt_asset_id,
            self.prompt_asset_source_revision,
        )
        if any(not value.strip() for value in texts):
            raise ValueError("interactive execution identity is incomplete")
        if self.max_steps_cap <= 0:
            raise ValueError("interactive execution step cap must be positive")
        if (
            self.required_max_steps is not None
            and not 0 < self.required_max_steps <= self.max_steps_cap
        ):
            raise ValueError("required interactive horizon exceeds its cap")
        if self.history_maximum_characters is not None and self.history_maximum_characters <= 0:
            raise ValueError("interactive history character limit must be positive or null")
        if self.history_window_steps is not None and self.history_window_steps <= 0:
            raise ValueError("history window must be positive or null")
        if self.include_reasoning_in_history:
            raise ValueError("private reasoning cannot enter later model context")

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizon_policy": self.horizon_policy,
            "max_steps_cap": self.max_steps_cap,
            "required_max_steps": self.required_max_steps,
            "history_window_steps": self.history_window_steps,
            "history_maximum_characters": self.history_maximum_characters,
            "include_reasoning_in_history": self.include_reasoning_in_history,
            "invalid_candidate_policy": self.invalid_candidate_policy,
            "invalid_environment_action_policy": self.invalid_environment_action_policy,
            "prompt_asset_id": self.prompt_asset_id,
            "prompt_asset_source_revision": self.prompt_asset_source_revision,
        }


@dataclass(frozen=True, slots=True)
class ExecutionContractV3:
    protocol_version: str
    benchmark: Protocol13Benchmark
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
    panel_manifest_id: str
    evaluator_profile: str | None = None
    interactive: InteractiveExecutionExtension | None = None
    code_test_suite: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.protocol_version,
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
            self.panel_manifest_id,
        )
        if any(not value.strip() for value in required):
            raise ValueError("execution identity is incomplete")
        if self.protocol_version != "skillev-benchmark-protocol@13":
            raise ValueError("execution does not belong to Protocol 13")
        if self.expected_count != expected_final_count(self.benchmark):
            raise ValueError("execution count differs from Protocol 13 catalog")
        if type(self.context_length) is not int or self.context_length <= 0:
            raise ValueError("context length must be a positive integer")
        if self.adapter_policy != "forbidden":
            raise ValueError("Protocol 13 backbone execution forbids adapters")
        if len(set(self.tool_surface)) != len(self.tool_surface):
            raise ValueError("tool surface must not contain duplicates")
        if any(not item.strip() for item in self.tool_surface):
            raise ValueError("tool surface entries must be non-empty")
        if self.lane is ExecutionLane.REFERENCE_BACKBONE:
            if (
                self.tool_surface
                or self.environment_profile is not None
                or self.interactive is not None
            ):
                raise ValueError("reference-backbone lane cannot expose tools/environment")
        if (
            self.lane is ExecutionLane.ASSISTED_DIAGNOSTIC
            and self.reference_eligibility is ReferenceEligibility.FORMAL
        ):
            raise ValueError("assisted diagnostic cannot be formal")
        if self.lane is ExecutionLane.BENCHMARK_NATIVE_SCAFFOLDED:
            if (
                self.environment_profile is None
                or not self.tool_surface
                or self.interactive is None
            ):
                raise ValueError("native scaffold requires environment and tool surface")
        elif self.interactive is not None:
            raise ValueError("non-native execution cannot carry an interactive extension")
        if self.code_test_suite is not None and not self.code_test_suite.strip():
            raise ValueError("code test suite must be non-empty or null")
        if self.evaluator_profile is not None and not self.evaluator_profile.strip():
            raise ValueError("evaluator profile must be non-empty or null")
        code_benchmarks = {
            Protocol13Benchmark.MBPP_PLUS,
            Protocol13Benchmark.HUMAN_EVAL,
        }
        is_code = self.benchmark in code_benchmarks
        if is_code != (self.code_test_suite is not None):
            raise ValueError("code test suite identity differs from benchmark")
        if is_code != (self.evaluator_profile is not None):
            raise ValueError("code evaluator identity differs from benchmark")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical, JSON-compatible execution identity.

        Runtime receipts, targets, and publication admission all compare this
        explicit field set.  This deliberately avoids ``asdict`` so enum and
        tuple representation cannot drift silently.
        """

        return {
            "protocol_version": self.protocol_version,
            "benchmark": self.benchmark.value,
            "condition_id": self.condition_id,
            "lane": self.lane.value,
            "reference_eligibility": self.reference_eligibility.value,
            "actor_model": self.actor_model,
            "actor_route": self.actor_route,
            "actor_service_profile": self.actor_service_profile,
            "context_length": self.context_length,
            "adapter_policy": self.adapter_policy,
            "population_id": self.population_id,
            "dataset_revision": self.dataset_revision,
            "selection_rule": self.selection_rule,
            "expected_count": self.expected_count,
            "prompt_profile": self.prompt_profile,
            "decoding_profile": self.decoding_profile,
            "parser_profile": self.parser_profile,
            "tool_surface": list(self.tool_surface),
            "completion_profile": self.completion_profile,
            "environment_profile": self.environment_profile,
            "scorer_profile": self.scorer_profile,
            "grader_profile": self.grader_profile,
            "metric_profile": self.metric_profile,
            "seed_aggregation": self.seed_aggregation,
            "panel_manifest_id": self.panel_manifest_id,
            "evaluator_profile": self.evaluator_profile,
            "interactive": self.interactive.to_mapping() if self.interactive else None,
            "code_test_suite": self.code_test_suite,
        }


__all__ = [
    "ExecutionContractV3",
    "ExecutionLane",
    "InteractiveExecutionExtension",
    "ReferenceEligibility",
]
