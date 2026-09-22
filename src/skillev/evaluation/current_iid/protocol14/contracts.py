"""Complete, immutable execution identities for corrected Protocol 14 lanes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .catalog import Protocol14Benchmark, expected_final_count


class ExecutionLane(StrEnum):
    REFERENCE_BACKBONE = "reference-backbone"
    BENCHMARK_NATIVE = "benchmark-native"
    EXTERNAL_JUDGE = "external-judge"
    DIAGNOSTIC = "diagnostic"


class FormalEligibility(StrEnum):
    FORMAL = "formal"
    DIAGNOSTIC_ONLY = "diagnostic-only"


class ThinkingMode(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class DecodingParameters:
    sampling_mode: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sampling_mode not in {"greedy", "sampling"}:
            raise ValueError("sampling mode must be greedy or sampling")
        numeric = (
            self.temperature,
            self.top_p,
            self.min_p,
            self.presence_penalty,
            self.repetition_penalty,
        )
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in numeric):
            raise ValueError("decoding controls must be numeric")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if any(not item for item in self.stop):
            raise ValueError("stop sequences must be non-empty")
        if self.sampling_mode == "greedy" and self.temperature != 0:
            raise ValueError("greedy decoding requires temperature zero")

    def to_mapping(self) -> dict[str, object]:
        return {
            "sampling_mode": self.sampling_mode,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@dataclass(frozen=True, slots=True)
class AggregationIdentity:
    seeds: tuple[int, ...]
    run_count: int
    reducer: str
    dispersion: str | None

    def __post_init__(self) -> None:
        if self.run_count <= 0 or not self.reducer.strip():
            raise ValueError("aggregation identity is incomplete")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("aggregation seeds must be unique")
        if self.seeds and len(self.seeds) != self.run_count:
            raise ValueError("aggregation run count differs from its seed list")
        if self.run_count > 1 and (self.dispersion is None or not self.dispersion.strip()):
            raise ValueError("multi-run aggregation must publish dispersion")

    def to_mapping(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "run_count": self.run_count,
            "reducer": self.reducer,
            "dispersion": self.dispersion,
        }


@dataclass(frozen=True, slots=True)
class InteractiveContract:
    horizon_policy: str
    max_steps: int
    history_window_steps: int | None
    history_maximum_characters: int
    history_observation_characters: int | None
    include_reasoning_in_history: bool
    invalid_candidate_policy: str
    invalid_environment_action_policy: str
    prompt_source_repository: str
    prompt_source_revision: str
    prompt_source_path: str
    prompt_source_symbol: str

    def __post_init__(self) -> None:
        texts = (
            self.horizon_policy,
            self.invalid_candidate_policy,
            self.invalid_environment_action_policy,
            self.prompt_source_repository,
            self.prompt_source_revision,
            self.prompt_source_path,
            self.prompt_source_symbol,
        )
        if any(not value.strip() for value in texts):
            raise ValueError("interactive contract is incomplete")
        if self.max_steps <= 0 or self.history_maximum_characters <= 0:
            raise ValueError("interactive limits must be positive")
        if self.history_window_steps is not None and self.history_window_steps <= 0:
            raise ValueError("history window must be positive or null")
        if (
            self.history_observation_characters is not None
            and self.history_observation_characters <= 0
        ):
            raise ValueError("history observation limit must be positive or null")
        if self.include_reasoning_in_history:
            raise ValueError("reasoning must not be replayed into interactive history")

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizon_policy": self.horizon_policy,
            "max_steps": self.max_steps,
            "history_window_steps": self.history_window_steps,
            "history_maximum_characters": self.history_maximum_characters,
            "history_observation_characters": self.history_observation_characters,
            "include_reasoning_in_history": self.include_reasoning_in_history,
            "invalid_candidate_policy": self.invalid_candidate_policy,
            "invalid_environment_action_policy": self.invalid_environment_action_policy,
            "prompt_source_repository": self.prompt_source_repository,
            "prompt_source_revision": self.prompt_source_revision,
            "prompt_source_path": self.prompt_source_path,
            "prompt_source_symbol": self.prompt_source_symbol,
        }


@dataclass(frozen=True, slots=True)
class ExecutionContractV4:
    benchmark: Protocol14Benchmark
    condition_id: str
    lane: ExecutionLane
    formal_eligibility: FormalEligibility
    actor_model: str
    actor_model_revision: str
    actor_route: str
    actor_service_profile: str
    context_length: int
    adapter_policy: str
    population_id: str
    dataset_revision: str
    selection_rule: str
    expected_count: int
    panel_manifest_id: str
    prompt_profile: str
    thinking_mode: ThinkingMode
    decoding_profile: str
    decoding: DecodingParameters
    parser_profile: str
    completion_profile: str | None
    tool_surface: tuple[str, ...]
    environment_profile: str | None
    scorer_profile: str
    grader_profile: str | None
    metric_profile: str
    aggregation: AggregationIdentity
    interactive: InteractiveContract | None = None
    code_test_suite: str | None = None
    protocol_version: str = "skillev-benchmark-protocol@14"

    def __post_init__(self) -> None:
        required = (
            self.condition_id,
            self.actor_model,
            self.actor_model_revision,
            self.actor_route,
            self.actor_service_profile,
            self.population_id,
            self.dataset_revision,
            self.selection_rule,
            self.panel_manifest_id,
            self.prompt_profile,
            self.decoding_profile,
            self.parser_profile,
            self.scorer_profile,
            self.metric_profile,
        )
        if any(not value.strip() for value in required):
            raise ValueError("execution identity is incomplete")
        if self.protocol_version != "skillev-benchmark-protocol@14":
            raise ValueError("execution does not belong to Protocol 14")
        if self.expected_count != expected_final_count(self.benchmark):
            raise ValueError("execution count differs from Protocol 14 catalog")
        if type(self.context_length) is not int or self.context_length <= 0:
            raise ValueError("context length must be positive")
        if self.adapter_policy != "forbidden":
            raise ValueError("Protocol 14 backbone execution forbids adapters")
        if len(self.tool_surface) != len(set(self.tool_surface)):
            raise ValueError("tool surface contains duplicates")
        if any(not item.strip() for item in self.tool_surface):
            raise ValueError("tool surface entries must be non-empty")
        native = self.lane is ExecutionLane.BENCHMARK_NATIVE
        if native != (self.interactive is not None):
            raise ValueError("interactive identity differs from execution lane")
        if native != (self.environment_profile is not None):
            raise ValueError("environment identity differs from execution lane")
        if native != bool(self.tool_surface):
            raise ValueError("native lane tool surface is incomplete")
        code = self.benchmark in {
            Protocol14Benchmark.MBPP_PLUS,
            Protocol14Benchmark.HUMAN_EVAL,
        }
        if code != (self.code_test_suite is not None):
            raise ValueError("code test suite identity differs from benchmark")
        if self.lane is ExecutionLane.DIAGNOSTIC and (
            self.formal_eligibility is FormalEligibility.FORMAL
        ):
            raise ValueError("diagnostic execution cannot be formal")
        if (
            self.benchmark is Protocol14Benchmark.MBPP_PLUS
            and "hard"
            in (self.condition_id + self.population_id + (self.code_test_suite or "")).lower()
        ):
            raise ValueError("Protocol 14 forbids the non-official MBPP+ hard name")

    def to_mapping(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "benchmark": self.benchmark.value,
            "condition_id": self.condition_id,
            "lane": self.lane.value,
            "formal_eligibility": self.formal_eligibility.value,
            "actor_model": self.actor_model,
            "actor_model_revision": self.actor_model_revision,
            "actor_route": self.actor_route,
            "actor_service_profile": self.actor_service_profile,
            "context_length": self.context_length,
            "adapter_policy": self.adapter_policy,
            "population_id": self.population_id,
            "dataset_revision": self.dataset_revision,
            "selection_rule": self.selection_rule,
            "expected_count": self.expected_count,
            "panel_manifest_id": self.panel_manifest_id,
            "prompt_profile": self.prompt_profile,
            "thinking_mode": self.thinking_mode.value,
            "decoding_profile": self.decoding_profile,
            "decoding": self.decoding.to_mapping(),
            "parser_profile": self.parser_profile,
            "completion_profile": self.completion_profile,
            "tool_surface": list(self.tool_surface),
            "environment_profile": self.environment_profile,
            "scorer_profile": self.scorer_profile,
            "grader_profile": self.grader_profile,
            "metric_profile": self.metric_profile,
            "aggregation": self.aggregation.to_mapping(),
            "interactive": self.interactive.to_mapping() if self.interactive else None,
            "code_test_suite": self.code_test_suite,
        }


__all__ = [
    "AggregationIdentity",
    "DecodingParameters",
    "ExecutionContractV4",
    "ExecutionLane",
    "FormalEligibility",
    "InteractiveContract",
    "ThinkingMode",
]
