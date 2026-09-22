"""Typed, immutable contracts for the paper direct-reference lane."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class DirectBenchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    MED_QA = "medqa"
    WEB_SHOP = "webshop"
    ALF_WORLD = "alfworld"
    SWE_BENCH = "swe-bench"
    MUSIQUE = "musique"
    NQ_OPEN = "nq-open"
    MATH_HARD = "math-hard"
    GPQA_DIAMOND = "gpqa-diamond"
    HUMAN_EVAL = "humaneval"
    MBPP_PLUS = "mbpp-plus"
    SCIENCE_WORLD = "scienceworld"
    MIND2WEB = "mind2web"


class BenchmarkComparability(StrEnum):
    EXACT_PAPER = "exact-paper"
    EXACT_EXTERNAL = "exact-external"
    APPROXIMATE_ONLY = "approximate-only"
    TARGET_UNDEFINED = "target-undefined"


class SeedAggregationMode(StrEnum):
    SINGLE_RUN = "single-run"
    MEAN_OVER_RUNS = "mean-over-runs"
    UNKNOWN = "comparison-semantics-unknown"


class MetricScale(StrEnum):
    UNIT_INTERVAL = "unit-interval"
    PERCENT = "percent"


class ParityMetric(StrEnum):
    ABSOLUTE_PERCENTAGE_POINT_GAP = "absolute-percentage-point-gap"


class EvidenceStatus(StrEnum):
    EXACT = "exact"
    RECONSTRUCTED = "reconstructed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not-applicable"


@dataclass(frozen=True, slots=True)
class ComparabilityEvidence:
    population: EvidenceStatus
    prompt: EvidenceStatus
    decoding: EvidenceStatus
    seed_aggregation: EvidenceStatus
    scorer: EvidenceStatus
    environment: EvidenceStatus

    @property
    def is_exact(self) -> bool:
        required = (
            self.population,
            self.prompt,
            self.decoding,
            self.seed_aggregation,
            self.scorer,
        )
        return all(value is EvidenceStatus.EXACT for value in required) and self.environment in {
            EvidenceStatus.EXACT,
            EvidenceStatus.NOT_APPLICABLE,
        }


@dataclass(frozen=True, slots=True)
class UpstreamEvidenceSpec:
    repository: str
    revision: str
    iid_dataset: str
    iid_dataset_revision: str
    preparation_seed: int

    def __post_init__(self) -> None:
        for value in (
            self.repository,
            self.revision,
            self.iid_dataset,
            self.iid_dataset_revision,
        ):
            if not value.strip():
                raise ValueError("upstream evidence fields must be non-empty")
        if self.preparation_seed < 0:
            raise ValueError("upstream preparation seed must be non-negative")


@dataclass(frozen=True, slots=True)
class DirectDecodingProfile:
    """Every generation control sent to the base-model service."""

    profile_id: str
    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int
    stop: tuple[str, ...] = ()
    seed: int = 42
    sampling_mode: str = "sampling"

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        if self.sampling_mode not in {"sampling", "greedy"}:
            raise ValueError("sampling_mode must be sampling or greedy")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if any(not value for value in self.stop):
            raise ValueError("stop sequences must be non-empty")
        if self.sampling_mode == "greedy":
            if self.temperature != 0.0 or self.top_p != 1.0 or self.top_k != 1:
                raise ValueError("greedy mode requires temperature/top_p/top_k = 0/1/1")
        elif self.temperature <= 0.0:
            raise ValueError("sampling mode requires positive temperature")


@dataclass(frozen=True, slots=True)
class DirectParityPolicy:
    metric: ParityMetric
    max_gap_pp_exclusive: Decimal
    require_complete_population: bool
    require_zero_infrastructure_failures: bool
    require_all_declared_metrics: bool

    def __post_init__(self) -> None:
        if self.max_gap_pp_exclusive <= 0:
            raise ValueError("parity threshold must be positive")


@dataclass(frozen=True, slots=True)
class MetricContract:
    metric_id: str
    reference_percent: Decimal
    source_scale: MetricScale = MetricScale.UNIT_INTERVAL
    required: bool = True

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric_id must be non-empty")
        if not Decimal("0") <= self.reference_percent <= Decimal("100"):
            raise ValueError("reference_percent must lie in [0, 100]")


@dataclass(frozen=True, slots=True)
class SeedAggregationSpec:
    mode: SeedAggregationMode
    seeds: tuple[int, ...]
    dispersion: str | None = None

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed aggregation requires unique seeds")
        if self.mode is SeedAggregationMode.SINGLE_RUN and len(self.seeds) != 1:
            raise ValueError("single-run aggregation requires exactly one seed")
        if self.mode is SeedAggregationMode.MEAN_OVER_RUNS and self.dispersion not in {
            "sample-std",
            "population-std",
        }:
            raise ValueError("multi-run aggregation requires an explicit dispersion")
        if self.mode is SeedAggregationMode.MEAN_OVER_RUNS and len(self.seeds) < 2:
            raise ValueError("multi-run aggregation requires at least two seeds")


@dataclass(frozen=True, slots=True)
class PaperBenchmarkSpec:
    benchmark: DirectBenchmark
    role: str
    population: str
    dataset_revision: str
    selection_rule: str
    sample_count: int
    comparability: BenchmarkComparability
    evidence: ComparabilityEvidence
    prompt_profile: str
    decoding_profile: str
    parser_profile: str
    scorer_profile: str
    metrics: tuple[MetricContract, ...]
    seed_aggregation: SeedAggregationSpec
    environment_contract: str | None = None
    invalid_candidate_policy: str | None = None
    invalid_environment_action_policy: str | None = None
    history_window_steps: int | None = None
    horizon_policy: str | None = None
    max_steps_cap: int | None = None
    required_max_steps: int | None = None
    include_reasoning_in_history: bool = False
    dataset_variant: str | None = None
    action_f1_contract: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"iid", "ood"}:
            raise ValueError("paper benchmark role must be iid or ood")
        if not 1 <= self.sample_count <= 128:
            raise ValueError("paper benchmark count must lie in [1, 128]")
        for field_name in (
            "population",
            "dataset_revision",
            "selection_rule",
            "prompt_profile",
            "decoding_profile",
            "parser_profile",
            "scorer_profile",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not self.metrics:
            raise ValueError("benchmark requires at least one metric contract")
        if self.history_window_steps is not None and self.history_window_steps <= 0:
            raise ValueError("history_window_steps must be positive or null")
        if self.max_steps_cap is not None and self.max_steps_cap <= 0:
            raise ValueError("max_steps_cap must be positive or null")
        if self.required_max_steps is not None and (
            self.required_max_steps <= 0
            or self.max_steps_cap is None
            or self.required_max_steps > self.max_steps_cap
        ):
            raise ValueError("required_max_steps must be positive and no larger than the cap")
        if self.include_reasoning_in_history:
            raise ValueError("private reasoning cannot enter subsequent model context")
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("benchmark metric IDs must be unique")


@dataclass(frozen=True, slots=True)
class DirectModelSpec:
    repo_id: str
    route: str
    served_model_name: str
    adapters: str
    skill_library: str
    rollout_engine: str


@dataclass(frozen=True, slots=True)
class FormalRunPolicy:
    final_attempts: int
    max_samples_per_benchmark: int
    retries_after_candidate_failure: int
    result_based_sampling: str
    private_outputs_required: bool


@dataclass(frozen=True, slots=True)
class AggregateComponent:
    benchmark: DirectBenchmark
    metric_id: str


@dataclass(frozen=True, slots=True)
class AggregateMetricSpec:
    aggregate_id: str
    reference_percent: Decimal
    components: tuple[AggregateComponent, ...]
    required: bool = False
    required_for_public_table: bool = True
    required_for_formal_gate: bool = False


@dataclass(frozen=True, slots=True)
class DirectReferenceProtocol:
    format_version: str
    model: DirectModelSpec
    upstream: UpstreamEvidenceSpec
    parity: DirectParityPolicy
    decoding_profiles: tuple[DirectDecodingProfile, ...]
    benchmarks: tuple[PaperBenchmarkSpec, ...]
    formal_run: FormalRunPolicy
    aggregates: tuple[AggregateMetricSpec, ...] = ()

    def profile(self, profile_id: str) -> DirectDecodingProfile:
        matches = tuple(
            profile for profile in self.decoding_profiles if profile.profile_id == profile_id
        )
        if len(matches) != 1:
            raise ValueError(f"no unique decoding profile: {profile_id}")
        return matches[0]

    def benchmark(self, benchmark: DirectBenchmark) -> PaperBenchmarkSpec:
        matches = tuple(spec for spec in self.benchmarks if spec.benchmark is benchmark)
        if len(matches) != 1:
            raise ValueError(f"no unique benchmark spec: {benchmark.value}")
        return matches[0]
