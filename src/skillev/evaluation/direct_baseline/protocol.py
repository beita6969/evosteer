"""Parse and validate the executable direct-reference protocol."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import yaml

from .config import (
    AggregateComponent,
    AggregateMetricSpec,
    BenchmarkComparability,
    ComparabilityEvidence,
    DirectBenchmark,
    DirectDecodingProfile,
    DirectModelSpec,
    DirectParityPolicy,
    DirectReferenceProtocol,
    EvidenceStatus,
    FormalRunPolicy,
    MetricContract,
    MetricScale,
    PaperBenchmarkSpec,
    ParityMetric,
    SeedAggregationMode,
    SeedAggregationSpec,
    UpstreamEvidenceSpec,
)

_EXPECTED_TOP_LEVEL = {
    "format",
    "model",
    "upstream",
    "parity",
    "profiles",
    "benchmarks",
    "formal_run",
    "aggregates",
}

_UPSTREAM_FIELDS = {
    "repository",
    "revision",
    "iid_dataset",
    "iid_dataset_revision",
    "preparation_seed",
}

_EVIDENCE_FIELDS = {
    "population",
    "prompt",
    "decoding",
    "seed_aggregation",
    "scorer",
    "environment",
}


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be an object with text keys")
    return cast(dict[str, object], value)


def _sequence(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return cast(list[object], value)


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{label} must be decimal-compatible")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be decimal-compatible") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _load_profiles(value: object) -> tuple[DirectDecodingProfile, ...]:
    raw_profiles = _mapping(value, "profiles")
    profiles: list[DirectDecodingProfile] = []
    for profile_id, item in raw_profiles.items():
        raw = _mapping(item, f"profile {profile_id}")
        stop = raw.get("stop", [])
        if type(stop) is not list or any(type(entry) is not str or not entry for entry in stop):
            raise ValueError("profile stop must be a sequence of non-empty text")
        profiles.append(
            DirectDecodingProfile(
                profile_id=profile_id,
                enable_thinking=_bool(raw["enable_thinking"], "enable_thinking"),
                temperature=_float(raw["temperature"], "temperature"),
                top_p=_float(raw["top_p"], "top_p"),
                top_k=_int(raw["top_k"], "top_k"),
                min_p=_float(raw["min_p"], "min_p"),
                presence_penalty=_float(raw["presence_penalty"], "presence_penalty"),
                repetition_penalty=_float(raw["repetition_penalty"], "repetition_penalty"),
                max_new_tokens=_int(raw["max_new_tokens"], "max_new_tokens"),
                stop=tuple(cast(list[str], stop)),
                seed=_int(raw["seed"], "seed"),
                sampling_mode=str(raw.get("sampling_mode", "sampling")),
            )
        )
    if not profiles:
        raise ValueError("direct-reference config has no profiles")
    return tuple(profiles)


def _load_metrics(value: object, benchmark: str) -> tuple[MetricContract, ...]:
    result: list[MetricContract] = []
    for item in _sequence(value, f"{benchmark} metrics"):
        raw = _mapping(item, f"{benchmark} metric")
        result.append(
            MetricContract(
                metric_id=_text(raw["id"], "metric id"),
                reference_percent=_decimal(raw["reference_percent"], "reference_percent"),
                source_scale=MetricScale(str(raw.get("source_scale", "unit-interval"))),
                required=_bool(raw.get("required", True), "metric required"),
            )
        )
    return tuple(result)


def _load_benchmarks(value: object) -> tuple[PaperBenchmarkSpec, ...]:
    result: list[PaperBenchmarkSpec] = []
    for item in _sequence(value, "benchmarks"):
        raw = _mapping(item, "benchmark")
        benchmark = DirectBenchmark(_text(raw["id"], "benchmark id"))
        seed_raw = _mapping(raw["seed_aggregation"], "seed aggregation")
        evidence_raw = _mapping(raw["evidence"], "comparability evidence")
        if set(evidence_raw) != _EVIDENCE_FIELDS:
            raise ValueError(f"{benchmark.value} comparability evidence fields changed")
        seeds_raw = _sequence(seed_raw["seeds"], "seed aggregation seeds")
        result.append(
            PaperBenchmarkSpec(
                benchmark=benchmark,
                role=_text(raw["role"], "benchmark role"),
                population=_text(raw["population"], "population"),
                dataset_revision=_text(raw["dataset_revision"], "dataset revision"),
                selection_rule=_text(raw["selection_rule"], "selection rule"),
                sample_count=_int(raw["n"], "sample count"),
                comparability=BenchmarkComparability(str(raw["comparability"])),
                evidence=ComparabilityEvidence(
                    population=EvidenceStatus(str(evidence_raw["population"])),
                    prompt=EvidenceStatus(str(evidence_raw["prompt"])),
                    decoding=EvidenceStatus(str(evidence_raw["decoding"])),
                    seed_aggregation=EvidenceStatus(str(evidence_raw["seed_aggregation"])),
                    scorer=EvidenceStatus(str(evidence_raw["scorer"])),
                    environment=EvidenceStatus(str(evidence_raw["environment"])),
                ),
                prompt_profile=_text(raw["prompt"], "prompt profile"),
                decoding_profile=_text(raw["decoding"], "decoding profile"),
                parser_profile=_text(raw["parser"], "parser profile"),
                scorer_profile=_text(raw["scorer"], "scorer profile"),
                metrics=_load_metrics(raw["metrics"], benchmark.value),
                seed_aggregation=SeedAggregationSpec(
                    mode=SeedAggregationMode(str(seed_raw["mode"])),
                    seeds=tuple(_int(seed, "seed") for seed in seeds_raw),
                    dispersion=_optional_text(seed_raw.get("dispersion"), "seed dispersion"),
                ),
                environment_contract=_optional_text(
                    raw.get("environment_contract"), "environment contract"
                ),
                invalid_candidate_policy=_optional_text(
                    raw.get("invalid_candidate_policy"), "invalid candidate policy"
                ),
                invalid_environment_action_policy=_optional_text(
                    raw.get("invalid_environment_action_policy"),
                    "invalid environment action policy",
                ),
                history_window_steps=(
                    _int(raw["history_window_steps"], "history window")
                    if raw.get("history_window_steps") is not None
                    else None
                ),
                horizon_policy=_optional_text(raw.get("horizon_policy"), "horizon policy"),
                max_steps_cap=(
                    _int(raw["max_steps_cap"], "maximum step cap")
                    if raw.get("max_steps_cap") is not None
                    else None
                ),
                required_max_steps=(
                    _int(raw["required_max_steps"], "required maximum steps")
                    if raw.get("required_max_steps") is not None
                    else None
                ),
                include_reasoning_in_history=_bool(
                    raw.get("include_reasoning_in_history", False),
                    "reasoning history policy",
                ),
                dataset_variant=_optional_text(raw.get("dataset_variant"), "dataset variant"),
                action_f1_contract=_optional_text(
                    raw.get("action_f1_contract"), "action F1 contract"
                ),
            )
        )
    return tuple(result)


def _load_aggregates(value: object) -> tuple[AggregateMetricSpec, ...]:
    result: list[AggregateMetricSpec] = []
    for item in _sequence(value, "aggregates"):
        raw = _mapping(item, "aggregate")
        components: list[AggregateComponent] = []
        for component in _sequence(raw["components"], "aggregate components"):
            pair = _sequence(component, "aggregate component")
            if len(pair) != 2:
                raise ValueError("aggregate component requires benchmark and metric")
            components.append(AggregateComponent(DirectBenchmark(str(pair[0])), str(pair[1])))
        if raw.get("reducer") != "arithmetic-mean" or not components:
            raise ValueError("aggregate requires arithmetic-mean and components")
        result.append(
            AggregateMetricSpec(
                aggregate_id=_text(raw["id"], "aggregate id"),
                reference_percent=_decimal(raw["reference_percent"], "aggregate reference"),
                components=tuple(components),
                required=_bool(raw.get("required", False), "aggregate required"),
                required_for_public_table=_bool(
                    raw.get("required_for_public_table", True),
                    "aggregate public-table policy",
                ),
                required_for_formal_gate=_bool(
                    raw.get("required_for_formal_gate", False),
                    "aggregate formal-gate policy",
                ),
            )
        )
    return tuple(result)


def load_direct_reference_protocol(path: Path) -> DirectReferenceProtocol:
    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "protocol")
    if set(raw) != _EXPECTED_TOP_LEVEL:
        raise ValueError("direct-reference protocol top-level fields changed")
    if raw["format"] != "skillev-qwen35-direct-reference@2":
        raise ValueError("incompatible direct-reference protocol")
    profiles = _load_profiles(raw["profiles"])
    model_raw = _mapping(raw["model"], "model")
    upstream_raw = _mapping(raw["upstream"], "upstream")
    if set(upstream_raw) != _UPSTREAM_FIELDS:
        raise ValueError("upstream evidence fields changed")
    parity_raw = _mapping(raw["parity"], "parity")
    formal_raw = _mapping(raw["formal_run"], "formal run")
    protocol = DirectReferenceProtocol(
        format_version=str(raw["format"]),
        model=DirectModelSpec(
            repo_id=_text(model_raw["repo_id"], "model repo_id"),
            route=_text(model_raw["route"], "model route"),
            served_model_name=_text(model_raw["served_model_name"], "served model"),
            adapters=_text(model_raw["adapters"], "adapter policy"),
            skill_library=_text(model_raw["skill_library"], "skill policy"),
            rollout_engine=_text(model_raw["rollout_engine"], "rollout policy"),
        ),
        upstream=UpstreamEvidenceSpec(
            repository=_text(upstream_raw["repository"], "upstream repository"),
            revision=_text(upstream_raw["revision"], "upstream revision"),
            iid_dataset=_text(upstream_raw["iid_dataset"], "upstream iid dataset"),
            iid_dataset_revision=_text(
                upstream_raw["iid_dataset_revision"], "upstream iid dataset revision"
            ),
            preparation_seed=_int(upstream_raw["preparation_seed"], "upstream preparation seed"),
        ),
        parity=DirectParityPolicy(
            metric=ParityMetric(_text(parity_raw["metric"], "parity metric")),
            max_gap_pp_exclusive=_decimal(
                parity_raw["max_gap_pp_exclusive"], "max_gap_pp_exclusive"
            ),
            require_complete_population=_bool(
                parity_raw["require_complete_population"], "complete population policy"
            ),
            require_zero_infrastructure_failures=_bool(
                parity_raw["require_zero_infrastructure_failures"], "infrastructure policy"
            ),
            require_all_declared_metrics=_bool(
                parity_raw["require_all_declared_metrics"], "metric policy"
            ),
        ),
        decoding_profiles=profiles,
        benchmarks=_load_benchmarks(raw["benchmarks"]),
        formal_run=FormalRunPolicy(
            final_attempts=_int(formal_raw["final_attempts"], "final attempts"),
            max_samples_per_benchmark=_int(
                formal_raw["max_samples_per_benchmark"], "maximum samples"
            ),
            retries_after_candidate_failure=_int(
                formal_raw["retries_after_candidate_failure"], "candidate retries"
            ),
            result_based_sampling=_text(
                formal_raw["result_based_sampling"], "result based sampling"
            ),
            private_outputs_required=_bool(
                formal_raw["private_outputs_required"], "private outputs policy"
            ),
        ),
        aggregates=_load_aggregates(raw["aggregates"]),
    )
    _validate_protocol(protocol)
    return protocol


def _validate_protocol(protocol: DirectReferenceProtocol) -> None:
    if (
        protocol.model.adapters,
        protocol.model.skill_library,
        protocol.model.rollout_engine,
    ) != ("forbidden", "forbidden", "forbidden"):
        raise ValueError("direct protocol must forbid adapters, skills, and rollout engine")
    if protocol.formal_run.final_attempts != 1:
        raise ValueError("formal direct run must have one final attempt")
    if protocol.formal_run.retries_after_candidate_failure != 0:
        raise ValueError("candidate retries must remain zero")
    if protocol.formal_run.result_based_sampling != "forbidden":
        raise ValueError("result-based sampling must remain forbidden")
    if not all(
        (
            protocol.parity.require_complete_population,
            protocol.parity.require_zero_infrastructure_failures,
            protocol.parity.require_all_declared_metrics,
        )
    ):
        raise ValueError("formal direct protocol requires strict parity policies")
    benchmark_ids = tuple(spec.benchmark for spec in protocol.benchmarks)
    if len(benchmark_ids) != 14 or len(set(benchmark_ids)) != 14:
        raise ValueError("direct protocol requires fourteen unique benchmarks")
    for spec in protocol.benchmarks:
        protocol.profile(spec.decoding_profile)
        expected = 30 if spec.benchmark is DirectBenchmark.AIME_2026 else 128
        if spec.sample_count != expected:
            raise ValueError(f"{spec.benchmark.value} must contain {expected} formal records")
        if spec.benchmark in {
            DirectBenchmark.WEB_SHOP,
            DirectBenchmark.ALF_WORLD,
            DirectBenchmark.SCIENCE_WORLD,
        } and (
            spec.environment_contract is None
            or spec.horizon_policy is None
            or spec.max_steps_cap is None
            or spec.invalid_candidate_policy not in {"terminate-zero", "consume-step-and-continue"}
            or spec.invalid_environment_action_policy
            not in {"delegate-to-official-env", "terminate-zero"}
        ):
            raise ValueError(f"{spec.benchmark.value} lacks a complete interactive contract")
    declared = {
        (spec.benchmark, metric.metric_id)
        for spec in protocol.benchmarks
        for metric in spec.metrics
    }
    for aggregate in protocol.aggregates:
        if any((item.benchmark, item.metric_id) not in declared for item in aggregate.components):
            raise ValueError(f"aggregate {aggregate.aggregate_id} references an unknown metric")


def validate_runtime_registries(protocol: DirectReferenceProtocol) -> None:
    from .parsing import PARSER_REGISTRY
    from .prompts import PROMPT_REGISTRY, PromptKind

    for spec in protocol.benchmarks:
        if spec.prompt_profile not in PROMPT_REGISTRY:
            raise ValueError(f"unknown prompt profile {spec.prompt_profile}")
        expected_kind = (
            PromptKind.INTERACTIVE if spec.environment_contract is not None else PromptKind.STATIC
        )
        if PROMPT_REGISTRY[spec.prompt_profile].kind is not expected_kind:
            raise ValueError("benchmark prompt kind differs from execution family")
        if spec.parser_profile not in PARSER_REGISTRY:
            raise ValueError(f"unknown parser profile {spec.parser_profile}")
