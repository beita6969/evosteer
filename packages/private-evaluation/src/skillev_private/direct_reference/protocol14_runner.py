"""Private manifest adapters for corrected Protocol 14 runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.contracts import ExecutionContractV4
from skillev.evaluation.direct_baseline.config import (
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
    PaperBenchmarkSpec,
    ParityMetric,
    SeedAggregationMode,
    SeedAggregationSpec,
    UpstreamEvidenceSpec,
)
from skillev_private.benchmarks.humaneval_identity import validate_humaneval_manifest

from .manifests import PopulationManifest, load_population_manifest
from .populations import PrivateDirectCase, load_humaneval_cases, load_skillflow_iid_cases

_LEGACY_DIRECT_MANIFEST_IDENTITIES = {
    Protocol14Benchmark.HOTPOT_QA: (
        "skillflow-released-iid-v3",
        "07bb38bcc62fa8bebab6af86c39ba23b0293c97d",
        "released-panel-order",
    ),
    Protocol14Benchmark.TRIVIA_QA: (
        "skillflow-released-iid-v3",
        "07bb38bcc62fa8bebab6af86c39ba23b0293c97d",
        "released-panel-order",
    ),
    Protocol14Benchmark.AIME_2026: (
        "skillflow-released-iid-v3-full",
        "07bb38bcc62fa8bebab6af86c39ba23b0293c97d",
        "released-panel-order",
    ),
    Protocol14Benchmark.MBPP_PLUS: (
        "mbpp-plus-v0.2.0-random0-128-v13",
        "evalplus-mbppplus-v0.2.0",
        "frozen-manifest-random0-128",
    ),
    Protocol14Benchmark.HUMAN_EVAL: (
        "official-v1-seed42-reconstruction",
        "07bb38bcc62fa8bebab6af86c39ba23b0293c97d",
        "released-panel-order",
    ),
}


@dataclass(frozen=True, slots=True)
class LoadedProtocol14Panel:
    benchmark: Protocol14Benchmark
    logical_population_id: str
    adapter_population_id: str
    dataset_revision: str
    selection_rule: str
    panel_manifest_id: str
    task_ids: tuple[str, ...]
    source_identities: tuple[str, ...]


def profile_from_execution(execution: ExecutionContractV4) -> DirectDecodingProfile:
    decoding = execution.decoding
    return DirectDecodingProfile(
        profile_id=execution.decoding_profile,
        sampling_mode=decoding.sampling_mode,
        enable_thinking=execution.thinking_mode.value == "enabled",
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        top_k=decoding.top_k,
        min_p=decoding.min_p,
        presence_penalty=decoding.presence_penalty,
        repetition_penalty=decoding.repetition_penalty,
        max_new_tokens=decoding.max_new_tokens,
        seed=decoding.seed,
        stop=decoding.stop,
    )


def load_protocol14_panel_manifest(
    path: Path, *, execution: ExecutionContractV4
) -> LoadedProtocol14Panel:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("panel manifest must be an object")
    if raw.get("format") == "skillev-direct-population-manifest@1":
        manifest = load_population_manifest(path)
        if manifest.benchmark.value != execution.benchmark.value:
            raise ValueError("direct manifest benchmark differs from Protocol 14")
        expected_identity = _LEGACY_DIRECT_MANIFEST_IDENTITIES.get(execution.benchmark)
        if (
            expected_identity is None
            or (
                manifest.population_id,
                manifest.dataset_revision,
                manifest.selection_rule,
            )
            != expected_identity
        ):
            raise ValueError("direct manifest adapter identity differs from Protocol 14")
        panel = LoadedProtocol14Panel(
            benchmark=execution.benchmark,
            logical_population_id=execution.population_id,
            adapter_population_id=manifest.population_id,
            dataset_revision=manifest.dataset_revision,
            selection_rule=manifest.selection_rule,
            panel_manifest_id=execution.panel_manifest_id,
            task_ids=tuple(row.task_id for row in manifest.entries),
            source_identities=tuple(row.source_identity for row in manifest.entries),
        )
    elif raw.get("format") == "skillev-qwen35-direct-interactive@3":
        if _text(raw.get("panel_manifest_id"), "panel manifest ID") != (
            execution.panel_manifest_id
        ):
            raise ValueError("interactive manifest identity differs from Protocol 14")
        cases = raw.get("cases")
        if not isinstance(cases, list):
            raise ValueError("interactive manifest cases must be an array")
        selected = [
            cast(dict[str, object], row)
            for row in cases
            if isinstance(row, dict) and row.get("benchmark") == execution.benchmark.value
        ]
        panel = LoadedProtocol14Panel(
            benchmark=execution.benchmark,
            logical_population_id=execution.population_id,
            adapter_population_id=_text(raw.get("population_id"), "population ID"),
            dataset_revision=_text(raw.get("dataset_revision"), "dataset revision"),
            selection_rule=_text(raw.get("selection_rule"), "selection rule"),
            panel_manifest_id=execution.panel_manifest_id,
            task_ids=tuple(_text(row.get("task_id"), "task ID") for row in selected),
            source_identities=tuple(
                _text(row.get("source_identity"), "source identity") for row in selected
            ),
        )
    else:
        raise ValueError("unsupported Protocol 14 panel manifest format")
    if (
        len(panel.task_ids) != execution.expected_count
        or len(panel.task_ids) != len(set(panel.task_ids))
        or len(panel.source_identities) != len(set(panel.source_identities))
    ):
        raise ValueError("panel rows differ from Protocol 14 execution")
    return panel


def load_protocol14_static_cases(
    *,
    source: Path,
    manifest_path: Path,
    execution: ExecutionContractV4,
    profile: DirectDecodingProfile,
) -> tuple[PrivateDirectCase, ...]:
    if execution.benchmark not in {
        Protocol14Benchmark.HOTPOT_QA,
        Protocol14Benchmark.TRIVIA_QA,
        Protocol14Benchmark.AIME_2026,
        Protocol14Benchmark.HUMAN_EVAL,
    }:
        raise ValueError("benchmark is not a Protocol 14 static panel")
    panel = load_protocol14_panel_manifest(manifest_path, execution=execution)
    manifest = load_population_manifest(manifest_path)
    protocol = _single_benchmark_protocol(execution, profile, manifest)
    benchmark = DirectBenchmark(execution.benchmark.value)
    if benchmark is DirectBenchmark.HUMAN_EVAL:
        expected_identity = _LEGACY_DIRECT_MANIFEST_IDENTITIES[execution.benchmark]
        validate_humaneval_manifest(
            manifest,
            expected_population_id=expected_identity[0],
            expected_dataset_revision=expected_identity[1],
            expected_selection_rule=expected_identity[2],
        )
        cases = load_humaneval_cases(source, protocol=protocol, manifest=manifest)
    else:
        cases = load_skillflow_iid_cases(
            source,
            protocol=protocol,
            include=frozenset({benchmark}),
            manifests={benchmark: manifest},
        )
    if tuple(case.public_task.task_id for case in cases) != panel.task_ids:
        raise ValueError("case loader changed Protocol 14 manifest order")
    if tuple(str(case.private_metadata["source_identity"]) for case in cases) != (
        panel.source_identities
    ):
        raise ValueError("case loader changed Protocol 14 source identities")
    return cases


def build_protocol14_interactive_adapter(
    execution: ExecutionContractV4,
    profile: DirectDecodingProfile,
    panel: LoadedProtocol14Panel,
) -> DirectReferenceProtocol:
    if execution.interactive is None or execution.environment_profile is None:
        raise ValueError("interactive adapter requires a native execution")
    metric_ids = (
        ("average_score", "success_rate")
        if execution.benchmark is Protocol14Benchmark.WEB_SHOP
        else ("success_rate",)
    )
    extension = execution.interactive
    spec = PaperBenchmarkSpec(
        benchmark=DirectBenchmark(execution.benchmark.value),
        role="iid",
        population=panel.adapter_population_id,
        dataset_revision=panel.dataset_revision,
        selection_rule=panel.selection_rule,
        sample_count=execution.expected_count,
        comparability=BenchmarkComparability.EXACT_EXTERNAL,
        evidence=_source_evidence(),
        prompt_profile=execution.prompt_profile,
        decoding_profile=execution.decoding_profile,
        parser_profile=execution.parser_profile,
        scorer_profile=execution.scorer_profile,
        metrics=tuple(MetricContract(item, Decimal(0)) for item in metric_ids),
        seed_aggregation=SeedAggregationSpec(SeedAggregationMode.SINGLE_RUN, (profile.seed,)),
        environment_contract=execution.environment_profile,
        invalid_candidate_policy=extension.invalid_candidate_policy,
        invalid_environment_action_policy=extension.invalid_environment_action_policy,
        history_window_steps=extension.history_window_steps,
        horizon_policy=extension.horizon_policy,
        max_steps_cap=extension.max_steps,
        include_reasoning_in_history=extension.include_reasoning_in_history,
    )
    return _protocol(execution, profile, spec)


def _single_benchmark_protocol(
    execution: ExecutionContractV4,
    profile: DirectDecodingProfile,
    manifest: PopulationManifest,
) -> DirectReferenceProtocol:
    metric_ids = {
        Protocol14Benchmark.HOTPOT_QA: ("em", "f1"),
        Protocol14Benchmark.TRIVIA_QA: ("em", "f1"),
        Protocol14Benchmark.AIME_2026: ("accuracy",),
        Protocol14Benchmark.HUMAN_EVAL: ("pass_at_1",),
    }[execution.benchmark]
    spec = PaperBenchmarkSpec(
        benchmark=DirectBenchmark(execution.benchmark.value),
        role="iid",
        population=manifest.population_id,
        dataset_revision=manifest.dataset_revision,
        selection_rule=manifest.selection_rule,
        sample_count=execution.expected_count,
        comparability=BenchmarkComparability.EXACT_EXTERNAL,
        evidence=_source_evidence(),
        prompt_profile=execution.prompt_profile,
        decoding_profile=execution.decoding_profile,
        parser_profile=execution.parser_profile,
        scorer_profile=execution.scorer_profile,
        metrics=tuple(MetricContract(item, Decimal(0)) for item in metric_ids),
        seed_aggregation=SeedAggregationSpec(SeedAggregationMode.SINGLE_RUN, (profile.seed,)),
    )
    return _protocol(execution, profile, spec)


def _source_evidence() -> ComparabilityEvidence:
    return ComparabilityEvidence(
        EvidenceStatus.EXACT,
        EvidenceStatus.EXACT,
        EvidenceStatus.EXACT,
        EvidenceStatus.EXACT,
        EvidenceStatus.EXACT,
        EvidenceStatus.NOT_APPLICABLE,
    )


def _protocol(
    execution: ExecutionContractV4,
    profile: DirectDecodingProfile,
    spec: PaperBenchmarkSpec,
) -> DirectReferenceProtocol:
    return DirectReferenceProtocol(
        format_version="skillev-protocol14-runtime-adapter@1",
        model=DirectModelSpec(
            execution.actor_model,
            execution.actor_route,
            execution.actor_route,
            "forbidden",
            "forbidden",
            execution.actor_service_profile,
        ),
        upstream=UpstreamEvidenceSpec(
            "protocol14-condition-registry",
            execution.actor_model_revision,
            execution.population_id,
            execution.dataset_revision,
            profile.seed,
        ),
        parity=DirectParityPolicy(
            ParityMetric.ABSOLUTE_PERCENTAGE_POINT_GAP,
            Decimal("7"),
            True,
            True,
            True,
        ),
        decoding_profiles=(profile,),
        benchmarks=(spec,),
        formal_run=FormalRunPolicy(1, 128, 0, "forbidden", True),
    )


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


__all__ = [
    "LoadedProtocol14Panel",
    "build_protocol14_interactive_adapter",
    "load_protocol14_panel_manifest",
    "load_protocol14_static_cases",
    "profile_from_execution",
]
