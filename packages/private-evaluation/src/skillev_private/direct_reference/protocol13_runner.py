"""Manifest-driven private panel loading for Protocol 13 runners."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.contracts import ExecutionContractV3
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

from .manifests import load_population_manifest
from .populations import PrivateDirectCase, load_humaneval_cases, load_skillflow_iid_cases

_ALFWORLD_MATCHED_HORIZON = re.compile(
    r"outer-(?P<outer>[1-9][0-9]*)-simulator-(?P<simulator>[1-9][0-9]*)-reference-matched"
)


@dataclass(frozen=True, slots=True)
class LoadedProtocol13Panel:
    benchmark: Protocol13Benchmark
    population_id: str
    dataset_revision: str
    selection_rule: str
    panel_manifest_id: str
    task_ids: tuple[str, ...]
    source_identities: tuple[str, ...]


def _validate(panel: LoadedProtocol13Panel, execution: ExecutionContractV3) -> None:
    expected = (
        panel.benchmark is execution.benchmark,
        panel.population_id == execution.population_id,
        panel.dataset_revision == execution.dataset_revision,
        panel.selection_rule == execution.selection_rule,
        panel.panel_manifest_id == execution.panel_manifest_id,
        len(panel.task_ids) == execution.expected_count,
        len(panel.task_ids) == len(set(panel.task_ids)),
        len(panel.source_identities) == len(set(panel.source_identities)),
    )
    if not all(expected):
        raise ValueError("panel manifest differs from Protocol 13 execution")


def load_protocol13_panel_manifest(
    path: Path, *, execution: ExecutionContractV3
) -> LoadedProtocol13Panel:
    """Load either the direct v1 or interactive v3 manifest without guessing."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("panel manifest must be an object")
    if raw.get("format") == "skillev-direct-population-manifest@1":
        manifest = load_population_manifest(path)
        panel = LoadedProtocol13Panel(
            benchmark=Protocol13Benchmark(manifest.benchmark.value),
            population_id=manifest.population_id,
            dataset_revision=manifest.dataset_revision,
            selection_rule=manifest.selection_rule,
            panel_manifest_id=execution.panel_manifest_id,
            task_ids=tuple(row.task_id for row in manifest.entries),
            source_identities=tuple(row.source_identity for row in manifest.entries),
        )
    elif raw.get("format") == "skillev-qwen35-direct-interactive@3":
        cases = raw.get("cases")
        if not isinstance(cases, list):
            raise ValueError("interactive manifest cases must be an array")
        benchmark = execution.benchmark.value
        selected = [
            cast(dict[str, object], row)
            for row in cases
            if isinstance(row, dict) and row.get("benchmark") == benchmark
        ]
        panel = LoadedProtocol13Panel(
            benchmark=execution.benchmark,
            population_id=_text(raw.get("population_id"), "population ID"),
            dataset_revision=_text(raw.get("dataset_revision"), "dataset revision"),
            selection_rule=_text(raw.get("selection_rule"), "selection rule"),
            panel_manifest_id=_text(raw.get("panel_manifest_id"), "panel manifest ID"),
            task_ids=tuple(_text(row.get("task_id"), "task ID") for row in selected),
            source_identities=tuple(
                _text(row.get("source_identity"), "source identity") for row in selected
            ),
        )
    else:
        raise ValueError("unsupported Protocol 13 panel manifest format")
    _validate(panel, execution)
    return panel


def load_protocol13_environment_manifest_identity(
    path: Path,
    *,
    execution: ExecutionContractV3,
) -> dict[str, object]:
    """Project the authoritative interactive manifest without task text or labels."""

    if execution.interactive is None:
        raise ValueError("environment manifest requires an interactive condition")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != "skillev-qwen35-direct-interactive@3":
        raise ValueError("interactive environment identity requires a v3 manifest")
    cases = raw.get("cases")
    deployments = raw.get("deployments")
    runtimes = raw.get("runtimes")
    if (
        not isinstance(cases, list)
        or not isinstance(deployments, dict)
        or not isinstance(runtimes, dict)
    ):
        raise ValueError("interactive environment manifest is incomplete")
    selected = [
        cast(dict[str, object], row)
        for row in cases
        if isinstance(row, dict) and row.get("benchmark") == execution.benchmark.value
    ]
    if len(selected) != execution.expected_count:
        raise ValueError("interactive environment manifest count differs")
    deployment_ids = tuple(dict.fromkeys(str(row.get("deployment")) for row in selected))
    if len(deployment_ids) != 1:
        raise ValueError("interactive environment manifest must use one deployment")
    deployment_id = deployment_ids[0]
    deployment = deployments.get(deployment_id)
    if not isinstance(deployment, dict):
        raise ValueError("interactive deployment is absent")
    if execution.benchmark is Protocol13Benchmark.ALF_WORLD:
        match = _ALFWORLD_MATCHED_HORIZON.fullmatch(execution.interactive.horizon_policy)
        if match is None:
            raise ValueError("ALFWorld condition does not separate outer and simulator horizons")
        outer_steps = int(match.group("outer"))
        simulator_steps = int(match.group("simulator"))
        if (
            execution.interactive.required_max_steps != outer_steps
            or deployment.get("simulator_max_steps") != simulator_steps
            or simulator_steps < outer_steps
        ):
            raise ValueError("ALFWorld deployment horizons differ from the condition")
    runtime_id = deployment.get("runtime")
    runtime = runtimes.get(runtime_id)
    if not isinstance(runtime_id, str) or not isinstance(runtime, dict):
        raise ValueError("interactive runtime is absent")
    public_deployment = {key: value for key, value in deployment.items() if key != "games"}
    games = deployment.get("games")
    if games is not None:
        if not isinstance(games, dict):
            raise ValueError("ALFWorld deployment games are invalid")
        public_deployment["games"] = {
            str(game_id): {
                key: value
                for key, value in cast(dict[str, object], game).items()
                if key != "instruction_text"
            }
            for game_id, game in games.items()
            if isinstance(game, dict)
        }
        if len(cast(dict[str, object], public_deployment["games"])) != len(games):
            raise ValueError("ALFWorld game deployment is invalid")
    case_identities: list[dict[str, object]] = []
    for row in selected:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("interactive case payload is absent")
        max_steps = row.get("max_steps")
        if type(max_steps) is not int or not 0 < max_steps <= execution.interactive.max_steps_cap:
            raise ValueError("interactive case horizon differs from the condition")
        if (
            execution.interactive.required_max_steps is not None
            and max_steps != execution.interactive.required_max_steps
        ):
            raise ValueError("interactive case horizon differs from the required matched horizon")
        case_identities.append(
            {
                "task_id": row.get("task_id"),
                "source_identity": row.get("source_identity"),
                "deployment": row.get("deployment"),
                "max_steps": max_steps,
                "payload": payload,
            }
        )
    return {
        "deployment_id": deployment_id,
        "deployment": public_deployment,
        "runtime_id": runtime_id,
        "runtime": runtime,
        "case_identities": case_identities,
    }


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def load_protocol13_static_cases(
    *,
    source: Path,
    manifest_path: Path,
    execution: ExecutionContractV3,
    profile: DirectDecodingProfile,
) -> tuple[PrivateDirectCase, ...]:
    """Materialize one static panel directly from the Protocol 13 condition."""

    if execution.benchmark not in {
        Protocol13Benchmark.HOTPOT_QA,
        Protocol13Benchmark.TRIVIA_QA,
        Protocol13Benchmark.AIME_2026,
        Protocol13Benchmark.HUMAN_EVAL,
    }:
        raise ValueError("benchmark is not a Protocol 13 static panel")
    panel = load_protocol13_panel_manifest(manifest_path, execution=execution)
    manifest = load_population_manifest(manifest_path)
    protocol = _single_benchmark_protocol(execution, profile)
    direct_benchmark = DirectBenchmark(execution.benchmark.value)
    if direct_benchmark is DirectBenchmark.HUMAN_EVAL:
        validate_humaneval_manifest(manifest)
        cases = load_humaneval_cases(source, protocol=protocol, manifest=manifest)
    else:
        cases = load_skillflow_iid_cases(
            source,
            protocol=protocol,
            include=frozenset({direct_benchmark}),
            manifests={direct_benchmark: manifest},
        )
    if tuple(case.public_task.task_id for case in cases) != panel.task_ids:
        raise ValueError("case loader changed manifest task order")
    if tuple(str(case.private_metadata["source_identity"]) for case in cases) != (
        panel.source_identities
    ):
        raise ValueError("case loader changed manifest source identities")
    return cases


def _single_benchmark_protocol(
    execution: ExecutionContractV3, profile: DirectDecodingProfile
) -> DirectReferenceProtocol:
    benchmark = DirectBenchmark(execution.benchmark.value)
    metric_ids = {
        Protocol13Benchmark.HOTPOT_QA: ("em", "f1"),
        Protocol13Benchmark.TRIVIA_QA: ("em", "f1"),
        Protocol13Benchmark.AIME_2026: ("accuracy",),
        Protocol13Benchmark.HUMAN_EVAL: ("pass_at_1",),
    }[execution.benchmark]
    evidence = ComparabilityEvidence(
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.NOT_APPLICABLE,
    )
    spec = PaperBenchmarkSpec(
        benchmark=benchmark,
        role="iid",
        population=execution.population_id,
        dataset_revision=execution.dataset_revision,
        selection_rule=execution.selection_rule,
        sample_count=execution.expected_count,
        comparability=BenchmarkComparability.APPROXIMATE_ONLY,
        evidence=evidence,
        prompt_profile=execution.prompt_profile,
        decoding_profile=execution.decoding_profile,
        parser_profile=execution.parser_profile,
        scorer_profile=execution.scorer_profile,
        metrics=tuple(MetricContract(item, Decimal(0)) for item in metric_ids),
        seed_aggregation=SeedAggregationSpec(SeedAggregationMode.SINGLE_RUN, (profile.seed,)),
    )
    return DirectReferenceProtocol(
        format_version="skillev-protocol13-runtime-adapter@1",
        model=DirectModelSpec(
            execution.actor_model,
            execution.actor_route,
            execution.actor_route,
            "forbidden",
            "forbidden",
            execution.actor_service_profile,
        ),
        upstream=UpstreamEvidenceSpec(
            "protocol13-condition-registry",
            "runtime",
            execution.population_id,
            execution.dataset_revision,
            profile.seed,
        ),
        parity=DirectParityPolicy(
            ParityMetric.ABSOLUTE_PERCENTAGE_POINT_GAP,
            Decimal("2"),
            True,
            True,
            True,
        ),
        decoding_profiles=(profile,),
        benchmarks=(spec,),
        formal_run=FormalRunPolicy(1, 128, 0, "forbidden", True),
    )


def build_protocol13_interactive_adapter(
    execution: ExecutionContractV3, profile: DirectDecodingProfile
) -> DirectReferenceProtocol:
    """Expose one condition through legacy task DTOs without loading legacy YAML."""

    if execution.interactive is None:
        raise ValueError("interactive adapter requires an interactive condition")
    benchmark = DirectBenchmark(execution.benchmark.value)
    metric_ids = (
        ("average_score", "success_rate")
        if execution.benchmark is Protocol13Benchmark.WEB_SHOP
        else ("success_rate",)
    )
    extension = execution.interactive
    spec = PaperBenchmarkSpec(
        benchmark=benchmark,
        role="iid",
        population=execution.population_id,
        dataset_revision=execution.dataset_revision,
        selection_rule=execution.selection_rule,
        sample_count=execution.expected_count,
        comparability=BenchmarkComparability.APPROXIMATE_ONLY,
        evidence=ComparabilityEvidence(
            EvidenceStatus.UNKNOWN,
            EvidenceStatus.UNKNOWN,
            EvidenceStatus.UNKNOWN,
            EvidenceStatus.UNKNOWN,
            EvidenceStatus.EXACT,
            EvidenceStatus.RECONSTRUCTED,
        ),
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
        max_steps_cap=extension.max_steps_cap,
        required_max_steps=extension.required_max_steps,
        include_reasoning_in_history=extension.include_reasoning_in_history,
    )
    base = _single_benchmark_protocol_for_spec(execution, profile, spec)
    return base


def _single_benchmark_protocol_for_spec(
    execution: ExecutionContractV3,
    profile: DirectDecodingProfile,
    spec: PaperBenchmarkSpec,
) -> DirectReferenceProtocol:
    return DirectReferenceProtocol(
        format_version="skillev-protocol13-runtime-adapter@1",
        model=DirectModelSpec(
            execution.actor_model,
            execution.actor_route,
            execution.actor_route,
            "forbidden",
            "forbidden",
            execution.actor_service_profile,
        ),
        upstream=UpstreamEvidenceSpec(
            "protocol13-condition-registry",
            "runtime",
            execution.population_id,
            execution.dataset_revision,
            profile.seed,
        ),
        parity=DirectParityPolicy(
            ParityMetric.ABSOLUTE_PERCENTAGE_POINT_GAP,
            Decimal("2"),
            True,
            True,
            True,
        ),
        decoding_profiles=(profile,),
        benchmarks=(spec,),
        formal_run=FormalRunPolicy(1, 128, 0, "forbidden", True),
    )


__all__ = [
    "LoadedProtocol13Panel",
    "build_protocol13_interactive_adapter",
    "load_protocol13_environment_manifest_identity",
    "load_protocol13_panel_manifest",
    "load_protocol13_static_cases",
]
