#!/usr/bin/env python3
"""Run one final-disjoint WebShop/ALFWorld ReAct memory ablation.

This diagnostic runner deliberately does not create a Protocol 13 formal
receipt.  It uses the same pinned model service, official environment bridge,
decoding profile, and native scorer as the formal runner, but consumes a small
train/validation manifest whose public source identities must be disjoint from
the frozen final manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

try:
    import run_qwen35_direct_interactive as legacy  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Imported as ``scripts.*`` by the test suite.
    from scripts import run_qwen35_direct_interactive as legacy
from skillev_private.direct_reference.journal import require_fresh_output_directory
from skillev_private.direct_reference.protocol13_runner import (
    build_protocol13_interactive_adapter,
)

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.current_iid.protocol13.runner_profiles import (
    load_protocol13_runner_profiles,
)
from skillev.evaluation.current_iid.protocol13.service_receipts import (
    load_model_service_contract,
    load_model_service_receipt,
    validate_replica_services,
)
from skillev.evaluation.direct_baseline import (
    DirectBenchmark,
    InvalidCandidatePolicy,
    NativeInteractiveAttempt,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_native_interactive_task,
)
from skillev.evaluation.direct_baseline.client import DeterministicReplicaClient
from skillev.evaluation.direct_baseline.interactive_tasks import InteractiveReasoningMode
from skillev.evaluation.interactive_prompt_assets import (
    InteractivePromptAsset,
    load_interactive_prompt_asset,
)
from skillev.experiments.protocol_v13 import load_protocol_v13


class ReactAblationVariant(StrEnum):
    CURRENT_V4 = "current-v4"
    FULL_HISTORY = "full-history"
    STRUCTURED_MEMORY = "structured-memory"
    TASK_SPECIFIC_DEMOS = "task-specific-demos"
    V6_STATE_MEMORY = "v6-state-memory"
    V6_HORIZON_POLICY = "v6-horizon-policy"
    V6_TASK_RECIPE = "v6-task-recipe"
    V6_TASK_SPECIFIC_DEMOS = "v6-task-specific-demos"
    V7_SURFACE_GROUNDING = "v7-surface-grounding"
    V8_PUBLISHED_WORKFLOW = "v8-published-workflow"
    V9_STATEACT_NO_THOUGHT = "v9-stateact-no-thought"
    V10_BOUNDED_COMMITMENT = "v10-bounded-commitment"
    V11_SINGLE_PRODUCT_COMMITMENT = "v11-single-product-commitment"
    V12_SKILLFLOW_SOURCE_ACTION_ONLY = "v12-skillflow-source-action-only"
    V13_EXPEL_BM25_DEMONSTRATIONS = "v13-expel-bm25-demonstrations"
    V14_EFFICIENT_TRAIN_REPLAYS = "v14-efficient-train-replays"
    V15_EFFICIENT_COMPACT_TRAIN_REPLAYS = "v15-efficient-compact-train-replays"
    V16_DIVERSE_EFFICIENT_BM25_REPLAYS = "v16-diverse-efficient-bm25-replays"
    V17_SINGLE_EFFICIENT_BM25_REPLAY = "v17-single-efficient-bm25-replay"


@dataclass(frozen=True, slots=True)
class ReactAblationSettings:
    prompt_profile: str
    parser_profile: str
    history_window_steps: int | None
    history_maximum_characters: int | None
    invalid_candidate_policy: InvalidCandidatePolicy
    demonstration_mode: str
    prompt_asset_version: str = "v5"
    include_thought: bool = True
    demonstration_selection_profile: str = "constraint-tags@1"
    compact_demonstration_action_surfaces: bool | None = None


def ablation_settings(
    benchmark: DirectBenchmark,
    variant: ReactAblationVariant,
) -> ReactAblationSettings:
    """Return the cumulative ladder requested by the ReAct repair plan."""

    prefix = benchmark.value
    old_window = 8 if benchmark is DirectBenchmark.WEB_SHOP else 16
    if variant is ReactAblationVariant.CURRENT_V4:
        return ReactAblationSettings(
            f"{prefix}-native-react-v4@1",
            "native-action@2",
            old_window,
            30_000,
            InvalidCandidatePolicy.TERMINATE_ZERO,
            "all-safe-train-replays",
        )
    if variant is ReactAblationVariant.FULL_HISTORY:
        return ReactAblationSettings(
            f"{prefix}-native-react-v4@1",
            "native-action@2",
            None,
            None,
            InvalidCandidatePolicy.TERMINATE_ZERO,
            "all-safe-train-replays",
        )
    if variant is ReactAblationVariant.STRUCTURED_MEMORY:
        return ReactAblationSettings(
            f"{prefix}-native-react-memory-v5@1",
            "native-action-memory-v5@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "none",
        )
    if variant is ReactAblationVariant.TASK_SPECIFIC_DEMOS:
        return ReactAblationSettings(
            f"{prefix}-native-react-memory-v5@1",
            "native-action-memory-v5@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
        )
    if variant is ReactAblationVariant.V6_STATE_MEMORY:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v6-state-memory is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-react-memory-v6-state@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "none",
        )
    if variant is ReactAblationVariant.V6_HORIZON_POLICY:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v6-horizon-policy is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-react-memory-v6@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "none",
        )
    if variant is ReactAblationVariant.V6_TASK_RECIPE:
        if benchmark is not DirectBenchmark.ALF_WORLD:
            raise ValueError("v6-task-recipe is an ALFWorld-only ablation")
        return ReactAblationSettings(
            "alfworld-native-react-memory-v6@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "none",
        )
    if variant is ReactAblationVariant.V6_TASK_SPECIFIC_DEMOS:
        return ReactAblationSettings(
            f"{prefix}-native-react-memory-v6@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
        )
    if variant is ReactAblationVariant.V7_SURFACE_GROUNDING:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v7-surface-grounding is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-react-memory-v7@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
        )
    if variant is ReactAblationVariant.V8_PUBLISHED_WORKFLOW:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v8-published-workflow is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-react-memory-v8@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
        )
    if variant is ReactAblationVariant.V9_STATEACT_NO_THOUGHT:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v9-stateact-no-thought is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v9@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
            False,
        )
    if variant is ReactAblationVariant.V10_BOUNDED_COMMITMENT:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v10-bounded-commitment is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v10@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
            False,
        )
    if variant is ReactAblationVariant.V11_SINGLE_PRODUCT_COMMITMENT:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v11-single-product-commitment is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v11@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
            False,
        )
    if variant is ReactAblationVariant.V12_SKILLFLOW_SOURCE_ACTION_ONLY:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v12-skillflow-source-action-only is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-skillflow-action-only-v1@1",
            "native-action@2",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "none",
            "v6",
            False,
        )
    if variant is ReactAblationVariant.V13_EXPEL_BM25_DEMONSTRATIONS:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v13-expel-bm25-demonstrations is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v13@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v6",
            False,
            "lexical-bm25@1",
        )
    if variant is ReactAblationVariant.V14_EFFICIENT_TRAIN_REPLAYS:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v14-efficient-train-replays is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v14@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v5",
            False,
        )
    if variant is ReactAblationVariant.V15_EFFICIENT_COMPACT_TRAIN_REPLAYS:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v15-efficient-compact-train-replays is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v15@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v5",
            False,
            "constraint-tags@1",
            True,
        )
    if variant is ReactAblationVariant.V16_DIVERSE_EFFICIENT_BM25_REPLAYS:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v16-diverse-efficient-bm25-replays is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v16@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v7",
            False,
            "lexical-bm25@1",
        )
    if variant is ReactAblationVariant.V17_SINGLE_EFFICIENT_BM25_REPLAY:
        if benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("v17-single-efficient-bm25-replay is a WebShop-only ablation")
        return ReactAblationSettings(
            "webshop-native-stateact-memory-v17@1",
            "native-action-memory-v6@1",
            None,
            None,
            InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
            "targeted-safe-train-replays",
            "v7",
            False,
            "lexical-bm25-top1@1",
        )
    raise AssertionError(f"unhandled ReAct ablation variant: {variant}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("webshop", "alfworld"), required=True)
    parser.add_argument("--variant", choices=tuple(ReactAblationVariant), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--forbidden-final-manifest", type=Path, required=True)
    parser.add_argument("--prompt-asset-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", action="append", required=True)
    parser.add_argument("--service-contract", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, action="append", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _manifest_rows(value: dict[str, Any], benchmark: str) -> tuple[dict[str, Any], ...]:
    if value.get("format") != "skillev-qwen35-direct-interactive@3":
        raise ValueError("ReAct ablation requires a Protocol 13 interactive manifest")
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise ValueError("interactive manifest cases must be an array")
    selected = tuple(
        _mapping(row, "interactive case")
        for row in rows
        if isinstance(row, dict) and row.get("benchmark") == benchmark
    )
    if not selected or len(selected) > 128:
        raise ValueError("ReAct ablation panel must contain between 1 and 128 cases")
    return selected


def _case_identities(rows: tuple[dict[str, Any], ...], benchmark: str) -> frozenset[str]:
    source_key = "goal_id" if benchmark == "webshop" else "game_id"
    identities: list[str] = []
    for row in rows:
        payload = _mapping(row.get("payload"), "interactive case payload")
        identities.extend(
            (str(row.get("task_id")), str(row.get("source_identity")), str(payload[source_key]))
        )
    if len(identities) != len(set(identities)):
        raise ValueError("ReAct ablation manifest contains duplicate public identities")
    return frozenset(identities)


def validate_final_disjointness(
    diagnostic_manifest: dict[str, Any],
    final_manifest: dict[str, Any],
    *,
    benchmark: str,
) -> tuple[dict[str, Any], ...]:
    """Reject task, source, or native environment identity overlap with final IID."""

    diagnostic_rows = _manifest_rows(diagnostic_manifest, benchmark)
    final_rows = _manifest_rows(final_manifest, benchmark)
    if _case_identities(diagnostic_rows, benchmark).intersection(
        _case_identities(final_rows, benchmark)
    ):
        raise ValueError("ReAct diagnostic panel overlaps the frozen final IID population")
    return diagnostic_rows


def _diagnostic_protocol(
    base: Any,
    manifest: dict[str, Any],
    *,
    count: int,
) -> Any:
    spec = base.benchmarks[0]
    adjusted = replace(
        spec,
        population=str(manifest["population_id"]),
        dataset_revision=str(manifest["dataset_revision"]),
        selection_rule=str(manifest["selection_rule"]),
        sample_count=count,
    )
    return replace(base, benchmarks=(adjusted,))


def _all_safe_demonstrations(asset: InteractivePromptAsset) -> tuple[str, ...]:
    return asset.render_demonstrations(task=None, task_type=None, structured_memory=False)


def _configure_cases(
    cases: tuple[legacy._Case, ...],
    *,
    settings: ReactAblationSettings,
    all_safe_demonstrations: tuple[str, ...],
    counter: QwenChatTokenCounter,
    context_length: int,
    replica_count: int,
    targeted_asset: InteractivePromptAsset | None = None,
) -> tuple[legacy._Case, ...]:
    configured: list[legacy._Case] = []
    for index, case in enumerate(cases):
        if settings.demonstration_mode == "all-safe-train-replays":
            demonstrations = all_safe_demonstrations
        elif settings.demonstration_mode == "none":
            demonstrations = ()
        else:
            if targeted_asset is None:
                raise ValueError("targeted ReAct ablation requires a prompt asset")
            demonstrations = targeted_asset.render_demonstrations(
                task=case.task.task,
                task_type=case.task.task_type,
                structured_memory=True,
                include_thought=settings.include_thought,
                selection_profile=settings.demonstration_selection_profile,
                compact_action_surfaces=settings.compact_demonstration_action_surfaces,
            )
        configured.append(
            replace(
                case,
                task=replace(
                    case.task,
                    prompt_profile_id=settings.prompt_profile,
                    parser_profile_id=settings.parser_profile,
                    invalid_candidate_policy=settings.invalid_candidate_policy,
                    history_window_steps=settings.history_window_steps,
                    history_maximum_characters=settings.history_maximum_characters,
                    reasoning_mode=InteractiveReasoningMode.VISIBLE_REACT,
                    demonstrations=demonstrations,
                    panel_index=index,
                    service_slot=index % replica_count,
                    history_token_counter=counter,
                    context_length=context_length,
                ),
            )
        )
    return tuple(configured)


def _group_summary(rows: list[NativeInteractiveAttempt]) -> dict[str, object]:
    definitive = [row for row in rows if row.infrastructure_error is None]
    trace = [step for row in definitive for step in row.trace]
    return {
        "count": len(rows),
        "definitive_count": len(definitive),
        "average_score_percent": (
            100 * sum(cast(float, row.reward) for row in definitive) / len(definitive)
            if definitive
            else None
        ),
        "success_count": sum(row.success is True for row in definitive),
        "success_rate_percent": (
            100 * sum(row.success is True for row in definitive) / len(definitive)
            if definitive
            else None
        ),
        "budget_exhausted_count": sum(row.budget_exhausted for row in definitive),
        "horizon_count": sum(row.terminated_by_horizon for row in definitive),
        "early_terminal_failure_count": sum(
            row.terminal_reached
            and row.success is False
            and not row.terminated_by_horizon
            and not row.budget_exhausted
            for row in definitive
        ),
        "candidate_invalid_termination_count": sum(
            row.termination_reason == "candidate-invalid" for row in definitive
        ),
        "repeated_action_count": sum(step.repeated_action for step in trace),
        "unchanged_state_count": sum(step.unchanged_state for step in trace),
        "revisited_product_or_location_count": sum(
            step.revisited_product_or_location for step in trace
        ),
        "purchase_with_unmet_constraints_count": sum(
            step.purchase_with_unmet_constraints for step in trace
        ),
        "memory_missing_count": sum(
            step.memory_update_status in {"missing-carried", "missing-empty"} for step in trace
        ),
        "unfinished_subgoal_at_horizon_count": sum(
            row.unfinished_subgoal_at_horizon is not None for row in definitive
        ),
        "infrastructure_failure_count": len(rows) - len(definitive),
    }


def _aggregate(
    attempts: tuple[NativeInteractiveAttempt, ...],
    metadata: dict[str, dict[str, object]],
    *,
    benchmark: str,
    variant: ReactAblationVariant,
    settings: ReactAblationSettings,
) -> dict[str, object]:
    grouped: dict[str, list[NativeInteractiveAttempt]] = defaultdict(list)
    for attempt in attempts:
        label = str(metadata[attempt.task_id].get("task_type") or "all")
        grouped[label].append(attempt)
    return {
        "format": "skillev-react-memory-ablation@1",
        "benchmark": benchmark,
        "variant": variant.value,
        "matched_outer_horizon": True,
        "prompt_profile": settings.prompt_profile,
        "parser_profile": settings.parser_profile,
        "history_window_steps": settings.history_window_steps,
        "history_maximum_characters": settings.history_maximum_characters,
        "invalid_candidate_policy": settings.invalid_candidate_policy.value,
        "demonstration_mode": settings.demonstration_mode,
        "demonstration_selection_profile": settings.demonstration_selection_profile,
        "prompt_asset_version": settings.prompt_asset_version,
        "compact_demonstration_action_surfaces": (settings.compact_demonstration_action_surfaces),
        "overall": _group_summary(list(attempts)),
        "by_task_type": {label: _group_summary(rows) for label, rows in sorted(grouped.items())},
    }


def require_no_infrastructure_failures(
    attempts: tuple[NativeInteractiveAttempt, ...],
) -> None:
    failures = sum(attempt.infrastructure_error is not None for attempt in attempts)
    if failures:
        raise RuntimeError(
            f"ReAct ablation has {failures} infrastructure failures; stop before next variant"
        )


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    endpoints = tuple(arguments.endpoint_base)
    if not endpoints or len(endpoints) != len(set(endpoints)):
        raise ValueError("ReAct ablation endpoint order must be non-empty and unique")
    if len(arguments.service_receipt) != len(endpoints):
        raise ValueError("endpoint and service receipt counts differ")

    protocol = load_protocol_v13(arguments.protocol, arguments.sources)
    executions = load_execution_contracts_v3(arguments.conditions, protocol=protocol)
    benchmark = Protocol13Benchmark(arguments.benchmark)
    execution = executions[benchmark]
    if execution.interactive is None:
        raise ValueError("ReAct ablation requires an interactive execution contract")
    registry = load_protocol13_runner_profiles(arguments.runner_config)
    profile = registry.require(execution.decoding_profile)
    if (
        arguments.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
    ):
        raise ValueError("ReAct ablation model identity differs from Protocol 13")
    receipts = tuple(load_model_service_receipt(path) for path in arguments.service_receipt)
    validate_replica_services(
        receipts,
        contract=load_model_service_contract(arguments.service_contract),
        execution=execution,
    )

    diagnostic_manifest = _mapping(
        json.loads(arguments.private_manifest.read_text(encoding="utf-8")),
        "diagnostic manifest",
    )
    final_manifest = _mapping(
        json.loads(arguments.forbidden_final_manifest.read_text(encoding="utf-8")),
        "final manifest",
    )
    rows = validate_final_disjointness(
        diagnostic_manifest,
        final_manifest,
        benchmark=benchmark.value,
    )
    base = _diagnostic_protocol(
        build_protocol13_interactive_adapter(execution, profile),
        diagnostic_manifest,
        count=len(rows),
    )
    cases = legacy._load_cases(
        arguments.private_manifest,
        base,
        frozenset({DirectBenchmark(benchmark.value)}),
        prompt_asset_dir=arguments.prompt_asset_dir,
    )
    settings = ablation_settings(
        DirectBenchmark(benchmark.value),
        ReactAblationVariant(arguments.variant),
    )
    base_asset = load_interactive_prompt_asset(
        arguments.prompt_asset_dir / f"{benchmark.value}_native_react_v5.yaml"
    )
    targeted_asset = load_interactive_prompt_asset(
        arguments.prompt_asset_dir
        / f"{benchmark.value}_native_react_{settings.prompt_asset_version}.yaml"
    )
    final_identities = _case_identities(
        _manifest_rows(final_manifest, benchmark.value), benchmark.value
    )
    base_asset.validate_final_isolation(final_identities)
    targeted_asset.validate_final_isolation(final_identities)
    counter = QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path)
    cases = _configure_cases(
        cases,
        settings=settings,
        all_safe_demonstrations=_all_safe_demonstrations(base_asset),
        targeted_asset=targeted_asset,
        counter=counter,
        context_length=execution.context_length,
        replica_count=len(endpoints),
    )

    require_fresh_output_directory(arguments.private_output_dir)
    clients = tuple(
        OpenAICompatibleDirectClient(
            endpoint_base=endpoint,
            served_model_name=execution.actor_route,
            api_key="EMPTY",
            timeout_seconds=arguments.request_timeout_seconds,
            context_length=execution.context_length,
            token_counter=counter,  # type: ignore[arg-type]
            service_instance_id=receipt.service_instance_id,
        )
        for endpoint, receipt in zip(endpoints, receipts, strict=True)
    )
    if any(
        routes != (execution.actor_route,)
        for routes in await asyncio.gather(*(client.model_routes() for client in clients))
    ):
        raise RuntimeError("ReAct ablation endpoint exposes another model route")
    client = DeterministicReplicaClient(clients)
    semaphore = asyncio.Semaphore(arguments.concurrency)

    async def run_case(case: legacy._Case) -> NativeInteractiveAttempt:
        async with semaphore:
            try:
                environment = await asyncio.to_thread(case.create_environment)
            except (OSError, RuntimeError, ValueError):
                return NativeInteractiveAttempt(
                    case.task.task_id,
                    case.task.benchmark,
                    None,
                    None,
                    0,
                    0,
                    0,
                    False,
                    False,
                    "EnvironmentCreationError",
                )
            return await run_native_interactive_task(client, case.task, environment)

    attempts = tuple(await asyncio.gather(*(run_case(case) for case in cases)))
    metadata = {
        str(row["task_id"]): {
            "source_split": _mapping(row["payload"], "case payload").get("split"),
            "task_type": _mapping(row["payload"], "case payload").get("task_type"),
            "max_steps": row.get("max_steps"),
        }
        for row in rows
    }
    with (arguments.private_output_dir / "per-task-results.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for attempt in attempts:
            stream.write(
                json.dumps(
                    {**legacy._attempt_value(attempt), **metadata[attempt.task_id]},
                    sort_keys=True,
                )
                + "\n"
            )
    aggregate = _aggregate(
        attempts,
        metadata,
        benchmark=benchmark.value,
        variant=ReactAblationVariant(arguments.variant),
        settings=settings,
    )
    (arguments.private_output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    require_no_infrastructure_failures(attempts)
    return aggregate


def main() -> None:
    print(json.dumps(asyncio.run(run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
