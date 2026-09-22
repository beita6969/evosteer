#!/usr/bin/env python3
"""Run one condition-authoritative Protocol 13 native agent panel."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import run_qwen35_direct_interactive as legacy  # type: ignore[import-not-found]
from skillev_private.direct_reference.journal import (
    require_fresh_output_directory,
)
from skillev_private.direct_reference.protocol13_runner import (
    build_protocol13_interactive_adapter,
    load_protocol13_environment_manifest_identity,
    load_protocol13_panel_manifest,
)

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.current_iid.protocol13.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol13ExecutionReceipt,
    RuntimeEnvironmentIdentity,
    RuntimeGenerationProfile,
    write_execution_receipt,
)
from skillev.evaluation.current_iid.protocol13.runner_profiles import (
    load_protocol13_runner_profiles,
)
from skillev.evaluation.current_iid.protocol13.service_receipts import (
    load_model_service_contract,
    load_model_service_receipt,
    validate_replica_services,
)
from skillev.evaluation.direct_baseline import (
    NativeInteractiveAttempt,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_native_interactive_task,
)
from skillev.evaluation.direct_baseline.client import DeterministicReplicaClient
from skillev.experiments.protocol_v13 import load_protocol_v13


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("webshop", "alfworld"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--prompt-asset-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--endpoint-base", action="append", required=True)
    parser.add_argument("--service-contract", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, action="append", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    endpoints = tuple(arguments.endpoint_base)
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("Protocol 13 endpoint order must be unique")
    if len(arguments.service_receipt) != len(endpoints):
        raise ValueError("endpoint and service receipt counts differ")
    protocol = load_protocol_v13(arguments.protocol, arguments.sources)
    executions = load_execution_contracts_v3(arguments.conditions, protocol=protocol)
    benchmark = Protocol13Benchmark(arguments.benchmark)
    execution = executions[benchmark]
    if execution.interactive is None or execution.environment_profile is None:
        raise ValueError("benchmark is not an interactive Protocol 13 condition")
    registry = load_protocol13_runner_profiles(arguments.runner_config)
    profile = registry.require(execution.decoding_profile)
    service_contract = load_model_service_contract(arguments.service_contract)
    service_receipts = tuple(load_model_service_receipt(path) for path in arguments.service_receipt)
    validate_replica_services(
        service_receipts,
        contract=service_contract,
        execution=execution,
    )
    if (
        arguments.served_model_name != execution.actor_route
        or registry.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
        or registry.context_length != execution.context_length
    ):
        raise ValueError("interactive runtime model identity differs from Protocol 13")
    panel = load_protocol13_panel_manifest(arguments.private_manifest, execution=execution)
    adapter = build_protocol13_interactive_adapter(execution, profile)
    counter = QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path)
    cases = legacy._load_cases(
        arguments.private_manifest,
        adapter,
        frozenset({legacy.DirectBenchmark(benchmark.value)}),
        prompt_asset_dir=arguments.prompt_asset_dir,
    )
    if tuple(case.task.task_id for case in cases) != panel.task_ids:
        raise ValueError("interactive case loader changed manifest order")
    cases = tuple(
        replace(
            case,
            task=replace(
                case.task,
                panel_index=index,
                service_slot=index % len(endpoints),
                history_token_counter=counter,
                context_length=execution.context_length,
            ),
        )
        for index, case in enumerate(cases)
    )
    require_fresh_output_directory(arguments.private_output_dir)
    manifest_identity = load_protocol13_environment_manifest_identity(
        arguments.private_manifest,
        execution=execution,
    )
    deployment_id = str(manifest_identity["deployment_id"])
    runtime_identity = manifest_identity["runtime"]
    if not isinstance(runtime_identity, dict):
        raise ValueError("interactive runtime identity is invalid")
    source_revision = str(runtime_identity.get("source_revision", ""))
    if not source_revision.strip():
        raise ValueError("interactive environment source revision is absent")
    extension = execution.interactive
    environment = RuntimeEnvironmentIdentity(
        environment_profile=execution.environment_profile,
        runtime_source_revision=source_revision,
        deployment_id=deployment_id,
        horizon_policy=extension.horizon_policy,
        max_steps_cap=extension.max_steps_cap,
        required_max_steps=extension.required_max_steps,
        history_window_steps=extension.history_window_steps,
        history_maximum_characters=extension.history_maximum_characters,
        include_reasoning_in_history=extension.include_reasoning_in_history,
        invalid_candidate_policy=extension.invalid_candidate_policy,
        invalid_environment_action_policy=extension.invalid_environment_action_policy,
        prompt_asset_id=extension.prompt_asset_id,
        prompt_asset_source_revision=extension.prompt_asset_source_revision,
        manifest_identity=manifest_identity,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- project Git identity
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    started_at = datetime.now(UTC).isoformat()
    common = {
        "attempt_id": arguments.attempt_id,
        "execution": execution.to_mapping(),
        "generation_profile": RuntimeGenerationProfile.from_direct_profile(profile),
        "environment": environment,
        "model_route": execution.actor_route,
        "service_instance_ids": tuple(item.service_instance_id for item in service_receipts),
        "context_length": execution.context_length,
        "adapter_active": False,
        "panel_manifest_id": execution.panel_manifest_id,
        "planned_task_ids": panel.task_ids,
        "generation_code_revision": revision,
        "scoring_code_revision": revision,
        "evaluator_version": execution.scorer_profile,
        "evaluator_contract": None,
        "started_at": started_at,
    }
    write_execution_receipt(
        arguments.private_output_dir / "execution-start.json",
        Protocol13ExecutionReceipt(
            format_version=START_FORMAT,
            status="running",
            response_model_ids=(),
            generated_task_ids=(),
            scored_task_ids=(),
            completed_at=None,
            **common,
        ),
    )
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
        for endpoint, receipt in zip(endpoints, service_receipts, strict=True)
    )
    if any(
        routes != (execution.actor_route,)
        for routes in await asyncio.gather(*(client.model_routes() for client in clients))
    ):
        raise RuntimeError("interactive endpoint exposes another model route")
    client = DeterministicReplicaClient(clients)
    semaphore = asyncio.Semaphore(arguments.concurrency)

    async def run_case(case: legacy._Case) -> NativeInteractiveAttempt:
        async with semaphore:
            try:
                native_environment = await asyncio.to_thread(case.create_environment)
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
            return await run_native_interactive_task(client, case.task, native_environment)

    attempts = tuple(await asyncio.gather(*(run_case(case) for case in cases)))
    manifest_value = json.loads(arguments.private_manifest.read_text(encoding="utf-8"))
    manifest_rows = manifest_value.get("cases")
    if not isinstance(manifest_rows, list):
        raise ValueError("interactive manifest cases are absent")
    case_metadata: dict[str, dict[str, object]] = {}
    for raw_case in manifest_rows:
        if not isinstance(raw_case, dict) or raw_case.get("benchmark") != benchmark.value:
            continue
        payload = raw_case.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("interactive case payload is absent")
        case_metadata[str(raw_case["task_id"])] = {
            "source_split": payload.get("split"),
            "task_type": payload.get("task_type"),
            "max_steps": raw_case.get("max_steps"),
        }
    if tuple(case_metadata) != panel.task_ids:
        raise ValueError("interactive diagnostic metadata changed panel order")
    with (arguments.private_output_dir / "per-task-results.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for attempt in attempts:
            stream.write(
                json.dumps(
                    {**legacy._attempt_value(attempt), **case_metadata[attempt.task_id]},
                    sort_keys=True,
                )
                + "\n"
            )
    aggregate = cast(dict[str, object], legacy._aggregate(attempts, adapter))
    (arguments.private_output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    task_ids = tuple(attempt.task_id for attempt in attempts)
    response_models = tuple(
        dict.fromkeys(
            step.response_model
            for attempt in attempts
            for step in attempt.trace
            if step.response_model is not None
        )
    )
    observed_service_ids = {
        step.service_instance_id
        for attempt in attempts
        for step in attempt.trace
        if step.service_instance_id is not None
    }
    if observed_service_ids != {item.service_instance_id for item in service_receipts}:
        raise ValueError("interactive generation did not exercise the declared replica set")
    complete = Protocol13ExecutionReceipt(
        format_version=COMPLETE_FORMAT,
        status="complete",
        response_model_ids=response_models,
        generated_task_ids=task_ids,
        scored_task_ids=task_ids,
        completed_at=datetime.now(UTC).isoformat(),
        **common,
    )
    complete.validate_against(
        execution=execution,
        expected_task_ids=panel.task_ids,
        expected_generation_profile=profile,
        expected_environment_manifest=environment.manifest_identity,
    )
    write_execution_receipt(arguments.private_output_dir / "execution-complete.json", complete)
    return aggregate


def main() -> None:
    print(json.dumps(asyncio.run(run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
