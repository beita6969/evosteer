#!/usr/bin/env python3
"""Run one source-authoritative corrected Protocol 14 native agent panel."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import run_qwen35_direct_interactive as legacy  # type: ignore[import-not-found]
from skillev_private.direct_reference.journal import (  # type: ignore[import-untyped]
    require_fresh_output_directory,
)
from skillev_private.direct_reference.protocol14_runner import (
    build_protocol14_interactive_adapter,
    load_protocol14_panel_manifest,
    profile_from_execution,
)  # type: ignore[import-untyped]

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.contracts import ExecutionContractV4
from skillev.evaluation.current_iid.protocol14.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol14ExecutionReceipt,
    RuntimeEnvironmentIdentity,
    RuntimeGenerationProfile,
    write_execution_receipt,
)
from skillev.evaluation.direct_baseline import (
    NativeInteractiveAttempt,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_native_interactive_task,
)
from skillev.experiments.protocol_v14 import load_protocol_v14


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("webshop", "alfworld"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--attempt-role", choices=("reference", "candidate", "canary"), required=True
    )
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--runtime-model-revision", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def _environment_identity(
    manifest_path: Path,
    benchmark: Protocol14Benchmark,
    execution_environment: str,
) -> tuple[str, str]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [row for row in raw["cases"] if row["benchmark"] == benchmark.value]
    deployments = tuple(dict.fromkeys(str(row["deployment"]) for row in cases))
    if len(deployments) != 1:
        raise ValueError("formal interactive panel requires one deployment identity")
    deployment = raw["deployments"][deployments[0]]
    runtime = raw["runtimes"][deployment["runtime"]]
    revision = str(runtime["source_revision"])
    if not revision.strip() or not execution_environment.strip():
        raise ValueError("interactive environment identity is incomplete")
    return deployments[0], revision


def _write_runtime_manifest(
    source: Path,
    destination: Path,
    benchmark: Protocol14Benchmark,
    execution: ExecutionContractV4,
) -> Path:
    """Bind the frozen payload to the corrected source contract without editing it in place."""

    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or execution.interactive is None:
        raise ValueError("interactive manifest or execution is invalid")
    contracts = raw.get("benchmarks")
    cases = raw.get("cases")
    if not isinstance(contracts, dict) or not isinstance(cases, list):
        raise ValueError("interactive manifest contract fields are invalid")
    contract = contracts.get(benchmark.value)
    if not isinstance(contract, dict):
        raise ValueError("interactive benchmark contract is missing")
    extension = execution.interactive
    contract.update(
        {
            "expected_count": execution.expected_count,
            "prompt_profile": execution.prompt_profile,
            "decoding_profile": execution.decoding_profile,
            "parser_profile": execution.parser_profile,
            "environment_contract": execution.environment_profile,
            "invalid_candidate_policy": extension.invalid_candidate_policy,
            "invalid_environment_action_policy": extension.invalid_environment_action_policy,
            "history_window_steps": extension.history_window_steps,
            "history_maximum_characters": extension.history_maximum_characters,
            "horizon_policy": extension.horizon_policy,
            "max_steps_cap": extension.max_steps,
            "include_reasoning_in_history": extension.include_reasoning_in_history,
        }
    )
    for value in cases:
        if isinstance(value, dict) and value.get("benchmark") == benchmark.value:
            value["max_steps"] = min(int(value["max_steps"]), extension.max_steps)
    destination.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    return destination


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_protocol_v14(arguments.protocol, arguments.sources)
    if arguments.attempt_role == "candidate":
        protocol.require_execution_ready()
    executions = load_execution_contracts_v4(arguments.conditions, protocol=protocol)
    benchmark = Protocol14Benchmark(arguments.benchmark)
    execution = executions[benchmark]
    if execution.interactive is None or execution.environment_profile is None:
        raise ValueError("benchmark is not an interactive Protocol 14 condition")
    profile = profile_from_execution(execution)
    if (
        arguments.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
        or arguments.runtime_model_revision != execution.actor_model_revision
    ):
        raise ValueError("interactive runtime model identity differs from Protocol 14")
    panel = load_protocol14_panel_manifest(arguments.private_manifest, execution=execution)
    adapter = build_protocol14_interactive_adapter(execution, profile, panel)
    require_fresh_output_directory(arguments.private_output_dir)
    runtime_manifest = _write_runtime_manifest(
        arguments.private_manifest,
        arguments.private_output_dir / "runtime-manifest.json",
        benchmark,
        execution,
    )
    cases = legacy._load_cases(
        runtime_manifest,
        adapter,
        frozenset({legacy.DirectBenchmark(benchmark.value)}),
        prompt_asset_dir=None,
    )
    if tuple(case.task.task_id for case in cases) != panel.task_ids:
        raise ValueError("interactive case loader changed manifest order")
    deployment_id, source_revision = _environment_identity(
        arguments.private_manifest, benchmark, execution.environment_profile
    )
    extension = execution.interactive
    environment = RuntimeEnvironmentIdentity(
        environment_profile=execution.environment_profile,
        runtime_source_revision=source_revision,
        deployment_id=deployment_id,
        horizon_policy=extension.horizon_policy,
        max_steps=extension.max_steps,
        history_window_steps=extension.history_window_steps,
        history_maximum_characters=extension.history_maximum_characters,
        history_observation_characters=extension.history_observation_characters,
        include_reasoning_in_history=extension.include_reasoning_in_history,
        invalid_candidate_policy=extension.invalid_candidate_policy,
        invalid_environment_action_policy=extension.invalid_environment_action_policy,
        prompt_source_revision=extension.prompt_source_revision,
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
        "attempt_role": arguments.attempt_role,
        "execution": execution.to_mapping(),
        "generation_profile": RuntimeGenerationProfile.from_direct_profile(profile),
        "environment": environment,
        "model_route": execution.actor_route,
        "runtime_model_revision": arguments.runtime_model_revision,
        "context_length": execution.context_length,
        "adapter_active": False,
        "panel_manifest_id": execution.panel_manifest_id,
        "planned_task_ids": panel.task_ids,
        "generation_code_revision": revision,
        "scoring_code_revision": revision,
        "evaluator_version": execution.scorer_profile,
        "started_at": started_at,
    }
    write_execution_receipt(
        arguments.private_output_dir / "execution-start.json",
        Protocol14ExecutionReceipt(
            format_version=START_FORMAT,
            status="running",
            response_model_ids=(),
            generated_task_ids=(),
            terminal_task_ids=(),
            completed_at=None,
            **common,
        ),
    )
    client = OpenAICompatibleDirectClient(
        endpoint_base=arguments.endpoint_base,
        served_model_name=execution.actor_route,
        api_key="EMPTY",
        timeout_seconds=arguments.request_timeout_seconds,
        context_length=execution.context_length,
        token_counter=QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path),  # type: ignore[arg-type]
    )
    if await client.model_routes() != (execution.actor_route,):
        raise RuntimeError("interactive endpoint exposes another model route")
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
    with (arguments.private_output_dir / "per-task-results.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for attempt in attempts:
            stream.write(json.dumps(legacy._attempt_value(attempt), sort_keys=True) + "\n")
    aggregate = cast(dict[str, object], legacy._aggregate(attempts, adapter))
    (arguments.private_output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    task_ids = tuple(attempt.task_id for attempt in attempts)
    terminal_ids = tuple(
        attempt.task_id for attempt in attempts if attempt.infrastructure_error is None
    )
    complete = Protocol14ExecutionReceipt(
        format_version=COMPLETE_FORMAT,
        status="complete",
        response_model_ids=(execution.actor_route,),
        generated_task_ids=task_ids,
        terminal_task_ids=terminal_ids,
        completed_at=datetime.now(UTC).isoformat(),
        **common,
    )
    complete.validate_against(execution=execution, expected_task_ids=panel.task_ids)
    write_execution_receipt(arguments.private_output_dir / "execution-complete.json", complete)
    return aggregate


def main() -> None:
    print(json.dumps(asyncio.run(run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
