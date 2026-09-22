#!/usr/bin/env python3
"""Run one manifest-authoritative static Protocol 13 benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from skillev_private.benchmarks.code_math import (
    CodeExecutionInfrastructureError,
    MathEquivalenceInfrastructureError,
)
from skillev_private.benchmarks.humaneval_official import (
    current_humaneval_executor_contract,
)
from skillev_private.direct_reference.evaluators import (
    SCORER_REGISTRY,
    require_protocol13_scorer_profile,
)
from skillev_private.direct_reference.journal import (
    PrivateGenerationJournal,
    require_fresh_output_directory,
)
from skillev_private.direct_reference.protocol13_runner import (
    load_protocol13_panel_manifest,
    load_protocol13_static_cases,
)

from skillev.evaluation.current_iid.protocol13.aime_profiles import (
    load_aime_profile_registry,
)
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.current_iid.protocol13.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol13ExecutionReceipt,
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
    DirectAttempt,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_direct_tasks,
)
from skillev.evaluation.direct_baseline.client import (
    DeterministicReplicaClient,
)
from skillev.experiments.protocol_v13 import load_protocol_v13

_STATIC = {
    Protocol13Benchmark.HOTPOT_QA,
    Protocol13Benchmark.TRIVIA_QA,
    Protocol13Benchmark.AIME_2026,
    Protocol13Benchmark.HUMAN_EVAL,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=sorted(item.value for item in _STATIC), required=True
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--endpoint-base", action="append", required=True)
    parser.add_argument("--service-contract", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, action="append", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--aime-profile-config",
        type=Path,
        default=Path("configs/evaluation/aime_protocol_candidates.yaml"),
    )
    parser.add_argument("--aime-profile-id", default="qwen35-thinking-aime-boxed-candidate@3")
    return parser.parse_args()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- project Git identity
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    endpoints = tuple(arguments.endpoint_base)
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("Protocol 13 endpoint order must be unique")
    if len(arguments.service_receipt) != len(endpoints):
        raise ValueError("endpoint and service receipt counts differ")
    protocol = load_protocol_v13(arguments.protocol, arguments.sources)
    executions = load_execution_contracts_v3(arguments.conditions, protocol=protocol)
    registry = load_protocol13_runner_profiles(arguments.runner_config)
    benchmark = Protocol13Benchmark(arguments.benchmark)
    execution = executions[benchmark]
    require_protocol13_scorer_profile(execution.scorer_profile)
    profile = registry.require(execution.decoding_profile)
    if benchmark is Protocol13Benchmark.AIME_2026:
        aime_profile = load_aime_profile_registry(arguments.aime_profile_config).require(
            arguments.aime_profile_id,
            calibration_year=2025,
        )
        observed = (
            aime_profile.prompt_profile,
            aime_profile.parser_profile,
            aime_profile.decoding_profile,
            aime_profile.enable_thinking,
            aime_profile.temperature,
            aime_profile.top_p,
            aime_profile.top_k,
            aime_profile.presence_penalty,
            aime_profile.max_new_tokens,
            aime_profile.seed,
        )
        expected = (
            execution.prompt_profile,
            execution.parser_profile,
            execution.decoding_profile,
            profile.enable_thinking,
            profile.temperature,
            profile.top_p,
            profile.top_k,
            profile.presence_penalty,
            profile.max_new_tokens,
            profile.seed,
        )
        if observed != expected or aime_profile.formal_reference_eligible:
            raise ValueError("AIME runtime differs from its typed pre-final evidence")
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
        raise ValueError("runtime model route or context differs from Protocol 13")
    panel = load_protocol13_panel_manifest(arguments.manifest, execution=execution)
    cases = load_protocol13_static_cases(
        source=arguments.source,
        manifest_path=arguments.manifest,
        execution=execution,
        profile=profile,
    )
    cases = tuple(
        replace(
            case,
            public_task=replace(
                case.public_task,
                panel_index=index,
                service_slot=index % len(endpoints),
            ),
        )
        for index, case in enumerate(cases)
    )
    require_fresh_output_directory(arguments.private_output_dir)
    evaluator_contract = (
        current_humaneval_executor_contract()
        if benchmark is Protocol13Benchmark.HUMAN_EVAL
        else None
    )
    if execution.evaluator_profile != (
        evaluator_contract.profile_id if evaluator_contract is not None else None
    ):
        raise ValueError("runtime evaluator profile differs from Protocol 13")
    revision = _revision()
    started_at = datetime.now(UTC).isoformat()
    common = {
        "attempt_id": arguments.attempt_id,
        "execution": execution.to_mapping(),
        "generation_profile": RuntimeGenerationProfile.from_direct_profile(profile),
        "environment": None,
        "model_route": execution.actor_route,
        "service_instance_ids": tuple(item.service_instance_id for item in service_receipts),
        "context_length": execution.context_length,
        "adapter_active": False,
        "panel_manifest_id": execution.panel_manifest_id,
        "planned_task_ids": panel.task_ids,
        "generation_code_revision": revision,
        "scoring_code_revision": revision,
        "evaluator_version": (
            evaluator_contract.profile_id
            if evaluator_contract is not None
            else execution.scorer_profile
        ),
        "evaluator_contract": (
            evaluator_contract.to_mapping() if evaluator_contract is not None else None
        ),
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
    counter = QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path)
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
    journal = PrivateGenerationJournal(
        (arguments.private_output_dir / "generations.jsonl").resolve(),
        append_existing=False,
    )
    async with journal:
        attempts = await run_direct_tasks(
            DeterministicReplicaClient(clients),
            tuple(case.public_task for case in cases),
            concurrency=arguments.concurrency,
            generation_observer=journal.append,
            request_attempt=1,
        )
    case_by_id = {case.public_task.task_id: case for case in cases}
    rows: list[dict[str, object]] = []
    scored_task_ids: list[str] = []
    for attempt in attempts:
        row, definitive = await _score(
            attempt, case_by_id[attempt.task_id], execution.scorer_profile
        )
        rows.append(row)
        if definitive:
            scored_task_ids.append(attempt.task_id)
    with (arguments.private_output_dir / "per-task-results.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    generated_ids = tuple(item.task_id for item in attempts)
    response_models = tuple(
        dict.fromkeys(item.response_model for item in attempts if item.response_model is not None)
    )
    complete = Protocol13ExecutionReceipt(
        format_version=COMPLETE_FORMAT,
        status="complete",
        response_model_ids=response_models,
        generated_task_ids=generated_ids,
        scored_task_ids=tuple(scored_task_ids),
        completed_at=datetime.now(UTC).isoformat(),
        **common,
    )
    observed_services = {
        item.service_instance_id for item in attempts if item.service_instance_id is not None
    }
    if observed_services != {item.service_instance_id for item in service_receipts}:
        raise RuntimeError("static run did not exercise the declared model services")
    complete.validate_against(
        execution=execution,
        expected_task_ids=panel.task_ids,
        expected_generation_profile=profile,
        expected_evaluator_contract=(
            evaluator_contract.to_mapping() if evaluator_contract is not None else None
        ),
    )
    write_execution_receipt(arguments.private_output_dir / "execution-complete.json", complete)
    summary = {
        "format": "skillev-protocol13-static-summary@1",
        "benchmark": benchmark.value,
        "planned": len(panel.task_ids),
        "generated": len(generated_ids),
        "scored": len(scored_task_ids),
        "attempt_id": arguments.attempt_id,
    }
    (arguments.private_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


async def _score(
    attempt: DirectAttempt, case: object, scorer_profile: str
) -> tuple[dict[str, object], bool]:
    row: dict[str, object] = {
        "task_id": attempt.task_id,
        "benchmark": attempt.benchmark.value,
        "parse_status": attempt.parsed.status.value if attempt.parsed else None,
        "parse_reason": attempt.parsed.reason.value if attempt.parsed else None,
        "metrics": None,
        "finish_reason": attempt.finish_reason,
        "generation_infrastructure_error": attempt.infrastructure_error,
        "scorer_infrastructure_error": None,
    }
    if attempt.infrastructure_error is not None:
        return row, False
    if attempt.parsed is None:
        raise RuntimeError("successful generation lacks a parsed response")
    try:
        score = await SCORER_REGISTRY[scorer_profile](case, attempt.parsed)  # type: ignore[arg-type]
    except (CodeExecutionInfrastructureError, MathEquivalenceInfrastructureError) as exc:
        row["scorer_infrastructure_error"] = type(exc).__name__
        return row, False
    row["metrics"] = score.metrics
    return row, True


def main() -> None:
    print(json.dumps(asyncio.run(run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
