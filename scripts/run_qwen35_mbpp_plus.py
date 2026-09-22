#!/usr/bin/env python3
"""Run the frozen Random(0)-128 MBPP+ Base+Plus backbone diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from skillev_private.benchmarks.evalplus_adapter import EvalPlusCarrierReceipt
from skillev_private.benchmarks.evalplus_process import (
    EvalPlusExecutionContract,
    run_evalplus_26d6d00,
)
from skillev_private.benchmarks.evalplus_result_v26d6d00 import (
    parse_evalplus_26d6d00_task,
)
from skillev_private.direct_reference.manifests import load_population_manifest
from skillev_private.direct_reference.protocol13_runner import (
    load_protocol13_panel_manifest,
)

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.current_iid.protocol13.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol13ExecutionReceipt,
    RuntimeGenerationProfile,
    load_execution_receipt,
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
from skillev.evaluation.direct_baseline.client import (
    DirectGenerationError,
    DirectGenerationRequest,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from skillev.evaluation.direct_baseline.config import DirectDecodingProfile
from skillev.evaluation.direct_baseline.parsing import parse_python_source
from skillev.evaluation.direct_baseline.prompts import render_static_messages
from skillev.experiments.protocol_v13 import load_protocol_v13


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-base", action="append", required=True)
    parser.add_argument("--served-model-name", default="qwen35-direct-base")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--service-contract", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, action="append", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--evaluator-timeout-seconds", type=int, default=7_200)
    parser.add_argument("--evalplus-source-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("generate", "score", "all"), default="all")
    parser.add_argument(
        "--runner-config",
        type=Path,
        default=Path("configs/evaluation/qwen35_protocol13_iid.yaml"),
    )
    return parser.parse_args()


def _profile(path: Path) -> DirectDecodingProfile:
    return load_protocol13_runner_profiles(path).require("qwen35-mbpp-plus-deterministic@1")


def _current_python_executable() -> Path:
    """Keep the venv launcher path instead of resolving it to the base interpreter."""

    return Path(os.path.abspath(sys.executable))


def _load_generation_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("MBPP+ generation journal row is not an object")
            rows.append(value)
    return rows


async def _main(arguments: argparse.Namespace) -> None:
    if arguments.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    protocol13 = load_protocol_v13(arguments.protocol, arguments.sources)
    executions = load_execution_contracts_v3(arguments.conditions, protocol=protocol13)
    execution = executions[Protocol13Benchmark.MBPP_PLUS]
    runner_registry = load_protocol13_runner_profiles(arguments.runner_config)
    if (
        arguments.served_model_name != execution.actor_route
        or runner_registry.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
        or runner_registry.context_length != execution.context_length
    ):
        raise ValueError("MBPP+ runtime model identity differs from Protocol 13")
    panel = load_protocol13_panel_manifest(arguments.manifest, execution=execution)
    if arguments.phase in {"generate", "all"}:
        arguments.output.mkdir(parents=True, exist_ok=False)
    elif not arguments.output.is_dir():
        raise ValueError("MBPP+ score phase requires an existing generation directory")
    evalplus_root = arguments.evalplus_source_root.resolve()
    if str(evalplus_root) not in sys.path:
        sys.path.insert(0, str(evalplus_root))
    from evalplus.data import get_mbpp_plus  # type: ignore[import-not-found]

    official = get_mbpp_plus(version="v0.2.0")
    task_ids = sorted(official)
    manifest = load_population_manifest(arguments.manifest)
    if (
        manifest.benchmark.value != "mbpp-plus"
        or manifest.population_id != "mbpp-plus-v0.2.0-random0-128-v13"
        or manifest.dataset_revision != "evalplus-mbppplus-v0.2.0"
        or manifest.selection_rule != "frozen-manifest-random0-128"
    ):
        raise ValueError("MBPP+ manifest differs from Protocol 13")
    selected_ids = tuple(entry.source_identity for entry in manifest.entries)
    if tuple(entry.task_id for entry in manifest.entries) != panel.task_ids:
        raise ValueError("MBPP+ manifest task order differs from Protocol 13 panel")
    if len(set(selected_ids)) != 128 or any(task_id not in official for task_id in selected_ids):
        raise ValueError("MBPP+ manifest does not select 128 unique official tasks")
    prompts = {task_id: str(official[task_id]["prompt"]) for task_id in selected_ids}
    if len(prompts) != 128 or any(not value.strip() for value in prompts.values()):
        raise RuntimeError("official MBPP+ prompt coverage differs")
    profile = _profile(arguments.runner_config)
    evalplus_contract = EvalPlusExecutionContract(
        source_root=evalplus_root,
        python_executable=_current_python_executable(),
        code_revision="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        dataset_name="mbpp",
        dataset_version="v0.2.0",
        result_schema_profile="evalplus-task-list-base-plus-status@26d6d00",
        min_time_limit=0.1,
        gt_time_limit_factor=2.0,
        parallelism=min(arguments.concurrency, 8),
        timeout_seconds=arguments.evaluator_timeout_seconds,
    )
    if execution.evaluator_profile != evalplus_contract.profile_id:
        raise ValueError("MBPP+ evaluator profile differs from Protocol 13")
    endpoints = tuple(dict.fromkeys(arguments.endpoint_base))
    service_contract = load_model_service_contract(arguments.service_contract)
    service_receipts = tuple(load_model_service_receipt(path) for path in arguments.service_receipt)
    if len(endpoints) != len(service_receipts):
        raise ValueError("MBPP+ requires one ordered service receipt per endpoint")
    validate_replica_services(
        service_receipts,
        contract=service_contract,
        execution=execution,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- project Git identity
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_start_path = arguments.output / "execution-start.json"
    existing_start = (
        load_execution_receipt(runtime_start_path) if arguments.phase == "score" else None
    )
    started_at = (
        existing_start.started_at if existing_start is not None else datetime.now(UTC).isoformat()
    )
    receipt_common = {
        "attempt_id": arguments.attempt_id,
        "execution": execution.to_mapping(),
        "generation_profile": RuntimeGenerationProfile.from_direct_profile(profile),
        "environment": None,
        "model_route": execution.actor_route,
        "service_instance_ids": tuple(receipt.service_instance_id for receipt in service_receipts),
        "context_length": execution.context_length,
        "adapter_active": False,
        "panel_manifest_id": execution.panel_manifest_id,
        "planned_task_ids": panel.task_ids,
        "generation_code_revision": (
            existing_start.generation_code_revision if existing_start is not None else revision
        ),
        "scoring_code_revision": revision,
        "evaluator_version": evalplus_contract.profile_id,
        "evaluator_contract": evalplus_contract.to_mapping(),
        "started_at": started_at,
    }
    if existing_start is None:
        write_execution_receipt(
            runtime_start_path,
            Protocol13ExecutionReceipt(
                format_version=START_FORMAT,
                status="running",
                response_model_ids=(),
                generated_task_ids=(),
                scored_task_ids=(),
                completed_at=None,
                **receipt_common,
            ),
        )
    else:
        expected_start = Protocol13ExecutionReceipt(
            format_version=START_FORMAT,
            status="running",
            response_model_ids=(),
            generated_task_ids=(),
            scored_task_ids=(),
            completed_at=None,
            **receipt_common,
        )
        if existing_start != expected_start:
            raise ValueError("MBPP+ score phase differs from its generation phase")
    generation_path = arguments.output / "generations.jsonl"
    generation_summary_path = arguments.output / "generation-summary.json"
    carrier = arguments.output / "official-samples-full-carrier.jsonl"
    if arguments.phase in {"generate", "all"}:
        counter = QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path)
        clients = tuple(
            OpenAICompatibleDirectClient(
                endpoint_base=endpoint,
                served_model_name=arguments.served_model_name,
                timeout_seconds=3_600,
                context_length=arguments.context_length,
                token_counter=counter,  # type: ignore[arg-type]
                service_instance_id=receipt.service_instance_id,
            )
            for endpoint, receipt in zip(endpoints, service_receipts, strict=True)
        )
        if any(
            routes != (execution.actor_route,)
            for routes in await asyncio.gather(*(client.model_routes() for client in clients))
        ):
            raise RuntimeError("MBPP+ endpoint exposes another model route")
        semaphore = asyncio.Semaphore(arguments.concurrency)
        generations: list[dict[str, object] | None] = [None] * 128

        async def generate(index: int, task_id: str) -> None:
            async with semaphore:
                started = time.monotonic()
                try:
                    result = await clients[index % len(clients)].generate(
                        DirectGenerationRequest(
                            request_id=task_id,
                            messages=render_static_messages(
                                "evalplus-mbpp-complete-code@1", prompts[task_id]
                            ),
                            profile=profile,
                            service_slot=index % len(clients),
                        )
                    )
                except DirectGenerationError as error:
                    generations[index] = {
                        "task_id": task_id,
                        "infrastructure_error": type(error).__name__,
                        "wall_time_seconds": time.monotonic() - started,
                    }
                    return
                parsed = parse_python_source(result.text)
                generations[index] = {
                    "task_id": task_id,
                    "raw_text": result.text,
                    "reasoning_text": result.reasoning_text,
                    "submission": parsed.value,
                    "parse_status": parsed.status.value,
                    "parse_reason": parsed.reason.value,
                    "finish_reason": result.finish_reason,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "response_model": result.response_model,
                    "response_id": result.response_id,
                    "service_instance_id": result.service_instance_id,
                    "service_slot": index % len(clients),
                    "wall_time_seconds": time.monotonic() - started,
                }

        generation_started = time.monotonic()
        await asyncio.gather(
            *(generate(index, task_id) for index, task_id in enumerate(selected_ids))
        )
        generation_seconds = time.monotonic() - generation_started
        generation_rows = [item for item in generations if item is not None]
        if len(generation_rows) != 128:
            raise RuntimeError("MBPP+ generation outcomes do not conserve the panel")
        with generation_path.open("x", encoding="utf-8") as stream:
            for item in generation_rows:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        by_task_id = {str(item["task_id"]): item for item in generation_rows}
        with carrier.open("x", encoding="utf-8") as stream:
            for task_id in task_ids:
                generated = by_task_id.get(task_id)
                solution = None if generated is None else generated.get("submission")
                if not isinstance(solution, str) or not solution.strip():
                    solution = "raise NotImplementedError"
                stream.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
        generation_summary_path.write_text(
            json.dumps(
                {
                    "format": "skillev-mbpp-plus-generation-summary@1",
                    "attempt_id": arguments.attempt_id,
                    "generation_seconds": generation_seconds,
                    "generated_count": len(generation_rows),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if arguments.phase == "generate":
            print(generation_summary_path.read_text(encoding="utf-8"), flush=True)
            return
    else:
        generation_rows = _load_generation_rows(generation_path)
        summary = json.loads(generation_summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("attempt_id") != arguments.attempt_id:
            raise ValueError("MBPP+ generation summary belongs to another attempt")
        generation_seconds = float(summary["generation_seconds"])
        by_task_id = {str(item.get("task_id")): item for item in generation_rows}
        if tuple(by_task_id) != selected_ids or not carrier.is_file():
            raise ValueError("MBPP+ score phase generation population differs")
    carrier_receipt = EvalPlusCarrierReceipt(
        dataset_task_count=len(task_ids),
        selected_task_count=len(selected_ids),
        placeholder_task_count=len(task_ids) - len(selected_ids),
        selected_task_ids=selected_ids,
        headline_denominator=len(selected_ids),
    )
    scoring_started = time.monotonic()
    result_path = arguments.output / "evalplus-results.json"
    evaluator_payload = await asyncio.to_thread(
        run_evalplus_26d6d00,
        evalplus_contract,
        carrier=carrier,
        output_file=result_path,
        stdout=arguments.output / "official-evaluator.stdout.log",
        stderr=arguments.output / "official-evaluator.stderr.log",
    )
    scoring_seconds = time.monotonic() - scoring_started
    evaluator_results = evaluator_payload["eval"]
    if not isinstance(evaluator_results, dict):
        raise RuntimeError("EvalPlus result is not keyed by task ID")
    verdict_rows: list[dict[str, object]] = []
    for task_id in selected_ids:
        generation = by_task_id[task_id]
        if "infrastructure_error" in generation:
            verdict_rows.append({"task_id": task_id, "outcome": "generation-infrastructure"})
            continue
        submission = generation.get("submission")
        if not isinstance(submission, str) or not submission.strip():
            verdict_rows.append({"task_id": task_id, "outcome": "candidate-invalid"})
            continue
        result = evaluator_results.get(task_id)
        if result is None:
            verdict_rows.append({"task_id": task_id, "outcome": "scorer-infrastructure"})
            continue
        try:
            verdict = parse_evalplus_26d6d00_task(task_id, result)
        except ValueError:
            verdict_rows.append({"task_id": task_id, "outcome": "scorer-infrastructure"})
            continue
        base_passed = verdict.base_passed
        plus_passed = verdict.plus_passed
        verdict_rows.append(
            {
                "task_id": task_id,
                "outcome": "scored-success" if base_passed and plus_passed else "scored-failure",
                "base_passed": base_passed,
                "plus_passed": plus_passed,
            }
        )
    if len(verdict_rows) != 128:
        raise RuntimeError("MBPP+ verdict outcomes do not conserve the panel")
    with (arguments.output / "verdicts.jsonl").open("x", encoding="utf-8") as stream:
        for item in verdict_rows:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = {
        kind: sum(item["outcome"] == kind for item in verdict_rows)
        for kind in (
            "scored-success",
            "scored-failure",
            "candidate-invalid",
            "generation-infrastructure",
            "scorer-infrastructure",
        )
    }
    base_failures = sum(
        item["outcome"] == "scored-failure" and item.get("base_passed") is False
        for item in verdict_rows
    )
    plus_failures = sum(
        item["outcome"] == "scored-failure"
        and item.get("base_passed") is True
        and item.get("plus_passed") is False
        for item in verdict_rows
    )
    if base_failures + plus_failures != counts["scored-failure"]:
        raise RuntimeError("MBPP+ failure taxonomy does not conserve scored failures")
    report = {
        "format": "skillev-current-mbpp-plus-result@2",
        "population_id": manifest.population_id,
        "decoding_profile": profile.profile_id,
        "count": 128,
        "outcomes": counts,
        "scored_failure_detail": {
            "base_failure": base_failures,
            "plus_only_failure": plus_failures,
        },
        "evalplus_version": "v0.2.0",
        "evalplus_code_revision": evalplus_contract.code_revision,
        "evalplus_result_schema": evalplus_contract.result_schema_profile,
        "carrier": {
            "dataset_task_count": carrier_receipt.dataset_task_count,
            "selected_task_count": carrier_receipt.selected_task_count,
            "placeholder_task_count": carrier_receipt.placeholder_task_count,
            "headline_denominator": carrier_receipt.headline_denominator,
        },
        "base_plus_pass_at_1_percent": 100.0 * counts["scored-success"] / 128,
        "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "wall_time_seconds": generation_seconds + scoring_seconds,
        "throughput_records_per_hour": 128 * 3_600 / (generation_seconds + scoring_seconds),
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if counts["generation-infrastructure"] or counts["scorer-infrastructure"]:
        raise RuntimeError(
            "MBPP+ run has infrastructure failures and cannot produce a complete receipt"
        )
    response_models = tuple(
        dict.fromkeys(
            str(item["response_model"])
            for item in generation_rows
            if isinstance(item.get("response_model"), str)
        )
    )
    complete = Protocol13ExecutionReceipt(
        format_version=COMPLETE_FORMAT,
        status="complete",
        response_model_ids=response_models,
        generated_task_ids=panel.task_ids,
        scored_task_ids=tuple(str(item["task_id"]) for item in verdict_rows),
        completed_at=datetime.now(UTC).isoformat(),
        **receipt_common,
    )
    observed_services = tuple(
        dict.fromkeys(
            str(item["service_instance_id"])
            for item in generation_rows
            if isinstance(item.get("service_instance_id"), str)
        )
    )
    if set(observed_services) != {receipt.service_instance_id for receipt in service_receipts}:
        raise RuntimeError("MBPP+ did not exercise the declared model services")
    complete.validate_against(
        execution=execution,
        expected_task_ids=panel.task_ids,
        expected_generation_profile=profile,
        expected_evaluator_contract=evalplus_contract.to_mapping(),
    )
    write_execution_receipt(arguments.output / "execution-complete.json", complete)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    asyncio.run(_main(_arguments()))
