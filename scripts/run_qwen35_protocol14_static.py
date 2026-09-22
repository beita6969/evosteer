#!/usr/bin/env python3
"""Run one manifest-authoritative corrected static Protocol 14 benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from skillev_private.benchmarks.code_math import (
    CodeExecutionInfrastructureError,
    MathEquivalenceInfrastructureError,
)
from skillev_private.direct_reference.evaluators import SCORER_REGISTRY
from skillev_private.direct_reference.journal import (
    PrivateGenerationJournal,
    require_fresh_output_directory,
)
from skillev_private.direct_reference.protocol14_runner import (
    load_protocol14_panel_manifest,
    load_protocol14_static_cases,
    profile_from_execution,
)

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol14ExecutionReceipt,
    RuntimeGenerationProfile,
    write_execution_receipt,
)
from skillev.evaluation.direct_baseline import (
    DirectAttempt,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_direct_tasks,
)
from skillev.evaluation.direct_baseline.client import (
    DirectGenerationRequest,
    DirectGenerationResult,
)
from skillev.experiments.protocol_v14 import load_protocol_v14

_STATIC = {
    Protocol14Benchmark.HOTPOT_QA,
    Protocol14Benchmark.TRIVIA_QA,
    Protocol14Benchmark.AIME_2026,
    Protocol14Benchmark.HUMAN_EVAL,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=sorted(item.value for item in _STATIC), required=True
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--attempt-role", choices=("reference", "candidate", "canary"), required=True
    )
    parser.add_argument("--endpoint-base", action="append", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--runtime-model-revision", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


@dataclass(slots=True)
class _RoundRobinClient:
    clients: tuple[OpenAICompatibleDirectClient, ...]
    next_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        async with self.lock:
            client = self.clients[self.next_index % len(self.clients)]
            self.next_index += 1
        return await client.generate(request)


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
    protocol = load_protocol_v14(arguments.protocol, arguments.sources)
    if arguments.attempt_role == "candidate":
        protocol.require_execution_ready()
    executions = load_execution_contracts_v4(arguments.conditions, protocol=protocol)
    benchmark = Protocol14Benchmark(arguments.benchmark)
    execution = executions[benchmark]
    profile = profile_from_execution(execution)
    if (
        arguments.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
        or arguments.runtime_model_revision != execution.actor_model_revision
    ):
        raise ValueError("runtime model identity differs from Protocol 14")
    panel = load_protocol14_panel_manifest(arguments.manifest, execution=execution)
    cases = load_protocol14_static_cases(
        source=arguments.source,
        manifest_path=arguments.manifest,
        execution=execution,
        profile=profile,
    )
    require_fresh_output_directory(arguments.private_output_dir)
    revision = _revision()
    started_at = datetime.now(UTC).isoformat()
    common = {
        "attempt_id": arguments.attempt_id,
        "attempt_role": arguments.attempt_role,
        "execution": execution.to_mapping(),
        "generation_profile": RuntimeGenerationProfile.from_direct_profile(profile),
        "environment": None,
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
    start = Protocol14ExecutionReceipt(
        format_version=START_FORMAT,
        status="running",
        response_model_ids=(),
        generated_task_ids=(),
        terminal_task_ids=(),
        completed_at=None,
        **common,
    )
    start.validate_against(execution=execution, expected_task_ids=panel.task_ids)
    write_execution_receipt(arguments.private_output_dir / "execution-start.json", start)
    counter = QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path)
    clients = tuple(
        OpenAICompatibleDirectClient(
            endpoint_base=endpoint,
            served_model_name=execution.actor_route,
            api_key="EMPTY",
            timeout_seconds=arguments.request_timeout_seconds,
            context_length=execution.context_length,
            token_counter=counter,  # type: ignore[arg-type]
        )
        for endpoint in dict.fromkeys(arguments.endpoint_base)
    )
    journal = PrivateGenerationJournal(
        (arguments.private_output_dir / "generations.jsonl").resolve(),
        append_existing=False,
    )
    async with journal:
        attempts = await run_direct_tasks(
            _RoundRobinClient(clients),
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
    complete = Protocol14ExecutionReceipt(
        format_version=COMPLETE_FORMAT,
        status="complete",
        response_model_ids=response_models,
        generated_task_ids=generated_ids,
        terminal_task_ids=tuple(scored_task_ids),
        completed_at=datetime.now(UTC).isoformat(),
        **common,
    )
    complete.validate_against(execution=execution, expected_task_ids=panel.task_ids)
    write_execution_receipt(arguments.private_output_dir / "execution-complete.json", complete)
    summary = {
        "format": "skillev-protocol14-static-summary@1",
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
