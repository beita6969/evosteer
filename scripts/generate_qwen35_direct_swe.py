#!/usr/bin/env python3
"""Legacy one-shot SWE generator retained to reject obsolete formal configs.

Use ``generate_qwen35_direct_swe_repo_agent.py`` for backbone parity and
``reparse_qwen35_direct_swe.py`` to replay already-frozen one-shot responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from skillev_private.direct_reference.journal import PrivateGenerationJournal
from skillev_private.direct_reference.manifests import load_population_manifest
from skillev_private.direct_reference.populations import load_skillflow_iid_cases
from skillev_private.direct_reference.swe_evaluator import (
    LocalCommandSWEEvaluator,
    RemoteSWEEvaluatorClient,
    SWEEvaluator,
)

from skillev.evaluation.direct_baseline import (
    DirectBenchmark,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_direct_tasks,
)
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iid-population", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    evaluator = parser.add_mutually_exclusive_group(required=True)
    evaluator.add_argument("--official-evaluator-base")
    evaluator.add_argument(
        "--local-evaluator-command-json",
        help="JSON string array containing the local evaluator argv",
    )
    parser.add_argument("--local-evaluator-work-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600)
    return parser.parse_args()


def _evaluator(arguments: argparse.Namespace) -> SWEEvaluator:
    if arguments.official_evaluator_base is not None:
        if arguments.local_evaluator_work_dir is not None:
            raise ValueError("local evaluator work directory requires a local command")
        return RemoteSWEEvaluatorClient(arguments.official_evaluator_base)
    if arguments.local_evaluator_work_dir is None:
        raise ValueError("local evaluator command requires --local-evaluator-work-dir")
    value = json.loads(arguments.local_evaluator_command_json)
    if type(value) is not list or not value or not all(type(item) is str for item in value):
        raise ValueError("local evaluator command must be a non-empty JSON string array")
    return LocalCommandSWEEvaluator(
        tuple(cast(list[str], value)),
        arguments.local_evaluator_work_dir.resolve(),
    )


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_direct_reference_protocol(arguments.config)
    if protocol.benchmark(DirectBenchmark.SWE_BENCH).prompt_profile != "direct-patch@2":
        raise RuntimeError("formal SWE generation requires the repository-agent runner")
    manifest = load_population_manifest(arguments.population_manifest)
    cases = load_skillflow_iid_cases(
        arguments.iid_population,
        protocol=protocol,
        include=frozenset({DirectBenchmark.SWE_BENCH}),
        manifests={DirectBenchmark.SWE_BENCH: manifest},
    )
    evaluator = _evaluator(arguments)
    await evaluator.preflight()
    served_model = arguments.served_model_name or protocol.model.served_model_name
    if served_model != protocol.model.served_model_name:
        raise ValueError("served model differs from executable protocol")
    client = OpenAICompatibleDirectClient(
        endpoint_base=arguments.endpoint_base,
        served_model_name=served_model,
        timeout_seconds=arguments.request_timeout_seconds,
        context_length=arguments.context_length,
        token_counter=QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path),
    )
    arguments.private_output_dir.mkdir(parents=True, exist_ok=True)
    journal = PrivateGenerationJournal(
        (arguments.private_output_dir / "generations.jsonl").resolve()
    )
    async with journal:
        attempts = await run_direct_tasks(
            client,
            tuple(case.public_task for case in cases),
            concurrency=arguments.concurrency,
            generation_observer=journal.append,
        )
    by_task = {case.public_task.task_id: case for case in cases}
    rows: list[dict[str, object]] = []
    for attempt in attempts:
        case = by_task[attempt.task_id]
        rows.append(
            {
                "task_id": attempt.task_id,
                "instance_id": case.private_metadata["instance_id"],
                "model_patch": attempt.parsed.value if attempt.parsed is not None else None,
                "parse_reason": (
                    attempt.parsed.reason.value if attempt.parsed is not None else None
                ),
                "generation_infrastructure_error": attempt.infrastructure_error,
            }
        )
    output = arguments.private_output_dir / "predictions.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "planned_count": len(cases),
        "candidate_response_count": sum(
            row["generation_infrastructure_error"] is None for row in rows
        ),
        "submission_count": sum(row["model_patch"] is not None for row in rows),
        "generation_infrastructure_failures": sum(
            row["generation_infrastructure_error"] is not None for row in rows
        ),
    }


def main() -> None:
    print(json.dumps(asyncio.run(_run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
