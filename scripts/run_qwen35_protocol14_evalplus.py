#!/usr/bin/env python3
"""Run frozen no-thinking EvalPlus full lanes and the MBPP+ panel projection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from skillev_private.benchmarks.evalplus_adapter import parse_evalplus_single_candidate
from skillev_private.benchmarks.evalplus_process import EvalPlusProcessConfig, run_evalplus
from skillev_private.direct_reference.protocol14_runner import (
    load_protocol14_panel_manifest,
    profile_from_execution,
)

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.code_eval import (
    CodeExtractionStatus,
    CodeOutcomeKind,
    CodeTerminalOutcome,
    classify_python_completion,
    evalplus_composite_percent,
    load_evalplus_protocol_profile,
    project_evalplus_outcomes,
)
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol14ExecutionReceipt,
    RuntimeGenerationProfile,
    write_execution_receipt,
)
from skillev.evaluation.direct_baseline.client import (
    DirectGenerationError,
    DirectGenerationRequest,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from skillev.evaluation.direct_baseline.prompts import render_static_messages
from skillev.experiments.protocol_v14 import load_protocol_v14


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("mbpp-plus", "humaneval-plus-crosscheck"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--evalplus-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mbpp-source", type=Path)
    parser.add_argument("--humaneval-source", type=Path)
    parser.add_argument("--mbpp-summary", type=Path)
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
    parser.add_argument(
        "--resume-scoring",
        action="store_true",
        help="Reuse a complete frozen generation journal after scorer infrastructure failure.",
    )
    return parser.parse_args()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- project Git identity
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _required_symbol(row: dict[str, Any]) -> str:
    entry_point = row.get("entry_point")
    if type(entry_point) is not str or not entry_point.strip():
        raise ValueError("EvalPlus task lacks an entry point")
    return entry_point


def _outcome_mapping(outcome: CodeTerminalOutcome) -> dict[str, object]:
    return {
        "outcome": outcome.kind.value,
        "extraction_status": (
            outcome.extraction_status.value if outcome.extraction_status is not None else None
        ),
        "base_passed": outcome.base_passed,
        "plus_passed": outcome.plus_passed,
        "definitive": outcome.definitive,
    }


def _resume_generation_state(
    output_dir: Path,
    *,
    marker: dict[str, object],
    full_task_ids: tuple[str, ...],
) -> tuple[list[dict[str, object]], Path, float]:
    if not output_dir.is_dir():
        raise ValueError("EvalPlus scoring resume requires an existing attempt directory")
    attempt_path = output_dir / "attempt.json"
    generations_path = output_dir / "generations.jsonl"
    carrier = output_dir / "official-samples.jsonl"
    for forbidden in (
        "evalplus-results.json",
        "verdicts.jsonl",
        "summary.json",
        "execution-complete.json",
    ):
        if (output_dir / forbidden).exists():
            raise ValueError("EvalPlus scoring resume requires an incomplete scorer attempt")
    observed_marker = json.loads(attempt_path.read_text(encoding="utf-8"))
    if observed_marker != marker:
        raise ValueError("EvalPlus scoring resume attempt identity differs")
    generation_rows = [
        json.loads(line)
        for line in generations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in generation_rows):
        raise ValueError("EvalPlus scoring resume generation journal is invalid")
    observed_ids = tuple(cast(str, row.get("task_id")) for row in generation_rows)
    if observed_ids != full_task_ids:
        raise ValueError("EvalPlus scoring resume generation population differs")
    carrier_rows = [
        json.loads(line)
        for line in carrier.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    carrier_ids = tuple(
        cast(str, row.get("task_id")) for row in carrier_rows if isinstance(row, dict)
    )
    if carrier_ids != full_task_ids:
        raise ValueError("EvalPlus scoring resume official carrier differs")
    generation_seconds = max(0.0, generations_path.stat().st_mtime - attempt_path.stat().st_mtime)
    return cast(list[dict[str, object]], generation_rows), carrier, generation_seconds


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if arguments.lane == "mbpp-plus" and arguments.manifest is None:
        raise ValueError("MBPP+ lane requires the frozen panel manifest")
    if arguments.lane == "mbpp-plus" and (
        arguments.mbpp_source is None
        or not arguments.mbpp_source.is_absolute()
        or not arguments.mbpp_source.is_file()
    ):
        raise ValueError("MBPP+ lane requires the absolute official v0.2.0 source asset")
    if arguments.lane == "humaneval-plus-crosscheck":
        if arguments.attempt_role != "reference" or arguments.mbpp_summary is None:
            raise ValueError("HumanEval+ cross-check requires the MBPP+ reference summary")
        if (
            arguments.humaneval_source is None
            or not arguments.humaneval_source.is_absolute()
            or not arguments.humaneval_source.is_file()
        ):
            raise ValueError(
                "HumanEval+ cross-check requires the absolute official v0.1.9 source asset"
            )
    protocol = load_protocol_v14(arguments.protocol, arguments.sources)
    if arguments.attempt_role == "candidate":
        protocol.require_execution_ready()
    execution = load_execution_contracts_v4(arguments.conditions, protocol=protocol)[
        Protocol14Benchmark.MBPP_PLUS
    ]
    evalplus = load_evalplus_protocol_profile(arguments.evalplus_config)
    profile = profile_from_execution(execution)
    if (
        execution.thinking_mode.value != "disabled"
        or profile.sampling_mode != "greedy"
        or arguments.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
        or arguments.runtime_model_revision != execution.actor_model_revision
        or version("evalplus") != evalplus.package_version
    ):
        raise ValueError("EvalPlus runtime differs from Protocol 14")
    if arguments.lane == "mbpp-plus":
        assert arguments.mbpp_source is not None
        os.environ["MBPP_OVERRIDE_PATH"] = str(arguments.mbpp_source.resolve())
        from evalplus.data import get_mbpp_plus  # type: ignore[import-not-found]

        official = cast(dict[str, dict[str, Any]], get_mbpp_plus())
        dataset = "mbpp"
        full_count = evalplus.mbpp_plus_task_count
        prompt_profile = execution.prompt_profile
    else:
        assert arguments.humaneval_source is not None
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(arguments.humaneval_source.resolve())
        from evalplus.data import get_human_eval_plus  # type: ignore[import-not-found]

        official = cast(dict[str, dict[str, Any]], get_human_eval_plus())
        dataset = "humaneval"
        full_count = evalplus.humaneval_plus_task_count
        prompt_profile = "direct-code@1"
    full_task_ids = tuple(sorted(official))
    if len(full_task_ids) != full_count or len(set(full_task_ids)) != full_count:
        raise ValueError("EvalPlus source population count differs")
    output_dir = arguments.private_output_dir
    revision = _revision()
    started_at = datetime.now(UTC).isoformat()
    marker = {
        "format": "skillev-protocol14-evalplus-attempt@1",
        "attempt_id": arguments.attempt_id,
        "attempt_role": arguments.attempt_role,
        "lane": arguments.lane,
        "dataset": dataset,
        "full_task_count": full_count,
        "source_revision": evalplus.source_revision,
        "package_version": evalplus.package_version,
        "actor_model_revision": arguments.runtime_model_revision,
        "thinking": False,
        "sampling_mode": "greedy",
    }
    if arguments.resume_scoring:
        generation_rows, carrier, generation_seconds = _resume_generation_state(
            output_dir, marker=marker, full_task_ids=full_task_ids
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "attempt.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
    panel_ids: tuple[str, ...] = ()
    common_receipt: dict[str, object] | None = None
    if arguments.lane == "mbpp-plus":
        assert arguments.manifest is not None
        panel = load_protocol14_panel_manifest(arguments.manifest, execution=execution)
        panel_ids = panel.source_identities
        common_receipt = {
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
        if arguments.resume_scoring:
            start_raw = json.loads(
                (output_dir / "execution-start.json").read_text(encoding="utf-8")
            )
            expected_start_fields = {
                "format": START_FORMAT,
                "status": "running",
                "attempt_id": arguments.attempt_id,
                "attempt_role": arguments.attempt_role,
                "generation_profile": RuntimeGenerationProfile.from_direct_profile(
                    profile
                ).to_mapping(),
                "model_route": execution.actor_route,
                "runtime_model_revision": arguments.runtime_model_revision,
                "context_length": execution.context_length,
                "adapter_active": False,
                "panel_manifest_id": execution.panel_manifest_id,
                "planned_task_ids": list(panel.task_ids),
            }
            if any(start_raw.get(key) != value for key, value in expected_start_fields.items()):
                raise ValueError("EvalPlus scoring resume start receipt differs")
            common_receipt["generation_code_revision"] = start_raw["generation_code_revision"]
            common_receipt["started_at"] = start_raw["started_at"]
        else:
            start = Protocol14ExecutionReceipt(
                format_version=START_FORMAT,
                status="running",
                response_model_ids=(),
                generated_task_ids=(),
                terminal_task_ids=(),
                completed_at=None,
                **common_receipt,
            )
            start.validate_against(execution=execution, expected_task_ids=panel.task_ids)
            write_execution_receipt(output_dir / "execution-start.json", start)
    if not arguments.resume_scoring:
        counter = QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path)
        clients = tuple(
            OpenAICompatibleDirectClient(
                endpoint_base=endpoint,
                served_model_name=execution.actor_route,
                api_key="EMPTY",
                timeout_seconds=3_600,
                context_length=execution.context_length,
                token_counter=counter,  # type: ignore[arg-type]
            )
            for endpoint in dict.fromkeys(arguments.endpoint_base)
        )
        routes = await asyncio.gather(*(client.model_routes() for client in clients))
        if any(route != (execution.actor_route,) for route in routes):
            raise RuntimeError("EvalPlus endpoint exposes an unexpected model route")
        semaphore = asyncio.Semaphore(arguments.concurrency)
        generations: list[dict[str, object] | None] = [None] * full_count

        async def generate(index: int, task_id: str) -> None:
            row = official[task_id]
            prompt = row.get("prompt")
            if type(prompt) is not str or not prompt.strip():
                raise ValueError("EvalPlus task prompt is absent")
            async with semaphore:
                started = time.monotonic()
                try:
                    response = await clients[index % len(clients)].generate(
                        DirectGenerationRequest(
                            request_id=task_id,
                            messages=render_static_messages(prompt_profile, prompt),
                            profile=profile,
                        )
                    )
                except DirectGenerationError as exc:
                    generations[index] = {
                        "task_id": task_id,
                        "generation_infrastructure_error": type(exc).__name__,
                        "wall_time_seconds": time.monotonic() - started,
                    }
                    return
                extraction = classify_python_completion(
                    final_text=response.text,
                    reasoning_text=response.reasoning_text,
                    required_symbols=(_required_symbol(row),),
                )
                generations[index] = {
                    "task_id": task_id,
                    "raw_text": response.text,
                    "reasoning_text": response.reasoning_text,
                    "submission": extraction.source,
                    "extraction_status": extraction.status.value,
                    "required_symbols": list(extraction.required_symbols),
                    "present_symbols": list(extraction.present_symbols),
                    "finish_reason": response.finish_reason,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "response_model": response.response_model,
                    "wall_time_seconds": time.monotonic() - started,
                }

        generation_started = time.monotonic()
        await asyncio.gather(
            *(generate(index, task_id) for index, task_id in enumerate(full_task_ids))
        )
        generation_seconds = time.monotonic() - generation_started
        generation_rows = [row for row in generations if row is not None]
        if len(generation_rows) != full_count:
            raise RuntimeError("EvalPlus generation did not conserve the source population")
        with (output_dir / "generations.jsonl").open("x", encoding="utf-8") as stream:
            for row in generation_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_task_id = {cast(str, row["task_id"]): row for row in generation_rows}
    generation_infrastructure = sum(
        "generation_infrastructure_error" in row for row in generation_rows
    )
    if generation_infrastructure:
        raise RuntimeError("EvalPlus generation has unresolved infrastructure rows")
    if not arguments.resume_scoring:
        carrier = output_dir / "official-samples.jsonl"
        with carrier.open("x", encoding="utf-8") as stream:
            for task_id in full_task_ids:
                row = by_task_id[task_id]
                submission = row.get("submission")
                if type(submission) is not str or not submission.strip():
                    submission = "raise NotImplementedError"
                stream.write(json.dumps({"task_id": task_id, "solution": submission}) + "\n")
    scoring_started = time.monotonic()
    evaluator_payload = await asyncio.to_thread(
        run_evalplus,
        config=EvalPlusProcessConfig(
            python_executable=Path(sys.executable),
            dataset=dataset,
            version=f"v{evalplus.package_version}",
            workers=min(arguments.concurrency, evalplus.workers),
            timeout_seconds=evalplus.total_timeout_seconds,
            minimum_time_limit_seconds=evalplus.minimum_time_limit_seconds,
            ground_truth_time_limit_factor=evalplus.ground_truth_time_limit_factor,
        ),
        samples=carrier,
        output=output_dir / "evalplus-results.json",
        stdout=output_dir / "official-evaluator.stdout.log",
        stderr=output_dir / "official-evaluator.stderr.log",
    )
    scoring_seconds = time.monotonic() - scoring_started
    evaluator_results = evaluator_payload["eval"]
    if not isinstance(evaluator_results, dict):
        raise RuntimeError("EvalPlus result is not keyed by task ID")
    outcomes: dict[str, CodeTerminalOutcome] = {}
    verdict_rows: list[dict[str, object]] = []
    for task_id in full_task_ids:
        generation = by_task_id[task_id]
        extraction_status = CodeExtractionStatus(cast(str, generation["extraction_status"]))
        if extraction_status is not CodeExtractionStatus.VALID:
            outcome = CodeTerminalOutcome(
                CodeOutcomeKind.CANDIDATE_INVALID,
                extraction_status,
                None,
                None,
            )
        else:
            raw_verdict = evaluator_results.get(task_id)
            if raw_verdict is None:
                outcome = CodeTerminalOutcome(
                    CodeOutcomeKind.SCORER_INFRASTRUCTURE, None, None, None
                )
            else:
                try:
                    verdict = parse_evalplus_single_candidate(task_id, raw_verdict)
                except ValueError:
                    outcome = CodeTerminalOutcome(
                        CodeOutcomeKind.SCORER_INFRASTRUCTURE, None, None, None
                    )
                else:
                    kind = (
                        CodeOutcomeKind.SCORED_SUCCESS
                        if verdict.base_plus_passed
                        else CodeOutcomeKind.SCORED_FAILURE
                    )
                    outcome = CodeTerminalOutcome(
                        kind,
                        CodeExtractionStatus.VALID,
                        verdict.base_passed,
                        verdict.plus_passed,
                    )
        outcomes[task_id] = outcome
        verdict_rows.append({"task_id": task_id, **_outcome_mapping(outcome)})
    with (output_dir / "verdicts.jsonl").open("x", encoding="utf-8") as stream:
        for row in verdict_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    unresolved = sum(not outcome.definitive for outcome in outcomes.values())
    full_successes = sum(outcome.success for outcome in outcomes.values())
    full_percent = 100.0 * full_successes / full_count
    result: dict[str, object] = {
        **marker,
        "format": "skillev-protocol14-evalplus-summary@1",
        "status": "complete" if unresolved == 0 else "infrastructure-incomplete",
        "full": {
            "count": full_count,
            "definitive": full_count - unresolved,
            "successes": full_successes,
            "base_plus_pass_at_1_percent": full_percent,
        },
        "panel": None,
        "cross_check": None,
        "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "throughput_records_per_hour": (
            full_count * 3600.0 / (generation_seconds + scoring_seconds)
        ),
    }
    if arguments.lane == "mbpp-plus":
        projection = project_evalplus_outcomes(
            full_task_ids=full_task_ids,
            panel_task_ids=panel_ids,
            outcomes=outcomes,
        )
        panel_unresolved = sum(not outcome.definitive for outcome in projection)
        panel_successes = sum(outcome.success for outcome in projection)
        panel_invalid = sum(
            outcome.kind is CodeOutcomeKind.CANDIDATE_INVALID for outcome in projection
        )
        panel_scored = sum(
            outcome.kind in {CodeOutcomeKind.SCORED_SUCCESS, CodeOutcomeKind.SCORED_FAILURE}
            for outcome in projection
        )
        result["panel"] = {
            "count": len(projection),
            "definitive": len(projection) - panel_unresolved,
            "scored_valid": panel_scored,
            "model_output_invalid": panel_invalid,
            "scorer_infrastructure": panel_unresolved,
            "successes": panel_successes,
            "base_plus_pass_at_1_percent": 100.0 * panel_successes / len(projection),
            "projection": "same-full378-outputs-in-frozen-manifest-order",
        }
        if unresolved == 0:
            assert common_receipt is not None
            panel = load_protocol14_panel_manifest(arguments.manifest, execution=execution)
            response_models = tuple(
                dict.fromkeys(
                    cast(str, row["response_model"])
                    for row in generation_rows
                    if type(row.get("response_model")) is str
                )
            ) or (execution.actor_route,)
            complete = Protocol14ExecutionReceipt(
                format_version=COMPLETE_FORMAT,
                status="complete",
                response_model_ids=response_models,
                generated_task_ids=panel.task_ids,
                terminal_task_ids=panel.task_ids,
                completed_at=datetime.now(UTC).isoformat(),
                **common_receipt,
            )
            complete.validate_against(execution=execution, expected_task_ids=panel.task_ids)
            write_execution_receipt(output_dir / "execution-complete.json", complete)
    else:
        assert arguments.mbpp_summary is not None
        mbpp_summary = json.loads(arguments.mbpp_summary.read_text(encoding="utf-8"))
        mbpp_full = mbpp_summary.get("full") if isinstance(mbpp_summary, dict) else None
        if not isinstance(mbpp_full, dict):
            raise ValueError("MBPP+ reference summary is invalid")
        mbpp_percent = mbpp_full.get("base_plus_pass_at_1_percent")
        if isinstance(mbpp_percent, bool) or not isinstance(mbpp_percent, int | float):
            raise ValueError("MBPP+ reference summary lacks a numeric full score")
        result["cross_check"] = {
            "metric": "mean(humaneval_plus_full,mbpp_plus_full)",
            "percent": evalplus_composite_percent(full_percent, float(mbpp_percent)),
            "published_external_anchor_percent": 71.8,
            "usable_as_mbpp_plus_target": False,
        }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if unresolved:
        raise RuntimeError("EvalPlus scorer has unresolved infrastructure outcomes")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    asyncio.run(run(_arguments()))
