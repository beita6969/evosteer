#!/usr/bin/env python3
"""Run the owner-authorized Qwen-local Protocol 14 HealthBench diagnostic."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.execution_receipts import (
    COMPLETE_FORMAT,
    START_FORMAT,
    Protocol14ExecutionReceipt,
    RuntimeGenerationProfile,
    write_execution_receipt,
)
from skillev.evaluation.healthbench_official import (
    HealthBenchQwenJudgeProfile,
    HealthBenchTransportError,
    JsonlJournal,
    QwenChatTransport,
    QwenHealthBenchRubricSampler,
    QwenOfficialCandidateSampler,
    RubricAttemptRegistry,
    TransportRetryPolicy,
    aggregate_healthbench_seed_metrics,
    aggregate_official_scores,
    candidate_usage_dict,
    generate_candidates,
    grade_candidates,
    load_qwen_judge_profile,
    normalize_healthbench_model_invalid_scores,
    validate_official_panel,
)
from skillev.experiments.protocol_v14 import load_protocol_v14


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--grader-config", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--population", choices=("exact128", "source500"), required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--attempt-role", choices=("reference", "candidate", "canary"), required=True
    )
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument(
        "--grader-endpoint-base",
        action="append",
        default=[],
        help="Repeat for equivalent adapter-free Qwen3.5-9B SGLang replicas.",
    )
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--runtime-model-revision", required=True)
    parser.add_argument("--phase", choices=("candidates", "grade", "all"), default="all")
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- project Git identity
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_revision(source_root: Path) -> str:
    return subprocess.run(  # noqa: S603 -- fixed Git query against operator source path
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_official(source_root: Path, *, expected_revision: str) -> tuple[Any, type[Any]]:
    if not source_root.is_absolute() or not (source_root / "healthbench_eval.py").is_file():
        raise ValueError("official simple-evals source root is invalid")
    if _source_revision(source_root) != expected_revision:
        raise ValueError("simple-evals checkout differs from the frozen rubric revision")
    package = "_skillev_protocol14_simple_evals"
    if package not in sys.modules:
        module = ModuleType(package)
        module.__package__ = package
        module.__path__ = [str(source_root)]
        sys.modules[package] = module
    healthbench = importlib.import_module(f"{package}.healthbench_eval")
    types_module = importlib.import_module(f"{package}.types")
    return healthbench, types_module.SamplerResponse


def _candidate_client(endpoint_base: str, *, timeout_seconds: float) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key="EMPTY",
        base_url=endpoint_base.rstrip("/") + "/",
        timeout=timeout_seconds,
        max_retries=0,
    )


def _candidate_retryable_errors() -> tuple[type[Exception], ...]:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, OSError)


def _require_candidate_route(client: Any, expected: str) -> None:
    routes = tuple(item.id for item in client.models.list().data)
    # Adapter-capable SGLang instances expose the frozen base route alongside
    # any dynamically loaded LoRA routes.  The request still pins ``model`` to
    # ``expected``; rejecting the endpoint merely because another route is
    # advertised prevents the base model from serving as the matched judge for
    # a trained candidate.  Absence of the requested route remains fatal.
    if expected not in routes:
        raise RuntimeError("candidate endpoint route differs from Protocol 14")


def _qwen_grader(
    *,
    endpoint_bases: list[str],
    profile: HealthBenchQwenJudgeProfile,
    sampler_response_type: type[Any],
) -> QwenHealthBenchRubricSampler:
    retry = TransportRetryPolicy(
        maximum_attempts=profile.maximum_attempts,
        request_timeout_seconds=profile.request_timeout_seconds,
        total_deadline_seconds=profile.total_deadline_seconds,
    )
    transports: list[QwenChatTransport] = []
    for endpoint_base in endpoint_bases:
        client = _candidate_client(endpoint_base, timeout_seconds=profile.request_timeout_seconds)
        _require_candidate_route(client, profile.model_route)
        transports.append(
            QwenChatTransport(
                client=client,
                model=profile.model_route,
                retry_policy=retry,
                transient_error_types=_candidate_retryable_errors(),
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                top_p=profile.top_p,
                top_k=profile.top_k,
                seed=profile.seed,
                enable_thinking=profile.enable_thinking,
            )
        )
    return QwenHealthBenchRubricSampler(
        transports=tuple(transports),
        response_type=sampler_response_type,
        semantic_attempts=RubricAttemptRegistry(),
    )


def _write_marker(path: Path, marker: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != marker:
            raise ValueError("HealthBench output directory belongs to another attempt")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _existing_generation_provenance(
    path: Path,
    *,
    attempt_id: str,
    execution: object,
    runtime_model_revision: str,
) -> tuple[str, str] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("HealthBench execution-start receipt is invalid")
    if (
        raw.get("format") != START_FORMAT
        or raw.get("status") != "running"
        or raw.get("attempt_id") != attempt_id
        or raw.get("execution") != execution
        or raw.get("runtime_model_revision") != runtime_model_revision
    ):
        raise ValueError("HealthBench existing execution-start receipt differs")
    started_at = raw.get("started_at")
    generation_revision = raw.get("generation_code_revision")
    if type(started_at) is not str or type(generation_revision) is not str:
        raise ValueError("HealthBench existing execution provenance is incomplete")
    return started_at, generation_revision


def _rubric_counts(scores: dict[str, dict[str, object]]) -> tuple[int, int]:
    total = 0
    valid = 0
    for record in scores.values():
        items = record.get("rubric_items")
        if not isinstance(items, list):
            raise HealthBenchTransportError("HealthBench rubric grades are absent")
        total += len(items)
        valid += sum(
            isinstance(item, dict) and isinstance(item.get("criteria_met"), bool) for item in items
        )
    return total, valid


def _seed_metrics(metrics: dict[str, object]) -> dict[str, object]:
    output = {
        name: value
        for name, value in metrics.items()
        if ":" not in name and isinstance(value, int | float) and not isinstance(value, bool)
    }
    if "overall_score" not in output:
        raise ValueError("HealthBench aggregate lacks overall_score")
    return output


def main(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.n_threads <= 0:
        raise ValueError("n_threads must be positive")
    if not arguments.input_path.is_absolute() or not arguments.input_path.is_file():
        raise ValueError("HealthBench input must be an absolute existing file")
    protocol = load_protocol_v14(arguments.protocol, arguments.sources)
    if arguments.attempt_role == "candidate":
        protocol.require_execution_ready()
    execution = load_execution_contracts_v4(arguments.conditions, protocol=protocol)[
        Protocol14Benchmark.HEALTHBENCH
    ]
    judge_profile = load_qwen_judge_profile(arguments.grader_config)
    if (
        execution.grader_profile != judge_profile.profile_id
        or execution.actor_route != judge_profile.model_route
        or arguments.served_model_name != execution.actor_route
        or arguments.context_length != execution.context_length
        or arguments.runtime_model_revision != execution.actor_model_revision
    ):
        raise ValueError("HealthBench runtime differs from Protocol 14")
    healthbench, sampler_response_type = _load_official(
        arguments.official_source_root,
        expected_revision=judge_profile.rubric_source_revision,
    )
    expected_count = 128 if arguments.population == "exact128" else 500
    if arguments.population == "source500" and arguments.attempt_role != "reference":
        raise ValueError("HealthBench source500 lane is reference-only")
    evaluator = healthbench.HealthBenchEval(
        grader_model=None,
        num_examples=expected_count,
        n_repeats=1,
        n_threads=arguments.n_threads,
        subset_name=None,
        input_path=str(arguments.input_path),
        length_adjustment_center=None,
        length_adjustment_penalty_per_500_chars=None,
    )
    examples = cast(list[dict[str, object]], evaluator.examples)
    validate_official_panel(examples, expected_count=expected_count)
    panel_ids = tuple(cast(str, row["prompt_id"]) for row in examples)
    output_dir = arguments.private_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = _revision()
    started_at = datetime.now(UTC).isoformat()
    generation_revision = revision
    start_path = output_dir / "execution-start.json"
    if arguments.population == "exact128":
        existing = _existing_generation_provenance(
            start_path,
            attempt_id=arguments.attempt_id,
            execution=execution.to_mapping(),
            runtime_model_revision=arguments.runtime_model_revision,
        )
        if existing is not None:
            started_at, generation_revision = existing
    marker = {
        "format": "skillev-protocol14-healthbench-attempt@1",
        "attempt_id": arguments.attempt_id,
        "attempt_role": arguments.attempt_role,
        "population": arguments.population,
        "expected_count": expected_count,
        "condition_id": execution.condition_id,
        "actor_model_revision": arguments.runtime_model_revision,
        "grader_profile": judge_profile.profile_id,
        "rubric_source_revision": judge_profile.rubric_source_revision,
        "seeds": list(execution.aggregation.seeds),
    }
    _write_marker(output_dir / "attempt.json", marker)
    if arguments.population == "exact128":
        runtime_profile = RuntimeGenerationProfile(
            execution.decoding_profile,
            execution.decoding.sampling_mode,
            False,
            execution.decoding.temperature,
            execution.decoding.top_p,
            execution.decoding.top_k,
            execution.decoding.min_p,
            execution.decoding.presence_penalty,
            execution.decoding.repetition_penalty,
            execution.decoding.max_new_tokens,
            execution.decoding.seed,
            execution.decoding.stop,
        )
        common_receipt = {
            "attempt_id": arguments.attempt_id,
            "attempt_role": arguments.attempt_role,
            "execution": execution.to_mapping(),
            "generation_profile": runtime_profile,
            "environment": None,
            "model_route": execution.actor_route,
            "runtime_model_revision": arguments.runtime_model_revision,
            "context_length": execution.context_length,
            "adapter_active": False,
            "panel_manifest_id": execution.panel_manifest_id,
            "planned_task_ids": panel_ids,
            "generation_code_revision": generation_revision,
            "scoring_code_revision": revision,
            "evaluator_version": execution.scorer_profile,
            "started_at": started_at,
        }
        if not start_path.exists():
            start_receipt = Protocol14ExecutionReceipt(
                format_version=START_FORMAT,
                status="running",
                response_model_ids=(),
                generated_task_ids=(),
                terminal_task_ids=(),
                completed_at=None,
                **common_receipt,
            )
            start_receipt.validate_against(execution=execution, expected_task_ids=panel_ids)
            write_execution_receipt(start_path, start_receipt)
    client = _candidate_client(
        arguments.endpoint_base, timeout_seconds=arguments.candidate_timeout_seconds
    )
    _require_candidate_route(client, execution.actor_route)
    metrics_by_seed: dict[int, dict[str, object]] = {}
    terminal_counts: dict[int, int] = {}
    infrastructure_failures: dict[int, int] = {}
    response_models: list[str] = []
    seed_receipts: list[dict[str, object]] = []
    for seed in execution.aggregation.seeds:
        seed_dir = output_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_marker = {**marker, "format": "skillev-protocol14-healthbench-seed@1", "seed": seed}
        _write_marker(seed_dir / "attempt.json", seed_marker)
        candidate_journal = JsonlJournal(seed_dir / "candidates.jsonl")
        candidate_failures = JsonlJournal(seed_dir / "candidate-infrastructure.jsonl")
        if arguments.phase in {"candidates", "all"}:
            sampler = QwenOfficialCandidateSampler(
                client=client,
                model=execution.actor_route,
                sampler_response_type=sampler_response_type,
                retryable_errors=_candidate_retryable_errors(),
                retry_policy=TransportRetryPolicy(
                    maximum_attempts=4,
                    request_timeout_seconds=arguments.candidate_timeout_seconds,
                    total_deadline_seconds=arguments.candidate_timeout_seconds + 60,
                ),
                temperature=execution.decoding.temperature,
                max_tokens=execution.decoding.max_new_tokens,
                top_p=execution.decoding.top_p,
                top_k=execution.decoding.top_k,
                seed=seed,
                enable_thinking=False,
            )
            candidates = generate_candidates(
                examples=examples,
                sampler=sampler,
                journal=candidate_journal,
                usage_parser=candidate_usage_dict,
                n_threads=arguments.n_threads,
                failure_journal=candidate_failures,
                expected_count=expected_count,
            )
        else:
            candidates = candidate_journal.load()
        if len(candidates) != expected_count:
            raise HealthBenchTransportError("HealthBench candidate seed is incomplete")
        response_models.extend(
            cast(str, item["response_model"])
            for item in candidates.values()
            if type(item.get("response_model")) is str
        )
        score_journal = JsonlJournal(seed_dir / "scores.jsonl")
        grader_failures = JsonlJournal(seed_dir / "grader-infrastructure.jsonl")
        if arguments.phase in {"grade", "all"}:
            grader_endpoints = arguments.grader_endpoint_base or [arguments.endpoint_base]
            grader = _qwen_grader(
                endpoint_bases=grader_endpoints,
                profile=judge_profile,
                sampler_response_type=sampler_response_type,
            )
            evaluator.grader_model = grader
            scores = grade_candidates(
                examples=examples,
                candidates=candidates,
                evaluator=evaluator,
                journal=score_journal,
                n_threads=arguments.n_threads,
                failure_journal=grader_failures,
                expected_count=expected_count,
                invocation_namespace=f"{arguments.attempt_id}:seed-{seed}",
            )
            formal_scores, model_output_invalid = normalize_healthbench_model_invalid_scores(
                scores=scores,
                candidates=candidates,
            )
            metrics = aggregate_official_scores(
                formal_scores.values(),
                clipped_stat=healthbench._compute_clipped_stats,
                expected_count=expected_count,
            )
            rubric_count, valid_rubrics = _rubric_counts(scores)
            if valid_rubrics != rubric_count:
                raise HealthBenchTransportError(
                    "HealthBench grader returned incomplete rubric verdicts"
                )
            metrics_by_seed[seed] = _seed_metrics(metrics)
            terminal_counts[seed] = len(scores)
            infrastructure_failures[seed] = 0
            seed_receipts.append(
                {
                    "seed": seed,
                    "candidate_count": len(candidates),
                    "terminal_count": len(scores),
                    "rubric_item_count": rubric_count,
                    "valid_rubric_verdict_count": valid_rubrics,
                    "judge_semantic_repair_count": (grader.semantic_attempts.semantic_repair_count),
                    "judge_transport_retry_count": sum(
                        transport.transport_retry_count for transport in grader.transports
                    ),
                    "infrastructure_failure_count": 0,
                    "model_output_invalid_count": model_output_invalid,
                    "metrics": _seed_metrics(metrics),
                }
            )
        else:
            terminal_counts[seed] = len(score_journal.load())
            infrastructure_failures[seed] = 0
    summary: dict[str, object] = {
        **marker,
        "format": "skillev-protocol14-healthbench-summary@1",
        "status": "candidates-complete" if arguments.phase == "candidates" else "complete",
        "judge": {
            "label": "Qwen-local-judge diagnostic",
            "backend": judge_profile.backend,
            "model_route": judge_profile.model_route,
            "replicas": len(arguments.grader_endpoint_base or [arguments.endpoint_base]),
            "unique_models": 1,
            "official_gpt_comparable": judge_profile.official_gpt_comparable,
        },
        "seed_receipts": seed_receipts,
        "aggregate": None,
    }
    if arguments.phase in {"grade", "all"}:
        aggregate = aggregate_healthbench_seed_metrics(
            metrics_by_seed=metrics_by_seed,
            terminal_count_by_seed=terminal_counts,
            infrastructure_failures_by_seed=infrastructure_failures,
            expected_seeds=execution.aggregation.seeds,
            expected_count=expected_count,
        )
        summary["aggregate"] = aggregate
        if arguments.population == "exact128":
            complete_receipt = Protocol14ExecutionReceipt(
                format_version=COMPLETE_FORMAT,
                status="complete",
                response_model_ids=tuple(dict.fromkeys(response_models))
                or (execution.actor_route,),
                generated_task_ids=panel_ids,
                terminal_task_ids=panel_ids,
                completed_at=datetime.now(UTC).isoformat(),
                **common_receipt,
            )
            complete_receipt.validate_against(execution=execution, expected_task_ids=panel_ids)
            write_execution_receipt(output_dir / "execution-complete.json", complete_receipt)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main(_arguments())
