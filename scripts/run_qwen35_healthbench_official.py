#!/usr/bin/env python3
"""Run the official HealthBench Full 128-sample diagnostic with Qwen3.5-9B."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

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
from skillev.evaluation.current_iid.protocol13.healthbench_grading import (
    HealthBenchGraderContract,
)
from skillev.evaluation.current_iid.protocol13.runner_profiles import (
    load_protocol13_runner_profiles,
)
from skillev.evaluation.current_iid.protocol13.service_receipts import (
    load_model_service_contract,
    load_model_service_receipt,
    validate_replica_services,
)
from skillev.evaluation.healthbench_official import (
    CANDIDATE_MAX_TOKENS,
    CANDIDATE_TEMPERATURE,
    OPENAI_SYSTEM_MESSAGE,
    SAMPLE_COUNT,
    HealthBenchGraderReceipt,
    JsonlJournal,
    QwenChatTransport,
    QwenHealthBenchRubricSampler,
    QwenOfficialCandidateSampler,
    RubricAttemptRegistry,
    TransportRetryPolicy,
    aggregate_official_scores,
    candidate_usage_dict,
    generate_candidates,
    grade_candidates,
    normalize_healthbench_model_invalid_scores,
)
from skillev.experiments.protocol_v13 import load_protocol_v13


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name", default="qwen35-direct-base")
    parser.add_argument("--service-contract", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, required=True)
    parser.add_argument("--grader-service-receipt", type=Path, action="append", required=True)
    parser.add_argument(
        "--grader-backend",
        choices=("qwen-sglang",),
        default="qwen-sglang",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("local-judge-diagnostic",),
        default="local-judge-diagnostic",
    )
    parser.add_argument(
        "--grader-profile",
        required=True,
        help="Explicit journal identity, for example qwen-local-rubric-per-item@4.",
    )
    parser.add_argument(
        "--grader-endpoint-base",
        action="append",
        default=[],
        help="Repeat for equivalent Qwen SGLang grader replicas.",
    )
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("candidates", "grade", "all"), default="all")
    parser.add_argument("--n-threads", type=int, default=120)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--transport-maximum-attempts", type=int, default=4)
    parser.add_argument("--transport-total-deadline-seconds", type=float, default=300.0)
    return parser.parse_args()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- project Git identity
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_official(source_root: Path) -> tuple[Any, Any]:
    if not source_root.is_absolute() or not (source_root / "healthbench_eval.py").is_file():
        raise ValueError("official simple-evals source root is invalid")
    # The pinned checkout may retain the upstream ``simple-evals`` directory
    # name, which is not an importable Python identifier.  Mount it under a
    # stable synthetic package name so upstream relative imports still work.
    package = "_skillev_pinned_simple_evals"
    if package not in sys.modules:
        module = ModuleType(package)
        module.__package__ = package
        module.__path__ = [str(source_root)]
        sys.modules[package] = module
    healthbench = importlib.import_module(f"{package}.healthbench_eval")
    types_module = importlib.import_module(f"{package}.types")
    return healthbench, types_module.SamplerResponse


def _candidate_client(endpoint_base: str, *, request_timeout_seconds: float) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key="EMPTY",
        base_url=endpoint_base.rstrip("/") + "/",
        timeout=request_timeout_seconds,
        max_retries=0,
    )


def _candidate_retryable_errors() -> tuple[type[Exception], ...]:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, OSError)


def _require_candidate_route(client: Any, expected: str) -> None:
    routes = tuple(item.id for item in client.models.list().data)
    if routes != (expected,):
        raise RuntimeError(f"candidate endpoint routes differ: {routes!r}")


def _qwen_grader(
    *,
    endpoint_bases: list[str],
    served_model_name: str,
    sampler_response_type: type[Any],
    retry_policy: TransportRetryPolicy,
) -> QwenHealthBenchRubricSampler:
    transports = []
    for endpoint_base in endpoint_bases:
        client = _candidate_client(
            endpoint_base,
            request_timeout_seconds=retry_policy.request_timeout_seconds,
        )
        _require_candidate_route(client, served_model_name)
        transports.append(
            QwenChatTransport(
                client=client,
                model=served_model_name,
                retry_policy=retry_policy,
                transient_error_types=_candidate_retryable_errors(),
            )
        )
    return QwenHealthBenchRubricSampler(
        transports=tuple(transports),
        response_type=sampler_response_type,
        semantic_attempts=RubricAttemptRegistry(),
    )


def main(arguments: argparse.Namespace) -> dict[str, object]:
    if not arguments.input_path.is_absolute() or not arguments.input_path.is_file():
        raise ValueError("HealthBench input path must be an absolute file")
    if arguments.n_threads <= 0:
        raise ValueError("n_threads must be positive")
    if arguments.evaluation_mode != "local-judge-diagnostic":
        raise ValueError("HealthBench is Qwen-local diagnostic only")
    if arguments.grader_backend != "qwen-sglang":
        raise ValueError("HealthBench must use the Qwen3.5-9B SGLang self-grader")
    if not arguments.grader_profile.startswith("qwen-local-"):
        raise ValueError("Qwen grader requires an explicit qwen-local profile")
    retry_policy = TransportRetryPolicy(
        maximum_attempts=arguments.transport_maximum_attempts,
        request_timeout_seconds=arguments.request_timeout_seconds,
        total_deadline_seconds=arguments.transport_total_deadline_seconds,
    )
    healthbench, sampler_response_type = _load_official(arguments.official_source_root)
    # Constructing HealthBenchEval performs the official Random(0).sample(..., 128)
    # selection and preserves the released example objects for both phases.
    evaluator = healthbench.HealthBenchEval(
        grader_model=None,
        num_examples=SAMPLE_COUNT,
        n_repeats=1,
        n_threads=arguments.n_threads,
        subset_name=None,
        input_path=str(arguments.input_path),
        length_adjustment_center=None,
        length_adjustment_penalty_per_500_chars=None,
    )
    output_dir = arguments.private_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol13 = load_protocol_v13(arguments.protocol, arguments.sources)
    execution = load_execution_contracts_v3(arguments.conditions, protocol=protocol13)[
        Protocol13Benchmark.HEALTHBENCH
    ]
    runner_registry = load_protocol13_runner_profiles(arguments.runner_config)
    profile = runner_registry.require(execution.decoding_profile)
    if (
        arguments.served_model_name != execution.actor_route
        or runner_registry.served_model_name != execution.actor_route
        or runner_registry.context_length != execution.context_length
    ):
        raise ValueError("HealthBench runtime identity differs from Protocol 13")
    if arguments.grader_profile != execution.grader_profile:
        raise ValueError("HealthBench grader profile differs from Protocol 13")
    panel_ids = tuple(str(row["prompt_id"]) for row in evaluator.examples)
    if len(panel_ids) != SAMPLE_COUNT or len(set(panel_ids)) != SAMPLE_COUNT:
        raise ValueError("HealthBench panel contains duplicate or missing prompt IDs")
    service_contract = load_model_service_contract(arguments.service_contract)
    candidate_service = load_model_service_receipt(arguments.service_receipt)
    validate_replica_services((candidate_service,), contract=service_contract, execution=execution)
    grader_services = tuple(
        load_model_service_receipt(path) for path in arguments.grader_service_receipt
    )
    grader_endpoints = tuple(arguments.grader_endpoint_base or [arguments.endpoint_base])
    if len(grader_endpoints) != len(grader_services):
        raise ValueError("HealthBench requires one ordered service receipt per grader endpoint")
    validate_replica_services(grader_services, contract=service_contract, execution=execution)
    runtime_start_path = output_dir / "execution-start.json"
    if runtime_start_path.exists():
        runtime_start = load_execution_receipt(runtime_start_path)
        if (
            runtime_start.status != "running"
            or runtime_start.attempt_id != arguments.attempt_id
            or runtime_start.execution != execution.to_mapping()
            or runtime_start.planned_task_ids != panel_ids
            or runtime_start.service_instance_ids != (candidate_service.service_instance_id,)
            or runtime_start.generation_profile
            != RuntimeGenerationProfile.from_direct_profile(profile)
            or runtime_start.evaluator_version != execution.scorer_profile
            or runtime_start.evaluator_contract is not None
        ):
            raise ValueError("HealthBench runtime receipt differs from this phase")
    else:
        revision = _revision()
        runtime_start = Protocol13ExecutionReceipt(
            format_version=START_FORMAT,
            status="running",
            attempt_id=arguments.attempt_id,
            execution=execution.to_mapping(),
            generation_profile=RuntimeGenerationProfile.from_direct_profile(profile),
            environment=None,
            model_route=execution.actor_route,
            service_instance_ids=(candidate_service.service_instance_id,),
            response_model_ids=(),
            context_length=execution.context_length,
            adapter_active=False,
            panel_manifest_id=execution.panel_manifest_id,
            planned_task_ids=panel_ids,
            generated_task_ids=(),
            scored_task_ids=(),
            generation_code_revision=revision,
            scoring_code_revision=revision,
            evaluator_version=execution.scorer_profile,
            evaluator_contract=None,
            started_at=datetime.now(UTC).isoformat(),
            completed_at=None,
        )
        write_execution_receipt(runtime_start_path, runtime_start)
    candidate_directory = output_dir / "candidate"
    candidate_directory.mkdir(parents=True, exist_ok=True)
    candidate_journal = JsonlJournal(candidate_directory / "candidates.jsonl")
    candidate_failure_journal = JsonlJournal(candidate_directory / "infrastructure-failures.jsonl")
    grader_directory = output_dir / "grades" / arguments.grader_profile
    grader_directory.mkdir(parents=True, exist_ok=True)
    score_journal = JsonlJournal(grader_directory / "scores.jsonl")
    grader_failure_journal = JsonlJournal(grader_directory / "infrastructure-failures.jsonl")
    start_path = candidate_directory / "execution-start.json"
    start: dict[str, object] = {
        "format": "skillev-healthbench-candidate-execution-start@2",
        "panel_ids": list(panel_ids),
        "candidate_condition_id": "healthbench-qwen-direct-candidate@1",
        "evaluation_mode": arguments.evaluation_mode,
        "candidate_route": arguments.served_model_name,
        "candidate_service_instance_id": candidate_service.service_instance_id,
        "selection_rule": "openai-simple-evals-random.Random(0).sample",
        "expected_sample_count": SAMPLE_COUNT,
    }
    if start_path.exists():
        existing = json.loads(start_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("HealthBench candidate journal marker is not an object")
        if existing != start:
            raise ValueError("HealthBench journal belongs to another attempt contract")
    else:
        start_path.write_text(json.dumps(start, indent=2) + "\n", encoding="utf-8")

    if arguments.phase in {"candidates", "all"}:
        client = _candidate_client(
            arguments.endpoint_base,
            request_timeout_seconds=retry_policy.request_timeout_seconds,
        )
        _require_candidate_route(client, arguments.served_model_name)
        candidate_sampler = QwenOfficialCandidateSampler(
            client=client,
            model=arguments.served_model_name,
            sampler_response_type=sampler_response_type,
            retryable_errors=_candidate_retryable_errors(),
            retry_policy=retry_policy,
        )
        candidates = generate_candidates(
            examples=evaluator.examples,
            sampler=candidate_sampler,
            journal=candidate_journal,
            usage_parser=candidate_usage_dict,
            n_threads=arguments.n_threads,
            failure_journal=candidate_failure_journal,
        )
    else:
        candidates = candidate_journal.load()
    if len(candidates) == SAMPLE_COUNT:
        (candidate_directory / "execution-complete.json").write_text(
            json.dumps(
                {
                    **start,
                    "format": "skillev-healthbench-candidate-execution-complete@2",
                    "status": "complete",
                    "candidate_count": len(candidates),
                    "empty_candidate_count": sum(
                        not str(item.get("response_text", "")).strip()
                        for item in candidates.values()
                    ),
                    "candidate_infrastructure_failures": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if arguments.phase in {"grade", "all"}:
        grader_contract = HealthBenchGraderContract(
            profile_id=arguments.grader_profile,
            backend="qwen-sglang",
            model=arguments.served_model_name,
            rubric_call_mode="simple-evals-per-rubric",
            response_schema="criteria_met-boolean-with-bounded-semantic-repair",
            max_tokens=2048,
            reasoning_effort=None,
            maximum_attempts=arguments.transport_maximum_attempts,
            aggregation_profile="simple-evals-clipped-mean@1",
        )
        grader_start_path = grader_directory / "execution-start.json"
        grader_start = {
            "format": "skillev-healthbench-grader-attempt-start@2",
            "grader_condition_id": f"healthbench-{arguments.grader_backend}-rubric-grader@1",
            "grader_profile": arguments.grader_profile,
            "evaluation_mode": arguments.evaluation_mode,
            "grader_service_instance_ids": [
                receipt.service_instance_id for receipt in grader_services
            ],
            "candidate_model": arguments.served_model_name,
            "grader_model": arguments.served_model_name,
            "grader_backend": arguments.grader_backend,
            "rubric_call_mode": "simple-evals-per-rubric",
            "response_schema": "criteria_met-boolean-with-bounded-semantic-repair",
            "grader_contract": grader_contract.to_mapping(),
        }
        if grader_start_path.exists():
            if json.loads(grader_start_path.read_text(encoding="utf-8")) != grader_start:
                raise ValueError("HealthBench grader journal belongs to another contract")
        else:
            grader_start_path.write_text(
                json.dumps(grader_start, indent=2) + "\n", encoding="utf-8"
            )

    result: dict[str, object] = {
        "format": "skillev-healthbench-qwen-local-diagnostic-summary@1",
        "benchmark": "HealthBench Full 2025",
        "sample_count": SAMPLE_COUNT,
        "selection": "openai-simple-evals-random.Random(0).sample",
        "candidate": {
            "model": arguments.served_model_name,
            "system_message": OPENAI_SYSTEM_MESSAGE,
            "temperature": CANDIDATE_TEMPERATURE,
            "max_tokens": CANDIDATE_MAX_TOKENS,
            "direct_completion": True,
            "thinking": False,
            "adapter_free": True,
        },
        "grader": None,
        "candidate_count": len(candidates),
        "score_count": len(score_journal.load()),
        "metrics": None,
    }
    if arguments.phase in {"grade", "all"}:
        grader = _qwen_grader(
            endpoint_bases=list(grader_endpoints),
            served_model_name=arguments.served_model_name,
            sampler_response_type=sampler_response_type,
            retry_policy=retry_policy,
        )
        grader_identity = {
            "model": arguments.served_model_name,
            "backend": "qwen-sglang",
            "service_instance_ids": [receipt.service_instance_id for receipt in grader_services],
            "replicas": len(grader_endpoints),
            "unique_grader_models": 1,
            "judge_ensemble_size": 1,
            "system_message": OPENAI_SYSTEM_MESSAGE,
            "temperature": 0.5,
            "max_tokens": 2048,
            "thinking": False,
            "parser_and_retry": "pinned-openai-simple-evals; qwen-rubric-schema-on-parser-retry",
            "official_gpt4_1_comparable": False,
        }
        evaluator.grader_model = grader
        scores = grade_candidates(
            examples=evaluator.examples,
            candidates=candidates,
            evaluator=evaluator,
            journal=score_journal,
            n_threads=arguments.n_threads,
            failure_journal=grader_failure_journal,
        )
        normalized_scores, empty_candidate_count = normalize_healthbench_model_invalid_scores(
            scores=scores,
            candidates=candidates,
        )
        metrics = aggregate_official_scores(
            normalized_scores.values(), clipped_stat=healthbench._compute_clipped_stats
        )
        result["grader"] = grader_identity
        result["score_count"] = len(scores)
        result["metrics"] = metrics
        result["empty_candidate_count"] = empty_candidate_count
        rubric_count = sum(
            len(cast(list[object], item["rubric_items"]))
            for item in scores.values()
            if isinstance(item.get("rubric_items"), list)
        )
        receipt = HealthBenchGraderReceipt(
            grader_backend="qwen-sglang",
            grader_model=arguments.served_model_name,
            grader_replicas=len(grader.transports),
            unique_grader_models=1,
            official_gpt4_1_comparable=False,
            semantic_repair_count=grader.semantic_attempts.semantic_repair_count,
            transport_retry_count=sum(
                transport.transport_retry_count for transport in grader.transports
            ),
            malformed_final_count=0,
            rubric_item_count=rubric_count,
        )
        result["grader_receipt"] = {
            name: getattr(receipt, name) for name in receipt.__dataclass_fields__
        }
        (grader_directory / "execution-complete.json").write_text(
            json.dumps(
                {
                    **grader_start,
                    "format": "skillev-healthbench-grader-attempt-complete@2",
                    "status": "complete",
                    "score_count": len(scores),
                    "infrastructure_failures": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if len(candidates) == SAMPLE_COUNT and len(scores) == SAMPLE_COUNT:
            runtime_complete = replace(
                runtime_start,
                format_version=COMPLETE_FORMAT,
                status="complete",
                response_model_ids=(execution.actor_route,),
                generated_task_ids=panel_ids,
                scored_task_ids=panel_ids,
                scoring_code_revision=_revision(),
                completed_at=datetime.now(UTC).isoformat(),
            )
            runtime_complete.validate_against(
                execution=execution,
                expected_task_ids=panel_ids,
                expected_generation_profile=profile,
            )
            write_execution_receipt(output_dir / "execution-complete.json", runtime_complete)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main(_arguments())
