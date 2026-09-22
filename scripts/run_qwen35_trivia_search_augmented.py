#!/usr/bin/env python3
"""Run the private 128-case Qwen3.5 TriviaQA local-search track."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import yaml
from skillev_private.benchmarks.qa_metrics import (
    normalize_triviaqa_answer,
    score_qa_answers,
)
from skillev_private.direct_reference.journal import (
    require_fresh_output_directory,
    write_attempt_marker,
)
from skillev_private.direct_reference.manifests import load_population_manifest
from skillev_private.direct_reference.populations import load_skillflow_iid_cases
from skillev_private.direct_reference.trivia_search import (
    DATABASE_FORMAT,
    DATABASE_FORMAT_V2,
    TriviaSearchDatabase,
    load_frozen_trivia_questions,
    load_official_trivia_aliases_for_questions,
)

from skillev.evaluation.direct_baseline import (
    DirectBenchmark,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from skillev.evaluation.direct_baseline.config import PaperBenchmarkSpec
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol
from skillev.evaluation.direct_baseline.search_augmented import (
    TriviaSearchAttempt,
    TriviaSearchStep,
    TriviaSearchTask,
    run_trivia_search_task,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--iid-population", required=True, type=Path)
    parser.add_argument("--population-manifest", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--official-alias-source", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name", default="qwen35-direct-base")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _load_search_config(path: Path, base: PaperBenchmarkSpec) -> dict[str, Any]:
    value = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "search config")
    retrieval = _mapping(value.get("retrieval"), "retrieval config")
    retry = _mapping(value.get("retry"), "retry config")
    track_id = value.get("track_id")
    tracks = {
        "qwen35-triviaqa-search-augmented@1": {
            "prompt_profile": "triviaqa-local-search-react@1",
            "database_format": DATABASE_FORMAT,
            "hits_per_search": 5,
            "max_search_calls": 3,
        },
        "qwen35-triviaqa-search-augmented@2": {
            "prompt_profile": "triviaqa-local-search-react-detailed@2",
            "database_format": DATABASE_FORMAT_V2,
            "hits_per_search": 8,
            "max_search_calls": 1,
        },
    }
    if track_id not in tracks:
        raise ValueError("search track ID is unsupported")
    track = tracks[cast(str, track_id)]
    expected = {
        "track_id": track_id,
        "benchmark": "triviaqa",
        "population": base.population,
        "dataset_revision": base.dataset_revision,
        "selection_rule": base.selection_rule,
        "sample_count": base.sample_count,
        "scorer_profile": base.scorer_profile,
        "decoding_profile": base.decoding_profile,
        "seed": base.seed_aggregation.seeds[0],
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("search config differs from the frozen TriviaQA base contract")
    if value.get("prompt_profile") != track["prompt_profile"]:
        raise ValueError("search prompt profile is incompatible")
    if value.get("parser_profile") != "triviaqa-search-or-final@1":
        raise ValueError("search parser profile is incompatible")
    if retrieval != {
        "database_format": track["database_format"],
        "max_search_calls": track["max_search_calls"],
        "hits_per_search": track["hits_per_search"],
        "max_hits_per_document": 2,
        "task_partitioned": True,
    }:
        raise ValueError("retrieval runtime contract is incompatible")
    if retry != {
        "infrastructure_maximum_attempts": 3,
        "candidate_failure_retry": False,
    }:
        raise ValueError("retry runtime contract is incompatible")
    return value


def _step_value(task_id: str, step: TriviaSearchStep) -> dict[str, object]:
    value = asdict(step)
    value["task_id"] = task_id
    value["command_kind"] = step.command_kind.value
    return value


def _attempt_value(attempt: TriviaSearchAttempt) -> dict[str, object]:
    return {
        "task_id": attempt.task_id,
        "benchmark": attempt.benchmark.value,
        "final_value": attempt.final_response.value,
        "parse_status": attempt.final_response.status.value,
        "parse_reason": attempt.final_response.reason.value,
        "search_calls": attempt.search_calls,
        "infrastructure_error": attempt.infrastructure_error,
        "trace": [_step_value(attempt.task_id, step) for step in attempt.trace],
    }


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_direct_reference_protocol(arguments.base_config)
    spec = protocol.benchmark(DirectBenchmark.TRIVIA_QA)
    search_config = _load_search_config(arguments.config, spec)
    if arguments.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if arguments.served_model_name != protocol.model.served_model_name:
        raise ValueError("served model name differs from the frozen base protocol")
    manifest = load_population_manifest(arguments.population_manifest)
    if manifest.benchmark is not DirectBenchmark.TRIVIA_QA:
        raise ValueError("population manifest is not TriviaQA")
    cases = load_skillflow_iid_cases(
        arguments.iid_population,
        protocol=protocol,
        include=frozenset({DirectBenchmark.TRIVIA_QA}),
        manifests={DirectBenchmark.TRIVIA_QA: manifest},
    )
    retrieval = _mapping(search_config["retrieval"], "retrieval config")
    retry = _mapping(search_config["retry"], "retry config")
    questions = dict(load_frozen_trivia_questions(arguments.iid_population))
    if set(questions) != {case.public_task.task_id for case in cases}:
        raise ValueError("public TriviaQA questions differ from the frozen evaluated cases")
    official_aliases = load_official_trivia_aliases_for_questions(
        arguments.official_alias_source,
        questions,
    )
    tasks = tuple(
        TriviaSearchTask(
            task_id=case.public_task.task_id,
            question=questions[case.public_task.task_id],
            profile=protocol.profile(spec.decoding_profile),
            population_id=spec.population,
            run_seed=spec.seed_aggregation.seeds[0],
            max_search_calls=int(retrieval["max_search_calls"]),
            hits_per_search=int(retrieval["hits_per_search"]),
            max_hits_per_document=int(retrieval["max_hits_per_document"]),
            infrastructure_maximum_attempts=int(retry["infrastructure_maximum_attempts"]),
            prompt_profile_id=str(search_config["prompt_profile"]),
        )
        for case in cases
    )
    require_fresh_output_directory(arguments.private_output_dir)
    with TriviaSearchDatabase(
        arguments.database,
        expected_format=str(retrieval["database_format"]),
    ) as database:
        if database.task_ids() != tuple(sorted(task.task_id for task in tasks)):
            raise ValueError("search database does not exactly cover the frozen task IDs")
        client = OpenAICompatibleDirectClient(
            endpoint_base=arguments.endpoint_base,
            served_model_name=arguments.served_model_name,
            api_key=arguments.api_key,
            timeout_seconds=arguments.request_timeout_seconds,
            context_length=arguments.context_length,
            token_counter=QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path),  # type: ignore[arg-type]
        )
        if await client.model_routes() != (arguments.served_model_name,):
            raise RuntimeError("SGLang route identity is incompatible")
        write_attempt_marker(
            arguments.private_output_dir / "attempt-start.json",
            {
                "format": "skillev-trivia-search-attempt@1",
                "status": "running",
                "track_id": search_config["track_id"],
                "planned_task_ids": [task.task_id for task in tasks],
                "prompt_profile": search_config["prompt_profile"],
                "parser_profile": search_config["parser_profile"],
                "decoding_profile": spec.decoding_profile,
                "population": spec.population,
                "seed": spec.seed_aggregation.seeds[0],
            },
        )
        step_path = arguments.private_output_dir / "step-journal.jsonl"
        result_path = arguments.private_output_dir / "per-task-results.jsonl"
        step_lock = asyncio.Lock()
        result_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(arguments.concurrency)
        completed = 0
        started = time.monotonic()

        async def observe(task_id: str, step: TriviaSearchStep) -> None:
            row = json.dumps(_step_value(task_id, step), ensure_ascii=False, sort_keys=True)
            async with step_lock:
                with step_path.open("a", encoding="utf-8") as stream:
                    stream.write(row + "\n")
                    stream.flush()

        async def run_one(task: TriviaSearchTask) -> TriviaSearchAttempt:
            nonlocal completed
            async with semaphore:
                attempt = await run_trivia_search_task(
                    client,
                    task,
                    database,
                    step_observer=observe,
                )
            async with result_lock:
                with result_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(_attempt_value(attempt), ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                    stream.flush()
            async with progress_lock:
                completed += 1
                elapsed = max(time.monotonic() - started, 0.001)
                rate = completed / elapsed
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "planned": len(tasks),
                            "tasks_per_minute": 60 * rate,
                            "eta_seconds": (len(tasks) - completed) / rate,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return attempt

        attempts = tuple(await asyncio.gather(*(run_one(task) for task in tasks)))

    scored: list[tuple[TriviaSearchAttempt, dict[str, float]]] = []
    for attempt in attempts:
        if attempt.infrastructure_error is None:
            if attempt.final_response.value is None:
                metrics = {"em": 0.0, "f1": 0.0}
            else:
                observed = score_qa_answers(
                    attempt.final_response.value,
                    official_aliases[attempt.task_id],
                    normalizer=normalize_triviaqa_answer,
                )
                metrics = {"em": observed.em, "f1": observed.f1}
            scored.append(
                (
                    attempt,
                    metrics,
                )
            )
    definitive = len(scored)
    em = sum(score["em"] for _attempt, score in scored)
    f1 = sum(score["f1"] for _attempt, score in scored)
    complete = len(attempts) == spec.sample_count and definitive == spec.sample_count
    observed_em = 100 * em / definitive if definitive else None
    observed_f1 = 100 * f1 / definitive if definitive else None
    trace_steps = [step for attempt in attempts for step in attempt.trace]
    search_steps = [step for step in trace_steps if step.command_kind.value == "search"]
    candidate_invalid = sum(
        attempt.infrastructure_error is None and attempt.final_response.value is None
        for attempt in attempts
    )

    def metric(metric_id: str, observed: float | None) -> dict[str, object]:
        contract = next(item for item in spec.metrics if item.metric_id == metric_id)
        formal = observed if complete else None
        gap = abs(Decimal(str(formal)) - contract.reference_percent) if formal is not None else None
        return {
            "metric": metric_id,
            "reference_percent": str(contract.reference_percent),
            "diagnostic_observed_percent": str(observed) if observed is not None else None,
            "formal_observed_percent": str(formal) if formal is not None else None,
            "absolute_gap_pp": str(gap) if gap is not None else None,
            "numeric_status": (
                "PASS"
                if gap is not None and gap < protocol.parity.max_gap_pp_exclusive
                else "INCOMPLETE"
                if not complete
                else "FAIL"
            ),
        }

    summary: dict[str, object] = {
        "track_id": search_config["track_id"],
        "benchmark": "triviaqa",
        "method": "adapter-free-search-augmented",
        "population": spec.population,
        "dataset_revision": spec.dataset_revision,
        "selection_rule": spec.selection_rule,
        "dataset_variant": search_config["dataset_variant"],
        "prompt_profile": search_config["prompt_profile"],
        "parser_profile": search_config["parser_profile"],
        "decoding_profile": spec.decoding_profile,
        "scorer_profile": spec.scorer_profile,
        "scorer_alias_source": "pinned-official-unfiltered-nocontext-validation",
        "database_format": retrieval["database_format"],
        "coverage": {
            "planned_count": spec.sample_count,
            "final_record_count": len(attempts),
            "candidate_response_count": definitive,
            "definitive_verdict_count": definitive,
            "generation_infrastructure_failures": sum(
                attempt.infrastructure_error == "DirectGenerationError" for attempt in attempts
            ),
            "retrieval_infrastructure_failures": sum(
                attempt.infrastructure_error not in {None, "DirectGenerationError"}
                for attempt in attempts
            ),
        },
        "metrics": [metric("em", observed_em), metric("f1", observed_f1)],
        "candidate_invalid_count": candidate_invalid,
        "search_usage_rate_percent": (
            100 * sum(attempt.search_calls > 0 for attempt in attempts) / len(attempts)
        ),
        "mean_search_calls": sum(attempt.search_calls for attempt in attempts) / len(attempts),
        "empty_hit_rate_percent": (
            100
            * sum(step.observation == "No matching passages." for step in search_steps)
            / len(search_steps)
            if search_steps
            else None
        ),
        "wall_time_seconds": time.monotonic() - started,
        "tasks_per_minute": 60 * len(attempts) / max(time.monotonic() - started, 0.001),
    }
    (arguments.private_output_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_attempt_marker(
        arguments.private_output_dir / "attempt-complete.json",
        {
            "format": "skillev-trivia-search-attempt@1",
            "status": "complete",
            "final_record_count": len(attempts),
        },
    )
    return summary


def main() -> None:
    print(json.dumps(asyncio.run(_run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
