#!/usr/bin/env python3
"""Generate, journal, score, and aggregate direct static benchmark panels."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from skillev_private.benchmarks.code_math import (
    CodeExecutionInfrastructureError,
    MathEquivalenceInfrastructureError,
)
from skillev_private.direct_reference import (
    PrivateDirectCase,
    load_gpqa_cases,
    load_humaneval_cases,
    load_math_hard_cases,
    load_mind2web_cases,
    load_musique_cases,
    load_nq_open_cases,
    load_skillflow_iid_cases,
)
from skillev_private.direct_reference.evaluators import SCORER_REGISTRY
from skillev_private.direct_reference.journal import (
    PrivateGenerationJournal,
    require_fresh_output_directory,
    write_attempt_marker,
)
from skillev_private.direct_reference.manifests import (
    PopulationManifest,
    load_population_manifest,
    validate_manifest_task_ids,
)

from skillev.evaluation.direct_baseline import (
    DirectAttempt,
    DirectBenchmark,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_direct_tasks,
)
from skillev.evaluation.direct_baseline.aggregation import (
    BenchmarkCoverage,
    CandidateOutcomeKind,
    TaskMetricVerdict,
    aggregate_metric,
)
from skillev.evaluation.direct_baseline.client import (
    DirectGenerationRequest,
    DirectGenerationResult,
)
from skillev.evaluation.direct_baseline.config import DirectReferenceProtocol
from skillev.evaluation.direct_baseline.protocol import (
    load_direct_reference_protocol,
    validate_runtime_registries,
)

_SCORABLE_IID = frozenset(
    {
        DirectBenchmark.HOTPOT_QA,
        DirectBenchmark.TRIVIA_QA,
        DirectBenchmark.AIME_2026,
        DirectBenchmark.MED_QA,
    }
)
_SCORABLE_OOD = frozenset(
    {
        DirectBenchmark.MUSIQUE,
        DirectBenchmark.NQ_OPEN,
        DirectBenchmark.MATH_HARD,
        DirectBenchmark.GPQA_DIAMOND,
        DirectBenchmark.HUMAN_EVAL,
        DirectBenchmark.MIND2WEB,
    }
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iid-population", type=Path)
    parser.add_argument("--population-manifest-dir", type=Path, required=True)
    parser.add_argument("--musique-archive", type=Path)
    parser.add_argument("--nq-open-jsonl", type=Path)
    parser.add_argument("--math-hard-parquet", type=Path)
    parser.add_argument("--gpqa-csv", type=Path)
    parser.add_argument("--humaneval-jsonl-gz", type=Path)
    parser.add_argument("--mind2web-archive", type=Path)
    parser.add_argument("--mind2web-scores", type=Path)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--additional-endpoint-base", action="append", default=[])
    parser.add_argument("--served-model-name")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(item.value for item in _SCORABLE_IID | _SCORABLE_OOD),
        default=sorted(item.value for item in _SCORABLE_IID | _SCORABLE_OOD),
    )
    return parser.parse_args()


@dataclass(slots=True)
class _RoundRobinDirectClient:
    clients: tuple[OpenAICompatibleDirectClient, ...]
    _next: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        async with self._lock:
            client = self.clients[self._next % len(self.clients)]
            self._next += 1
        return await client.generate(request)


def _marker(cases: tuple[PrivateDirectCase, ...]) -> dict[str, object]:
    return {
        "format": "skillev-direct-attempt@1",
        "status": "running",
        "planned_task_ids": [case.public_task.task_id for case in cases],
        "prompt_profiles": sorted({case.public_task.prompt_profile_id for case in cases}),
        "decoding_profiles": sorted({case.public_task.profile.profile_id for case in cases}),
        "population_ids": sorted({case.public_task.population_id for case in cases}),
        "seeds": sorted({case.public_task.run_seed for case in cases}),
    }


def _load_resume_attempts(
    path: Path, cases: tuple[PrivateDirectCase, ...]
) -> tuple[dict[str, DirectAttempt], dict[str, int]]:
    by_id = {case.public_task.task_id: case.public_task for case in cases}
    latest: dict[str, DirectAttempt] = {}
    attempts: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("generation journal row must be an object")
            task_id = str(raw.get("task_id") or "")
            task = by_id.get(task_id)
            request_attempt = int(raw.get("request_attempt", 0))
            if task is None or request_attempt != attempts[task_id] + 1:
                raise ValueError("generation journal identity or attempt sequence differs")
            if (
                raw.get("benchmark") != task.benchmark.value
                or raw.get("prompt_profile_id") != task.prompt_profile_id
                or raw.get("parser_profile_id") != task.parser_profile_id
                or raw.get("decoding_profile_id") != task.profile.profile_id
                or raw.get("population_id") != task.population_id
                or raw.get("run_seed") != task.run_seed
            ):
                raise ValueError("generation journal contract differs from the frozen task")
            infrastructure_error = raw.get("infrastructure_error")
            raw_text = raw.get("raw_text")
            if infrastructure_error is not None and not isinstance(infrastructure_error, str):
                raise ValueError("generation infrastructure error has an invalid type")
            if infrastructure_error is None and not isinstance(raw_text, str):
                raise ValueError("successful generation journal row has no response text")
            parsed = task.parser(raw_text) if isinstance(raw_text, str) else None
            latest[task_id] = DirectAttempt(
                task_id=task_id,
                benchmark=task.benchmark,
                raw_text=raw_text if isinstance(raw_text, str) else None,
                reasoning_text=(
                    str(raw["reasoning_text"])
                    if isinstance(raw.get("reasoning_text"), str)
                    else None
                ),
                parsed=parsed,
                finish_reason=(
                    str(raw["finish_reason"]) if isinstance(raw.get("finish_reason"), str) else None
                ),
                prompt_tokens=(
                    int(raw["prompt_tokens"]) if isinstance(raw.get("prompt_tokens"), int) else None
                ),
                completion_tokens=(
                    int(raw["completion_tokens"])
                    if isinstance(raw.get("completion_tokens"), int)
                    else None
                ),
                response_model=(
                    str(raw["response_model"])
                    if isinstance(raw.get("response_model"), str)
                    else None
                ),
                response_id=(
                    str(raw["response_id"]) if isinstance(raw.get("response_id"), str) else None
                ),
                service_instance_id=(
                    str(raw["service_instance_id"])
                    if isinstance(raw.get("service_instance_id"), str)
                    else None
                ),
                panel_index=task.panel_index,
                service_slot=task.service_slot,
                prompt_profile_id=task.prompt_profile_id,
                parser_profile_id=task.parser_profile_id,
                decoding_profile_id=task.profile.profile_id,
                population_id=task.population_id,
                run_seed=task.run_seed,
                request_attempt=request_attempt,
                infrastructure_error=infrastructure_error,
            )
            attempts[task_id] = request_attempt
    return latest, attempts


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_direct_reference_protocol(arguments.config)
    validate_runtime_registries(protocol)
    unknown_scorers = {
        spec.scorer_profile
        for spec in protocol.benchmarks
        if spec.scorer_profile not in SCORER_REGISTRY
    }.difference(
        {
            "webshop-official-reward@1",
            "alfworld-official-success@1",
            "scienceworld-official-reward@1",
            "swe-official-resolved@1",
        }
    )
    if unknown_scorers:
        raise ValueError(f"unknown private scorer profiles: {sorted(unknown_scorers)}")
    selected = frozenset(DirectBenchmark(value) for value in arguments.benchmarks)
    manifests = _load_manifests(arguments.population_manifest_dir, protocol, selected)
    cases = _load_cases(arguments, protocol, selected, manifests)
    marker = _marker(cases)
    marker_path = arguments.private_output_dir / "attempt-start.json"
    journal_path = (arguments.private_output_dir / "generations.jsonl").resolve()
    if arguments.resume:
        if not marker_path.is_file() or not journal_path.is_file():
            raise FileNotFoundError("resume requires the original attempt marker and journal")
        if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise RuntimeError("resume attempt marker differs from the frozen contract")
        if (arguments.private_output_dir / "attempt-complete.json").exists():
            raise RuntimeError("completed direct attempt cannot be resumed")
    else:
        require_fresh_output_directory(arguments.private_output_dir)
        write_attempt_marker(marker_path, marker)
    journal = PrivateGenerationJournal(journal_path, append_existing=arguments.resume)
    served_model_name = arguments.served_model_name or protocol.model.served_model_name
    if served_model_name != protocol.model.served_model_name:
        raise ValueError("served model name differs from the frozen protocol")
    endpoints = tuple(dict.fromkeys([arguments.endpoint_base, *arguments.additional_endpoint_base]))
    clients = tuple(
        OpenAICompatibleDirectClient(
            endpoint_base=endpoint,
            served_model_name=served_model_name,
            api_key=arguments.api_key,
            timeout_seconds=arguments.request_timeout_seconds,
            context_length=arguments.context_length,
            token_counter=QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path),  # type: ignore[arg-type]
        )
        for endpoint in endpoints
    )
    client = _RoundRobinDirectClient(clients)
    latest, prior_attempts = (
        _load_resume_attempts(journal_path, cases) if arguments.resume else ({}, {})
    )
    async with journal:
        pending_by_attempt: dict[int, list[PrivateDirectCase]] = defaultdict(list)
        for case in cases:
            prior = latest.get(case.public_task.task_id)
            if prior is None or prior.infrastructure_error is not None:
                pending_by_attempt[prior_attempts.get(case.public_task.task_id, 0) + 1].append(case)
        for request_attempt, pending in sorted(pending_by_attempt.items()):
            generated = await run_direct_tasks(
                client,
                tuple(case.public_task for case in pending),
                concurrency=arguments.concurrency,
                generation_observer=journal.append,
                request_attempt=request_attempt,
            )
            latest.update({attempt.task_id: attempt for attempt in generated})
    attempts = tuple(latest[case.public_task.task_id] for case in cases)
    by_id = {case.public_task.task_id: case for case in cases}
    verdicts: dict[DirectBenchmark, list[TaskMetricVerdict]] = defaultdict(list)
    submissions: dict[DirectBenchmark, int] = defaultdict(int)
    scorer_invocations: dict[DirectBenchmark, int] = defaultdict(int)
    generation_infra: dict[DirectBenchmark, int] = defaultdict(int)
    scorer_infra: dict[DirectBenchmark, int] = defaultdict(int)
    private_rows: list[dict[str, object]] = []
    for attempt in attempts:
        benchmark = attempt.benchmark
        spec = protocol.benchmark(benchmark)
        if attempt.infrastructure_error is not None:
            generation_infra[benchmark] += 1
            verdicts[benchmark].append(
                TaskMetricVerdict(
                    attempt.task_id,
                    benchmark,
                    CandidateOutcomeKind.GENERATION_INFRASTRUCTURE,
                    {},
                    False,
                )
            )
            private_rows.append(_private_attempt_row(attempt, None, None))
            continue
        if attempt.parsed is None or attempt.raw_text is None:
            raise RuntimeError("successful generation lacks raw or parsed response")
        submissions[benchmark] += int(attempt.parsed.value is not None)
        try:
            score = await SCORER_REGISTRY[spec.scorer_profile](
                by_id[attempt.task_id], attempt.parsed
            )
        except (CodeExecutionInfrastructureError, MathEquivalenceInfrastructureError) as exc:
            scorer_infra[benchmark] += 1
            verdicts[benchmark].append(
                TaskMetricVerdict(
                    attempt.task_id,
                    benchmark,
                    CandidateOutcomeKind.SCORER_INFRASTRUCTURE,
                    {},
                    False,
                )
            )
            private_rows.append(_private_attempt_row(attempt, None, type(exc).__name__))
            continue
        scorer_invocations[benchmark] += int(score.scorer_invoked)
        declared_metric_ids = {contract.metric_id for contract in spec.metrics}
        if set(score.metrics) != declared_metric_ids:
            raise ValueError(f"{benchmark.value} scorer returned an undeclared metric set")
        kind = (
            CandidateOutcomeKind.SCORED
            if score.verdict_kind.value == "scored"
            else CandidateOutcomeKind.PARSE_FAILURE
        )
        verdicts[benchmark].append(
            TaskMetricVerdict(attempt.task_id, benchmark, kind, score.metrics, True)
        )
        private_rows.append(_private_attempt_row(attempt, score.metrics, None))

    with (arguments.private_output_dir / "per-task-results.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in private_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary: dict[str, object] = {}
    for benchmark in sorted(selected, key=lambda item: item.value):
        benchmark_verdicts = tuple(verdicts[benchmark])
        spec = protocol.benchmark(benchmark)
        definitive = sum(item.definitive for item in benchmark_verdicts)
        candidate_responses = len(benchmark_verdicts) - generation_infra[benchmark]
        coverage = BenchmarkCoverage(
            planned_count=spec.sample_count,
            final_record_count=len(benchmark_verdicts),
            candidate_response_count=candidate_responses,
            definitive_verdict_count=definitive,
            generation_infrastructure_failures=generation_infra[benchmark],
            scorer_infrastructure_failures=scorer_infra[benchmark],
        )
        metrics = tuple(
            aggregate_metric(contract, benchmark_verdicts, coverage) for contract in spec.metrics
        )
        denominator = len(benchmark_verdicts)
        summary[benchmark.value] = {
            "benchmark": benchmark.value,
            "comparability": spec.comparability.value,
            "population": spec.population,
            "dataset_revision": spec.dataset_revision,
            "selection_rule": spec.selection_rule,
            "prompt_profile": spec.prompt_profile,
            "decoding_profile": spec.decoding_profile,
            "parser_profile": spec.parser_profile,
            "scorer_profile": spec.scorer_profile,
            "seed_aggregation": spec.seed_aggregation.mode.value,
            "dataset_variant": spec.dataset_variant,
            "comparability_evidence": {
                "population": spec.evidence.population.value,
                "prompt": spec.evidence.prompt.value,
                "decoding": spec.evidence.decoding.value,
                "seed_aggregation": spec.evidence.seed_aggregation.value,
                "scorer": spec.evidence.scorer.value,
                "environment": spec.evidence.environment.value,
            },
            "availability": coverage.availability.value,
            "coverage": {
                "planned_count": coverage.planned_count,
                "final_record_count": coverage.final_record_count,
                "candidate_response_count": coverage.candidate_response_count,
                "definitive_verdict_count": coverage.definitive_verdict_count,
                "generation_infrastructure_failures": generation_infra[benchmark],
                "scorer_infrastructure_failures": scorer_infra[benchmark],
                "environment_infrastructure_failures": 0,
            },
            "metrics": [
                {
                    "metric": metric.metric_id,
                    "reference_percent": str(metric.reference_percent),
                    "diagnostic_observed_percent": _decimal(metric.diagnostic_observed_percent),
                    "formal_observed_percent": _decimal(metric.formal_observed_percent),
                    "absolute_gap_pp": _decimal(metric.formal_gap_pp),
                    "numeric_status": (
                        "PASS"
                        if metric.numeric_passes(policy=protocol.parity, coverage=coverage)
                        else coverage.availability.value.upper()
                        if coverage.availability.value != "complete"
                        else "FAIL"
                    ),
                }
                for metric in metrics
            ],
            "submission_rate_percent": 100 * submissions[benchmark] / denominator
            if denominator
            else None,
            "scorer_reach_rate_percent": 100 * scorer_invocations[benchmark] / denominator
            if denominator
            else None,
        }
    with (arguments.private_output_dir / "aggregate.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    unresolved_infrastructure = sum(generation_infra.values()) + sum(scorer_infra.values())
    if unresolved_infrastructure:
        raise RuntimeError(
            f"{unresolved_infrastructure} direct-reference infrastructure failures remain"
        )
    write_attempt_marker(
        arguments.private_output_dir / "attempt-complete.json",
        {
            "format": "skillev-direct-attempt@1",
            "status": "complete",
            "final_record_count": len(attempts),
        },
    )
    return summary


def _private_attempt_row(
    attempt: DirectAttempt, metrics: object, scorer_error: str | None
) -> dict[str, object]:
    parsed = attempt.parsed
    return {
        "task_id": attempt.task_id,
        "benchmark": attempt.benchmark.value,
        "parse_status": parsed.status.value if parsed is not None else None,
        "parse_reason": parsed.reason.value if parsed is not None else None,
        "metrics": metrics,
        "finish_reason": attempt.finish_reason,
        "generation_infrastructure_error": attempt.infrastructure_error,
        "scorer_infrastructure_error": scorer_error,
    }


def _decimal(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _load_cases(
    arguments: argparse.Namespace,
    protocol: DirectReferenceProtocol,
    selected: frozenset[DirectBenchmark],
    manifests: dict[DirectBenchmark, PopulationManifest],
) -> tuple[PrivateDirectCase, ...]:
    cases: list[PrivateDirectCase] = []
    iid = selected & _SCORABLE_IID
    if iid:
        if arguments.iid_population is None:
            raise ValueError("--iid-population is required for selected IID benchmarks")
        cases.extend(
            load_skillflow_iid_cases(
                arguments.iid_population,
                protocol=protocol,
                include=iid,
                manifests=manifests,
            )
        )
    loaders = (
        (DirectBenchmark.MUSIQUE, arguments.musique_archive, load_musique_cases),
        (DirectBenchmark.NQ_OPEN, arguments.nq_open_jsonl, load_nq_open_cases),
        (DirectBenchmark.MATH_HARD, arguments.math_hard_parquet, load_math_hard_cases),
        (DirectBenchmark.GPQA_DIAMOND, arguments.gpqa_csv, load_gpqa_cases),
    )
    for benchmark, path, loader in loaders:
        if benchmark not in selected:
            continue
        if path is None:
            raise ValueError(f"private source path is required for {benchmark.value}")
        cases.extend(loader(path, protocol=protocol))
    if DirectBenchmark.HUMAN_EVAL in selected:
        if arguments.humaneval_jsonl_gz is None:
            raise ValueError("private source path is required for humaneval")
        cases.extend(
            load_humaneval_cases(
                arguments.humaneval_jsonl_gz,
                protocol=protocol,
                manifest=manifests[DirectBenchmark.HUMAN_EVAL],
            )
        )
    if DirectBenchmark.MIND2WEB in selected:
        if arguments.mind2web_archive is None or arguments.mind2web_scores is None:
            raise ValueError("Mind2Web requires both archive and candidate-score paths")
        cases.extend(
            load_mind2web_cases(
                arguments.mind2web_archive,
                arguments.mind2web_scores,
                protocol=protocol,
                archive_password=_mind2web_archive_password(),
            )
        )
    result = tuple(cases)
    for benchmark in selected:
        selected_cases = tuple(case for case in result if case.public_task.benchmark is benchmark)
        validate_manifest_task_ids(
            manifests[benchmark],
            population_id=protocol.benchmark(benchmark).population,
            task_ids=tuple(case.public_task.task_id for case in selected_cases),
            source_identities=tuple(
                str(case.private_metadata["source_identity"]) for case in selected_cases
            ),
        )
    return result


def _load_manifests(
    directory: Path,
    protocol: DirectReferenceProtocol,
    selected: frozenset[DirectBenchmark],
) -> dict[DirectBenchmark, PopulationManifest]:
    manifests: dict[DirectBenchmark, PopulationManifest] = {}
    for benchmark in selected:
        manifest = load_population_manifest(directory / f"{benchmark.value}.json")
        if manifest.benchmark is not benchmark:
            raise ValueError(f"manifest benchmark mismatch for {benchmark.value}")
        if manifest.population_id != protocol.benchmark(benchmark).population:
            raise ValueError(f"manifest population mismatch for {benchmark.value}")
        manifests[benchmark] = manifest
    return manifests


def _mind2web_archive_password() -> bytes:
    value = os.environ.get("SKILLEV_MIND2WEB_ZIP_PASSWORD")
    if not value:
        raise ValueError("SKILLEV_MIND2WEB_ZIP_PASSWORD is required for Mind2Web")
    return value.encode("utf-8")


def main() -> None:
    print(json.dumps(asyncio.run(_run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
