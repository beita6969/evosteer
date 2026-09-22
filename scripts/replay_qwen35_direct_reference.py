#!/usr/bin/env python3
"""Replay direct parsers and private scorers without a model request."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from skillev_private.benchmarks.code_math import (
    CodeExecutionInfrastructureError,
    MathEquivalenceInfrastructureError,
)
from skillev_private.direct_reference.evaluators import SCORER_REGISTRY
from skillev_private.direct_reference.manifests import load_population_manifest
from skillev_private.direct_reference.populations import load_skillflow_iid_cases
from skillev_private.direct_reference.replay import (
    collapse_generation_attempts,
    load_generation_records,
    validate_replay_contract,
)

from skillev.evaluation.direct_baseline.aggregation import (
    BenchmarkCoverage,
    CandidateOutcomeKind,
    TaskMetricVerdict,
    aggregate_metric,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.parsing import PARSER_REGISTRY
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol

_IID = frozenset(
    {
        DirectBenchmark.HOTPOT_QA,
        DirectBenchmark.TRIVIA_QA,
        DirectBenchmark.AIME_2026,
        DirectBenchmark.MED_QA,
    }
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--iid-population", type=Path, required=True)
    parser.add_argument("--population-manifest-dir", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(item.value for item in _IID),
        required=True,
    )
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_direct_reference_protocol(arguments.config)
    selected = frozenset(DirectBenchmark(value) for value in arguments.benchmarks)
    manifests = {
        benchmark: load_population_manifest(
            arguments.population_manifest_dir / f"{benchmark.value}.json"
        )
        for benchmark in selected
    }
    cases = load_skillflow_iid_cases(
        arguments.iid_population,
        protocol=protocol,
        include=selected,
        manifests=manifests,
    )
    by_id = {case.public_task.task_id: case for case in cases}
    records = tuple(
        record
        for record in collapse_generation_attempts(load_generation_records(arguments.journal))
        if record.benchmark in selected
    )
    if {record.task_id for record in records} != set(by_id):
        raise ValueError("replay journal does not exactly cover the selected frozen population")
    rows: list[dict[str, object]] = []
    verdicts: dict[DirectBenchmark, list[TaskMetricVerdict]] = defaultdict(list)
    generation_infrastructure: dict[DirectBenchmark, int] = defaultdict(int)
    scorer_infrastructure: dict[DirectBenchmark, int] = defaultdict(int)
    for record in records:
        spec = protocol.benchmark(record.benchmark)
        validate_replay_contract(record, spec, served_model_name=protocol.model.served_model_name)
        if record.infrastructure_error is not None:
            generation_infrastructure[record.benchmark] += 1
            verdicts[record.benchmark].append(
                TaskMetricVerdict(
                    record.task_id,
                    record.benchmark,
                    CandidateOutcomeKind.GENERATION_INFRASTRUCTURE,
                    {},
                    False,
                )
            )
            continue
        if record.raw_text is None:
            raise RuntimeError("validated successful replay record lacks raw text")
        parsed = PARSER_REGISTRY[spec.parser_profile](record.raw_text)
        try:
            score = await SCORER_REGISTRY[spec.scorer_profile](by_id[record.task_id], parsed)
        except (CodeExecutionInfrastructureError, MathEquivalenceInfrastructureError) as exc:
            # The CLI preserves a typed infrastructure category while retaining raw generation.
            scorer_infrastructure[record.benchmark] += 1
            verdicts[record.benchmark].append(
                TaskMetricVerdict(
                    record.task_id,
                    record.benchmark,
                    CandidateOutcomeKind.SCORER_INFRASTRUCTURE,
                    {},
                    False,
                )
            )
            rows.append({"task_id": record.task_id, "scorer_error": type(exc).__name__})
            continue
        verdicts[record.benchmark].append(
            TaskMetricVerdict(
                record.task_id,
                record.benchmark,
                (
                    CandidateOutcomeKind.SCORED
                    if score.verdict_kind.value == "scored"
                    else CandidateOutcomeKind.PARSE_FAILURE
                ),
                score.metrics,
                True,
            )
        )
        rows.append(
            {
                "task_id": record.task_id,
                "benchmark": record.benchmark.value,
                "parse_status": parsed.status.value,
                "parse_reason": parsed.reason.value,
                "metrics": score.metrics,
            }
        )
    summary: dict[str, object] = {}
    for benchmark in sorted(selected, key=lambda item: item.value):
        spec = protocol.benchmark(benchmark)
        benchmark_verdicts = tuple(verdicts[benchmark])
        generation_failures = generation_infrastructure[benchmark]
        scorer_failures = scorer_infrastructure[benchmark]
        coverage = BenchmarkCoverage(
            planned_count=spec.sample_count,
            final_record_count=len(benchmark_verdicts),
            candidate_response_count=len(benchmark_verdicts) - generation_failures,
            definitive_verdict_count=sum(item.definitive for item in benchmark_verdicts),
            generation_infrastructure_failures=generation_failures,
            scorer_infrastructure_failures=scorer_failures,
        )
        metrics = tuple(
            aggregate_metric(contract, benchmark_verdicts, coverage) for contract in spec.metrics
        )
        summary[benchmark.value] = {
            "planned_count": spec.sample_count,
            "definitive_verdict_count": coverage.definitive_verdict_count,
            "generation_infrastructure_failures": generation_failures,
            "scorer_infrastructure_failures": scorer_failures,
            "metrics": {
                metric.metric_id: (
                    str(metric.formal_observed_percent)
                    if metric.formal_observed_percent is not None
                    else None
                )
                for metric in metrics
            },
        }
    arguments.private_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.private_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(asyncio.run(_run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
