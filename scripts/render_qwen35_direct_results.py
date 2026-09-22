#!/usr/bin/env python3
"""Render public direct-reference Markdown from aggregate JSON and protocol YAML."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import cast

from skillev.evaluation.direct_baseline.aggregation import (
    AggregatedMetric,
    BenchmarkCoverage,
    compute_published_aggregate,
)
from skillev.evaluation.direct_baseline.config import (
    DirectBenchmark,
    DirectParityPolicy,
    MetricContract,
)
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol
from skillev.evaluation.direct_baseline.reporting import (
    BenchmarkResult,
    DirectParityGateResult,
    EvaluationScope,
    MetricResult,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aggregate-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("iid", "ood", "all"), default="all")
    parser.add_argument(
        "--engineering-check", choices=("passed", "failed", "not-run"), default="not-run"
    )
    parser.add_argument(
        "--engineering-receipt",
        type=Path,
        help="Public JSON receipt whose status is passed, failed, or not-run",
    )
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _load_results(paths: list[Path]) -> dict[DirectBenchmark, dict[str, object]]:
    merged: dict[DirectBenchmark, dict[str, object]] = {}
    for path in paths:
        raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "aggregate result")
        for key, value in raw.items():
            benchmark = DirectBenchmark(key)
            if benchmark in merged:
                raise ValueError(f"duplicate aggregate result for {benchmark.value}")
            merged[benchmark] = _mapping(value, benchmark.value)
    return merged


def _metric_rows(value: dict[str, object]) -> list[dict[str, object]]:
    rows = value.get("metrics")
    if type(rows) is not list:
        raise ValueError("benchmark result lacks metric rows")
    return [_mapping(row, "metric") for row in cast(list[object], rows)]


def _engineering_status(arguments: argparse.Namespace) -> tuple[str, str | None]:
    if arguments.role == "all" and arguments.engineering_receipt is None:
        raise ValueError("all-scope formal reporting requires an engineering receipt")
    if arguments.engineering_receipt is None:
        return str(arguments.engineering_check), None
    receipt = _mapping(
        json.loads(arguments.engineering_receipt.read_text(encoding="utf-8")),
        "engineering receipt",
    )
    status = receipt.get("status")
    if not isinstance(status, str) or status not in {"passed", "failed", "not-run"}:
        raise ValueError("engineering receipt has an invalid status")
    command = receipt.get("command")
    if command is not None:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("engineering receipt command must be non-empty text")
    return status, command


def _decimal(value: object) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _display_percent(value: Decimal | None) -> str:
    return "N/A" if value is None else f"{value.quantize(Decimal('0.01'))}%"


def _raw_percent(value: Decimal | None) -> str:
    return "N/A" if value is None else f"{value}%"


def _rate(value: object) -> str:
    return _display_percent(_decimal(value))


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _coverage_infrastructure(coverage: dict[str, object]) -> tuple[int, int, int, int]:
    generation = _integer(
        coverage.get("generation_infrastructure_failures", 0),
        "generation infrastructure failures",
    )
    scorer = _integer(
        coverage.get("scorer_infrastructure_failures", 0),
        "scorer infrastructure failures",
    )
    environment = _integer(
        coverage.get("environment_infrastructure_failures", 0),
        "environment infrastructure failures",
    )
    return generation, scorer, environment, generation + scorer + environment


def _coverage(value: dict[str, object]) -> BenchmarkCoverage:
    return BenchmarkCoverage(
        planned_count=_integer(value["planned_count"], "planned count"),
        final_record_count=_integer(value["final_record_count"], "final record count"),
        candidate_response_count=_integer(
            value["candidate_response_count"], "candidate response count"
        ),
        definitive_verdict_count=_integer(
            value["definitive_verdict_count"], "definitive verdict count"
        ),
        generation_infrastructure_failures=_integer(
            value.get("generation_infrastructure_failures", 0),
            "generation infrastructure failures",
        ),
        scorer_infrastructure_failures=_integer(
            value.get("scorer_infrastructure_failures", 0),
            "scorer infrastructure failures",
        ),
        environment_infrastructure_failures=_integer(
            value.get("environment_infrastructure_failures", 0),
            "environment infrastructure failures",
        ),
    )


def _recompute_metric(
    *,
    row: dict[str, object],
    contract: MetricContract,
    coverage: BenchmarkCoverage,
    policy: DirectParityPolicy,
) -> MetricResult:
    result = MetricResult(
        metric_id=contract.metric_id,
        reference_percent=contract.reference_percent,
        diagnostic_observed_percent=_decimal(row.get("diagnostic_observed_percent")),
        formal_observed_percent=_decimal(row.get("formal_observed_percent")),
    )
    declared_reference = _decimal(row.get("reference_percent"))
    declared_gap = _decimal(row.get("absolute_gap_pp"))
    expected_status = result.numeric_status(policy=policy, coverage=coverage)
    if declared_reference != contract.reference_percent:
        raise ValueError("aggregate reference differs from protocol")
    if declared_gap != result.gap_pp:
        raise ValueError("aggregate gap differs from recomputed gap")
    if row.get("numeric_status") != expected_status:
        raise ValueError("aggregate status differs from recomputed status")
    return result


def _render(arguments: argparse.Namespace) -> str:
    protocol = load_direct_reference_protocol(arguments.config)
    results = _load_results(arguments.aggregate_json)
    specs = tuple(
        spec
        for spec in protocol.benchmarks
        if arguments.role == "all" or spec.role == arguments.role
    )
    if set(results) != {spec.benchmark for spec in specs}:
        raise ValueError("public aggregate does not exactly cover the requested protocol role")
    metric_lookup: dict[tuple[DirectBenchmark, str], AggregatedMetric] = {}
    public_results: list[BenchmarkResult] = []
    engineering_status, engineering_command = _engineering_status(arguments)
    total_planned = sum(spec.sample_count for spec in specs)
    lines = [
        "# Qwen3.5-9B Direct Reference Results",
        "",
        "> Generated from the executable protocol and public aggregate JSON; no private "
        "predictions or targets are read by this renderer.",
        "",
        "This is the adapter-free Paper Direct-Qwen track. It does not use LoRA, skills, "
        "retrieval, the posterior, the Z head, or the structured action codec.",
        "",
        f"Requested role: **{arguments.role}**; planned records: **{total_planned}**. AIME "
        "uses its complete 30-record panel and every other selected benchmark uses 128.",
        "",
        "## Component metrics",
        "",
        f"The numeric rule is strict: `absolute gap < {protocol.parity.max_gap_pp_exclusive} "
        "percentage points`. The "
        "full-precision value drives the status; the two-decimal value is display only.",
        "",
        "| Benchmark | Metric | Reference | Observed (full) | Observed (display) | Gap "
        "(pp) | Threshold | Numeric | Scientific comparability | Planned/scored | Infra "
        "(gen/scorer/env) | Prompt | Decoding | Population | Seed aggregation |",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---|---|---|---|",
    ]
    for spec in specs:
        value = results[spec.benchmark]
        if value.get("benchmark") != spec.benchmark.value:
            raise ValueError("result benchmark identity differs from protocol")
        coverage = _mapping(value.get("coverage"), "coverage")
        if coverage.get("planned_count") != spec.sample_count:
            raise ValueError("result planned population differs from protocol")
        coverage_contract = _coverage(coverage)
        expected_bindings = {
            "comparability": spec.comparability.value,
            "population": spec.population,
            "dataset_revision": spec.dataset_revision,
            "selection_rule": spec.selection_rule,
            "prompt_profile": spec.prompt_profile,
            "decoding_profile": spec.decoding_profile,
            "parser_profile": spec.parser_profile,
            "scorer_profile": spec.scorer_profile,
            "seed_aggregation": spec.seed_aggregation.mode.value,
        }
        for key, expected in expected_bindings.items():
            if value.get(key) != expected:
                raise ValueError(f"aggregate {key} differs from protocol")
        if "dataset_variant" in value and value.get("dataset_variant") != spec.dataset_variant:
            raise ValueError("aggregate dataset_variant differs from protocol")
        metrics = _metric_rows(value)
        if {row.get("metric") for row in metrics} != {
            contract.metric_id for contract in spec.metrics
        }:
            raise ValueError("result metric set differs from protocol")
        seed_mode = spec.seed_aggregation.mode.value
        row_scientific = spec.evidence.is_exact
        generation_infra, scorer_infra, environment_infra, _ = _coverage_infrastructure(coverage)
        scientific_label = (
            f"{spec.comparability.value}; {seed_mode}; "
            f"{'eligible' if row_scientific else 'scientific NO-GO'}"
        )
        seed_label = (
            f"{seed_mode}; seeds={','.join(str(seed) for seed in spec.seed_aggregation.seeds)}"
        )
        recomputed_metrics: list[MetricResult] = []
        for contract in spec.metrics:
            row = next(item for item in metrics if item.get("metric") == contract.metric_id)
            metric_id = str(row["metric"])
            metric = _recompute_metric(
                row=row,
                contract=contract,
                coverage=coverage_contract,
                policy=protocol.parity,
            )
            recomputed_metrics.append(metric)
            observed = metric.formal_observed_percent
            reference = metric.reference_percent
            gap = metric.gap_pp
            metric_lookup[(spec.benchmark, metric_id)] = AggregatedMetric(
                metric.metric_id,
                metric.reference_percent,
                metric.diagnostic_observed_percent,
                metric.formal_observed_percent,
            )
            status = metric.numeric_status(policy=protocol.parity, coverage=coverage_contract)
            lines.append(
                "| {benchmark} | {metric} | {reference} | {observed_full} | "
                "{observed_display} | {gap} | `<{threshold} pp` | {status} | {scientific} | "
                "{planned}/{scored} | {gen}/{scorer}/{env} | `{prompt}` | `{decoding}` | "
                "`{population}` | {seed} |".format(
                    benchmark=spec.benchmark.value,
                    metric=metric_id,
                    reference=_raw_percent(reference),
                    observed_full=_raw_percent(observed),
                    observed_display=_display_percent(observed),
                    gap=(str(gap) if gap is not None else "N/A"),
                    threshold=protocol.parity.max_gap_pp_exclusive,
                    status=status,
                    scientific=scientific_label,
                    planned=coverage.get("planned_count"),
                    scored=coverage.get("definitive_verdict_count"),
                    gen=generation_infra,
                    scorer=scorer_infra,
                    env=environment_infra,
                    prompt=spec.prompt_profile,
                    decoding=spec.decoding_profile,
                    population=spec.population,
                    seed=seed_label,
                )
            )
        public_results.append(
            BenchmarkResult(
                benchmark=spec.benchmark,
                comparability=spec.comparability,
                evidence=spec.evidence,
                population=spec.population,
                prompt_profile=spec.prompt_profile,
                decoding_profile=spec.decoding_profile,
                parser_profile=spec.parser_profile,
                scorer_profile=spec.scorer_profile,
                seed_aggregation=seed_mode,
                coverage=coverage_contract,
                metrics=tuple(recomputed_metrics),
                submission_rate_percent=_decimal(value.get("submission_rate_percent"))
                or Decimal("0"),
                scorer_reach_rate_percent=_decimal(value.get("scorer_reach_rate_percent")),
                valid_native_action_rate_percent=_decimal(
                    value.get("valid_native_action_rate_percent")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage and telemetry",
            "",
            "Submission is an output-contract event, not benchmark success. A missing "
            "official verdict remains infrastructure and is never converted to a zero.",
            "",
            "| Benchmark | Final | Candidate responses | Definitive verdicts | Submission | "
            "Scorer/terminal reach | Valid native action | Parse invalid | Env invalid | "
            "Partial reward | Infra total | Availability |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for spec in specs:
        value = results[spec.benchmark]
        coverage = _mapping(value.get("coverage"), "coverage")
        _, _, _, infra_total = _coverage_infrastructure(coverage)
        reach = value.get("scorer_reach_rate_percent")
        if reach is None:
            reach = value.get("official_terminal_rate_percent")
        lines.append(
            "| {benchmark} | {final}/{planned} | {candidate} | {verdicts} | {submission} | "
            "{reach} | {valid} | {parse_invalid} | {env_invalid} | {partial_reward} | "
            "{infra} | {availability} |".format(
                benchmark=spec.benchmark.value,
                final=coverage.get("final_record_count"),
                planned=coverage.get("planned_count"),
                candidate=coverage.get("candidate_response_count"),
                verdicts=coverage.get("definitive_verdict_count"),
                submission=_rate(value.get("submission_rate_percent")),
                reach=_rate(reach),
                valid=_rate(value.get("valid_action_rate_percent")),
                parse_invalid=_rate(value.get("parse_invalid_rate_percent")),
                env_invalid=_rate(value.get("environment_invalid_rate_percent")),
                partial_reward=_rate(value.get("partial_reward_rate_percent")),
                infra=infra_total,
                availability=value.get("availability", "unknown"),
            )
        )
    lines.extend(["", "## Published aggregate rows", ""])
    lines.extend(
        [
            "| Aggregate | Observed (full) | Observed (display) | Reference | Gap (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    aggregate_results: list[MetricResult] = []
    for aggregate in protocol.aggregates:
        if not aggregate.required_for_public_table:
            continue
        if any(component.benchmark not in results for component in aggregate.components):
            continue
        computed = compute_published_aggregate(aggregate, metric_lookup)
        aggregate_result = MetricResult(
            metric_id=computed.metric_id,
            reference_percent=computed.reference_percent,
            diagnostic_observed_percent=computed.diagnostic_observed_percent,
            formal_observed_percent=computed.formal_observed_percent,
        )
        aggregate_results.append(aggregate_result)
        observed = aggregate_result.formal_observed_percent
        gap = aggregate_result.gap_pp
        lines.append(
            f"| {aggregate.aggregate_id} | {_raw_percent(observed)} | "
            f"{_display_percent(observed)} | {aggregate.reference_percent}% | "
            f"{gap if gap is not None else 'N/A'} |"
        )
    scope = EvaluationScope(arguments.role)
    gate = DirectParityGateResult(
        scope=scope,
        benchmark_results=tuple(public_results),
        aggregate_results=tuple(aggregate_results),
        engineering_check_passed=engineering_status == "passed",
        protocol_semantics_known=True,
        policy=protocol.parity,
        expected_all_benchmarks=frozenset(spec.benchmark for spec in protocol.benchmarks),
    )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Scope numeric parity: **{'GO' if gate.scope_numeric_pass else 'NO-GO'}**",
            f"- Scope scientific parity: **{'GO' if gate.scope_scientific_pass else 'NO-GO'}**",
            f"- Engineering validation: **{engineering_status.upper()}**",
            f"- Formal training: **{gate.formal_training_status.value.upper()}**",
            "",
            "Unknown paper seed/protocol semantics remain scientific NO-GO even when a numeric "
            "single-run score falls inside the strict "
            f"`<{protocol.parity.max_gap_pp_exclusive}` percentage-point band.",
            "",
        ]
    )
    if engineering_command is not None:
        lines.extend([f"Engineering receipt command: `{engineering_command}`.", ""])
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    arguments.output.write_text(_render(arguments), encoding="utf-8")


if __name__ == "__main__":
    main()
