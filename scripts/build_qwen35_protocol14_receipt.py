#!/usr/bin/env python3
"""Build answer-free Protocol 14 receipts, freeze targets, and publish results."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import yaml

from skillev.evaluation.current_iid.protocol14.aggregation import aggregate_protocol_v14
from skillev.evaluation.current_iid.protocol14.catalog import (
    ACTIVE_PROTOCOL14_BENCHMARKS,
    Protocol14Benchmark,
)
from skillev.evaluation.current_iid.protocol14.config import (
    load_execution_contracts_v4,
    load_targets_v5,
)
from skillev.evaluation.current_iid.protocol14.contracts import (
    ExecutionContractV4,
    FormalEligibility,
)
from skillev.evaluation.current_iid.protocol14.publication import (
    render_protocol_v14_markdown,
)
from skillev.evaluation.current_iid.protocol14.receipts import (
    AttemptRole,
    MetricObservationV4,
    OutcomeCountsV5,
    Protocol14RunReceipt,
    load_run_receipt,
    write_run_receipt,
)
from skillev.evaluation.current_iid.protocol14.targets import MetricProjection
from skillev.experiments.protocol_v14 import load_protocol_v14

_METRICS: dict[Protocol14Benchmark, tuple[tuple[str, str, str, str, MetricProjection], ...]] = {
    Protocol14Benchmark.HOTPOT_QA: (
        (
            "em",
            "planned-panel-128",
            "official_hotpotqa_exact_match",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
        (
            "f1",
            "planned-panel-128",
            "official_hotpotqa_token_f1",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
    Protocol14Benchmark.TRIVIA_QA: (
        (
            "em",
            "planned-panel-128",
            "official_triviaqa_alias_exact_match",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
        (
            "f1",
            "planned-panel-128",
            "official_triviaqa_alias_token_f1",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
    Protocol14Benchmark.AIME_2026: (
        (
            "accuracy",
            "planned-panel-30",
            "official_aime_integer_exact_match",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
    Protocol14Benchmark.HEALTHBENCH: (
        (
            "overall_score",
            "planned-panel-128",
            "simple_evals_mean_then_clip",
            "single-seed-identity",
            MetricProjection.CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT,
        ),
    ),
    Protocol14Benchmark.WEB_SHOP: (
        (
            "average_score",
            "planned-panel-128",
            "sum(native_task_score) / planned_count * 100",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
        (
            "success_rate",
            "planned-panel-128",
            "count(native_task_score == 1) / planned_count * 100",
            "success-rate",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
    Protocol14Benchmark.ALF_WORLD: (
        (
            "success_rate",
            "planned-panel-128",
            "count(native_won) / planned_count * 100",
            "success-rate",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
    Protocol14Benchmark.MBPP_PLUS: (
        (
            "pass_at_1",
            "planned-panel-128",
            "count(base_and_plus_pass) / planned_count * 100",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
    Protocol14Benchmark.HUMAN_EVAL: (
        (
            "pass_at_1",
            "planned-panel-128",
            "count(official_humaneval_pass) / planned_count * 100",
            "mean-over-planned-records",
            MetricProjection.IDENTITY_PERCENT,
        ),
    ),
}


def _required_for_formal_gate(execution: ExecutionContractV4) -> bool:
    return execution.formal_eligibility is FormalEligibility.FORMAL


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-run")
    build.add_argument(
        "--benchmark", choices=[item.value for item in ACTIVE_PROTOCOL14_BENCHMARKS], required=True
    )
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--role", choices=("reference", "candidate"), required=True)
    build.add_argument("--output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-targets")
    freeze.add_argument("--targets", type=Path, required=True)
    freeze.add_argument("--reference-receipts-dir", type=Path, required=True)
    freeze.add_argument("--frozen-targets-output", type=Path, required=True)
    freeze.add_argument("--reference-package-output", type=Path, required=True)
    freeze.add_argument("--reference-package-public-path", required=True)
    freeze.add_argument("--frozen-at")
    publish = subparsers.add_parser("publish")
    publish.add_argument("--targets", type=Path, required=True)
    publish.add_argument("--candidate-receipts-dir", type=Path, required=True)
    publish.add_argument("--json-output", type=Path, required=True)
    publish.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} contains a non-object row")
            rows.append(value)
    return rows


def _metric(
    execution: ExecutionContractV4, metric_id: str, numerator: Decimal
) -> MetricObservationV4:
    matches = [row for row in _METRICS[execution.benchmark] if row[0] == metric_id]
    if len(matches) != 1:
        raise ValueError("Protocol 14 metric identity is not unique")
    _, denominator_id, formula, aggregation, projection = matches[0]
    raw = numerator / Decimal(execution.expected_count) * Decimal(100)
    observed = (
        min(Decimal(100), max(Decimal(0), raw))
        if projection is MetricProjection.CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT
        else raw
    )
    return MetricObservationV4(
        metric_id=metric_id,
        unit="percent",
        numerator=numerator,
        denominator=execution.expected_count,
        denominator_id=denominator_id,
        projection=projection,
        observed_percent=observed,
        formula=formula,
        aggregation=aggregation,
    )


def _provenance(run_dir: Path, execution: ExecutionContractV4) -> dict[str, str]:
    complete = _load_json(run_dir / "execution-complete.json")
    if complete.get("status") != "complete" or complete.get("execution") != execution.to_mapping():
        raise ValueError("private execution receipt differs from Protocol 14")
    return {
        "attempt_id": _text(complete["attempt_id"], "attempt ID"),
        "attempt_role": _text(complete["attempt_role"], "attempt role"),
        "generation_code_revision": _text(
            complete["generation_code_revision"], "generation revision"
        ),
        "scoring_code_revision": _text(complete["scoring_code_revision"], "scoring revision"),
        "evaluator_version": _text(complete["evaluator_version"], "evaluator version"),
        "started_at": _text(complete["started_at"], "start time"),
        "completed_at": _text(complete["completed_at"], "completion time"),
    }


def _static_receipt(
    execution: ExecutionContractV4, rows: list[dict[str, object]]
) -> tuple[OutcomeCountsV5, tuple[MetricObservationV4, ...], dict[str, Decimal | int | str]]:
    if len(rows) != execution.expected_count:
        raise ValueError("static run does not conserve the panel")
    generation_infra = sum(row.get("generation_infrastructure_error") is not None for row in rows)
    scorer_infra = sum(row.get("scorer_infrastructure_error") is not None for row in rows)
    definitive_rows = [
        row
        for row in rows
        if row.get("generation_infrastructure_error") is None
        and row.get("scorer_infrastructure_error") is None
    ]
    invalid = sum(row.get("parse_status") != "extracted" for row in definitive_rows)
    numerators: dict[str, Decimal] = {row[0]: Decimal(0) for row in _METRICS[execution.benchmark]}
    for row in definitive_rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("definitive static row lacks metrics")
        for metric_id in numerators:
            value = metrics.get(metric_id)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("static metric must be numeric")
            numerators[metric_id] += Decimal(str(value))
    metrics = tuple(_metric(execution, name, value) for name, value in numerators.items())
    binary = execution.benchmark in {
        Protocol14Benchmark.AIME_2026,
        Protocol14Benchmark.HUMAN_EVAL,
    }
    successes = int(numerators[next(iter(numerators))]) if binary else None
    definitive = len(definitive_rows)
    outcomes = OutcomeCountsV5(
        scored_valid=definitive - invalid,
        model_output_invalid=invalid,
        generation_infrastructure=generation_infra,
        environment_infrastructure=0,
        scorer_infrastructure=scorer_infra,
        binary_success=successes,
        binary_failure=None if successes is None else definitive - successes,
        definitive_detail={"scored-valid": definitive - invalid, "model-invalid": invalid},
    )
    return outcomes, metrics, {}


def _interactive_receipt(
    execution: ExecutionContractV4, rows: list[dict[str, object]]
) -> tuple[OutcomeCountsV5, tuple[MetricObservationV4, ...], dict[str, Decimal | int | str]]:
    if len(rows) != execution.expected_count:
        raise ValueError("interactive run does not conserve the panel")
    generation_infra = sum(
        row.get("infrastructure_error") == "DirectGenerationError" for row in rows
    )
    environment_infra = sum(
        (row.get("infrastructure_error") not in {None, "DirectGenerationError"})
        or row.get("cleanup_error") is not None
        for row in rows
    )
    definitive_rows = [
        row
        for row in rows
        if row.get("infrastructure_error") is None and row.get("cleanup_error") is None
    ]
    invalid = sum(row.get("termination_reason") == "candidate-invalid" for row in definitive_rows)
    reward = sum(Decimal(str(row.get("reward") or 0)) for row in definitive_rows)
    successes = sum(row.get("success") is True for row in definitive_rows)
    metrics = [_metric(execution, "success_rate", Decimal(successes))]
    if execution.benchmark is Protocol14Benchmark.WEB_SHOP:
        metrics.insert(0, _metric(execution, "average_score", reward))
    definitive = len(definitive_rows)
    outcomes = OutcomeCountsV5(
        scored_valid=definitive - invalid,
        model_output_invalid=invalid,
        generation_infrastructure=generation_infra,
        environment_infrastructure=environment_infra,
        scorer_infrastructure=0,
        binary_success=successes,
        binary_failure=definitive - successes,
        definitive_detail={"scored-valid": definitive - invalid, "model-invalid": invalid},
    )
    diagnostics: dict[str, Decimal | int | str] = {
        "total_steps": sum(_integer(row.get("steps"), "steps") for row in definitive_rows),
        "invalid_actions": sum(
            _integer(row.get("invalid_actions"), "invalid actions") for row in definitive_rows
        ),
    }
    return outcomes, tuple(metrics), diagnostics


def _health_receipt(
    execution: ExecutionContractV4, run_dir: Path, summary: dict[str, object]
) -> tuple[OutcomeCountsV5, tuple[MetricObservationV4, ...], dict[str, Decimal | int | str]]:
    if summary.get("status") != "complete":
        raise ValueError("HealthBench summary is incomplete")
    judge = _mapping(summary.get("judge"), "HealthBench judge")
    if (
        judge.get("label") != "Qwen-local-judge diagnostic"
        or judge.get("backend") != "qwen-sglang"
        or judge.get("model_route") != execution.actor_route
        or judge.get("official_gpt_comparable") is not False
    ):
        raise ValueError("HealthBench result does not use the owner-authorized Qwen judge")
    aggregate = _mapping(summary.get("aggregate"), "HealthBench aggregate")
    metrics_raw = _mapping(aggregate.get("metrics"), "HealthBench metrics")
    overall = _mapping(metrics_raw.get("overall_score"), "HealthBench overall score")
    observed_percent = Decimal(str(overall.get("mean_percent")))
    numerator = observed_percent / Decimal(100) * Decimal(execution.expected_count)
    empty_ids_by_seed: list[set[str]] = []
    for seed in execution.aggregation.seeds:
        candidates = _load_jsonl(run_dir / f"seed-{seed}" / "candidates.jsonl")
        if len(candidates) != execution.expected_count:
            raise ValueError("HealthBench candidate seed is incomplete")
        empty_ids_by_seed.append(
            {
                _text(row.get("prompt_id"), "prompt ID")
                for row in candidates
                if not str(row.get("response_text", "")).strip()
            }
        )
    invalid = len(set().union(*empty_ids_by_seed))
    definitive = execution.expected_count
    outcomes = OutcomeCountsV5(
        scored_valid=definitive - invalid,
        model_output_invalid=invalid,
        generation_infrastructure=0,
        environment_infrastructure=0,
        scorer_infrastructure=0,
        definitive_detail={"scored-valid": definitive - invalid, "model-invalid": invalid},
    )
    population_std = overall.get("population_std_percent")
    diagnostics = {
        "judge_backend": "qwen-sglang",
        "judge_label": "Qwen-local-judge diagnostic",
        "official_gpt_comparable": "false",
        "population_std_percent": (
            "not-applicable-single-seed" if population_std is None else str(population_std)
        ),
        "seed_count": len(execution.aggregation.seeds),
    }
    return outcomes, (_metric(execution, "overall_score", numerator),), diagnostics


def _mbpp_receipt(
    execution: ExecutionContractV4, summary: dict[str, object]
) -> tuple[OutcomeCountsV5, tuple[MetricObservationV4, ...], dict[str, Decimal | int | str]]:
    panel = _mapping(summary.get("panel"), "MBPP+ panel")
    count = _integer(panel.get("count"), "panel count")
    if count != execution.expected_count:
        raise ValueError("MBPP+ panel count differs")
    scored = _integer(panel.get("scored_valid"), "scored valid")
    invalid = _integer(panel.get("model_output_invalid"), "model invalid")
    scorer_infra = _integer(panel.get("scorer_infrastructure"), "scorer infrastructure")
    successes = _integer(panel.get("successes"), "successes")
    definitive = scored + invalid
    outcomes = OutcomeCountsV5(
        scored_valid=scored,
        model_output_invalid=invalid,
        generation_infrastructure=0,
        environment_infrastructure=0,
        scorer_infrastructure=scorer_infra,
        binary_success=successes,
        binary_failure=definitive - successes,
        definitive_detail={"scored-valid": scored, "model-invalid": invalid},
    )
    full = _mapping(summary.get("full"), "MBPP+ full")
    diagnostics = {
        "source_full_count": _integer(full.get("count"), "full count"),
        "source_full_percent": str(full.get("base_plus_pass_at_1_percent")),
    }
    return outcomes, (_metric(execution, "pass_at_1", Decimal(successes)),), diagnostics


def build_run_receipt(
    *,
    execution: ExecutionContractV4,
    run_dir: Path,
    role: AttemptRole,
) -> Protocol14RunReceipt:
    provenance = _provenance(run_dir, execution)
    if provenance["attempt_role"] != role.value:
        raise ValueError("requested receipt role differs from the execution")
    if execution.benchmark in {
        Protocol14Benchmark.HOTPOT_QA,
        Protocol14Benchmark.TRIVIA_QA,
        Protocol14Benchmark.AIME_2026,
        Protocol14Benchmark.HUMAN_EVAL,
    }:
        outcomes, metrics, diagnostics = _static_receipt(
            execution, _load_jsonl(run_dir / "per-task-results.jsonl")
        )
    elif execution.benchmark in {
        Protocol14Benchmark.WEB_SHOP,
        Protocol14Benchmark.ALF_WORLD,
    }:
        outcomes, metrics, diagnostics = _interactive_receipt(
            execution, _load_jsonl(run_dir / "per-task-results.jsonl")
        )
    elif execution.benchmark is Protocol14Benchmark.HEALTHBENCH:
        outcomes, metrics, diagnostics = _health_receipt(
            execution, run_dir, _load_json(run_dir / "summary.json")
        )
    else:
        outcomes, metrics, diagnostics = _mbpp_receipt(
            execution, _load_json(run_dir / "summary.json")
        )
    return Protocol14RunReceipt(
        benchmark=execution.benchmark,
        role=role,
        execution=execution,
        attempt_id=provenance["attempt_id"],
        planned_count=execution.expected_count,
        outcomes=outcomes,
        metrics=metrics,
        generation_code_revision=provenance["generation_code_revision"],
        scoring_code_revision=provenance["scoring_code_revision"],
        evaluator_version=provenance["evaluator_version"],
        started_at=provenance["started_at"],
        completed_at=provenance["completed_at"],
        runtime_contract_matched=True,
        final_panel_used_for_selection=False,
        diagnostics=diagnostics,
    )


def _receipt_paths(directory: Path) -> dict[Protocol14Benchmark, Path]:
    return {
        benchmark: directory / f"{benchmark.value}.json"
        for benchmark in ACTIVE_PROTOCOL14_BENCHMARKS
    }


def freeze_targets(
    arguments: argparse.Namespace,
    executions: dict[Protocol14Benchmark, ExecutionContractV4],
) -> None:
    root = yaml.safe_load(arguments.targets.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or not isinstance(root.get("targets"), dict):
        raise ValueError("target registry is invalid")
    target_rows = cast(dict[str, dict[str, object]], root["targets"])
    paths = _receipt_paths(arguments.reference_receipts_dir)
    receipts = {
        benchmark: load_run_receipt(path, execution=executions[benchmark])
        for benchmark, path in paths.items()
    }
    if any(
        receipt.role is not AttemptRole.REFERENCE or not receipt.is_formally_complete
        for receipt in receipts.values()
    ):
        raise ValueError("all reference receipts must be complete and infrastructure-clear")
    frozen_at = arguments.frozen_at or datetime.now(UTC).isoformat()
    for benchmark in ACTIVE_PROTOCOL14_BENCHMARKS:
        receipt = receipts[benchmark]
        row = target_rows[benchmark.value]
        old_evidence = row.get("evidence")
        old_metrics = row.get("metrics")
        anchors = row.get("published_anchors")
        if not isinstance(anchors, list):
            raise ValueError("published anchors must be an array")
        if isinstance(old_evidence, dict) and isinstance(old_metrics, list):
            for old_metric in old_metrics:
                if not isinstance(old_metric, dict):
                    raise ValueError("pending target metric is invalid")
                anchors.append(
                    {
                        "label": f"published source anchor for {benchmark.value}",
                        "scope": "standalone-benchmark",
                        "metric_id": old_metric["metric_id"],
                        "value_percent": old_metric["reference_percent"],
                        "usable_as_formal_target": False,
                        "evidence": old_evidence,
                    }
                )
        row["status"] = "frozen-matched-reference"
        row["reference_condition_id"] = executions[benchmark].condition_id
        row["reference_receipt_id"] = receipt.attempt_id
        row["frozen_at"] = frozen_at
        row["evidence"] = {
            "source_kind": "independent-matched-reference-run",
            "repository_or_paper": "https://github.com/YJLi-new/SKILLEV-new",
            "revision_or_version": receipt.generation_code_revision,
            "path_or_section": arguments.reference_package_public_path,
            "supported_fields": [
                "benchmark",
                "model_revision",
                "population",
                "panel",
                "prompt",
                "thinking_mode",
                "decoding",
                "parser",
                "environment",
                "scorer",
                "grader",
                "metric",
                "aggregation",
                "coverage",
            ],
        }
        templates = {item[0]: item for item in _METRICS[benchmark]}
        row["metrics"] = [
            {
                "metric_id": metric.metric_id,
                "unit": metric.unit,
                "denominator_id": metric.denominator_id,
                "projection": metric.projection.value,
                "formula": metric.formula,
                "aggregation": metric.aggregation,
                "reference_percent": str(metric.observed_percent),
                "maximum_gap_pp_exclusive": "7.0",
                "required_for_formal_gate": _required_for_formal_gate(executions[benchmark]),
            }
            for metric in receipt.metrics
            if metric.metric_id in templates
        ]
    arguments.frozen_targets_output.write_text(
        yaml.safe_dump(root, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    load_targets_v5(arguments.frozen_targets_output, executions=executions)
    package = {
        "format": "skillev-protocol14-reference-targets@1",
        "frozen_at": frozen_at,
        "catalog": [item.value for item in ACTIVE_PROTOCOL14_BENCHMARKS],
        "receipts": {
            benchmark.value: receipts[benchmark].to_mapping()
            for benchmark in ACTIVE_PROTOCOL14_BENCHMARKS
        },
    }
    arguments.reference_package_output.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def publish(
    arguments: argparse.Namespace,
    executions: dict[Protocol14Benchmark, ExecutionContractV4],
) -> None:
    targets = load_targets_v5(arguments.targets, executions=executions)
    paths = _receipt_paths(arguments.candidate_receipts_dir)
    receipts = {
        benchmark: load_run_receipt(path, execution=executions[benchmark])
        for benchmark, path in paths.items()
    }
    aggregate = aggregate_protocol_v14(receipts, targets)
    arguments.json_output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arguments.markdown_output.write_text(render_protocol_v14_markdown(aggregate), encoding="utf-8")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def main() -> None:
    arguments = _arguments()
    protocol = load_protocol_v14(arguments.protocol, arguments.sources)
    executions = load_execution_contracts_v4(arguments.conditions, protocol=protocol)
    if arguments.command == "build-run":
        benchmark = Protocol14Benchmark(arguments.benchmark)
        receipt = build_run_receipt(
            execution=executions[benchmark],
            run_dir=arguments.run_dir,
            role=AttemptRole(arguments.role),
        )
        write_run_receipt(arguments.output, receipt)
    elif arguments.command == "freeze-targets":
        freeze_targets(arguments, executions)
    else:
        publish(arguments, executions)


if __name__ == "__main__":
    main()
