#!/usr/bin/env python3
"""Build one answer-free Protocol 13 receipt from private runner artifacts."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import cast

from skillev_private.direct_reference.protocol13_runner import (
    load_protocol13_environment_manifest_identity,
    load_protocol13_panel_manifest,
)

from skillev.evaluation.current_iid.protocol13.adapters import (
    PrivatePanelOutcome,
    require_exact_private_panel,
)
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.current_iid.protocol13.execution_receipts import (
    load_execution_receipt,
)
from skillev.evaluation.current_iid.protocol13.receipts import (
    MetricObservationV3,
    OutcomeCountsV4,
    Protocol13RunReceipt,
    RunProvenanceV3,
)
from skillev.evaluation.current_iid.protocol13.runner_profiles import (
    load_protocol13_runner_profiles,
)
from skillev.evaluation.current_iid.protocol13.targets import MetricProjection
from skillev.experiments.protocol_v13 import load_protocol_v13


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[item.value for item in Protocol13Benchmark],
        required=True,
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument(
        "--grader-profile",
        help="HealthBench only: select exactly one explicit grader journal.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-code-revision", required=True)
    return parser.parse_args()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            rows.append(cast(dict[str, object], value))
    return tuple(rows)


def _required_text(value: object, key: str) -> str:
    if not isinstance(value, dict) or type(value.get(key)) is not str:
        raise ValueError(f"record lacks {key}")
    result = cast(str, value[key])
    if not result.strip():
        raise ValueError(f"record has empty {key}")
    return result


def _unique_rows_by_id(
    rows: tuple[dict[str, object], ...],
    *,
    key: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        identity = _required_text(row, key)
        if identity in result:
            raise ValueError(f"duplicate {key}: {identity}")
        result[identity] = row
    return result


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{label} is not decimal-compatible")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _metric(
    metric_id: str,
    numerator: Decimal,
    planned: int,
    *,
    formula: str,
    aggregation: str,
    projection: MetricProjection = MetricProjection.IDENTITY_PERCENT,
) -> MetricObservationV3:
    raw = numerator / Decimal(planned) * Decimal(100)
    observed = (
        min(Decimal(100), max(Decimal(0), raw))
        if projection is MetricProjection.CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT
        else raw
    )
    return MetricObservationV3(
        metric_id=metric_id,
        unit="percent",
        numerator=numerator,
        denominator=planned,
        denominator_id=f"planned-panel-{planned}",
        projection=projection,
        observed_percent=observed,
        formula=formula,
        aggregation=aggregation,
    )


def _static(
    artifact_dir: Path,
    benchmark: Protocol13Benchmark,
    task_ids: tuple[str, ...],
) -> tuple[OutcomeCountsV4, tuple[MetricObservationV3, ...], dict[str, int | str]]:
    rows = tuple(
        row
        for row in _jsonl(artifact_dir / "per-task-results.jsonl")
        if row.get("benchmark") == benchmark.value
    )
    joined = require_exact_private_panel(
        expected_task_ids=task_ids,
        outcomes=tuple(
            PrivatePanelOutcome(
                task_id=_required_text(row, "task_id"),
                outcome=_static_outcome(row),
                metrics={
                    str(name): str(value)
                    for name, value in cast(dict[str, object], row.get("metrics", {})).items()
                },
                infrastructure_error=(
                    str(row.get("generation_infrastructure_error"))
                    if row.get("generation_infrastructure_error") is not None
                    else str(row.get("scorer_infrastructure_error"))
                    if row.get("scorer_infrastructure_error") is not None
                    else None
                ),
            )
            for row in rows
        ),
    )
    by_id = {_required_text(row, "task_id"): row for row in rows}
    ordered = tuple(by_id[item.task_id] for item in joined)
    kinds = tuple(item.outcome for item in joined)
    success = sum(item == "scored-success" for item in kinds)
    failure = sum(item == "scored-failure" for item in kinds)
    counts = OutcomeCountsV4(
        definitive_scored=success + failure,
        candidate_invalid=sum(item == "candidate-invalid" for item in kinds),
        generation_infrastructure=sum(item == "generation-infrastructure" for item in kinds),
        environment_infrastructure=0,
        scorer_infrastructure=sum(item == "scorer-infrastructure" for item in kinds),
        binary_success=success,
        binary_failure=failure,
    )
    metric_ids = {
        str(name)
        for row in ordered
        if isinstance(row.get("metrics"), dict)
        for name in cast(dict[str, object], row["metrics"])
    }
    formula_by_benchmark = {
        Protocol13Benchmark.HOTPOT_QA: {
            "em": "official_hotpotqa_exact_match",
            "f1": "official_hotpotqa_token_f1",
        },
        Protocol13Benchmark.TRIVIA_QA: {
            "em": "official_triviaqa_alias_exact_match",
            "f1": "official_triviaqa_alias_token_f1",
        },
        Protocol13Benchmark.AIME_2026: {"accuracy": "official_aime_integer_exact_match"},
        Protocol13Benchmark.HUMAN_EVAL: {
            "pass_at_1": "count(official_humaneval_pass) / planned_count * 100"
        },
    }
    formulas = formula_by_benchmark[benchmark]
    if metric_ids != set(formulas):
        raise ValueError("static artifact metric set differs from Protocol 13")
    metrics = tuple(
        _metric(
            metric_id,
            sum(
                (
                    _decimal(cast(dict[str, object], row["metrics"])[metric_id], metric_id)
                    for row in ordered
                    if isinstance(row.get("metrics"), dict)
                ),
                Decimal(0),
            ),
            len(task_ids),
            formula=formula,
            aggregation="mean-over-records",
        )
        for metric_id, formula in formulas.items()
    )
    diagnostics: dict[str, int | str] = {
        "submission_count": sum(row.get("parse_status") == "extracted" for row in ordered),
        "manifest_join": "exact",
    }
    return counts, metrics, diagnostics


def _static_outcome(row: dict[str, object]) -> str:
    if row.get("generation_infrastructure_error") is not None:
        return "generation-infrastructure"
    if row.get("scorer_infrastructure_error") is not None:
        return "scorer-infrastructure"
    if row.get("parse_status") != "extracted":
        return "candidate-invalid"
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("definitive static row lacks metrics")
    passed = all(_decimal(value, "metric") == 1 for value in metrics.values())
    return "scored-success" if passed else "scored-failure"


def _interactive(
    artifact_dir: Path,
    benchmark: Protocol13Benchmark,
    task_ids: tuple[str, ...],
) -> tuple[OutcomeCountsV4, tuple[MetricObservationV3, ...], dict[str, int | str]]:
    rows = tuple(
        row
        for row in _jsonl(artifact_dir / "per-task-results.jsonl")
        if row.get("benchmark") == benchmark.value
    )
    outcomes: list[PrivatePanelOutcome] = []
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        task_id = _required_text(row, "task_id")
        if row.get("infrastructure_error") is not None:
            kind = (
                "generation-infrastructure"
                if row.get("infrastructure_error") == "DirectGenerationError"
                else "environment-infrastructure"
            )
        elif row.get("termination_reason") == "candidate-invalid":
            kind = "candidate-invalid"
        else:
            kind = "scored-success" if row.get("success") is True else "scored-failure"
        outcomes.append(PrivatePanelOutcome(task_id, kind, {}, None))
        if task_id in by_id:
            raise ValueError("duplicate interactive task result")
        by_id[task_id] = row
    joined = require_exact_private_panel(expected_task_ids=task_ids, outcomes=tuple(outcomes))
    ordered = tuple(by_id[item.task_id] for item in joined)
    kinds = tuple(item.outcome for item in joined)
    success = sum(item == "scored-success" for item in kinds)
    failure = sum(item == "scored-failure" for item in kinds)
    counts = OutcomeCountsV4(
        definitive_scored=success + failure,
        candidate_invalid=sum(item == "candidate-invalid" for item in kinds),
        generation_infrastructure=sum(item == "generation-infrastructure" for item in kinds),
        environment_infrastructure=sum(item == "environment-infrastructure" for item in kinds),
        scorer_infrastructure=0,
        binary_success=success,
        binary_failure=failure,
    )
    if benchmark is Protocol13Benchmark.WEB_SHOP:
        rewards = tuple(_decimal(row.get("reward", 0), "native reward") for row in ordered)
        metrics: tuple[MetricObservationV3, ...] = (
            _metric(
                "average_score",
                sum(rewards, Decimal(0)),
                len(task_ids),
                formula="sum(native_task_score) / planned_count * 100",
                aggregation="mean-over-records",
            ),
            _metric(
                "success_rate",
                Decimal(sum(value == 1 for value in rewards)),
                len(task_ids),
                formula="count(native_task_score == 1) / planned_count * 100",
                aggregation="success-rate",
            ),
        )
    else:
        metrics = (
            _metric(
                "success_rate",
                Decimal(sum(row.get("success") is True for row in ordered)),
                len(task_ids),
                formula="count(native_won) / planned_count * 100",
                aggregation="success-rate",
            ),
        )
    steps = sorted(int(row.get("steps", 0)) for row in ordered)
    diagnostics: dict[str, int | str] = {
        "manifest_join": "exact",
        "horizon_failures": sum(row.get("terminated_by_horizon") is True for row in ordered),
        "budget_exhausted": sum(row.get("budget_exhausted") is True for row in ordered),
        "terminal_success_count": sum(row.get("terminal_success") is True for row in ordered),
        "terminal_failure_at_budget_count": sum(
            row.get("terminal_success") is False and row.get("budget_exhausted") is True
            for row in ordered
        ),
        "terminal_failure_before_budget_count": sum(
            row.get("terminal_success") is False and row.get("budget_exhausted") is not True
            for row in ordered
        ),
        "mean_steps_milli": sum(steps) * 1000 // len(steps),
        "p95_steps": steps[min(len(steps) - 1, int(len(steps) * 0.95))],
        "cleanup_errors": sum(row.get("cleanup_error") is not None for row in ordered),
    }
    if benchmark is Protocol13Benchmark.ALF_WORLD:
        task_types = sorted(
            {str(row["task_type"]) for row in ordered if isinstance(row.get("task_type"), str)}
        )
        splits = sorted(
            {
                str(row["source_split"])
                for row in ordered
                if isinstance(row.get("source_split"), str)
            }
        )
        for task_type in task_types:
            diagnostics[f"task_type_{task_type}_count"] = sum(
                row.get("task_type") == task_type for row in ordered
            )
            diagnostics[f"task_type_{task_type}_success"] = sum(
                row.get("task_type") == task_type and row.get("success") is True for row in ordered
            )
        for split in splits:
            diagnostics[f"split_{split}_count"] = sum(
                row.get("source_split") == split for row in ordered
            )
            diagnostics[f"split_{split}_success"] = sum(
                row.get("source_split") == split and row.get("success") is True for row in ordered
            )
    return (
        counts,
        metrics,
        diagnostics,
    )


def _health(
    artifact_dir: Path,
    task_ids: tuple[str, ...],
    grader_profile: str,
) -> tuple[OutcomeCountsV4, tuple[MetricObservationV3, ...], dict[str, int | str]]:
    candidate_dir = artifact_dir / "candidate"
    start = _json(candidate_dir / "execution-start.json")
    complete = _json(candidate_dir / "execution-complete.json")
    if not isinstance(start, dict) or not isinstance(start.get("panel_ids"), list):
        raise ValueError("HealthBench attempt marker lacks panel IDs")
    marker_ids = tuple(str(item) for item in cast(list[object], start["panel_ids"]))
    if marker_ids != task_ids:
        raise ValueError("HealthBench marker order differs from runtime receipt")
    if not isinstance(complete, dict) or complete.get("status") != "complete":
        raise ValueError("HealthBench candidate phase is incomplete")
    candidate_rows = _jsonl(candidate_dir / "candidates.jsonl")
    grader_dir = artifact_dir / "grades" / grader_profile
    grader_start = _json(grader_dir / "execution-start.json")
    grader_complete = _json(grader_dir / "execution-complete.json")
    if not isinstance(grader_start, dict) or grader_start.get("grader_profile") != grader_profile:
        raise ValueError("HealthBench grader profile marker differs")
    grader_contract = grader_start.get("grader_contract")
    if (
        not isinstance(grader_contract, dict)
        or grader_contract.get("profile_id") != grader_profile
        or grader_contract.get("backend") != "qwen-sglang"
        or grader_contract.get("model") != "qwen35-direct-base"
        or grader_contract.get("rubric_call_mode") != "simple-evals-per-rubric"
    ):
        raise ValueError("HealthBench grader contract differs")
    if not isinstance(grader_complete, dict) or grader_complete.get("status") != "complete":
        raise ValueError("HealthBench grader phase is incomplete")
    score_rows = _jsonl(grader_dir / "scores.jsonl")
    candidate_by_id = _unique_rows_by_id(candidate_rows, key="prompt_id")
    score_by_id = _unique_rows_by_id(score_rows, key="prompt_id")
    candidate_ids = tuple(candidate_by_id)
    require_exact_private_panel(
        expected_task_ids=task_ids,
        outcomes=tuple(PrivatePanelOutcome(item, "candidate", {}, None) for item in candidate_ids),
    )
    joined = require_exact_private_panel(
        expected_task_ids=task_ids,
        outcomes=tuple(PrivatePanelOutcome(item, "scored", {}, None) for item in score_by_id),
    )
    empty_ids = {
        prompt_id
        for prompt_id, row in candidate_by_id.items()
        if not str(row.get("response_text", "")).strip()
    }
    scores = tuple(
        Decimal(0)
        if item.task_id in empty_ids
        else _decimal(
            cast(dict[str, object], score_by_id[item.task_id]["metrics"])["overall_score"],
            "overall score",
        )
        for item in joined
    )
    counts = OutcomeCountsV4(
        definitive_scored=len(scores) - len(empty_ids),
        candidate_invalid=len(empty_ids),
        generation_infrastructure=0,
        environment_infrastructure=0,
        scorer_infrastructure=0,
    )
    metrics = (
        _metric(
            "native_rubric_mean",
            sum(scores, Decimal(0)),
            len(task_ids),
            formula="clip(mean(simple_evals_overall_score),0,1)*100",
            aggregation="official-aggregate",
            projection=MetricProjection.CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT,
        ),
    )
    return (
        counts,
        metrics,
        {
            "manifest_join": "exact",
            "grader_profile": grader_profile,
            "grader_score_count": len(scores),
            "empty_candidate_count": len(empty_ids),
            "grader_backend": str(grader_contract["backend"]),
            "grader_model": str(grader_contract["model"]),
            "rubric_call_mode": str(grader_contract["rubric_call_mode"]),
        },
    )


def _mbpp(
    artifact_dir: Path, task_ids: tuple[str, ...]
) -> tuple[OutcomeCountsV4, tuple[MetricObservationV3, ...], dict[str, int | str]]:
    rows = _jsonl(artifact_dir / "verdicts.jsonl")
    joined = require_exact_private_panel(
        expected_task_ids=task_ids,
        outcomes=tuple(
            PrivatePanelOutcome(
                _required_text(row, "task_id"),
                _required_text(row, "outcome"),
                {},
                None,
            )
            for row in rows
        ),
    )
    kinds = tuple(item.outcome for item in joined)
    success = sum(item == "scored-success" for item in kinds)
    failure = sum(item == "scored-failure" for item in kinds)
    counts = OutcomeCountsV4(
        definitive_scored=success + failure,
        candidate_invalid=sum(item == "candidate-invalid" for item in kinds),
        generation_infrastructure=sum(item == "generation-infrastructure" for item in kinds),
        environment_infrastructure=0,
        scorer_infrastructure=sum(item == "scorer-infrastructure" for item in kinds),
        binary_success=success,
        binary_failure=failure,
    )
    metrics = (
        _metric(
            "pass_at_1",
            Decimal(counts.scored_success),
            len(task_ids),
            formula="count(evalplus_base_and_plus_pass) / planned_count * 100",
            aggregation="mean-over-records",
        ),
    )
    return counts, metrics, {"manifest_join": "exact"}


def _value(receipt: Protocol13RunReceipt) -> dict[str, object]:
    return {
        "format": "skillev-protocol13-receipt@2",
        "benchmark": receipt.benchmark.value,
        "condition_id": receipt.execution.condition_id,
        "planned_count": receipt.planned_count,
        "outcomes": {
            "definitive_scored": receipt.outcomes.definitive_scored,
            "candidate_invalid": receipt.outcomes.candidate_invalid,
            "generation_infrastructure": receipt.outcomes.generation_infrastructure,
            "environment_infrastructure": receipt.outcomes.environment_infrastructure,
            "scorer_infrastructure": receipt.outcomes.scorer_infrastructure,
            "binary_success": receipt.outcomes.binary_success,
            "binary_failure": receipt.outcomes.binary_failure,
            "detail": receipt.outcomes.detail,
        },
        "metrics": [
            {
                "metric_id": item.metric_id,
                "unit": item.unit,
                "numerator": str(item.numerator),
                "denominator": item.denominator,
                "denominator_id": item.denominator_id,
                "projection": item.projection.value,
                "observed_percent": str(item.observed_percent),
                "formula": item.formula,
                "aggregation": item.aggregation,
            }
            for item in receipt.metrics
        ],
        "provenance": {
            name: getattr(receipt.provenance, name)
            for name in (
                "attempt_id",
                "generation_attempt_id",
                "scoring_attempt_id",
                "protocol_version",
                "generation_code_revision",
                "scoring_code_revision",
                "renderer_code_revision",
                "evaluator_version",
                "started_at",
                "completed_at",
            )
        },
        "runtime_execution_attempt_id": receipt.runtime_execution_attempt_id,
        "runtime_contract_matched": receipt.runtime_contract_matched,
        "final_panel_used_for_selection": receipt.final_panel_used_for_selection,
        "diagnostics": receipt.diagnostics,
    }


def main() -> None:
    args = _arguments()
    protocol = load_protocol_v13(args.protocol, args.sources)
    executions = load_execution_contracts_v3(args.conditions, protocol=protocol)
    benchmark = Protocol13Benchmark(args.benchmark)
    execution = executions[benchmark]
    profile = load_protocol13_runner_profiles(args.runner_config).require(
        execution.decoding_profile
    )
    runtime = load_execution_receipt(args.execution_receipt)
    if benchmark is Protocol13Benchmark.HEALTHBENCH:
        if args.grader_profile is None:
            raise ValueError("HealthBench requires --grader-profile")
        if args.grader_profile != execution.grader_profile:
            raise ValueError("HealthBench grader profile differs from execution")
        task_ids = runtime.planned_task_ids
        counts, metrics, diagnostics = _health(args.artifact_dir, task_ids, args.grader_profile)
    else:
        if args.grader_profile is not None:
            raise ValueError("--grader-profile is HealthBench-only")
        if args.manifest is None:
            raise ValueError("this benchmark requires --manifest")
        panel = load_protocol13_panel_manifest(args.manifest, execution=execution)
        task_ids = panel.task_ids
        if benchmark in {
            Protocol13Benchmark.HOTPOT_QA,
            Protocol13Benchmark.TRIVIA_QA,
            Protocol13Benchmark.AIME_2026,
            Protocol13Benchmark.HUMAN_EVAL,
        }:
            counts, metrics, diagnostics = _static(args.artifact_dir, benchmark, task_ids)
        elif benchmark in {Protocol13Benchmark.WEB_SHOP, Protocol13Benchmark.ALF_WORLD}:
            counts, metrics, diagnostics = _interactive(args.artifact_dir, benchmark, task_ids)
        else:
            counts, metrics, diagnostics = _mbpp(args.artifact_dir, task_ids)
    runtime.validate_against(
        execution=execution,
        expected_task_ids=task_ids,
        expected_generation_profile=profile,
        expected_environment_manifest=(
            load_protocol13_environment_manifest_identity(args.manifest, execution=execution)
            if benchmark in {Protocol13Benchmark.WEB_SHOP, Protocol13Benchmark.ALF_WORLD}
            and args.manifest is not None
            else None
        ),
    )
    if runtime.completed_at is None:
        raise ValueError("complete runtime receipt lacks completion time")
    provenance = RunProvenanceV3(
        attempt_id=runtime.attempt_id,
        generation_attempt_id=f"{runtime.attempt_id}:generation",
        scoring_attempt_id=f"{runtime.attempt_id}:scoring",
        protocol_version=execution.protocol_version,
        generation_code_revision=runtime.generation_code_revision,
        scoring_code_revision=runtime.scoring_code_revision,
        renderer_code_revision=args.renderer_code_revision,
        evaluator_version=runtime.evaluator_version,
        started_at=runtime.started_at,
        completed_at=runtime.completed_at,
    )
    public_diagnostics: dict[str, Decimal | int | str] = dict(diagnostics)
    receipt = Protocol13RunReceipt(
        benchmark=benchmark,
        execution=execution,
        planned_count=execution.expected_count,
        outcomes=counts,
        metrics=metrics,
        provenance=provenance,
        diagnostics=public_diagnostics,
        runtime_execution_attempt_id=runtime.attempt_id,
        runtime_contract_matched=True,
        final_panel_used_for_selection=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_value(receipt), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
