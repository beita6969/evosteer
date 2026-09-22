"""Answer-free aggregate construction for Protocol 13."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from .admission import (
    AdmittedMetricV3,
    MetricComparisonMode,
    Protocol13MismatchError,
    admit_against_target_v3,
)
from .catalog import ACTIVE_PROTOCOL13_BENCHMARKS, Protocol13Benchmark
from .receipts import Protocol13RunReceipt
from .targets import ReferenceTargetV3, TargetStatus


def aggregate_protocol_v13(
    receipts: Mapping[Protocol13Benchmark, Protocol13RunReceipt],
    targets: Mapping[Protocol13Benchmark, ReferenceTargetV3],
) -> dict[str, object]:
    expected = set(ACTIVE_PROTOCOL13_BENCHMARKS)
    if set(receipts) != expected or set(targets) != expected:
        raise ValueError("Protocol 13 aggregate requires exactly the authoritative eight")
    rows: dict[str, object] = {}
    overall_pass = True
    for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS:
        receipt = receipts[benchmark]
        target = targets[benchmark]
        formal_admitted = False
        try:
            admitted = admit_against_target_v3(receipt, target)
        except Protocol13MismatchError as exc:
            reason: str | None = str(exc)
            admitted = compare_against_numeric_target_v3(receipt, target)
            numeric_status = (
                "not-evaluated"
                if not admitted
                else "pass"
                if all(metric.passed for metric in admitted)
                else "fail"
            )
            scientific_status = {
                TargetStatus.UNDEFINED: "missing",
                TargetStatus.DIAGNOSTIC: "diagnostic",
                TargetStatus.OWNER_DEFINED_GOAL: "owner-defined-goal",
                TargetStatus.EXTERNAL_AGGREGATE_ANCHOR: "external-anchor",
                TargetStatus.PAPER_REPORTED_PROTOCOL_INCOMPLETE: "protocol-incomplete",
            }.get(target.status, "incomplete")
            formal_status = "blocked"
            overall_pass = False
        else:
            formal_admitted = True
            numeric_status = "pass" if all(metric.passed for metric in admitted) else "fail"
            scientific_status = "matched"
            formal_status = numeric_status
            reason = None
            overall_pass &= formal_status == "pass"
        rows[benchmark.value] = {
            "status": formal_status,
            "numeric_status": numeric_status,
            "scientific_status": scientific_status,
            "formal_status": formal_status,
            "target_evidence_status": target.evidence_status,
            "formal_target_admitted": formal_admitted,
            "nonformal_reason": reason,
            "nonformal_reasons": [] if reason is None else [reason],
            "condition_id": receipt.execution.condition_id,
            "reference_eligibility": receipt.execution.reference_eligibility.value,
            "coverage": {
                "planned": receipt.planned_count,
                "definitive_scored": receipt.outcomes.definitive_scored,
                "binary_success": receipt.outcomes.binary_success,
                "binary_failure": receipt.outcomes.binary_failure,
                "candidate_invalid": receipt.outcomes.candidate_invalid,
                "generation_infrastructure": receipt.outcomes.generation_infrastructure,
                "environment_infrastructure": receipt.outcomes.environment_infrastructure,
                "scorer_infrastructure": receipt.outcomes.scorer_infrastructure,
            },
            "runtime_contract_status": (
                "matched" if receipt.runtime_contract_matched else "mismatch"
            ),
            "execution_receipt": {
                "attempt_id": receipt.runtime_execution_attempt_id,
                "condition_id": receipt.execution.condition_id,
                "population_id": receipt.execution.population_id,
                "prompt_profile": receipt.execution.prompt_profile,
                "decoding_profile": receipt.execution.decoding_profile,
                "parser_profile": receipt.execution.parser_profile,
                "environment_profile": receipt.execution.environment_profile,
                "grader_profile": receipt.execution.grader_profile,
                "evaluator_profile": receipt.execution.evaluator_profile,
                "code_test_suite": receipt.execution.code_test_suite,
                "seed_aggregation": receipt.execution.seed_aggregation,
                "context_length": receipt.execution.context_length,
                "adapter_active": False,
            },
            "metrics": {
                metric.metric_id: {
                    "unit": metric.unit,
                    "numerator": str(metric.numerator),
                    "denominator": metric.denominator,
                    "denominator_id": metric.denominator_id,
                    "projection": metric.projection.value,
                    "observed_percent": str(metric.observed_percent),
                    "formula": metric.formula,
                    "aggregation": metric.aggregation,
                }
                for metric in receipt.metrics
            },
            "admission": {
                metric.metric_id: {
                    "reference_percent": str(metric.reference_percent),
                    "gap_pp": str(metric.gap_pp),
                    "maximum_gap_pp_exclusive": str(metric.maximum_gap_pp_exclusive),
                    "passed": metric.passed,
                    "comparison_mode": metric.comparison_mode.value,
                }
                for metric in admitted
            },
            "provenance": {
                "attempt_id": receipt.provenance.attempt_id,
                "generation_attempt_id": receipt.provenance.generation_attempt_id,
                "scoring_attempt_id": receipt.provenance.scoring_attempt_id,
                "generation_code_revision": receipt.provenance.generation_code_revision,
                "scoring_code_revision": receipt.provenance.scoring_code_revision,
                "renderer_code_revision": receipt.provenance.renderer_code_revision,
                "evaluator_version": receipt.provenance.evaluator_version,
            },
            "diagnostics": {name: str(value) for name, value in receipt.diagnostics.items()},
        }
    return {
        "format": "skillev-current-iid-result@4",
        "protocol_version": "skillev-benchmark-protocol@13",
        "catalog": [item.value for item in ACTIVE_PROTOCOL13_BENCHMARKS],
        "overall_status": "pass" if overall_pass else "fail",
        "benchmarks": rows,
    }


def compare_against_numeric_target_v3(
    receipt: Protocol13RunReceipt,
    target: ReferenceTargetV3,
) -> tuple[AdmittedMetricV3, ...]:
    """Compute display-only gaps without upgrading scientific eligibility."""

    observed = {metric.metric_id: metric for metric in receipt.metrics}
    compared: list[AdmittedMetricV3] = []
    for contract in target.metrics:
        row = observed.get(contract.metric_id)
        if row is None:
            continue
        if (
            row.unit != contract.unit
            or row.denominator_id != contract.denominator_id
            or row.projection is not contract.projection
            or row.formula != contract.formula
            or row.aggregation != contract.aggregation
            or row.denominator != receipt.planned_count
        ):
            continue
        if target.status is TargetStatus.OWNER_DEFINED_GOAL:
            comparison_mode = MetricComparisonMode.MINIMUM_GOAL
            gap = max(contract.reference_percent - row.observed_percent, Decimal(0))
            passed = row.observed_percent >= contract.reference_percent
        else:
            comparison_mode = MetricComparisonMode.ABSOLUTE_PARITY
            gap = abs(row.observed_percent - contract.reference_percent)
            passed = gap < contract.maximum_gap_pp_exclusive
        compared.append(
            AdmittedMetricV3(
                metric_id=contract.metric_id,
                observed_percent=row.observed_percent,
                reference_percent=contract.reference_percent,
                gap_pp=gap,
                maximum_gap_pp_exclusive=contract.maximum_gap_pp_exclusive,
                passed=passed,
                comparison_mode=comparison_mode,
            )
        )
    return tuple(compared)


__all__ = ["aggregate_protocol_v13", "compare_against_numeric_target_v3"]
