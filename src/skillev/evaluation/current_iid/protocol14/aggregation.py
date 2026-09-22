"""Answer-free Protocol 14 aggregation without cross-condition macro averages."""

from __future__ import annotations

from collections.abc import Mapping

from .admission import (
    AdmittedMetricV4,
    Protocol14MismatchError,
    admit_against_target_v4,
    compare_against_diagnostic_target_v4,
)
from .catalog import ACTIVE_PROTOCOL14_BENCHMARKS, Protocol14Benchmark
from .contracts import FormalEligibility
from .receipts import Protocol14RunReceipt
from .targets import ReferenceTargetV5


def aggregate_protocol_v14(
    receipts: Mapping[Protocol14Benchmark, Protocol14RunReceipt],
    targets: Mapping[Protocol14Benchmark, ReferenceTargetV5],
) -> dict[str, object]:
    expected = set(ACTIVE_PROTOCOL14_BENCHMARKS)
    if set(receipts) != expected or set(targets) != expected:
        raise ValueError("Protocol 14 aggregate requires exactly the authoritative eight")
    rows: dict[str, object] = {}
    formal_gate_pass = True
    diagnostic_complete = True
    formal_benchmarks: list[str] = []
    diagnostic_benchmarks: list[str] = []
    for benchmark in ACTIVE_PROTOCOL14_BENCHMARKS:
        receipt = receipts[benchmark]
        target = targets[benchmark]
        diagnostic_only = receipt.execution.formal_eligibility is FormalEligibility.DIAGNOSTIC_ONLY
        admitted: tuple[AdmittedMetricV4, ...]
        diagnostic_comparison: tuple[AdmittedMetricV4, ...]
        if diagnostic_only:
            diagnostic_benchmarks.append(benchmark.value)
            admitted = ()
            try:
                diagnostic_comparison = compare_against_diagnostic_target_v4(receipt, target)
            except Protocol14MismatchError as exc:
                diagnostic_comparison = ()
                diagnostic_status = "incomparable"
                diagnostic_complete = False
                nonformal_reasons = [
                    "execution is diagnostic-only and excluded from the formal gate",
                    str(exc),
                ]
            else:
                diagnostic_status = _comparison_status(diagnostic_comparison)
                nonformal_reasons = [
                    "execution is diagnostic-only and excluded from the formal gate"
                ]
            parity_status = "not-applicable"
            floor_status = "not-applicable"
            formal_status = "diagnostic-only"
            formal_target_admitted = False
        else:
            formal_benchmarks.append(benchmark.value)
            diagnostic_comparison = ()
            diagnostic_status = "not-applicable"
            try:
                admitted = admit_against_target_v4(receipt, target)
            except Protocol14MismatchError as exc:
                admitted = ()
                parity_status = "incomparable"
                floor_status = "incomparable"
                formal_status = "blocked"
                formal_target_admitted = False
                nonformal_reasons = [str(exc)]
                formal_gate_pass = False
            else:
                parity_status = _comparison_status(admitted)
                floor_status = (
                    "pass"
                    if all(metric.capability_floor_status.value == "pass" for metric in admitted)
                    else "fail"
                )
                formal_status = "pass" if parity_status == "pass" else "fail"
                formal_target_admitted = True
                nonformal_reasons = []
                formal_gate_pass &= formal_status == "pass"
        rows[benchmark.value] = {
            "condition_id": receipt.execution.condition_id,
            "formal_eligibility": receipt.execution.formal_eligibility.value,
            "parity_status": parity_status,
            "capability_floor_status": floor_status,
            "formal_status": formal_status,
            "diagnostic_status": diagnostic_status,
            "formal_target_admitted": formal_target_admitted,
            "nonformal_reasons": nonformal_reasons,
            "coverage": {
                "planned": receipt.planned_count,
                "scored_valid": receipt.outcomes.scored_valid,
                "model_output_invalid": receipt.outcomes.model_output_invalid,
                "generation_infrastructure": receipt.outcomes.generation_infrastructure,
                "environment_infrastructure": receipt.outcomes.environment_infrastructure,
                "scorer_infrastructure": receipt.outcomes.scorer_infrastructure,
                "binary_success": receipt.outcomes.binary_success,
                "binary_failure": receipt.outcomes.binary_failure,
            },
            "execution_receipt": {
                "attempt_id": receipt.attempt_id,
                "population_id": receipt.execution.population_id,
                "panel_manifest_id": receipt.execution.panel_manifest_id,
                "prompt_profile": receipt.execution.prompt_profile,
                "thinking_mode": receipt.execution.thinking_mode.value,
                "decoding_profile": receipt.execution.decoding_profile,
                "parser_profile": receipt.execution.parser_profile,
                "environment_profile": receipt.execution.environment_profile,
                "scorer_profile": receipt.execution.scorer_profile,
                "grader_profile": receipt.execution.grader_profile,
                "code_test_suite": receipt.execution.code_test_suite,
                "aggregation": receipt.execution.aggregation.to_mapping(),
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
                    "signed_gap_pp": str(metric.signed_gap_pp),
                    "absolute_gap_pp": str(metric.absolute_gap_pp),
                    "relative_error_percent": (
                        None
                        if metric.relative_error_percent is None
                        else str(metric.relative_error_percent)
                    ),
                    "maximum_gap_pp_exclusive": str(metric.maximum_gap_pp_exclusive),
                    "parity_status": metric.parity_status.value,
                    "capability_floor_status": metric.capability_floor_status.value,
                }
                for metric in admitted
            },
            "diagnostic_comparison": {
                metric.metric_id: _comparison_mapping(metric) for metric in diagnostic_comparison
            },
            "target": {
                "status": target.status.value,
                "scope": target.scope.value,
                "reference_receipt_id": target.reference_receipt_id,
                "frozen_at": target.frozen_at,
                "published_anchors": [
                    {
                        "label": anchor.label,
                        "scope": anchor.scope.value,
                        "metric_id": anchor.metric_id,
                        "value_percent": str(anchor.value_percent),
                        "usable_as_formal_target": anchor.usable_as_formal_target,
                    }
                    for anchor in target.published_anchors
                ],
            },
            "provenance": {
                "generation_code_revision": receipt.generation_code_revision,
                "scoring_code_revision": receipt.scoring_code_revision,
                "evaluator_version": receipt.evaluator_version,
                "started_at": receipt.started_at,
                "completed_at": receipt.completed_at,
            },
            "diagnostics": {name: str(value) for name, value in receipt.diagnostics.items()},
        }
    return {
        "format": "skillev-current-iid-result@5",
        "protocol_version": "skillev-benchmark-protocol@14",
        "gate_mode": "absolute_percentage_points",
        "threshold_exclusive": "7.0",
        "catalog": [item.value for item in ACTIVE_PROTOCOL14_BENCHMARKS],
        "planned_record_count": 926,
        "overall_status": ("pass" if formal_gate_pass and diagnostic_complete else "fail"),
        "formal_gate_status": "pass" if formal_gate_pass else "fail",
        "formal_gate_benchmarks": formal_benchmarks,
        "diagnostic_completion_status": ("complete" if diagnostic_complete else "incomplete"),
        "diagnostic_benchmarks": diagnostic_benchmarks,
        "benchmarks": rows,
    }


def _comparison_status(metrics: tuple[AdmittedMetricV4, ...]) -> str:
    metric_parity = {metric.parity_status.value for metric in metrics}
    if metric_parity == {"pass"}:
        return "pass"
    if "below-reference" in metric_parity and "above-reference" in metric_parity:
        return "incomparable"
    if "below-reference" in metric_parity:
        return "below-reference"
    return "above-reference"


def _comparison_mapping(metric: AdmittedMetricV4) -> dict[str, object]:
    return {
        "reference_percent": str(metric.reference_percent),
        "signed_gap_pp": str(metric.signed_gap_pp),
        "absolute_gap_pp": str(metric.absolute_gap_pp),
        "relative_error_percent": (
            None if metric.relative_error_percent is None else str(metric.relative_error_percent)
        ),
        "maximum_gap_pp_exclusive": str(metric.maximum_gap_pp_exclusive),
        "numeric_status": metric.parity_status.value,
    }


__all__ = ["aggregate_protocol_v14"]
