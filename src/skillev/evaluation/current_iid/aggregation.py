"""Answer-free machine aggregate construction."""

from __future__ import annotations

from collections.abc import Mapping

from skillev.evaluation.current_iid.admission import ProtocolMismatchError, admit_against_target
from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CurrentIIDBenchmark,
)
from skillev.evaluation.current_iid.coverage import project_status
from skillev.evaluation.current_iid.receipts import CurrentIIDRunReceipt, Protocol12RunReceipt
from skillev.evaluation.current_iid.targets import BenchmarkTarget, ReferenceTargetContract
from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11, BenchmarkV11


def aggregate_current_iid(
    receipts: Mapping[BenchmarkV11, CurrentIIDRunReceipt],
    targets: Mapping[BenchmarkV11, BenchmarkTarget],
    *,
    source_commit: str,
) -> dict[str, object]:
    if not source_commit.strip():
        raise ValueError("source commit is required")
    if set(receipts) != set(ACTIVE_BENCHMARKS_V11) or set(targets) != set(ACTIVE_BENCHMARKS_V11):
        raise ValueError("aggregate requires exactly the Protocol 12 current nine IID benchmarks")
    rows: dict[str, object] = {}
    for benchmark in ACTIVE_BENCHMARKS_V11:
        receipt = receipts[benchmark]
        status, metrics = project_status(receipt, targets[benchmark])
        rows[benchmark.value] = {
            "condition_id": receipt.condition_id,
            "attempt_id": receipt.attempt_id,
            "status": status.value,
            "coverage": {
                "planned": receipt.planned_count,
                "final_records": receipt.final_record_count,
                "definitive": receipt.definitive_verdict_count,
                "candidate_failures": receipt.candidate_failure_count,
                "infrastructure_failures": receipt.infrastructure_failure_count,
                "generation_infrastructure_failures": (receipt.generation_infrastructure_failures),
                "scorer_infrastructure_failures": receipt.scorer_infrastructure_failures,
                "environment_infrastructure_failures": (
                    receipt.environment_infrastructure_failures
                ),
            },
            "execution_contract": {
                "population_id": receipt.population_id,
                "prompt_profile": receipt.prompt_profile,
                "decoding_profile": receipt.decoding_profile,
                "actor_route": receipt.actor_route,
                "context_length": receipt.context_length,
                "adapter_active": receipt.adapter_active,
                "tool_surface": list(receipt.tool_surface),
                "grader_profile": receipt.grader_profile,
                "auxiliary_models": list(receipt.auxiliary_models),
                "scaffold_id": receipt.scaffold_id,
            },
            "metrics": {
                metric.metric_id: {
                    "observed": metric.observed_percent,
                    "reference": metric.reference_percent,
                    "gap_pp": metric.gap_pp,
                }
                for metric in metrics
            },
        }
    return {
        "format": "skillev-current-iid-result@1",
        "source_commit": source_commit,
        "benchmarks": rows,
    }


def aggregate_protocol_v12(
    *,
    receipts: Mapping[CurrentIIDBenchmark, Protocol12RunReceipt],
    targets: Mapping[CurrentIIDBenchmark, ReferenceTargetContract],
) -> dict[str, object]:
    expected = set(ACTIVE_CURRENT_IID_BENCHMARKS)
    if set(receipts) != expected or set(targets) != expected:
        raise ValueError("Protocol 12 aggregate requires exactly the authoritative nine")
    rows: dict[str, object] = {}
    overall_pass = True
    for benchmark in ACTIVE_CURRENT_IID_BENCHMARKS:
        receipt = receipts[benchmark]
        target = targets[benchmark]
        reason: str | None
        try:
            admitted = admit_against_target(receipt, target)
        except ProtocolMismatchError as exc:
            admitted = ()
            status = "diagnostic" if target.status.value == "diagnostic" else "non-formal"
            reason = str(exc)
            overall_pass = False
        else:
            status = "pass" if all(item.passed for item in admitted) else "fail"
            reason = None
            overall_pass &= status == "pass"
        rows[benchmark.value] = {
            "status": status,
            "nonformal_reason": reason,
            "condition_id": receipt.execution.condition_id,
            "reference_eligibility": receipt.execution.reference_eligibility.value,
            "coverage": {
                "planned": receipt.planned_count,
                "scored_success": receipt.outcomes.scored_success,
                "scored_failure": receipt.outcomes.scored_failure,
                "candidate_invalid": receipt.outcomes.candidate_invalid,
                "generation_infrastructure": receipt.outcomes.generation_infrastructure,
                "environment_infrastructure": receipt.outcomes.environment_infrastructure,
                "scorer_infrastructure": receipt.outcomes.scorer_infrastructure,
            },
            "metrics": {
                metric.metric_id: {
                    "numerator": str(metric.numerator),
                    "denominator": metric.denominator,
                    "observed_percent": str(metric.observed_percent),
                    "formula": metric.formula,
                    "aggregation": metric.aggregation,
                }
                for metric in receipt.metrics
            },
            "admission": {
                item.metric_id: {
                    "reference_percent": str(item.reference_percent),
                    "gap_pp": str(item.gap_pp),
                    "passed": item.passed,
                }
                for item in admitted
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
            "diagnostics": {key: str(value) for key, value in receipt.diagnostics.items()},
        }
    return {
        "format": "skillev-current-iid-result@2",
        "protocol_version": "skillev-benchmark-protocol@12",
        "catalog": [item.value for item in ACTIVE_CURRENT_IID_BENCHMARKS],
        "overall_status": "pass" if overall_pass else "fail",
        "benchmarks": rows,
    }


__all__ = ["aggregate_current_iid", "aggregate_protocol_v12"]
