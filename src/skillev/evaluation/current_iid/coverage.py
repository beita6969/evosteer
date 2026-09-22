"""Coverage and final-status projection for current IID runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillev.evaluation.current_iid.receipts import CurrentIIDRunReceipt
from skillev.evaluation.current_iid.targets import BenchmarkTarget, TargetStatus


class CurrentIIDResultStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - result status, not a credential
    FAIL = "fail"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"
    TARGET_UNDEFINED = "target-undefined"
    PROTOCOL_MISMATCH = "protocol-mismatch"
    ASSISTED_DIAGNOSTIC = "assisted-diagnostic"


@dataclass(frozen=True, slots=True)
class MetricResult:
    metric_id: str
    observed_percent: float
    reference_percent: float | None
    gap_pp: float | None


def project_status(
    receipt: CurrentIIDRunReceipt,
    target: BenchmarkTarget,
    *,
    maximum_gap_pp_exclusive: float = 7.0,
) -> tuple[CurrentIIDResultStatus, tuple[MetricResult, ...]]:
    if receipt.infrastructure_failure_count:
        return CurrentIIDResultStatus.INCOMPLETE, ()
    if target.status in {TargetStatus.UNDEFINED, TargetStatus.DIAGNOSTIC}:
        status = (
            CurrentIIDResultStatus.TARGET_UNDEFINED
            if target.status is TargetStatus.UNDEFINED
            else CurrentIIDResultStatus.ASSISTED_DIAGNOSTIC
        )
        return status, tuple(
            MetricResult(name, value, None, None) for name, value in receipt.metrics.items()
        )
    if not target.permits_parity(receipt):
        return CurrentIIDResultStatus.PROTOCOL_MISMATCH, ()
    references = {item.metric_id: item.reference_percent for item in target.metrics}
    if set(receipt.metrics) != set(references):
        return CurrentIIDResultStatus.INCOMPLETE, ()
    metrics = tuple(
        MetricResult(name, value, references[name], abs(value - references[name]))
        for name, value in receipt.metrics.items()
    )
    status = (
        CurrentIIDResultStatus.PASS
        if all(
            item.gap_pp is not None and item.gap_pp < maximum_gap_pp_exclusive for item in metrics
        )
        else CurrentIIDResultStatus.FAIL
    )
    return status, metrics
