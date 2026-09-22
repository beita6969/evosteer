"""Fail-closed admission of Protocol 13 receipts against matched targets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .receipts import Protocol13RunReceipt
from .targets import ReferenceTargetV3


class Protocol13MismatchError(ValueError):
    pass


class MetricComparisonMode(StrEnum):
    ABSOLUTE_PARITY = "absolute-parity"
    MINIMUM_GOAL = "minimum-goal"


@dataclass(frozen=True, slots=True)
class AdmittedMetricV3:
    metric_id: str
    observed_percent: Decimal
    reference_percent: Decimal
    gap_pp: Decimal
    maximum_gap_pp_exclusive: Decimal
    passed: bool
    comparison_mode: MetricComparisonMode


def admit_against_target_v3(
    receipt: Protocol13RunReceipt,
    target: ReferenceTargetV3,
) -> tuple[AdmittedMetricV3, ...]:
    if not target.can_enter_formal_gate:
        raise Protocol13MismatchError("target is not eligible for the formal gate")
    if receipt.execution != target.observed_execution:
        raise Protocol13MismatchError("receipt execution differs from target execution")
    if not receipt.is_formally_complete:
        raise Protocol13MismatchError("receipt is incomplete or contains infrastructure failures")
    observed = {metric.metric_id: metric for metric in receipt.metrics}
    required = {
        metric.metric_id: metric for metric in target.metrics if metric.required_for_formal_gate
    }
    if set(observed) != set(required):
        raise Protocol13MismatchError("receipt metric set differs from required target metrics")
    admitted: list[AdmittedMetricV3] = []
    for metric_id, contract in required.items():
        row = observed[metric_id]
        if (
            row.unit != contract.unit
            or row.denominator_id != contract.denominator_id
            or row.projection is not contract.projection
            or row.formula != contract.formula
            or row.aggregation != contract.aggregation
            or row.denominator != receipt.planned_count
        ):
            raise Protocol13MismatchError(f"metric contract differs for {metric_id}")
        gap = abs(row.observed_percent - contract.reference_percent)
        admitted.append(
            AdmittedMetricV3(
                metric_id=metric_id,
                observed_percent=row.observed_percent,
                reference_percent=contract.reference_percent,
                gap_pp=gap,
                maximum_gap_pp_exclusive=contract.maximum_gap_pp_exclusive,
                passed=gap < contract.maximum_gap_pp_exclusive,
                comparison_mode=MetricComparisonMode.ABSOLUTE_PARITY,
            )
        )
    return tuple(admitted)


__all__ = [
    "AdmittedMetricV3",
    "MetricComparisonMode",
    "Protocol13MismatchError",
    "admit_against_target_v3",
]
