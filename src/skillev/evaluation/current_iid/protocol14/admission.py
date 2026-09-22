"""Protocol 14 parity, capability-floor, and formal admission semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from .receipts import AttemptRole, Protocol14RunReceipt
from .targets import ReferenceTargetV5, TargetScope, TargetStatus


class Protocol14MismatchError(ValueError):
    pass


class ParityStatus(StrEnum):
    PASS = "pass"  # noqa: S105 -- scientific verdict, not a credential
    BELOW_REFERENCE = "below-reference"
    ABOVE_REFERENCE = "above-reference"
    INCOMPARABLE = "incomparable"


class CapabilityFloorStatus(StrEnum):
    PASS = "pass"  # noqa: S105 -- scientific verdict, not a credential
    FAIL = "fail"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True, slots=True)
class AdmittedMetricV4:
    metric_id: str
    observed_percent: Decimal
    reference_percent: Decimal
    signed_gap_pp: Decimal
    absolute_gap_pp: Decimal
    relative_error_percent: Decimal | None
    maximum_gap_pp_exclusive: Decimal
    parity_status: ParityStatus
    capability_floor_status: CapabilityFloorStatus

    @property
    def parity_passed(self) -> bool:
        return self.parity_status is ParityStatus.PASS


def admit_against_target_v4(
    receipt: Protocol14RunReceipt,
    target: ReferenceTargetV5,
) -> tuple[AdmittedMetricV4, ...]:
    if receipt.role is not AttemptRole.CANDIDATE:
        raise Protocol14MismatchError("formal admission requires a candidate receipt")
    if not target.can_enter_formal_gate:
        raise Protocol14MismatchError("target is not eligible for the formal gate")
    return _compare_against_matched_target(receipt, target, formal=True)


def compare_against_diagnostic_target_v4(
    receipt: Protocol14RunReceipt,
    target: ReferenceTargetV5,
) -> tuple[AdmittedMetricV4, ...]:
    """Compare a complete diagnostic receipt without granting formal admission."""

    if receipt.role is not AttemptRole.CANDIDATE:
        raise Protocol14MismatchError("diagnostic comparison requires a candidate receipt")
    if target.can_enter_formal_gate:
        raise Protocol14MismatchError("formal target must use formal admission")
    if (
        target.status is not TargetStatus.FROZEN_MATCHED_REFERENCE
        or target.scope is not TargetScope.STANDALONE_BENCHMARK
        or not target.metrics
        or any(metric.required_for_formal_gate for metric in target.metrics)
    ):
        raise Protocol14MismatchError("target is not eligible for diagnostic comparison")
    return _compare_against_matched_target(receipt, target, formal=False)


def _compare_against_matched_target(
    receipt: Protocol14RunReceipt,
    target: ReferenceTargetV5,
    *,
    formal: bool,
) -> tuple[AdmittedMetricV4, ...]:
    if receipt.execution != target.observed_execution:
        raise Protocol14MismatchError("receipt execution differs from target execution")
    if not receipt.is_formally_complete:
        raise Protocol14MismatchError("receipt is incomplete or has unresolved infrastructure")
    if target.frozen_at is None:
        raise Protocol14MismatchError("target freeze time is absent")
    try:
        frozen_at = datetime.fromisoformat(target.frozen_at.replace("Z", "+00:00"))
        started_at = datetime.fromisoformat(receipt.started_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Protocol14MismatchError("target or candidate time is invalid") from exc
    if frozen_at >= started_at:
        raise Protocol14MismatchError("target was not frozen before the candidate run")
    observed = {metric.metric_id: metric for metric in receipt.metrics}
    required = {
        metric.metric_id: metric
        for metric in target.metrics
        if metric.required_for_formal_gate is formal
    }
    if set(observed) != set(required):
        raise Protocol14MismatchError("receipt metric set differs from required target metrics")
    admitted: list[AdmittedMetricV4] = []
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
            raise Protocol14MismatchError(f"metric contract differs for {metric_id}")
        signed = row.observed_percent - contract.reference_percent
        absolute = abs(signed)
        threshold = contract.maximum_gap_pp_exclusive
        if absolute < threshold:
            parity = ParityStatus.PASS
        elif signed < 0:
            parity = ParityStatus.BELOW_REFERENCE
        else:
            parity = ParityStatus.ABOVE_REFERENCE
        floor = (
            CapabilityFloorStatus.PASS
            if row.observed_percent > contract.reference_percent - threshold
            else CapabilityFloorStatus.FAIL
        )
        relative = (
            None
            if contract.reference_percent == 0
            else absolute / contract.reference_percent * Decimal(100)
        )
        admitted.append(
            AdmittedMetricV4(
                metric_id=metric_id,
                observed_percent=row.observed_percent,
                reference_percent=contract.reference_percent,
                signed_gap_pp=signed,
                absolute_gap_pp=absolute,
                relative_error_percent=relative,
                maximum_gap_pp_exclusive=threshold,
                parity_status=parity,
                capability_floor_status=floor,
            )
        )
    return tuple(admitted)


__all__ = [
    "AdmittedMetricV4",
    "CapabilityFloorStatus",
    "ParityStatus",
    "Protocol14MismatchError",
    "admit_against_target_v4",
    "compare_against_diagnostic_target_v4",
]
