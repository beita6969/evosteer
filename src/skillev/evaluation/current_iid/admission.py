"""Fail-closed Protocol 12 receipt/target admission."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .conditions import ExecutionContract
from .receipts import Protocol12RunReceipt
from .targets import ReferenceTargetContract


class ProtocolMismatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AdmittedMetric:
    metric_id: str
    observed_percent: Decimal
    reference_percent: Decimal
    gap_pp: Decimal
    passed: bool


def validate_receipt_against_contract(
    receipt: Protocol12RunReceipt,
    execution: ExecutionContract,
) -> None:
    if receipt.execution != execution:
        raise ProtocolMismatchError("receipt execution contract differs from protocol")
    if not receipt.is_formally_complete:
        raise ProtocolMismatchError("formal receipt is incomplete")


def admit_against_target(
    receipt: Protocol12RunReceipt,
    target: ReferenceTargetContract,
) -> tuple[AdmittedMetric, ...]:
    if not target.can_enter_formal_gate:
        raise ProtocolMismatchError("target is not formal")
    assert target.execution is not None
    validate_receipt_against_contract(receipt, target.execution)
    observed = {item.metric_id: item for item in receipt.metrics}
    expected = {item.metric_id: item for item in target.metrics if item.required_for_formal_gate}
    if set(observed) != set(expected):
        raise ProtocolMismatchError("receipt metric set differs from target")
    admitted: list[AdmittedMetric] = []
    for metric_id, contract in expected.items():
        row = observed[metric_id]
        if row.formula != contract.formula or row.aggregation != contract.aggregation.value:
            raise ProtocolMismatchError(f"metric contract differs for {metric_id}")
        gap = abs(row.observed_percent - contract.reference_percent)
        admitted.append(
            AdmittedMetric(
                metric_id,
                row.observed_percent,
                contract.reference_percent,
                gap,
                gap < contract.maximum_gap_pp_exclusive,
            )
        )
    return tuple(admitted)


__all__ = [
    "AdmittedMetric",
    "ProtocolMismatchError",
    "admit_against_target",
    "validate_receipt_against_contract",
]
