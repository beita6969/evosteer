"""Protocol 12 formal training gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .admission import ProtocolMismatchError, admit_against_target
from .catalog import ACTIVE_CURRENT_IID_BENCHMARKS, CurrentIIDBenchmark
from .receipts import Protocol12RunReceipt
from .targets import ReferenceTargetContract


@dataclass(frozen=True, slots=True)
class CurrentIIDGateResult:
    catalog_complete: bool
    targets_complete: bool
    receipts_complete: bool
    protocol_compatible: bool
    infrastructure_clear: bool
    all_metrics_within_7pp: bool

    @property
    def go(self) -> bool:
        return all(
            (
                self.catalog_complete,
                self.targets_complete,
                self.receipts_complete,
                self.protocol_compatible,
                self.infrastructure_clear,
                self.all_metrics_within_7pp,
            )
        )


def evaluate_current_iid_gate(
    targets: Mapping[CurrentIIDBenchmark, ReferenceTargetContract],
    receipts: Mapping[CurrentIIDBenchmark, Protocol12RunReceipt],
) -> CurrentIIDGateResult:
    expected = set(ACTIVE_CURRENT_IID_BENCHMARKS)
    catalog_complete = set(receipts) == expected == set(targets)
    targets_complete = catalog_complete and all(
        targets[item].can_enter_formal_gate for item in ACTIVE_CURRENT_IID_BENCHMARKS
    )
    receipts_complete = catalog_complete and all(
        receipts[item].is_formally_complete for item in ACTIVE_CURRENT_IID_BENCHMARKS
    )
    infrastructure_clear = catalog_complete and all(
        receipts[item].outcomes.infrastructure == 0 for item in ACTIVE_CURRENT_IID_BENCHMARKS
    )
    protocol_compatible = catalog_complete
    all_within = catalog_complete and targets_complete
    if catalog_complete and targets_complete:
        for benchmark in ACTIVE_CURRENT_IID_BENCHMARKS:
            try:
                admitted = admit_against_target(receipts[benchmark], targets[benchmark])
            except ProtocolMismatchError:
                protocol_compatible = False
                all_within = False
                continue
            all_within &= all(item.passed for item in admitted)
    else:
        protocol_compatible = False
        all_within = False
    return CurrentIIDGateResult(
        catalog_complete,
        targets_complete,
        receipts_complete,
        protocol_compatible,
        infrastructure_clear,
        all_within,
    )


__all__ = ["CurrentIIDGateResult", "evaluate_current_iid_gate"]
