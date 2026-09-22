"""Formal corrected Protocol 14 gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .admission import Protocol14MismatchError, admit_against_target_v4
from .catalog import ACTIVE_PROTOCOL14_BENCHMARKS, Protocol14Benchmark
from .receipts import Protocol14RunReceipt
from .targets import ReferenceTargetV5


@dataclass(frozen=True, slots=True)
class Protocol14GateResult:
    catalog_complete: bool
    targets_complete: bool
    receipts_complete: bool
    runtime_contracts_matched: bool
    infrastructure_clear: bool
    target_frozen_before_candidate: bool
    all_metrics_within_threshold: bool
    final_panel_not_used_for_selection: bool

    @property
    def go(self) -> bool:
        return all(
            (
                self.catalog_complete,
                self.targets_complete,
                self.receipts_complete,
                self.runtime_contracts_matched,
                self.infrastructure_clear,
                self.target_frozen_before_candidate,
                self.all_metrics_within_threshold,
                self.final_panel_not_used_for_selection,
            )
        )


def evaluate_protocol14_gate(
    targets: Mapping[Protocol14Benchmark, ReferenceTargetV5],
    receipts: Mapping[Protocol14Benchmark, Protocol14RunReceipt],
) -> Protocol14GateResult:
    expected = set(ACTIVE_PROTOCOL14_BENCHMARKS)
    catalog = set(receipts) == expected == set(targets)
    targets_complete = catalog and all(
        targets[item].can_enter_formal_gate for item in ACTIVE_PROTOCOL14_BENCHMARKS
    )
    receipts_complete = catalog and all(
        receipts[item].is_formally_complete for item in ACTIVE_PROTOCOL14_BENCHMARKS
    )
    runtime = catalog and all(
        receipts[item].runtime_contract_matched for item in ACTIVE_PROTOCOL14_BENCHMARKS
    )
    infrastructure = catalog and all(
        receipts[item].outcomes.infrastructure == 0 for item in ACTIVE_PROTOCOL14_BENCHMARKS
    )
    no_selection = catalog and all(
        not receipts[item].final_panel_used_for_selection for item in ACTIVE_PROTOCOL14_BENCHMARKS
    )
    frozen_before = catalog and targets_complete
    within = catalog and targets_complete and receipts_complete
    if within:
        for benchmark in ACTIVE_PROTOCOL14_BENCHMARKS:
            try:
                admitted = admit_against_target_v4(receipts[benchmark], targets[benchmark])
            except Protocol14MismatchError as exc:
                if "frozen before" in str(exc):
                    frozen_before = False
                within = False
            else:
                within &= all(metric.parity_passed for metric in admitted)
    else:
        frozen_before = False
    return Protocol14GateResult(
        catalog_complete=catalog,
        targets_complete=targets_complete,
        receipts_complete=receipts_complete,
        runtime_contracts_matched=runtime,
        infrastructure_clear=infrastructure,
        target_frozen_before_candidate=frozen_before,
        all_metrics_within_threshold=within,
        final_panel_not_used_for_selection=no_selection,
    )


__all__ = ["Protocol14GateResult", "evaluate_protocol14_gate"]
