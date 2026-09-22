"""Formal Protocol 13 gate derived exclusively from receipt admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .admission import Protocol13MismatchError, admit_against_target_v3
from .catalog import ACTIVE_PROTOCOL13_BENCHMARKS, Protocol13Benchmark
from .receipts import Protocol13RunReceipt
from .targets import ReferenceTargetV3


@dataclass(frozen=True, slots=True)
class Protocol13GateResult:
    catalog_complete: bool
    targets_complete: bool
    receipts_complete: bool
    runtime_contracts_matched: bool
    protocol_compatible: bool
    infrastructure_clear: bool
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
                self.protocol_compatible,
                self.infrastructure_clear,
                self.all_metrics_within_threshold,
                self.final_panel_not_used_for_selection,
            )
        )


def evaluate_protocol13_gate(
    targets: Mapping[Protocol13Benchmark, ReferenceTargetV3],
    receipts: Mapping[Protocol13Benchmark, Protocol13RunReceipt],
) -> Protocol13GateResult:
    expected = set(ACTIVE_PROTOCOL13_BENCHMARKS)
    catalog_complete = set(receipts) == expected == set(targets)
    targets_complete = catalog_complete and all(
        targets[item].can_enter_formal_gate for item in ACTIVE_PROTOCOL13_BENCHMARKS
    )
    receipts_complete = catalog_complete and all(
        receipts[item].is_formally_complete for item in ACTIVE_PROTOCOL13_BENCHMARKS
    )
    runtime_contracts_matched = catalog_complete and all(
        receipts[item].runtime_contract_matched for item in ACTIVE_PROTOCOL13_BENCHMARKS
    )
    final_panel_not_used_for_selection = catalog_complete and all(
        not receipts[item].final_panel_used_for_selection for item in ACTIVE_PROTOCOL13_BENCHMARKS
    )
    infrastructure_clear = catalog_complete and all(
        receipts[item].outcomes.infrastructure == 0 for item in ACTIVE_PROTOCOL13_BENCHMARKS
    )
    compatible = catalog_complete and targets_complete
    within = compatible
    if compatible:
        for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS:
            try:
                admitted = admit_against_target_v3(receipts[benchmark], targets[benchmark])
            except Protocol13MismatchError:
                compatible = False
                within = False
            else:
                within &= all(item.passed for item in admitted)
    return Protocol13GateResult(
        catalog_complete=catalog_complete,
        targets_complete=targets_complete,
        receipts_complete=receipts_complete,
        runtime_contracts_matched=runtime_contracts_matched,
        protocol_compatible=compatible,
        infrastructure_clear=infrastructure_clear,
        all_metrics_within_threshold=within,
        final_panel_not_used_for_selection=final_panel_not_used_for_selection,
    )


__all__ = ["Protocol13GateResult", "evaluate_protocol13_gate"]
