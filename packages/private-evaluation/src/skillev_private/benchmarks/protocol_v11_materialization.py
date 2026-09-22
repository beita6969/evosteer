"""Coverage-only Protocol 11 materialization receipt."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11, BenchmarkV11, PopulationRoleV11

from .protocol_v11_sources import ProtocolV11PrivateRecord, validate_population_disjointness


@dataclass(frozen=True, slots=True)
class ProtocolV11MaterializationReceipt:
    counts: dict[BenchmarkV11, dict[PopulationRoleV11, int]]
    final_ids_frozen: bool

    @property
    def complete(self) -> bool:
        return self.final_ids_frozen and all(
            set(self.counts.get(benchmark, {})) == set(PopulationRoleV11)
            for benchmark in ACTIVE_BENCHMARKS_V11
        )


def materialization_receipt(
    records: tuple[ProtocolV11PrivateRecord, ...], *, final_ids_frozen: bool
) -> ProtocolV11MaterializationReceipt:
    validate_population_disjointness(records)
    counts: dict[BenchmarkV11, dict[PopulationRoleV11, int]] = {}
    for record in records:
        roles = counts.setdefault(record.benchmark, {})
        roles[record.role] = roles.get(record.role, 0) + 1
    return ProtocolV11MaterializationReceipt(counts, final_ids_frozen)


__all__ = ["ProtocolV11MaterializationReceipt", "materialization_receipt"]
