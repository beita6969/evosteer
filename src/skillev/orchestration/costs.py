"""Method-neutral logical and physical resource-cost aggregation.

Costs remain as non-fungible :class:`BudgetVector` values.  This module does
not assign prices or collapse unlike resource dimensions into a scalar.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from skillev.runtime.contracts import BudgetVector


def sum_budget_vectors(vectors: Iterable[BudgetVector]) -> BudgetVector:
    """Add resource vectors dimension by dimension."""

    total = BudgetVector()
    for vector in vectors:
        total = total.add(vector)
    return total


@dataclass(frozen=True, slots=True)
class CostRecord:
    """Cost of one logical operation and the work physically executed for it.

    ``logical`` is the usage attributed once to the semantic operation.
    ``physical`` includes all work that was actually executed, such as retry
    attempts.  They are kept separate because neither is a substitute for the
    other.
    """

    logical: BudgetVector
    physical: BudgetVector

    @classmethod
    def from_physical_attempts(
        cls,
        *,
        logical: BudgetVector,
        physical_attempts: Iterable[BudgetVector],
    ) -> CostRecord:
        """Create one record while aggregating its physical attempts."""

        return cls(
            logical=logical,
            physical=sum_budget_vectors(physical_attempts),
        )


@dataclass(frozen=True, slots=True)
class CostSummary:
    """Dimension-preserving totals over a collection of cost records."""

    logical: BudgetVector
    physical: BudgetVector
    record_count: int

    def __post_init__(self) -> None:
        if type(self.record_count) is not int or self.record_count < 0:
            raise ValueError("Cost summary count must be a non-negative integer")


def aggregate_costs(records: Iterable[CostRecord]) -> CostSummary:
    """Aggregate logical and physical usage without method-specific weighting."""

    logical = BudgetVector()
    physical = BudgetVector()
    record_count = 0
    for record in records:
        logical = logical.add(record.logical)
        physical = physical.add(record.physical)
        record_count += 1
    return CostSummary(
        logical=logical,
        physical=physical,
        record_count=record_count,
    )
