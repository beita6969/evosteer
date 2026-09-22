"""EvoSteer's combined input/output token cap on the existing exact ledger.

Every producer reserves and settles the same inherited ledger. The extra cap
counts outstanding maxima as well as actual settled input plus output tokens;
it never substitutes estimated usage for an exact settlement.
"""

from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

from skillev.runtime.budget_ledger import BudgetExceededError, BudgetLedger, LedgerEntry
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector


class EvoBudgetLedger(BudgetLedger):
    """Add one aggregate token limit without changing legacy budget semantics."""

    def __init__(
        self,
        *,
        run_id: str,
        attempt_id: str,
        cap: BudgetVector,
        total_token_cap: int,
    ) -> None:
        if type(total_token_cap) is not int or total_token_cap < 0:
            raise ValueError("total_token_cap must be a nonnegative integer")
        super().__init__(run_id=run_id, attempt_id=attempt_id, cap=cap)
        self._total_token_cap = total_token_cap
        # All mutations take this lock before the parent's lock. In particular,
        # the combined check and inherited reserve/restore are one transaction.
        self._combined_lock = Lock()

    @property
    def total_token_cap(self) -> int:
        return self._total_token_cap

    @property
    def total_tokens_available(self) -> int:
        with self._combined_lock:
            committed = self.reserved.add(self.settled)
            return self.total_token_cap - committed.input_tokens - committed.output_tokens

    def can_reserve(self, maximum: BudgetVector) -> bool:
        """Check the exact same caps that reserve() will enforce, without mutation."""
        if not isinstance(maximum, BudgetVector):
            raise TypeError("maximum must be BudgetVector")
        with self._combined_lock:
            proposed = self.reserved.add(self.settled).add(maximum)
            return (
                proposed.fits_within(self.cap)
                and proposed.input_tokens + proposed.output_tokens <= self.total_token_cap
            )

    def reserve(self, reservation: BudgetReservation) -> LedgerEntry:
        if not isinstance(reservation, BudgetReservation):
            raise TypeError("reservation must be BudgetReservation")
        with self._combined_lock:
            proposed = self.reserved.add(self.settled).add(reservation.maximum)
            if proposed.input_tokens + proposed.output_tokens > self.total_token_cap:
                raise BudgetExceededError("reservation exceeds the combined input/output token cap")
            return super().reserve(reservation)

    def settle(self, settlement: BudgetSettlement) -> LedgerEntry:
        with self._combined_lock:
            return super().settle(settlement)

    def restore_completed(self, entries: Iterable[LedgerEntry]) -> None:
        entries = tuple(entries)
        with self._combined_lock:
            actual = BudgetVector()
            for entry in entries:
                if not isinstance(entry, LedgerEntry) or entry.settlement is None:
                    raise ValueError("only completed ledger entries can be restored")
                actual = actual.add(entry.settlement.actual)
            proposed = self.reserved.add(self.settled).add(actual)
            if proposed.input_tokens + proposed.output_tokens > self.total_token_cap:
                raise BudgetExceededError(
                    "saved charges exceed the combined input/output token cap"
                )
            super().restore_completed(entries)


def can_reserve(ledger: BudgetLedger, maximum: BudgetVector) -> bool:
    """Support both legacy dimension caps and EvoSteer's aggregate cap."""
    if isinstance(ledger, EvoBudgetLedger):
        return ledger.can_reserve(maximum)
    return maximum.fits_within(ledger.available)


def total_token_budget(ledger: BudgetLedger) -> dict[str, int] | None:
    """Public aggregate budget evidence, absent for a legacy-only ledger."""
    if isinstance(ledger, EvoBudgetLedger):
        return {"cap": ledger.total_token_cap, "available": ledger.total_tokens_available}
    return None
