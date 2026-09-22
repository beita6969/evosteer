from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from skillev.runtime import (
    BudgetExceededError,
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetSettlementError,
    BudgetVector,
    DuplicateBudgetReservationError,
    DuplicateBudgetSettlementError,
    ReservationState,
)


def _ledger() -> BudgetLedger:
    return BudgetLedger(
        run_id="run-1",
        attempt_id="attempt-1",
        cap=BudgetVector(model_calls=2, output_tokens=20),
    )


def _reservation(identity: str, maximum: BudgetVector) -> BudgetReservation:
    return BudgetReservation(
        reservation_id=identity,
        run_id="run-1",
        attempt_id="attempt-1",
        invocation_id="trajectory-1",
        maximum=maximum,
    )


def test_reserve_then_settle_is_the_only_success_transition() -> None:
    ledger = _ledger()
    reservation = _reservation(
        "call-1",
        BudgetVector(model_calls=1, output_tokens=10),
    )

    reserved = ledger.reserve(reservation)
    assert reserved.state is ReservationState.RESERVED
    assert ledger.reserved == reservation.maximum

    settlement = BudgetSettlement(
        reservation_id="call-1",
        actual=BudgetVector(model_calls=1, output_tokens=7),
    )
    settled = ledger.settle(settlement)
    assert settled.state is ReservationState.SETTLED
    assert settled.settlement == settlement
    assert ledger.reserved == BudgetVector()
    assert ledger.settled == settlement.actual
    ledger.assert_fully_settled()


def test_duplicate_reservation_is_never_idempotent() -> None:
    ledger = _ledger()
    reservation = _reservation("call-1", BudgetVector(model_calls=1))
    ledger.reserve(reservation)
    with pytest.raises(DuplicateBudgetReservationError):
        ledger.reserve(reservation)


def test_duplicate_settlement_is_never_idempotent() -> None:
    ledger = _ledger()
    reservation = _reservation("call-1", BudgetVector(model_calls=1))
    settlement = BudgetSettlement("call-1", BudgetVector(model_calls=1))
    ledger.reserve(reservation)
    ledger.settle(settlement)
    with pytest.raises(DuplicateBudgetSettlementError):
        ledger.settle(settlement)


def test_reservation_over_cap_fails_before_call_start() -> None:
    ledger = _ledger()
    with pytest.raises(BudgetExceededError):
        ledger.reserve(
            _reservation(
                "too-large",
                BudgetVector(model_calls=3, output_tokens=1),
            )
        )


def test_exact_settlement_must_fit_its_reservation() -> None:
    ledger = _ledger()
    ledger.reserve(_reservation("small", BudgetVector(output_tokens=3)))
    with pytest.raises(BudgetSettlementError):
        ledger.settle(BudgetSettlement("small", BudgetVector(output_tokens=4)))


def test_success_boundary_rejects_unsettled_call_without_repairing_it() -> None:
    ledger = _ledger()
    ledger.reserve(_reservation("call-1", BudgetVector(model_calls=1)))
    with pytest.raises(RuntimeError):
        ledger.assert_fully_settled()
    assert ledger.entries[0].state is ReservationState.RESERVED


def test_concurrent_reservations_cannot_oversubscribe() -> None:
    ledger = BudgetLedger(
        run_id="run-1",
        attempt_id="attempt-1",
        cap=BudgetVector(model_calls=1),
    )

    def attempt(identity: str) -> bool:
        try:
            ledger.reserve(_reservation(identity, BudgetVector(model_calls=1)))
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(attempt, ("left", "right"))) == [False, True]


def test_removed_repair_api_is_absent() -> None:
    for name in (
        "scope",
        "release",
        "abort",
        "fail_attempt_with_unknown_usage",
        "reconcile_failed_attempt",
    ):
        assert not hasattr(BudgetLedger, name)


def test_budget_vector_scale_is_component_wise() -> None:
    maximum = BudgetVector(
        input_tokens=2,
        output_tokens=3,
        model_calls=4,
        agent_turns=5,
        tool_calls=6,
        wall_time_milliseconds=7,
    )
    assert maximum.scale(3) == BudgetVector(
        input_tokens=6,
        output_tokens=9,
        model_calls=12,
        agent_turns=15,
        tool_calls=18,
        wall_time_milliseconds=21,
    )
