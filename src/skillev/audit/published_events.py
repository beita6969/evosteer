"""Hash-covered operational event history for published-attempt audit."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from skillev.contracts import TrajectoryRecord, normalize_json
from skillev.rollout import StructuredJsonActionCodec
from skillev.runtime import (
    ActionKind,
    AttemptBuilderKind,
    AttemptSourceLogKind,
    BudgetVector,
    EventEnvelope,
    EventType,
    PublishedSuccessfulAttemptBundle,
)


def _budget_payload(
    value: object,
    *,
    vector_field: str,
) -> tuple[str, BudgetVector]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != {"reservation_id", vector_field}:
        raise ValueError("budget event has incompatible fields")
    reservation_id = normalized["reservation_id"]
    if type(reservation_id) is not str or not reservation_id:
        raise ValueError("budget event reservation_id must be non-empty text")
    return reservation_id, BudgetVector.from_value(normalized[vector_field])


def parse_budget_reserved(value: object) -> tuple[str, BudgetVector]:
    return _budget_payload(value, vector_field="maximum")


def parse_budget_settled(value: object) -> tuple[str, BudgetVector]:
    return _budget_payload(value, vector_field="actual")


@dataclass(frozen=True, slots=True)
class PublishedEventHistory:
    envelopes: tuple[EventEnvelope, ...]

    @classmethod
    def read_full_or_operational_log(
        cls,
        bundle: PublishedSuccessfulAttemptBundle,
    ) -> PublishedEventHistory:
        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        kind = (
            AttemptSourceLogKind.OPERATIONAL
            if exact.builder_kind is AttemptBuilderKind.NO_BAYESIAN
            else AttemptSourceLogKind.FULL_METHOD
        )
        descriptor = exact.source_log(kind)
        path = exact.source_log_path(descriptor)
        envelopes = tuple(
            EventEnvelope.from_value(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        sequences: dict[str, int] = {}
        for envelope in envelopes:
            if envelope.run_id != exact.run_id or envelope.attempt_id != exact.attempt_id:
                raise ValueError("published event identity differs from bundle")
            expected = sequences.get(envelope.producer_id, 0) + 1
            if envelope.producer_seq != expected:
                raise ValueError("published producer sequence is not contiguous")
            sequences[envelope.producer_id] = envelope.producer_seq
        return cls(envelopes)

    def budget_settlements(self) -> Mapping[str, BudgetVector]:
        reservations: dict[str, BudgetVector] = {}
        settlements: dict[str, BudgetVector] = {}
        for event in self.envelopes:
            if event.event_type is EventType.BUDGET_RESERVED:
                reservation_id, maximum = parse_budget_reserved(event.payload)
                if reservation_id in reservations:
                    raise ValueError("published log repeats budget reservation")
                reservations[reservation_id] = maximum
            elif event.event_type is EventType.BUDGET_SETTLED:
                reservation_id, actual = parse_budget_settled(event.payload)
                if reservation_id not in reservations:
                    raise ValueError("budget settlement has no reservation")
                if reservation_id in settlements:
                    raise ValueError("published log repeats budget settlement")
                if not actual.fits_within(reservations[reservation_id]):
                    raise ValueError("budget settlement exceeds reservation")
                settlements[reservation_id] = actual
        if set(reservations) != set(settlements):
            raise ValueError("published success has unsettled budget reservations")
        return settlements

    def require_exact_budget_reservations(self, expected: Iterable[str]) -> None:
        """Require the settled ledger to contain exactly the audited invocations.

        A published successful attempt has a closed set of model, tool, and
        authoring invocations.  This check catches a real failure mode where a
        source event was valid in isolation but the operational ledger also
        contained an unreferenced paid call.
        """

        expected_ids = tuple(expected)
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("audited budget reservation IDs repeat")
        settlements = self.budget_settlements()
        if set(settlements) != set(expected_ids):
            raise ValueError("published ledger differs from audited invocations")


def expected_rollout_reservation_ids(
    records: Iterable[TrajectoryRecord],
) -> tuple[str, ...]:
    """Reconstruct every generation and executable-action reservation.

    Action parsing deliberately reuses the rollout codec: malformed and
    completion actions have no tool invocation, while valid tool/skill actions
    have exactly one.  The trajectory record is the canonical source for both
    generated action text and the executed horizon.
    """

    codec = StructuredJsonActionCodec()
    reservation_ids: list[str] = []
    for record in records:
        for step in record.steps:
            reservation_ids.extend(
                (
                    f"{record.trajectory_id}:{step.index}:reasoning:model",
                    f"{record.trajectory_id}:{step.index}:action:model",
                )
            )
            parsed = codec.parse(step.action_text)
            if parsed.action is not None and parsed.action.kind is not ActionKind.COMPLETE:
                reservation_ids.append(f"{record.trajectory_id}:{step.index}:tool")
    if len(reservation_ids) != len(set(reservation_ids)):
        raise ValueError("published trajectories repeat a budget invocation")
    return tuple(reservation_ids)


__all__ = [
    "PublishedEventHistory",
    "expected_rollout_reservation_ids",
    "parse_budget_reserved",
    "parse_budget_settled",
]
