"""Durable author responses inside the existing optimizer-step transaction.

A rollback may redo an uncommitted training step, but must not ask the author
for a different draft under the same phase/proposal identity. Received drafts
are reused; a call with an unknown or failed response is not silently retried.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue, canonical_json
from skillev.policy import AuthoringTokenizerProtocol
from skillev.runtime import BudgetLedger, BudgetReservation, BudgetSettlement, BudgetVector

from .authoring import (
    AuthoringFailedError,
    AuthoringRequest,
    AuthoringResult,
    SkillAuthor,
    authoring_reservation_id,
    render_authoring_prompt,
    template_version_for,
)


class UnresolvedAuthoringCallError(AuthoringFailedError):
    """A previous response is unavailable; recovery needs an explicit decision."""


def require_no_saved_phase(directory: Path) -> None:
    if (directory / "decision.json").exists():
        raise UnresolvedAuthoringCallError(
            "recovery cannot skip the phase decision saved before interruption"
        )


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvolutionAuthoringJournal:
    """One phase decision owned by one durable training-step journal directory."""

    def __init__(self, directory: Path, *, decision: dict[str, JsonValue]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / "decision.json"
        if path.exists():
            if json.loads(path.read_text()) != decision:
                raise ValueError("recovered phase or proposal differs from its saved decision")
        else:
            _write(path, decision)

    def bind(self, author: SkillAuthor, ledger: BudgetLedger) -> SkillAuthor:
        return _DurableSkillAuthor(self.directory, author, ledger)


class _DurableSkillAuthor:
    def __init__(self, directory: Path, delegate: SkillAuthor, ledger: BudgetLedger) -> None:
        self._directory = directory
        self._delegate = delegate
        self._ledger = ledger
        self._position = 0

    @property
    def tokenizer(self) -> AuthoringTokenizerProtocol:
        return self._delegate.tokenizer

    def author(self, request: AuthoringRequest) -> AuthoringResult:
        path = self._directory / f"call-{self._position:04d}.json"
        self._position += 1
        request_value = {
            "seed": request.seed,
            "template": template_version_for(request),
            "prompt": render_authoring_prompt(request, tokenizer=self.tokenizer),
            "reservation_id": authoring_reservation_id(request),
        }
        transport_path = path.with_suffix(".transport.json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request"] != request_value:
                raise ValueError("recovered authoring request differs from its original evidence")
            if saved["state"] == "rejected":
                raise AuthoringFailedError(
                    "original author response was rejected; no replacement draft is permitted"
                )
            if saved["state"] not in {"received", "validated"} and not transport_path.exists():
                raise UnresolvedAuthoringCallError(
                    "previous authoring response is unknown or failed; "
                    "automatic re-authoring is forbidden"
                )
            if saved["state"] in {"received", "validated"}:
                result = AuthoringResult.from_value(saved["result"])
                self._restore_usage(saved["usage"])
                return result
        pending = {"request": request_value, "state": "pending"}
        _write(path, pending)
        from .authoring_transport import DurableAuthoringTransport
        from .external_sglang_authoring import ExternalSGLangSkillAuthor

        delegate = self._delegate
        if isinstance(delegate, ExternalSGLangSkillAuthor):
            delegate = replace(
                delegate,
                transport=DurableAuthoringTransport(delegate.transport, transport_path),
            )
        try:
            result = delegate.author(request)
        except AuthoringFailedError as error:
            received = (
                transport_path.exists()
                and json.loads(transport_path.read_text(encoding="utf-8"))["state"]
                == "response-received"
            )
            _write(
                path,
                {
                    **pending,
                    "state": "rejected" if received else "failed",
                    "failure_type": type(error).__name__,
                },
            )
            raise
        entries = tuple(
            entry
            for entry in self._ledger.entries
            if entry.reservation.reservation_id == authoring_reservation_id(request)
        )
        usage = []
        for entry in entries:
            if entry.settlement is None:
                raise UnresolvedAuthoringCallError("author returned without settling its usage")
            usage.append(
                {
                    "reservation": asdict(entry.reservation),
                    "settlement": asdict(entry.settlement),
                }
            )
        _write(path, {**pending, "state": "validated", "result": result.to_value(), "usage": usage})
        return result

    def _restore_usage(self, usage: list[dict[str, Any]]) -> None:
        """Charge original call evidence to a fresh recovery ledger, not a new call.

        The original budget events already exist. Do not publish them twice.
        A same-process lookup likewise must not add the original charge again.
        """
        existing = {entry.reservation.reservation_id: entry for entry in self._ledger.entries}
        for value in usage:
            raw = value["reservation"]
            reservation = BudgetReservation(
                reservation_id=raw["reservation_id"],
                run_id=self._ledger.run_id,
                attempt_id=self._ledger.attempt_id,
                invocation_id=raw["invocation_id"],
                maximum=BudgetVector.from_value(raw["maximum"]),
            )
            settlement = BudgetSettlement(
                reservation_id=value["settlement"]["reservation_id"],
                actual=BudgetVector.from_value(value["settlement"]["actual"]),
            )
            if reservation.reservation_id in existing:
                entry = existing[reservation.reservation_id]
                if entry.reservation != reservation or entry.settlement != settlement:
                    raise ValueError("recovered authoring usage differs from the current ledger")
            else:
                self._ledger.reserve(reservation)
                self._ledger.settle(settlement)
