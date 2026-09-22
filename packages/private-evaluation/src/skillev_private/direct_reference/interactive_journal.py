"""Durable, task-keyed journal for native interactive evaluations.

Only typed infrastructure outcomes may be superseded. Candidate failures,
horizon outcomes, native failures, partial rewards, and successes are final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class InteractiveJournalRecord:
    task_id: str
    request_attempt: int
    execution_contract_id: str
    outcome: dict[str, object]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.execution_contract_id.strip():
            raise ValueError("interactive journal identity is incomplete")
        if type(self.request_attempt) is not int or self.request_attempt <= 0:
            raise ValueError("interactive request attempt must be positive")


def can_resume(record: InteractiveJournalRecord) -> bool:
    return record.outcome.get("infrastructure_error") is not None


class InteractiveJournal:
    def __init__(self, path: Path, *, execution_contract_id: str) -> None:
        if not execution_contract_id.strip():
            raise ValueError("interactive execution contract ID is empty")
        self.path = path
        self.execution_contract_id = execution_contract_id
        self._records: dict[str, InteractiveJournalRecord] | None = None

    def load(self) -> dict[str, InteractiveJournalRecord]:
        if self._records is not None:
            return dict(self._records)
        records: dict[str, InteractiveJournalRecord] = {}
        if not self.path.exists():
            self._records = records
            return records
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value: object = json.loads(line)
                record = _record(value, line_number=line_number)
                if record.execution_contract_id != self.execution_contract_id:
                    raise ValueError("interactive journal belongs to another execution contract")
                previous = records.get(record.task_id)
                if previous is not None:
                    if not can_resume(previous):
                        raise ValueError("interactive journal retries a definitive outcome")
                    if record.request_attempt != previous.request_attempt + 1:
                        raise ValueError("interactive journal attempt sequence is not contiguous")
                elif record.request_attempt != 1:
                    raise ValueError("interactive journal first attempt must be one")
                records[record.task_id] = record
        self._records = records
        return dict(records)

    def append(self, record: InteractiveJournalRecord) -> None:
        if record.execution_contract_id != self.execution_contract_id:
            raise ValueError("interactive record belongs to another execution contract")
        current = self.load().get(record.task_id)
        expected_attempt = 1 if current is None else current.request_attempt + 1
        if current is not None and not can_resume(current):
            raise ValueError("cannot replace a definitive interactive outcome")
        if record.request_attempt != expected_attempt:
            raise ValueError("interactive request attempt is not the next attempt")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "execution_contract_id": record.execution_contract_id,
                        "outcome": record.outcome,
                        "request_attempt": record.request_attempt,
                        "task_id": record.task_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
        if self._records is None:
            self._records = {}
        self._records[record.task_id] = record


def _record(value: object, *, line_number: int) -> InteractiveJournalRecord:
    if not isinstance(value, dict) or set(value) != {
        "execution_contract_id",
        "outcome",
        "request_attempt",
        "task_id",
    }:
        raise ValueError(f"interactive journal line {line_number} has an incompatible shape")
    outcome = value["outcome"]
    if not isinstance(outcome, dict):
        raise ValueError(f"interactive journal line {line_number} outcome is not an object")
    return InteractiveJournalRecord(
        task_id=str(value["task_id"]),
        request_attempt=cast(int, value["request_attempt"]),
        execution_contract_id=str(value["execution_contract_id"]),
        outcome=cast(dict[str, Any], outcome),
    )


__all__ = ["InteractiveJournal", "InteractiveJournalRecord", "can_resume"]
