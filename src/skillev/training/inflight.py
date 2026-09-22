"""Private, uncommitted artifacts; never an optimizer or posterior checkpoint.

A previous real interruption discarded 28 completed rollouts. Persist each
successfully evaluated and cleaned-up artifact with its original call charges,
and admit it only into the same planned batch under the same execution condition.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json
from skillev.rollout import RolloutArtifact, RolloutTokenizerProtocol
from skillev.runtime import BudgetLedger, BudgetReservation, BudgetSettlement, BudgetVector
from skillev.runtime.budget_ledger import LedgerEntry

from .planning import PlannedRollout, TrainingBatchPlan


def durable_json(path: Path, value: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _entry_value(entry: LedgerEntry) -> dict[str, JsonValue]:
    reservation, settlement = entry.reservation, entry.settlement
    if settlement is None:
        raise ValueError("a completed artifact still has an unsettled call")
    return {
        "reservation_id": reservation.reservation_id,
        "run_id": reservation.run_id,
        "attempt_id": reservation.attempt_id,
        "invocation_id": reservation.invocation_id,
        "maximum": reservation.maximum.to_value(),
        "actual": settlement.actual.to_value(),
    }


class InFlightBatchStore:
    def __init__(self, root: Path, *, condition: dict[str, JsonValue]) -> None:
        self.root = root
        self.condition = condition

    def begin(self, plan: TrainingBatchPlan) -> InFlightBatch:
        directory = self.root / f"step-{plan.optimizer_step:08d}"
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        value: dict[str, JsonValue] = {
            "format": "uncommitted-batch@1",
            "condition": self.condition,
            "batch_id": plan.batch_id,
            "optimizer_step": plan.optimizer_step,
            "policy_snapshot_id": plan.policy_snapshot_id,
            "library_version": plan.library_version,
            "rollouts": [
                {
                    "position": item.position,
                    "trajectory_id": item.trajectory_id,
                    "task": item.task.to_value(),
                    "decoding": item.decoding.to_value(),
                }
                for item in plan.rollouts
            ],
        }
        path = directory / "batch.json"
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("in-flight evidence belongs to a different batch or condition")
        else:
            durable_json(path, value)
        return InFlightBatch(directory, plan)


class InFlightBatch:
    def __init__(self, directory: Path, plan: TrainingBatchPlan) -> None:
        self.directory, self.plan = directory, plan

    def _path(self, item: PlannedRollout) -> Path:
        return self.directory / f"trajectory-{item.position:06d}.json"

    def save(self, item: PlannedRollout, artifact: RolloutArtifact, ledger: BudgetLedger) -> None:
        self._require_artifact(item, artifact)
        entries: list[JsonValue] = [
            _entry_value(entry)
            for entry in ledger.entries
            if entry.reservation.invocation_id == item.trajectory_id
        ]
        value: dict[str, JsonValue] = {"artifact": artifact.to_value(), "budget_entries": entries}
        path = self._path(item)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("cannot replace the completed evidence of a planned rollout")
            return
        durable_json(path, value)

    def load(
        self, item: PlannedRollout, *, tokenizer: RolloutTokenizerProtocol, ledger: BudgetLedger
    ) -> RolloutArtifact | None:
        path = self._path(item)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        artifact = RolloutArtifact.from_value(value["artifact"], tokenizer=tokenizer)
        self._require_artifact(item, artifact)
        entries = []
        for raw in value["budget_entries"]:
            if raw["invocation_id"] != item.trajectory_id:
                raise ValueError("saved call charges belong to another trajectory")
            reservation = BudgetReservation(
                reservation_id=raw["reservation_id"],
                run_id=raw["run_id"],
                attempt_id=raw["attempt_id"],
                invocation_id=raw["invocation_id"],
                maximum=BudgetVector.from_value(raw["maximum"]),
            )
            settlement = BudgetSettlement(
                reservation.reservation_id, BudgetVector.from_value(raw["actual"])
            )
            entries.append(LedgerEntry.reserved(reservation).settled(settlement))
        # Atomic import: malformed or duplicate evidence must not partially debit
        # the attempt. This does not re-execute or re-publish the original calls.
        ledger.restore_completed(entries)
        return artifact

    def require_complete(
        self, artifacts: tuple[RolloutArtifact, ...], *, tokenizer: RolloutTokenizerProtocol
    ) -> dict[str, JsonValue]:
        """Read every authoritative artifact before an optimizer may consume B.

        This is a storage check, not a second rollout or budget settlement. A
        complete in-memory batch never substitutes for missing durable evidence.
        """
        if len(artifacts) != len(self.plan.rollouts):
            raise ValueError("durable population differs from the complete planned batch")
        rows: list[JsonValue] = []
        for item, expected in zip(self.plan.rollouts, artifacts, strict=True):
            path = self._path(item)
            raw = json.loads(path.read_text(encoding="utf-8"))
            stored = RolloutArtifact.from_value(raw["artifact"], tokenizer=tokenizer)
            self._require_artifact(item, stored)
            if stored.to_value() != expected.to_value():
                raise ValueError("durable artifact differs from the optimizer batch")
            rows.append(
                {
                    "position": item.position,
                    "trajectory_id": stored.record.trajectory_id,
                    "artifact_file": path.name,
                    "action_count": stored.record.horizon,
                    "terminal_result_count": 1,
                }
            )
        value: dict[str, JsonValue] = {
            "format": "readable-complete-batch-evidence@1",
            "batch_id": self.plan.batch_id,
            "optimizer_step": self.plan.optimizer_step,
            "policy_snapshot_id": self.plan.policy_snapshot_id,
            "library_version": self.plan.library_version,
            "trajectory_count": len(rows),
            "artifacts": rows,
            "state": "complete-artifacts-before-optimizer-not-a-commit",
        }
        path = self.directory / "complete-evidence.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("cannot replace the complete batch evidence identity")
        if not path.exists():
            durable_json(path, value)
        return value

    def _require_artifact(self, item: PlannedRollout, artifact: RolloutArtifact) -> None:
        manifest = artifact.manifest
        if (
            manifest.trajectory_id != item.trajectory_id
            or manifest.task_id != item.task.task_id
            or manifest.policy_snapshot.snapshot_id != self.plan.policy_snapshot_id
            or manifest.library_version != self.plan.library_version
            or manifest.decoding_snapshot_id != item.decoding.snapshot_id
        ):
            raise ValueError("saved artifact differs from its original rollout coordinates")
