"""SQLite canonical evidence store with a rebuildable append-only JSONL view."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

from skillev.contracts import canonical_json

from .schema import EVIDENCE_SCHEMA_VERSION, EvidenceRecord, EvidenceSplit


class EvidenceStore:
    """Persist immutable evidence and expose only eligible posterior updates."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        event_log_path: str | Path,
        read_only: bool = False,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.event_log_path = Path(event_log_path).resolve()
        if self.database_path == self.event_log_path:
            raise ValueError("database and event log paths must differ")
        self.read_only = read_only
        self._lock = threading.RLock()
        if read_only:
            self._connection = sqlite3.connect(
                f"file:{self.database_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        else:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (EVIDENCE_SCHEMA_VERSION,),
                )
            elif row[0] != EVIDENCE_SCHEMA_VERSION:
                raise ValueError("evidence database schema version is incompatible")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    event_id TEXT PRIMARY KEY,
                    split TEXT NOT NULL,
                    verifier_passed INTEGER NOT NULL,
                    train_step INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )

    def append(self, record: EvidenceRecord) -> bool:
        if self.read_only:
            raise PermissionError("read-only evidence stores cannot append")
        if not isinstance(record, EvidenceRecord):
            raise TypeError("record must be EvidenceRecord")
        serialized = canonical_json(record.to_value())
        with self._lock:
            existing = self._connection.execute(
                "SELECT record_json FROM evidence WHERE event_id = ?",
                (record.event_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != serialized:
                    raise ValueError("event ID already exists with different evidence")
                return False
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO evidence(event_id, split, verifier_passed, train_step, record_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.event_id,
                        record.split.value,
                        int(record.verifier_passed),
                        record.train_step,
                        serialized,
                    ),
                )
            with self.event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
        return True

    def get(self, event_id: str) -> EvidenceRecord | None:
        row = self._connection.execute(
            "SELECT record_json FROM evidence WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return None if row is None else EvidenceRecord.from_value(json.loads(row[0]))

    def records(self) -> Iterator[EvidenceRecord]:
        rows = self._connection.execute(
            "SELECT record_json FROM evidence ORDER BY train_step, event_id"
        )
        for (serialized,) in rows:
            yield EvidenceRecord.from_value(json.loads(serialized))

    def posterior_evidence(self) -> Iterator[EvidenceRecord]:
        rows = self._connection.execute(
            """
            SELECT record_json FROM evidence
            WHERE split = ? AND verifier_passed = 1
            ORDER BY train_step, event_id
            """,
            (EvidenceSplit.TRAIN.value,),
        )
        for (serialized,) in rows:
            record = EvidenceRecord.from_value(json.loads(serialized))
            if not record.may_update_posterior:
                raise RuntimeError("posterior evidence query admitted an ineligible record")
            yield record

    def snapshot_to(self, destination: str | Path) -> Path:
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as destination_connection, self._lock:
            self._connection.backup(destination_connection)
        return target

    def rebuild_event_log(self, destination: str | Path) -> Path:
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            for record in self.records():
                handle.write(canonical_json(record.to_value()) + "\n")
        return target

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class SplitEvidenceStores:
    """Route training and evaluation records to physically separate stores."""

    def __init__(self, *, train: EvidenceStore, evaluation: EvidenceStore) -> None:
        if train.database_path == evaluation.database_path:
            raise ValueError("train and evaluation evidence databases must be physically separate")
        if train.event_log_path == evaluation.event_log_path:
            raise ValueError("train and evaluation event logs must be physically separate")
        if train.read_only:
            raise ValueError("training evidence store must be writable")
        self.train = train
        self.evaluation = evaluation

    def append(self, record: EvidenceRecord) -> bool:
        if record.split is EvidenceSplit.TRAIN:
            return self.train.append(record)
        return self.evaluation.append(record)

    def posterior_evidence(self) -> Iterator[EvidenceRecord]:
        return self.train.posterior_evidence()


__all__ = ["EvidenceStore", "SplitEvidenceStores"]
