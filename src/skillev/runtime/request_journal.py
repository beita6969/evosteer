"""Private exact HTTP results with explicit ambiguous-execution semantics.

A durable DISPATCHED row precedes the actual send. Only a complete received
response advances it to COMPLETED. On crash/timeout the unresolved operation is
not resent: without server-side idempotency its outcome is genuinely unknown.
This store is not an optimizer checkpoint and never settles a training budget.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json


class UnknownRequestOutcomeError(RuntimeError):
    """A previous dispatch may have executed; do not sample a replacement."""


class DurableRequestJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # SQLite derives WAL/SHM permissions from the database at open time.
        # Create privately BEFORE the first connection, not after schema writes.
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        for suffix in ("-wal", "-shm"):
            sidefile = path.with_name(path.name + suffix)
            if sidefile.exists():
                sidefile.chmod(0o600)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                identity TEXT PRIMARY KEY, endpoint TEXT NOT NULL,
                payload BLOB NOT NULL, state TEXT NOT NULL,
                status INTEGER, response BLOB
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS episode_routes (
                episode TEXT PRIMARY KEY, policy TEXT NOT NULL, endpoint TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS authorized_retries (
                identity TEXT PRIMARY KEY, authorization_id TEXT NOT NULL,
                reason TEXT NOT NULL, prior_endpoint TEXT NOT NULL,
                prior_payload BLOB NOT NULL, prior_state TEXT NOT NULL,
                prior_status INTEGER, prior_response BLOB, retry_state TEXT NOT NULL
            )""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def episode_route(self, episode: str, policy: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT policy,endpoint FROM episode_routes WHERE episode=?", (episode,)
            ).fetchone()
        if row is None:
            return None
        if row[0] != policy:
            raise ValueError("saved episode route belongs to another policy")
        return str(row[1])

    def save_episode_route(self, episode: str, policy: str, endpoint: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT policy,endpoint FROM episode_routes WHERE episode=?", (episode,)
            ).fetchone()
            if row is not None and row != (policy, endpoint):
                raise ValueError("cannot replace an episode's original serving route")
            db.execute(
                "INSERT OR IGNORE INTO episode_routes VALUES(?,?,?)", (episode, policy, endpoint)
            )

    def require_resolved_prefix(self, prefix: tuple[str, str]) -> None:
        """Before restarting a concurrent logical operation, check ALL its calls.

        The caller invokes this before admitting new work, not while its own
        requests are active. A case cannot evade an unknown rubric response by
        issuing a different parse-repair prompt after restart.
        """
        if len(prefix) != 2 or any(not isinstance(v, str) or not v for v in prefix):
            raise ValueError("request scope needs non-empty coordinates")
        with self._connect() as db:
            found = db.execute(
                "SELECT 1 FROM requests WHERE state!='COMPLETED' "
                "AND json_extract(identity,'$[0]')=? "
                "AND json_extract(identity,'$[1]')=? LIMIT 1",
                prefix,
            ).fetchone()
        if found is not None:
            raise UnknownRequestOutcomeError("logical operation has an unresolved prior dispatch")

    def authorize_aborted_retry(
        self, *, identity: tuple[str, ...], authorization_id: str, reason: str
    ) -> None:
        """Operator-only, single-use permission after a confirmed request abort.

        This is not automatic timeout recovery. The operator must confirm the
        old sender has stopped and obtain permission to regenerate, not claim
        the missing response was recovered. Preserve the original dispatch.
        """
        if not identity or any(not isinstance(v, str) or not v for v in identity):
            raise ValueError("retry requires the exact original request identity")
        if not authorization_id.strip() or not reason.strip():
            raise ValueError("retry requires explicit authorization and abort evidence")
        key = canonical_json(list(identity))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT authorization_id,reason FROM authorized_retries WHERE identity=?", (key,)
            ).fetchone()
            if prior is not None:
                if prior != (authorization_id, reason):
                    raise ValueError("cannot replace or renew an existing retry authorization")
                return
            row = db.execute(
                "SELECT endpoint,payload,state,status,response FROM requests WHERE identity=?",
                (key,),
            ).fetchone()
            if row is None or row[2] != "DISPATCHED":
                raise ValueError("only an unresolved dispatched operation can be authorized")
            db.execute(
                "INSERT INTO authorized_retries VALUES(?,?,?,?,?,?,?,?,?)",
                (key, authorization_id, reason, *row, "READY"),
            )

    def request(
        self,
        *,
        identity: tuple[str, ...],
        endpoint: str,
        payload: dict[str, JsonValue],
        send: Callable[[], tuple[int, JsonValue]],
    ) -> tuple[int, JsonValue]:
        if not identity or any(not isinstance(v, str) or not v for v in identity):
            raise ValueError("persistent requests need non-empty execution coordinates")
        key = canonical_json(list(identity))
        encoded = canonical_json(payload).encode()
        # Transaction only covers admission; never hold a database lock across
        # an HTTP request. A second caller observes DISPATCHED rather than sends.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT endpoint,payload,state,status,response FROM requests WHERE identity=?",
                (key,),
            ).fetchone()
            if row is not None:
                old_endpoint, old_payload, state, status, response = row
                if old_endpoint != endpoint or zlib.decompress(old_payload) != encoded:
                    raise ValueError("saved request differs from its exact input/route")
                if state == "COMPLETED":
                    restored = json.loads(zlib.decompress(response))
                    if isinstance(restored, dict):
                        restored["skillev_restored_response"] = True
                    return status, restored
                permit = db.execute(
                    "UPDATE authorized_retries SET retry_state='DISPATCHED' "
                    "WHERE identity=? AND retry_state='READY'",
                    (key,),
                )
                if permit.rowcount != 1:
                    raise UnknownRequestOutcomeError("previous dispatch has no durable response")
            else:
                db.execute(
                    "INSERT INTO requests(identity,endpoint,payload,state) VALUES(?,?,?,?)",
                    (key, endpoint, zlib.compress(encoded), "DISPATCHED"),
                )
        status, response = send()
        saved = zlib.compress(canonical_json(response).encode())
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET state='COMPLETED',status=?,response=? WHERE identity=?",
                (status, saved, key),
            )
            db.execute(
                "UPDATE authorized_retries SET retry_state='COMPLETED' "
                "WHERE identity=? AND retry_state='DISPATCHED'",
                (key,),
            )
        return status, response
