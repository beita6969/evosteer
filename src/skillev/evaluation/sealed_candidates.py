"""Private durable candidate/side-effect journal, with insert-once final ownership.

SQLite transactions prevent two completions or resumed workers replacing the
evaluated policy's final answer. This is not an attestation or a hash protocol.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .model_output_provenance import CandidateStatus, StoredModelOutput


class EventRecorder(Protocol):
    def record(self, scope: tuple[str, str, str], stage: str, payload: object) -> None: ...


class EventOrigin(StrEnum):
    """The trusted process namespace that emitted a journal trace event."""

    ACTOR_DIAGNOSTIC = "actor-diagnostic"
    MODEL_TRANSPORT = "model-transport"
    ENVIRONMENT = "environment"
    SCORER = "scorer"
    LEGACY_UNVERIFIED = "legacy-unverified"


ACTOR_DIAGNOSTIC_STAGES = frozenset(
    {
        "input",
        "rendered-request",
        "model-response",
        "action-surface",
        "decision-transport",
        "control-attempt",
        "control-decoded",
        "control-resolved",
        "control-parse-failure",
        "terminal-parse-failure",
        "peer-budget-unavailable",
        "peer-delivery-failure",
        "agent-message",
        "agent-reply",
        "history-read",
        "public-transition",
        "public-episode-close",
        "candidate-budget-exhausted",
        "model-interface-repair",
        "call-accounting",
        "owner-final-submission",
        "skill-access",
        "skill-retrieval",
        "skill-tool",
        "skill-context",
    }
)


def authorize_actor_trace(stage: str) -> EventOrigin:
    """Fence the actor RPC to diagnostics that cannot impersonate native authority."""
    if stage not in ACTOR_DIAGNOSTIC_STAGES:
        raise ValueError("actor cannot write an authoritative runtime event")
    return EventOrigin.ACTOR_DIAGNOSTIC


@dataclass(frozen=True, slots=True)
class FinalCandidate:
    run_id: str
    arm_id: str
    episode_id: str
    policy_id: str
    final_message_id: str
    text: str
    parser_id: str
    prompt_tokens: int
    completion_tokens: int
    intervention_counts: dict[str, int] = field(default_factory=dict)
    submission: dict[str, str] | None = None
    attempt_id: str | None = None
    owner_call_id: str | None = None
    terminal_status: CandidateStatus = CandidateStatus.LEGACY_UNVERIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_status", CandidateStatus(self.terminal_status))
        if self.terminal_status is not CandidateStatus.LEGACY_UNVERIFIED and (
            not self.attempt_id or not self.owner_call_id
        ):
            raise ValueError("verified candidates require the generating attempt and owner call")
        if any(
            not value
            for value in (
                self.run_id,
                self.arm_id,
                self.episode_id,
                self.policy_id,
                self.final_message_id,
                self.parser_id,
            )
        ):
            raise ValueError("final candidate identity is incomplete")
        if min(self.prompt_tokens, self.completion_tokens) < 0:
            raise ValueError("candidate cost must be non-negative")
        if any(type(value) is not int or value < 0 for value in self.intervention_counts.values()):
            raise ValueError("observed intervention counts must be non-negative integers")
        if self.submission is not None:
            expected = {
                "owner_id",
                "message_id",
                "raw_response",
                "payload",
                "payload_type",
                "projection_id",
            }
            if set(self.submission) != expected or any(
                type(value) is not str or not value for value in self.submission.values()
            ):
                raise ValueError("owner final submission is structurally incomplete")
            if self.submission["message_id"] != self.final_message_id:
                raise ValueError("owner final submission message identity differs from candidate")


class CandidateReader:
    """Read-only scorer handle: no model, generation callback, or selection API."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        self._trace_has_origin = self._has_trace_origin()

    def _has_trace_origin(self) -> bool:
        return "origin" in {row[1] for row in self.connection.execute("PRAGMA table_info(trace)")}

    @property
    def trace_has_origin(self) -> bool:
        return self._trace_has_origin

    def get(self, run_id: str, arm_id: str, episode_id: str) -> FinalCandidate:
        row = self.connection.execute(
            "SELECT payload FROM candidates WHERE run_id=? AND arm_id=? AND episode_id=?",
            (run_id, arm_id, episode_id),
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return FinalCandidate(**json.loads(row[0]))

    def close(self) -> None:
        self.connection.close()

    def traces(
        self,
        scope: tuple[str, str, str],
        stage: str,
        *,
        origin: EventOrigin | None = None,
    ) -> tuple[object, ...]:
        if origin is not None and not isinstance(origin, EventOrigin):
            raise TypeError("trace origin must be an EventOrigin")
        if origin is not None and not self._trace_has_origin:
            rows = (
                self.connection.execute(
                    "SELECT payload FROM trace WHERE run_id=? AND arm_id=? AND episode_id=? "
                    "AND stage=? ORDER BY sequence",
                    (*scope, stage),
                ).fetchall()
                if origin is EventOrigin.LEGACY_UNVERIFIED
                else ()
            )
        else:
            query = (
                "SELECT payload FROM trace WHERE run_id=? AND arm_id=? AND episode_id=? AND stage=?"
            )
            parameters: tuple[object, ...] = (*scope, stage)
            if origin is not None:
                query += " AND origin=?"
                parameters += (origin.value,)
            rows = self.connection.execute(query + " ORDER BY sequence", parameters).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def stored_score(self, scope: tuple[str, str, str]) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT payload FROM scores WHERE run_id=? AND arm_id=? AND episode_id=?", scope
        ).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def model_outputs(self, scope: tuple[str, str, str]) -> tuple[StoredModelOutput, ...]:
        if not self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_outputs'"
        ).fetchone():
            return ()  # Old databases stay readable, without inventing source records.
        return tuple(
            StoredModelOutput.from_value(json.loads(payload))
            for (payload,) in self.connection.execute(
                "SELECT payload FROM model_outputs WHERE run_id=? AND arm_id=? AND episode_id=? "
                "ORDER BY sequence",
                scope,
            )
        )


class CandidateJournal(CandidateReader):
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                run_id TEXT, arm_id TEXT, episode_id TEXT, payload TEXT NOT NULL,
                PRIMARY KEY (run_id, arm_id, episode_id));
            CREATE TABLE IF NOT EXISTS executions (
                run_id TEXT, arm_id TEXT, episode_id TEXT, decision_id TEXT,
                action TEXT NOT NULL, state_revision INTEGER NOT NULL,
                status TEXT NOT NULL, observation TEXT,
                PRIMARY KEY (run_id, arm_id, episode_id, decision_id));
            CREATE TABLE IF NOT EXISTS trace (
                sequence INTEGER PRIMARY KEY, run_id TEXT, arm_id TEXT, episode_id TEXT,
                stage TEXT NOT NULL, payload TEXT NOT NULL,
                origin TEXT NOT NULL DEFAULT 'legacy-unverified');
            CREATE TABLE IF NOT EXISTS scores (
                run_id TEXT, arm_id TEXT, episode_id TEXT, payload TEXT NOT NULL,
                PRIMARY KEY (run_id, arm_id, episode_id));
            CREATE TABLE IF NOT EXISTS run_configuration (
                run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episode_attempts (
                run_id TEXT, arm_id TEXT, episode_id TEXT,
                attempt_id TEXT NOT NULL, policy_id TEXT NOT NULL,
                PRIMARY KEY (run_id, arm_id, episode_id));
            CREATE TABLE IF NOT EXISTS model_outputs (
                sequence INTEGER PRIMARY KEY,
                run_id TEXT, arm_id TEXT, episode_id TEXT,
                attempt_id TEXT NOT NULL, call_id TEXT NOT NULL, payload TEXT NOT NULL,
                UNIQUE (run_id, arm_id, episode_id, attempt_id, call_id));
        """)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(executions)")}
        if "acknowledged" not in columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE executions ADD COLUMN acknowledged INTEGER NOT NULL DEFAULT 0"
                )
                self.connection.execute(
                    "UPDATE executions SET acknowledged=1 "
                    "WHERE status!='unknown' OR observation IS NOT NULL"
                )
        trace_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(trace)")}
        if "origin" not in trace_columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE trace ADD COLUMN origin TEXT NOT NULL DEFAULT 'legacy-unverified'"
                )
        self._trace_has_origin = True
        # Semantic UNKNOWN and a missing transport acknowledgement are different.
        with self.connection:
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS trace_scope_stage_origin "
                "ON trace (run_id, arm_id, episode_id, stage, origin, sequence)"
            )
            self.connection.execute("DROP INDEX IF EXISTS one_pending_execution")
            self.connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS one_unacknowledged_execution "
                "ON executions (run_id, arm_id, episode_id) WHERE acknowledged=0"
            )

    def freeze_run(self, run_id: str, configuration: object) -> None:
        """Persist actual runtime controls so a resumed run cannot change its conditions."""
        value = json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        with self.connection:
            row = self.connection.execute(
                "SELECT payload FROM run_configuration WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is not None:
                if json.loads(row[0]) != json.loads(value):
                    raise ValueError("resumed run controls differ from the original execution")
            else:
                self.connection.execute(
                    "INSERT INTO run_configuration VALUES (?, ?)", (run_id, value)
                )

    def start_attempt(self, scope: tuple[str, str, str], *, policy_id: str) -> str:
        attempt_id = str(uuid4())
        with self.connection:
            self.connection.execute(
                "INSERT INTO episode_attempts VALUES (?, ?, ?, ?, ?)",
                (*scope, attempt_id, policy_id),
            )
        return attempt_id

    def record_model_output(self, output: StoredModelOutput) -> None:
        """Commit actual output and transport completion together before actor delivery."""
        expected = self.connection.execute(
            "SELECT attempt_id,policy_id FROM episode_attempts "
            "WHERE run_id=? AND arm_id=? AND episode_id=?",
            output.scope,
        ).fetchone()
        if expected != (output.attempt_id, output.policy_id):
            raise ValueError("model output belongs to a different generating attempt or policy")
        if output.result["policy_snapshot_id"] != output.policy_id:
            raise ValueError("served output has a different actual policy identity")
        if self.connection.execute(
            "SELECT 1 FROM candidates WHERE run_id=? AND arm_id=? AND episode_id=?", output.scope
        ).fetchone():
            raise ValueError("no generation can follow a committed final")
        with self.connection:
            self.connection.execute(
                "INSERT INTO model_outputs "
                "(run_id,arm_id,episode_id,attempt_id,call_id,payload) VALUES (?,?,?,?,?,?)",
                (*output.scope, output.attempt_id, output.call_id, json.dumps(asdict(output))),
            )
            self.connection.execute(
                "INSERT INTO trace (run_id,arm_id,episode_id,stage,payload,origin) "
                "VALUES (?,?,?,?,?,?)",
                (
                    *output.scope,
                    "model-transport-complete",
                    json.dumps(
                        {
                            "call_id": output.call_id,
                            "attempt_id": output.attempt_id,
                            "policy_id": output.policy_id,
                            "usage": output.result["usage"],
                            "participant": output.participant,
                            "finish_reason": output.result["finish_reason"],
                            "channel_status": output.channel_status.value,
                        }
                    ),
                    EventOrigin.MODEL_TRANSPORT.value,
                ),
            )

    def save_score(self, scope: tuple[str, str, str], payload: dict[str, object]) -> None:
        # This API has no generator callback; a submitted candidate cannot be replaced.
        self.get(*scope)
        if payload.get("status") == "infrastructure-failure":
            raise ValueError("infrastructure failure is not a definitive score")
        with self.connection:
            previous = self.stored_score(scope)
            if previous is not None:
                if previous != json.loads(json.dumps(payload)):
                    raise ValueError("definitive native scores cannot be silently regraded")
            else:
                self.connection.execute(
                    "INSERT INTO scores VALUES (?, ?, ?, ?)",
                    (*scope, json.dumps(payload, ensure_ascii=False)),
                )

    def seal(self, candidate: FinalCandidate) -> FinalCandidate:
        scope = (candidate.run_id, candidate.arm_id, candidate.episode_id)
        if candidate.terminal_status is not CandidateStatus.LEGACY_UNVERIFIED:
            from .integrity_final_validation import validate_persisted_owner_source

            outputs = self.model_outputs(scope)
            if not outputs:
                raise ValueError("verified final has no persisted actual model output")
            source = outputs[-1]
            validate_persisted_owner_source(
                self,
                candidate,
                benchmark=source.benchmark,
                native_thinking=source.native_thinking,
                adapter_name=source.adapter_name,
            )
        with self.connection:
            try:
                self.connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                    (*scope, json.dumps(asdict(candidate), ensure_ascii=False)),
                )
            except sqlite3.IntegrityError:
                if self.get(*scope) != candidate:
                    raise ValueError(
                        "a policy final has already been submitted; replacement is forbidden"
                    ) from None
        return self.get(*scope)

    def record(
        self,
        scope: tuple[str, str, str],
        stage: str,
        payload: object,
        *,
        origin: EventOrigin | None = None,
    ) -> None:
        if origin is not None and not isinstance(origin, EventOrigin):
            raise TypeError("trace origin must be an EventOrigin")
        # Legacy three-argument diagnostic clients remain writable, but their
        # observations cannot later be treated as trusted runtime evidence.
        stored_origin = EventOrigin.LEGACY_UNVERIFIED if origin is None else origin
        if stored_origin is EventOrigin.ACTOR_DIAGNOSTIC:
            authorize_actor_trace(stage)
        with self.connection:
            self.connection.execute(
                "INSERT INTO trace (run_id, arm_id, episode_id, stage, payload, origin) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (*scope, stage, json.dumps(payload, ensure_ascii=False), stored_origin.value),
            )

    def intent(
        self, scope: tuple[str, str, str], decision_id: str, action: str, revision: int
    ) -> None:
        """Insert before calling the environment. UNKNOWN is never blindly replayed."""
        with self.connection:
            pending = self.connection.execute(
                "SELECT 1 FROM executions WHERE run_id=? AND arm_id=? AND episode_id=? "
                "AND acknowledged=0",
                scope,
            ).fetchone()
            if pending is not None:
                raise RuntimeError(
                    "an environment execution is unacknowledged; reconcile before continuing"
                )
            try:
                self.connection.execute(
                    "INSERT INTO executions (run_id, arm_id, episode_id, decision_id, action, "
                    "state_revision, status, observation, acknowledged) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'unknown', NULL, 0)",
                    (*scope, decision_id, action, revision),
                )
            except sqlite3.IntegrityError:
                raise ValueError("a decision may only be executed once") from None

    def acknowledge(
        self, scope: tuple[str, str, str], decision_id: str, *, status: str, observation: str
    ) -> None:
        if status not in {"confirmed", "rejected", "unknown"}:
            raise ValueError("invalid execution acknowledgement")
        with self.connection:
            updated = self.connection.execute(
                "UPDATE executions SET status=?, observation=?, acknowledged=1 "
                "WHERE run_id=? AND arm_id=? AND episode_id=? AND decision_id=? AND acknowledged=0",
                (status, observation, *scope, decision_id),
            )
            if updated.rowcount != 1:
                raise ValueError("execution acknowledgement has no pending intent")
