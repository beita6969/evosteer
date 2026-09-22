"""Trace origins keep actor diagnostics separate from broker authority."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from skillev.evaluation.sealed_candidates import (
    CandidateJournal,
    CandidateReader,
    EventOrigin,
    FinalCandidate,
    authorize_actor_trace,
)

SCOPE = ("synthetic-run", "A2", "case-1")


def test_actor_allowlist_rejects_authoritative_stage_names() -> None:
    assert authorize_actor_trace("public-episode-close") is EventOrigin.ACTOR_DIAGNOSTIC
    with pytest.raises(ValueError):
        authorize_actor_trace("native-outcome")


def test_journal_filters_origins_and_three_argument_clients_remain_unverified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.sqlite"
    journal = CandidateJournal(path)
    journal.record(SCOPE, "rendered-request", {"diagnostic": True})
    journal.record(
        SCOPE,
        "rendered-request",
        {"transport": True},
        origin=EventOrigin.MODEL_TRANSPORT,
    )
    journal.record(SCOPE, "native-outcome", {"success": True}, origin=EventOrigin.ENVIRONMENT)
    with pytest.raises(ValueError):
        journal.record(SCOPE, "native-outcome", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    journal.close()

    reader = CandidateReader(path)
    assert reader.traces(SCOPE, "rendered-request", origin=EventOrigin.MODEL_TRANSPORT) == (
        {"transport": True},
    )
    assert reader.traces(SCOPE, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC) == ()
    assert reader.traces(SCOPE, "rendered-request", origin=EventOrigin.LEGACY_UNVERIFIED) == (
        {"diagnostic": True},
    )
    assert reader.traces(SCOPE, "native-outcome", origin=EventOrigin.ENVIRONMENT) == (
        {"success": True},
    )
    reader.close()


def test_schema_migration_marks_preexisting_trace_rows_legacy_unverified(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE trace (sequence INTEGER PRIMARY KEY, run_id TEXT, arm_id TEXT, "
        "episode_id TEXT, stage TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO trace VALUES (1, ?, ?, ?, 'native-outcome', '{\"success\": true}')", SCOPE
    )
    connection.commit()
    connection.close()

    journal = CandidateJournal(path)
    assert journal.traces(SCOPE, "native-outcome", origin=EventOrigin.ENVIRONMENT) == ()
    assert journal.traces(SCOPE, "native-outcome", origin=EventOrigin.LEGACY_UNVERIFIED) == (
        {"success": True},
    )
    journal.close()


def test_owner_submission_is_optional_for_old_artifacts_and_strict_when_present(
    tmp_path: Path,
) -> None:
    journal = CandidateJournal(tmp_path / "candidate.sqlite")
    legacy = FinalCandidate(*SCOPE, "policy", "owner-final", "text", "parser", 1, 1)
    assert journal.seal(legacy) == legacy
    with pytest.raises(ValueError):
        FinalCandidate(
            "other-run",
            "A2",
            "case-2",
            "policy",
            "owner-final",
            "text",
            "parser",
            1,
            1,
            submission={
                "owner_id": "owner",
                "message_id": "different-message",
                "raw_response": "Final: text",
                "payload": "text",
                "payload_type": "natural-language",
                "projection_id": "owner-final-v2",
            },
        )
    journal.close()
