"""Communication status follows observed, origin-aware control delivery evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillev.evaluation.integrity_communication_report import communication_summary
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin
from skillev.evaluation.step0_completion import native_action_constraint

SCOPE = ("synthetic-run", "A2", "case-1")


def _record_complete_model_boundary(journal: CandidateJournal, *, participant="owner") -> None:
    journal.record(
        SCOPE,
        "model-transport-start",
        {"participant": participant},
        origin=EventOrigin.MODEL_TRANSPORT,
    )
    journal.record(SCOPE, "model-transport-complete", {}, origin=EventOrigin.MODEL_TRANSPORT)
    journal.record(SCOPE, "rendered-request", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    journal.record(SCOPE, "model-response", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)


def test_diagnostic_transport_evidence_is_incomplete_and_never_passes(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    journal.record(SCOPE, "rendered-request", {})
    journal.record(SCOPE, "model-response", {})

    report = communication_summary(journal, (SCOPE,))

    assert report["status"] == "incomplete-evidence"
    assert report["legacy_unverified_events"] == 2
    journal.close()


@pytest.mark.parametrize("resolved", [False, True])
def test_owner_control_parse_failure_is_output_failure_not_delivery(
    tmp_path: Path, resolved: bool
) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    journal.record(
        SCOPE,
        "control-attempt",
        {"attempt_id": "attempt-1"},
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    journal.record(
        SCOPE,
        "control-parse-failure",
        {"attempt_id": "attempt-1"},
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    if resolved:
        journal.record(
            SCOPE,
            "control-resolved",
            {"attempt_id": "attempt-1", "resolution": "repaired"},
            origin=EventOrigin.ACTOR_DIAGNOSTIC,
        )

    report = communication_summary(journal, (SCOPE,))

    assert report["attempted_control_messages"] == 1
    assert report["control_parse_failures"] == 1
    assert report["unresolved_control_parse_failures"] == int(not resolved)
    assert report["repaired_control_parse_failures"] == int(resolved)
    assert report["undelivered_unresolved_requests"] == int(not resolved)
    assert report["unresolved_delivery_failures"] == report["repaired_delivery_failures"] == 0
    assert report["status"] == (
        "complete-with-repairs" if resolved else "unresolved-output-failure"
    )
    journal.close()


@pytest.mark.parametrize("delivery_failure", [False, True])
def test_parse_failure_cannot_hide_an_unresolved_delivery(tmp_path: Path, delivery_failure) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    for stage, attempt in (
        ("control-attempt", "malformed"),
        ("control-parse-failure", "malformed"),
        # Either a known delivery failure on the same ID or another incomplete
        # attempt without a known failure cause must remain visible.
        ("peer-delivery-failure", "malformed")
        if delivery_failure
        else ("control-attempt", "unresolved"),
    ):
        journal.record(SCOPE, stage, {"attempt_id": attempt}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    report = communication_summary(journal, (SCOPE,))
    assert report["unresolved_control_parse_failures"] == 1
    assert report["unresolved_delivery_failures"] == 1
    assert report["status"] == "unresolved-delivery-failure"
    journal.close()


def test_resolved_repair_and_actual_peer_calls_are_reported_by_benchmark(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    _record_complete_model_boundary(journal, participant="solver")
    for stage, payload in (
        ("control-attempt", {"attempt_id": "attempt-1"}),
        ("control-decoded", {"attempt_id": "attempt-1"}),
        ("peer-delivery-failure", {"attempt_id": "attempt-1"}),
        ("control-resolved", {"attempt_id": "attempt-1", "resolution": "repaired"}),
        ("model-interface-repair", {"call_id": "model-call-2", "status": "started"}),
        (
            "model-interface-repair",
            {"call_id": "model-call-2", "status": "completed", "success": True},
        ),
        (
            "agent-message",
            {"message_id": "message-1", "sender": "owner", "recipient": "solver"},
        ),
        (
            "agent-reply",
            {
                "parent_id": "message-1",
                "sender": "solver",
                "recipient": "owner",
                "late": False,
            },
        ),
    ):
        journal.record(SCOPE, stage, payload, origin=EventOrigin.ACTOR_DIAGNOSTIC)

    report = communication_summary(journal, (SCOPE,), benchmark_by_scope={SCOPE: "aime-2026"})

    assert report["successfully_decoded_messages"] == 1
    assert report["delivered_requests"] == report["returned_replies"] == 1
    assert report["repair_attempts"] == report["repair_successes"] == 1
    assert report["repaired_delivery_failures"] == 1
    assert report["actual_peer_calls_by_benchmark"] == {"aime-2026": 1}
    assert report["status"] == "complete-with-repairs"
    journal.close()


def test_no_model_boundary_is_not_observed(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    assert communication_summary(journal, (SCOPE,))["status"] == "not-observed"
    journal.close()


@pytest.mark.parametrize("advertised", [("act",), ("look around",), ()])
def test_scienceworld_freeform_surface_is_not_an_enumerated_action_menu(
    tmp_path: Path, advertised: tuple[str, ...]
) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    journal.record(SCOPE, "environment-reset", {}, origin=EventOrigin.ENVIRONMENT)
    journal.record(
        SCOPE,
        "action-surface",
        {
            "mode": "scienceworld",
            "native_actions": ("act",),
            "advertised_actions": advertised,
            "tool_schema": native_action_constraint("scienceworld").json_schema,
        },
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    report = communication_summary(journal, (SCOPE,))
    assert report["surface_mismatches"] == int(advertised != ("act",))
    assert report["status"] == ("complete" if advertised == ("act",) else "transport-defect")
    journal.close()


def test_failed_peer_transport_still_counts_as_actual_use(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    journal.record(SCOPE, "rendered-request", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    journal.record(
        SCOPE,
        "model-transport-start",
        {"participant": "researcher"},
        origin=EventOrigin.MODEL_TRANSPORT,
    )
    report = communication_summary(journal, (SCOPE,), benchmark_by_scope={SCOPE: "healthbench"})
    assert report["actual_peer_calls_by_benchmark"] == {"healthbench": 1}
    assert report["episodes_with_peer_calls"] == 1
    assert "researcher" in report["actual_agent_identities"]
    assert report["status"] == "transport-defect"
    journal.close()


def test_partial_missing_raw_response_evidence_cannot_report_complete(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    journal.record(SCOPE, "rendered-request", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    journal.record(SCOPE, "model-transport-start", {}, origin=EventOrigin.MODEL_TRANSPORT)
    journal.record(SCOPE, "model-transport-complete", {}, origin=EventOrigin.MODEL_TRANSPORT)
    report = communication_summary(journal, (SCOPE,))
    assert report["unobserved_required_boundary_count"] > 0
    assert report["status"] == "incomplete-evidence"
    journal.close()


@pytest.mark.parametrize("resolved", [False, True])
def test_terminal_failure_needs_a_later_nonempty_owner_final(
    tmp_path: Path, resolved: bool
) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    for stage, payload in (
        ("owner-final-submission", {"payload": "Earlier output"}),
        ("terminal-parse-failure", {"candidate_submitted": False}),
        ("owner-final-submission", {"payload": ""}),
    ):
        journal.record(SCOPE, stage, payload, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    if resolved:
        journal.record(
            SCOPE,
            "owner-final-submission",
            {"payload": "Later final"},
            origin=EventOrigin.ACTOR_DIAGNOSTIC,
        )
    report = communication_summary(journal, (SCOPE,))
    assert report["unresolved_terminal_parse_failures"] == int(not resolved)
    assert report["repaired_terminal_parse_failures"] == int(resolved)
    assert report["status"] == (
        "complete-with-repairs" if resolved else "unresolved-output-failure"
    )
    assert report["peer_execution_status"] == "not-observed"
    journal.close()


@pytest.mark.parametrize("resolved", [False, True])
def test_native_rejection_is_not_hidden_by_an_earlier_accepted_action(
    tmp_path: Path, resolved: bool
) -> None:
    journal = CandidateJournal(tmp_path / "trace.sqlite")
    _record_complete_model_boundary(journal)
    actions = ["look", None, None] + (["open door"] if resolved else [])
    for revision, action in enumerate(actions):
        journal.record(
            SCOPE,
            "decision-transport",
            {"result": {"action": action}, "decision": {"state_revision": revision}},
            origin=EventOrigin.ACTOR_DIAGNOSTIC,
        )
        if action is not None:
            journal.connection.execute(
                "INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    *SCOPE,
                    f"decision-{revision}",
                    action,
                    revision,
                    "executed",
                    "Public feedback",
                    1,
                ),
            )
            journal.record(SCOPE, "environment-result", {}, origin=EventOrigin.ENVIRONMENT)
            journal.record(
                SCOPE,
                "public-transition",
                {"observation": "Public feedback"},
                origin=EventOrigin.ACTOR_DIAGNOSTIC,
            )
    report = communication_summary(journal, (SCOPE,))
    assert report["unresolved_action_decisions"] == (0 if resolved else 2)
    assert report["repaired_action_decisions"] == (2 if resolved else 0)
    assert report["status"] == (
        "complete-with-repairs" if resolved else "unresolved-output-failure"
    )
    journal.close()
