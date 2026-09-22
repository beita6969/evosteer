"""Aggregate origin-aware communication evidence without task or message contents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .scienceworld_commands import TYPED_COMMANDS
from .sealed_candidates import CandidateReader, EventOrigin

Scope = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class CommunicationEvidence:
    observed_calls: int
    observation_gaps: int
    route_mismatches: int
    unacknowledged_executions: int
    unresolved_delivery_failures: int
    repaired_delivery_failures: int
    unresolved_output_failures: int = 0
    repaired_output_failures: int = 0


def communication_status(evidence: CommunicationEvidence) -> str:
    """Describe evidence completeness and transport health, never model quality."""
    if evidence.observed_calls == 0:
        return "not-observed"
    if evidence.observation_gaps:
        return "incomplete-evidence"
    if evidence.route_mismatches or evidence.unacknowledged_executions:
        return "transport-defect"
    if evidence.unresolved_delivery_failures:
        return "unresolved-delivery-failure"
    if evidence.unresolved_output_failures:
        return "unresolved-output-failure"
    return (
        "complete-with-repairs"
        if evidence.repaired_delivery_failures or evidence.repaired_output_failures
        else "complete"
    )


def _allowed(schema: Mapping[str, Any], mode: str) -> set[str]:
    if "oneOf" in schema:
        return set().union(*(_allowed(branch, mode) for branch in schema["oneOf"]))
    properties = schema.get("properties", {})
    name = properties.get("name", {}).get("const")
    arguments = properties.get("arguments", {}).get("properties", {})
    key = {
        "search": "query",
        "click": "target",
        "act": "command",
        **dict.fromkeys(TYPED_COMMANDS, "target"),
    }.get(name)
    if key is None:
        return set()
    values = arguments.get(key, {}).get("enum")
    if values is None:
        # ScienceWorld advertises a free-form act(command), not an oracle
        # enumeration of valid object/action combinations. Its actual surface
        # is {"act"}; an absent enum is not a missing or pruned action menu.
        return (
            {name}
            if name == "search" or (mode == "scienceworld" and name in {"act", *TYPED_COMMANDS})
            else set()
        )
    return set(values) if mode == "alfworld" else {f"{name}[{value}]" for value in values}


def _diagnostic(reader: CandidateReader, scope: Scope, stage: str) -> tuple[object, ...]:
    return reader.traces(scope, stage, origin=EventOrigin.ACTOR_DIAGNOSTIC)


def _transport(reader: CandidateReader, scope: Scope, stage: str) -> tuple[object, ...]:
    return reader.traces(scope, stage, origin=EventOrigin.MODEL_TRANSPORT)


def _environment(reader: CandidateReader, scope: Scope, stage: str) -> tuple[object, ...]:
    return reader.traces(scope, stage, origin=EventOrigin.ENVIRONMENT)


def _attempt_ids(events: tuple[object, ...]) -> tuple[set[str], int]:
    identifiers: set[str] = set()
    malformed = 0
    for event in events:
        if not isinstance(event, Mapping):
            malformed += 1
            continue
        attempt_id = event.get("attempt_id")
        if type(attempt_id) is not str or not attempt_id:
            malformed += 1
            continue
        identifiers.add(attempt_id)
    return identifiers, malformed


def _resolutions(events: tuple[object, ...]) -> tuple[dict[str, str], int]:
    resolved: dict[str, str] = {}
    malformed = 0
    for event in events:
        if not isinstance(event, Mapping):
            malformed += 1
            continue
        attempt_id, resolution = event.get("attempt_id"), event.get("resolution")
        if (
            type(attempt_id) is not str
            or not attempt_id
            or type(resolution) is not str
            or not resolution
        ):
            malformed += 1
            continue
        if attempt_id in resolved and resolved[attempt_id] != resolution:
            malformed += 1
            continue
        resolved[attempt_id] = resolution
    return resolved, malformed


def _repair_successes(events: tuple[object, ...]) -> int:
    successful_states = {"repaired", "resolved", "succeeded", "success"}
    successes = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("success") is True or event.get("repaired") is True:
            successes += 1
        elif (
            event.get("status") in successful_states or event.get("resolution") in successful_states
        ):
            successes += 1
    return successes


def _peer_rows(events: tuple[object, ...]) -> list[Mapping[str, object]]:
    return [event for event in events if isinstance(event, Mapping)]


def _output_failures(reader: CandidateReader, scope: Scope) -> dict[str, int]:
    """A later accepted output resolves earlier errors, never errors that follow it."""
    counts = dict.fromkeys(
        (
            "unresolved_terminal_parse_failures",
            "repaired_terminal_parse_failures",
            "unresolved_action_decisions",
            "repaired_action_decisions",
        ),
        0,
    )
    if not reader.trace_has_origin:
        return counts
    rows = reader.connection.execute(
        "SELECT stage,payload FROM trace WHERE run_id=? AND arm_id=? AND episode_id=? "
        "AND origin=? AND stage IN (?,?,?) ORDER BY sequence",
        (
            *scope,
            EventOrigin.ACTOR_DIAGNOSTIC.value,
            "terminal-parse-failure",
            "owner-final-submission",
            "decision-transport",
        ),
    )
    for stage, raw in rows:
        value = json.loads(raw)
        if stage == "terminal-parse-failure":
            counts["unresolved_terminal_parse_failures"] += 1
        elif stage == "owner-final-submission":
            if isinstance(value, Mapping) and isinstance(value.get("payload"), str):
                if value["payload"].strip():
                    counts["repaired_terminal_parse_failures"] += counts[
                        "unresolved_terminal_parse_failures"
                    ]
                    counts["unresolved_terminal_parse_failures"] = 0
        elif isinstance(value, Mapping) and isinstance(value.get("result"), Mapping):
            result = value["result"]
            if "action" in result and result["action"] is None:
                counts["unresolved_action_decisions"] += 1
            elif isinstance(result.get("action"), str) and result["action"].strip():
                counts["repaired_action_decisions"] += counts["unresolved_action_decisions"]
                counts["unresolved_action_decisions"] = 0
    return counts


def communication_summary(
    reader: CandidateReader,
    scopes: tuple[Scope, ...],
    *,
    benchmark_by_scope: Mapping[Scope, str] | None = None,
) -> dict[str, object]:
    """Summarize only origin-labelled observations; legacy rows never establish PASS."""
    counts = dict.fromkeys(
        (
            "model_requests",
            "model_responses",
            "authoritative_model_requests",
            "authoritative_model_responses",
            "diagnostic_model_requests",
            "diagnostic_model_responses",
            "native_surfaces",
            "surface_mismatches",
            "decisions",
            "accepted_decisions",
            "rejected_decisions",
            "control_parse_failures",
            "unresolved_control_parse_failures",
            "repaired_control_parse_failures",
            "terminal_parse_failures",
            "unresolved_terminal_parse_failures",
            "repaired_terminal_parse_failures",
            "unresolved_action_decisions",
            "repaired_action_decisions",
            "undelivered_peer_budget_requests",
            "attempted_control_messages",
            "successfully_decoded_messages",
            "delivered_requests",
            "returned_replies",
            "undelivered_unresolved_requests",
            "repair_attempts",
            "repair_successes",
            "repaired_delivery_failures",
            "unresolved_delivery_failures",
            "executions",
            "unmatched_executions",
            "missing_acknowledgements",
            "feedback_mismatches",
            "peer_requests",
            "peer_replies",
            "late_peer_replies",
            "peer_route_mismatches",
            "episodes_with_peer_calls",
            "voluntary_no_peer_episode_count",
            "unobserved_required_boundary_count",
        ),
        0,
    )
    identities = {"owner"}
    actual_peer_calls_by_benchmark: dict[str, int] = {}
    legacy_unverified_events = 0

    for scope in scopes:
        for name, value in _output_failures(reader, scope).items():
            counts[name] += value
        authoritative_requests = _transport(reader, scope, "model-transport-start")
        authoritative_responses = _transport(reader, scope, "model-transport-complete")
        diagnostic_requests = _diagnostic(reader, scope, "rendered-request")
        diagnostic_responses = _diagnostic(reader, scope, "model-response")
        counts["authoritative_model_requests"] += len(authoritative_requests)
        counts["authoritative_model_responses"] += len(authoritative_responses)
        counts["diagnostic_model_requests"] += len(diagnostic_requests)
        counts["diagnostic_model_responses"] += len(diagnostic_responses)
        if authoritative_requests or authoritative_responses:
            counts["model_requests"] += len(authoritative_requests)
            counts["model_responses"] += len(authoritative_responses)
            counts["unobserved_required_boundary_count"] += int(not authoritative_requests)
            counts["unobserved_required_boundary_count"] += int(not authoritative_responses)
            counts["unobserved_required_boundary_count"] += abs(
                len(authoritative_requests) - len(diagnostic_requests)
            ) + abs(len(authoritative_responses) - len(diagnostic_responses))
        else:
            counts["model_requests"] += len(diagnostic_requests)
            counts["model_responses"] += len(diagnostic_responses)
            if diagnostic_requests or diagnostic_responses:
                # Actor traces can diagnose a mismatch but cannot attest transport completion.
                counts["unobserved_required_boundary_count"] += 2

        legacy_unverified_events += sum(
            len(reader.traces(scope, stage, origin=EventOrigin.LEGACY_UNVERIFIED))
            for stage in (
                "rendered-request",
                "model-response",
                "decision-transport",
                "agent-message",
            )
        )
        control_attempt_events = _diagnostic(reader, scope, "control-attempt")
        control_decoded_events = _diagnostic(reader, scope, "control-decoded")
        control_resolved_events = _diagnostic(reader, scope, "control-resolved")
        parse_failure_events = _diagnostic(reader, scope, "control-parse-failure")
        delivery_failure_events = _diagnostic(
            reader, scope, "peer-budget-unavailable"
        ) + _diagnostic(reader, scope, "peer-delivery-failure")
        failure_events = parse_failure_events + delivery_failure_events
        counts["control_parse_failures"] += len(parse_failure_events)
        counts["terminal_parse_failures"] += len(
            _diagnostic(reader, scope, "terminal-parse-failure")
        )
        counts["undelivered_peer_budget_requests"] += len(
            _diagnostic(reader, scope, "peer-budget-unavailable")
        )
        attempt_ids, malformed_attempts = _attempt_ids(control_attempt_events)
        decoded_ids, malformed_decoded = _attempt_ids(control_decoded_events)
        parse_failure_ids, malformed_parse = _attempt_ids(parse_failure_events)
        delivery_failure_ids, malformed_delivery = _attempt_ids(delivery_failure_events)
        failure_ids = parse_failure_ids | delivery_failure_ids
        resolutions, malformed_resolutions = _resolutions(control_resolved_events)
        counts["attempted_control_messages"] += len(control_attempt_events)
        counts["successfully_decoded_messages"] += len(control_decoded_events)
        counts["unobserved_required_boundary_count"] += (
            malformed_attempts
            + malformed_decoded
            + malformed_parse
            + malformed_delivery
            + malformed_resolutions
        )
        counts["unobserved_required_boundary_count"] += len(decoded_ids - attempt_ids)
        counts["unobserved_required_boundary_count"] += len(set(resolutions) - attempt_ids)
        unresolved = (attempt_ids | failure_ids) - set(resolutions)
        # An incomplete or malformed owner tool call is output failure, not
        # evidence that the transport lost a valid request. Keep undecoded
        # attempts without a known cause unresolved, and retain an explicitly
        # observed delivery failure even if the same ID also failed parsing.
        unresolved_delivery = (unresolved - parse_failure_ids) | (unresolved & delivery_failure_ids)
        counts["unresolved_control_parse_failures"] += len(unresolved & parse_failure_ids)
        counts["repaired_control_parse_failures"] += len(parse_failure_ids & set(resolutions))
        counts["unresolved_delivery_failures"] += len(unresolved_delivery)
        counts["undelivered_unresolved_requests"] += len(unresolved)
        counts["repaired_delivery_failures"] += len(delivery_failure_ids & set(resolutions))
        repairs = _diagnostic(reader, scope, "model-interface-repair")
        counts["repair_attempts"] += sum(
            isinstance(event, Mapping) and event.get("status") == "started" for event in repairs
        )
        counts["repair_successes"] += _repair_successes(repairs)

        for surface in _diagnostic(reader, scope, "action-surface"):
            if not isinstance(surface, Mapping):
                counts["unobserved_required_boundary_count"] += 1
                continue
            try:
                native = set(surface["native_actions"])
                advertised = set(surface["advertised_actions"])
                schema = surface["tool_schema"]
                mode = surface["mode"]
                if not isinstance(schema, Mapping) or type(mode) is not str:
                    raise TypeError
            except (KeyError, TypeError):
                counts["unobserved_required_boundary_count"] += 1
                continue
            counts["native_surfaces"] += 1
            counts["surface_mismatches"] += int(
                native != advertised or native != _allowed(schema, mode)
            )

        accepted: list[tuple[str, object]] = []
        for item in _diagnostic(reader, scope, "decision-transport"):
            if not isinstance(item, Mapping):
                counts["unobserved_required_boundary_count"] += 1
                continue
            try:
                result, decision = item["result"], item["decision"]
                if not isinstance(result, Mapping) or not isinstance(decision, Mapping):
                    raise TypeError
                action, revision = result["action"], decision["state_revision"]
            except (KeyError, TypeError):
                counts["unobserved_required_boundary_count"] += 1
                continue
            counts["decisions"] += 1
            if action is None:
                counts["rejected_decisions"] += 1
            elif type(action) is str:
                counts["accepted_decisions"] += 1
                accepted.append((action, revision))
            else:
                counts["unobserved_required_boundary_count"] += 1

        executions = reader.connection.execute(
            "SELECT action, state_revision, acknowledged, observation FROM executions "
            "WHERE run_id=? AND arm_id=? AND episode_id=? ORDER BY state_revision",
            scope,
        ).fetchall()
        counts["executions"] += len(executions)
        counts["missing_acknowledgements"] += sum(not row[2] for row in executions)
        counts["unmatched_executions"] += sum(
            (row[0], row[1]) not in accepted for row in executions
        ) + sum(pair not in [(row[0], row[1]) for row in executions] for pair in accepted)
        environment_results = _environment(reader, scope, "environment-result")
        if executions and len(environment_results) != len(executions):
            counts["unobserved_required_boundary_count"] += abs(
                len(environment_results) - len(executions)
            )
        if _diagnostic(reader, scope, "action-surface") and not _environment(
            reader, scope, "environment-reset"
        ):
            counts["unobserved_required_boundary_count"] += 1
        transitions = _diagnostic(reader, scope, "public-transition")
        for row, transition in zip(executions, transitions, strict=False):
            counts["feedback_mismatches"] += int(
                not isinstance(transition, Mapping) or transition.get("observation") != row[3]
            )
        counts["feedback_mismatches"] += abs(len(executions) - len(transitions))

        requests = _peer_rows(_diagnostic(reader, scope, "agent-message"))
        replies = _peer_rows(_diagnostic(reader, scope, "agent-reply"))
        counts["delivered_requests"] += len(requests)
        counts["returned_replies"] += len(replies)
        counts["peer_requests"] += len(requests)
        counts["peer_replies"] += len(replies)
        peer_calls = [
            event
            for event in authoritative_requests
            if isinstance(event, Mapping) and event.get("participant") in {"solver", "researcher"}
        ]
        counts["episodes_with_peer_calls"] += bool(peer_calls)
        identities.update(str(event["participant"]) for event in peer_calls)
        peer_attempts = any(
            isinstance(event, Mapping) and event.get("kind") == "message"
            for event in control_attempt_events
        )
        if (diagnostic_requests or diagnostic_responses) and not (
            peer_attempts or requests or failure_events
        ):
            counts["voluntary_no_peer_episode_count"] += 1
        elif (authoritative_requests or authoritative_responses) and not (
            diagnostic_requests or diagnostic_responses
        ):
            counts["unobserved_required_boundary_count"] += 1
        benchmark = (
            benchmark_by_scope.get(scope, "unknown")
            if benchmark_by_scope is not None
            else "unknown"
        )
        actual_peer_calls_by_benchmark[benchmark] = actual_peer_calls_by_benchmark.get(
            benchmark, 0
        ) + len(peer_calls)
        by_id = {
            row.get("message_id"): row
            for row in requests
            if type(row.get("message_id")) is str and row.get("message_id")
        }
        for reply in replies:
            sender = reply.get("sender")
            if type(sender) is str and sender:
                identities.add(sender)
            parent = by_id.get(reply.get("parent_id"))
            counts["late_peer_replies"] += int(reply.get("late") is True)
            counts["peer_route_mismatches"] += int(
                parent is None
                or (parent.get("recipient"), parent.get("sender"))
                != (reply.get("sender"), reply.get("recipient"))
            )

    transport_defects = (
        counts["surface_mismatches"]
        + counts["unmatched_executions"]
        + counts["missing_acknowledgements"]
        + counts["feedback_mismatches"]
        + counts["peer_route_mismatches"]
        + abs(counts["model_requests"] - counts["model_responses"])
        + abs(counts["peer_requests"] - counts["peer_replies"])
    )
    evidence = CommunicationEvidence(
        observed_calls=counts["model_requests"] + counts["model_responses"],
        observation_gaps=counts["unobserved_required_boundary_count"],
        route_mismatches=transport_defects,
        unacknowledged_executions=counts["missing_acknowledgements"],
        unresolved_delivery_failures=counts["unresolved_delivery_failures"],
        repaired_delivery_failures=counts["repaired_delivery_failures"],
        repaired_output_failures=max(
            counts["repair_successes"],
            counts["repaired_control_parse_failures"]
            + counts["repaired_terminal_parse_failures"]
            + counts["repaired_action_decisions"],
        ),
        unresolved_output_failures=(
            counts["unresolved_control_parse_failures"]
            + counts["unresolved_terminal_parse_failures"]
            + counts["unresolved_action_decisions"]
        ),
    )
    return {
        "report_version": "communication-report@2",
        **counts,
        "executed_decisions": counts["executions"],
        "acknowledged_decisions": counts["executions"] - counts["missing_acknowledgements"],
        "legacy_unverified_events": legacy_unverified_events,
        "actual_peer_calls_by_benchmark": actual_peer_calls_by_benchmark,
        "actual_agent_identities": sorted(identities),
        "actual_agent_count": len(identities),
        "status": "incomplete-evidence"
        if legacy_unverified_events and not evidence.observed_calls
        else communication_status(evidence),
        "native_task_success": "reported-separately",
        "peer_execution_status": "observed"
        if counts["episodes_with_peer_calls"]
        else "not-observed",
        "native_execution_status": "observed"
        if counts["executions"]
        else "not-observed"
        if counts["native_surfaces"]
        else "not-applicable",
    }
