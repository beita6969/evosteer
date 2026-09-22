"""Trusted final identity checks; no scores, answer selection or generation API."""

from __future__ import annotations

from dataclasses import asdict

from .agent_communication import control_payload
from .model_output_provenance import CandidateStatus, StoredModelOutput, served_token_stream
from .native_channels import ChannelStatus
from .owner_final import FinalSubmission, project_owner_final
from .sealed_candidates import CandidateReader, EventOrigin, FinalCandidate
from .step0_completion import StepZeroTerminalMode


def validate_final_identity(
    final: FinalCandidate,
    *,
    expected_scope: tuple[str, str, str],
    expected_policy: str,
    expected_parser: str,
    expected_message_id: str,
) -> None:
    if (final.run_id, final.arm_id, final.episode_id) != expected_scope:
        raise ValueError("final belongs to another episode")
    if final.policy_id != expected_policy or final.parser_id != expected_parser:
        raise ValueError("final identity differs from the executed condition")
    if final.final_message_id != expected_message_id:
        raise ValueError("final is not the designated owner submission")


def validate_owner_projection(
    final: FinalCandidate, *, raw_response: str, participant: str, mode: StepZeroTerminalMode
) -> None:
    if not final.text:
        if final.submission is not None:
            raise ValueError("empty final cannot carry a submitted payload")
        if (
            participant == "owner"
            and project_completed_owner_final(mode, raw_response, message_id=final.final_message_id)
            is not None
        ):
            raise ValueError("an actual valid owner final cannot be discarded as an empty answer")
        return
    if participant != "owner":
        raise ValueError("a peer cannot submit the owner's final")
    expected = project_completed_owner_final(mode, raw_response, message_id=final.final_message_id)
    if expected is None or final.submission is None:
        raise ValueError("owner final has no identifiable executed projection")
    if final.submission != asdict(expected) or final.text != expected.payload:
        raise ValueError("submitted payload differs from the actual owner's response")


def terminal_mode(benchmark: str) -> StepZeroTerminalMode:
    return {
        "hotpotqa": StepZeroTerminalMode.SHORT_ANSWER,
        "triviaqa": StepZeroTerminalMode.SHORT_ANSWER,
        "musique": StepZeroTerminalMode.SHORT_ANSWER,
        "nq-open": StepZeroTerminalMode.SHORT_ANSWER,
        "aime-2026": StepZeroTerminalMode.AIME_INTEGER,
        "mbpp-plus": StepZeroTerminalMode.PYTHON_SOURCE,
        "humaneval": StepZeroTerminalMode.PYTHON_SOURCE,
        "livecodebench": StepZeroTerminalMode.PYTHON_SOURCE,
        "apps-introductory": StepZeroTerminalMode.PYTHON_SOURCE,
    }.get(benchmark, StepZeroTerminalMode.NATURAL_LANGUAGE)


def project_completed_owner_final(
    mode: StepZeroTerminalMode, text: str, *, message_id: str
) -> FinalSubmission | None:
    """Use the controller's control/final distinction, not an answer selector."""
    try:
        control = control_payload(text)
    except ValueError:
        return None
    if control is not None and control.get("kind") in {
        "message",
        "review",
        "history",
        "skill",
        "corpus_search",
    }:
        return None
    return project_owner_final(mode, text, owner_id="owner", message_id=message_id)


def validate_persisted_owner_source(
    reader: CandidateReader,
    final: FinalCandidate,
    *,
    benchmark: str,
    native_thinking: bool,
    adapter_name: str | None,
) -> StoredModelOutput:
    """Same authority check before scoring or returning a same-run stored final."""
    if final.terminal_status in {
        CandidateStatus.LEGACY_UNVERIFIED,
        CandidateStatus.INFRASTRUCTURE_UNRESOLVED,
    }:
        raise ValueError("unverified or unresolved infrastructure is not a verified candidate")
    scope = (final.run_id, final.arm_id, final.episode_id)
    outputs = reader.model_outputs(scope)
    if not outputs:
        raise ValueError("candidate has no persisted actual generation")
    attempt = reader.connection.execute(
        "SELECT attempt_id,policy_id FROM episode_attempts "
        "WHERE run_id=? AND arm_id=? AND episode_id=?",
        scope,
    ).fetchone()
    if attempt != (final.attempt_id, final.policy_id):
        raise ValueError("candidate does not belong to the original episode attempt")
    for output in outputs:
        if (
            output.scope != scope
            or output.attempt_id != final.attempt_id
            or output.policy_id != final.policy_id
            or output.result["policy_snapshot_id"] != final.policy_id
            or output.benchmark != benchmark
            or output.native_thinking != native_thinking
            or output.adapter_name != adapter_name
        ):
            raise ValueError("persisted output belongs to another scope, attempt or policy")
    source = outputs[-1]
    served_token_stream(outputs)
    if source.call_id != final.owner_call_id or source.participant != "owner":
        raise ValueError("final must identify the actual last completed owner call")
    source_is_decision = source.purpose in {"decision", "interface-repair"}
    costs = [output.result["usage"] for output in outputs]
    if (final.prompt_tokens, final.completion_tokens, final.intervention_counts["model_calls"]) != (
        sum(row["input_tokens"] for row in costs),
        sum(row["output_tokens"] for row in costs),
        len(outputs),
    ):
        raise ValueError("candidate accounting differs from its actual persisted calls")
    if benchmark in {"webshop", "alfworld", "scienceworld"}:
        import json

        actions = reader.traces(scope, "owner-action-source", origin=EventOrigin.ENVIRONMENT)
        by_call = {output.call_id: output for output in outputs}
        acknowledged = reader.connection.execute(
            "SELECT decision_id,action,state_revision FROM executions "
            "WHERE run_id=? AND arm_id=? AND episode_id=? AND acknowledged=1",
            scope,
        ).fetchall()
        if len(actions) != len(acknowledged):
            raise ValueError("native trajectory has no complete source/acknowledgement linkage")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("native action source is malformed")
            generated = by_call.get(action["call_id"])
            if (
                generated is None
                or generated.participant != "owner"
                or generated.purpose not in {"decision", "interface-repair"}
                or generated.channel_status is not ChannelStatus.COMPLETE
                or action["attempt_id"] != final.attempt_id
                or generated.public_revision != action["source_revision"]
                or (action["decision_id"], action["action"], action["source_revision"])
                not in acknowledged
            ):
                raise ValueError(
                    "native action is not linked to this owner's acknowledged execution"
                )
        if json.loads(final.text) != [
            action["action"] for action in actions if isinstance(action, dict)
        ]:
            raise ValueError("final trajectory differs from the acknowledged owner actions")
        if final.terminal_status is not CandidateStatus.SUBMITTED:
            raise ValueError(
                "a native trajectory must be submitted as a trajectory, not a text failure"
            )
        return source
    if final.text:
        if not source_is_decision:
            raise ValueError("a separate review draft is not the owner's submitted decision")
        if source.channel_status is not ChannelStatus.COMPLETE or not source.final_text.strip():
            raise ValueError("unfinished reasoning has no eligible final answer")
        if final.terminal_status is not CandidateStatus.SUBMITTED:
            raise ValueError("a valid owner final must be classified as submitted")
    elif final.terminal_status is CandidateStatus.SUBMITTED:
        raise ValueError("an empty static answer is not a submitted final")
    if final.terminal_status is CandidateStatus.BUDGET_EXHAUSTED and not (
        len(outputs) >= min(source.budgets["total_model_calls"], source.budgets["calls_per_turn"])
        or final.completion_tokens >= source.budgets["total_output_tokens"]
    ):
        raise ValueError("budget-exhausted candidate has no actual exhausted budget")
    if final.terminal_status is CandidateStatus.MODEL_NO_FINAL and (
        source.channel_status is ChannelStatus.COMPLETE and source.final_text.strip()
    ):
        raise ValueError("model-no-final contradicts the actual final-channel output")
    if final.terminal_status is CandidateStatus.MODEL_FORMAT_INVALID and (
        source.channel_status is not ChannelStatus.COMPLETE or not source.final_text.strip()
    ):
        raise ValueError("format-invalid requires an actual nonempty final-channel output")
    if source_is_decision:
        validate_owner_projection(
            final,
            raw_response=source.final_text
            if source.channel_status is ChannelStatus.COMPLETE
            else "",
            participant=source.participant,
            mode=terminal_mode(benchmark),
        )
    elif final.submission is not None:
        raise ValueError("a review draft cannot carry a submitted final payload")
    return source
