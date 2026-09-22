"""Action-derived declaration/execution links, not inferred causal skill credit."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from skillev.contracts import JsonValue, TrajectoryRecord
from skillev.contracts.skill_exposure import CATALOG_EXPOSURES
from skillev.contracts.skill_invocation import parse_action_invocation


@dataclass(frozen=True, slots=True)
class InvocationExecutionLink:
    step_index: int
    declared_skill_id: str
    admitted: bool
    observation_status: str
    following_execution_steps: tuple[int, ...]
    terminal_success: bool
    body_returned: bool | None = None
    returned_library_version: str | None = None
    returned_skill_version: str | None = None
    later_execution_steps: tuple[int, ...] | None = None
    body_visible_execution_steps: tuple[int, ...] | None = None
    read_ordinal: int | None = None
    repeat_same_version: bool | None = None
    preceding_observation_status: str | None = None
    following_task_action_steps: tuple[int, ...] | None = None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **(
                cast(
                    dict[str, JsonValue],
                    {
                        "read_ordinal": self.read_ordinal,
                        "repeat_same_version": self.repeat_same_version,
                        "preceding_observation_status": self.preceding_observation_status,
                        "following_task_action_steps": list(self.following_task_action_steps or ()),
                        "read_followed_by_task_action": bool(self.following_task_action_steps),
                        "efficacy": "not-inferred-from-read-or-temporal-following",
                    },
                )
                if self.read_ordinal is not None
                else {}
            ),
            **(
                {
                    "body_returned": self.body_returned,
                    "returned_library_version": self.returned_library_version,
                    "returned_skill_version": self.returned_skill_version,
                }
                if self.body_returned is not None
                else {}
            ),
            **(
                {"later_execution_steps": cast(JsonValue, list(self.later_execution_steps))}
                if self.later_execution_steps is not None
                else {}
            ),
            **(
                {
                    "body_visible_execution_steps": cast(
                        JsonValue, list(self.body_visible_execution_steps)
                    )
                }
                if self.body_visible_execution_steps is not None
                else {}
            ),
            "step_index": self.step_index,
            "declared_skill_id": self.declared_skill_id,
            "admitted": self.admitted,
            "observation_status": self.observation_status,
            "following_execution_steps": list(self.following_execution_steps),
            "terminal_success": self.terminal_success,
            "label_source": "TerminalReward.success",
            "interpretation": "explicit-strategy-declaration-not-independent-skill-execution",
        }

    @classmethod
    def from_value(cls, value: object) -> "InvocationExecutionLink":
        if not isinstance(value, dict):
            raise TypeError("invocation evidence must be an object")
        if value.get("label_source") != "TerminalReward.success":
            raise ValueError("invocation evidence must use the terminal Bernoulli label")
        return cls(
            value["step_index"],
            value["declared_skill_id"],
            value["admitted"],
            value["observation_status"],
            tuple(value["following_execution_steps"]),
            value["terminal_success"],
            value.get("body_returned"),
            value.get("returned_library_version"),
            value.get("returned_skill_version"),
            tuple(value["later_execution_steps"])
            if value.get("later_execution_steps") is not None
            else None,
            tuple(value["body_visible_execution_steps"])
            if value.get("body_visible_execution_steps") is not None
            else None,
            value.get("read_ordinal"),
            value.get("repeat_same_version"),
            value.get("preceding_observation_status"),
            tuple(value["following_task_action_steps"])
            if value.get("following_task_action_steps") is not None
            else None,
        )


def invocation_execution_links(
    record: TrajectoryRecord,
    input_evidence: tuple[dict[str, JsonValue], ...] | None = None,
) -> tuple[InvocationExecutionLink, ...]:
    """Associate subsequent public actions up to the next declaration, without
    assigning their causal contribution to the skill. Rejected calls stay visible
    but never become a posterior event; an empty following span stays empty.
    """
    parsed = [
        parse_action_invocation(step.action_text, initial_meta=record.initial_context.meta)
        for step in record.steps
    ]
    result: list[InvocationExecutionLink] = []
    seen_versions: set[tuple[str, str | None, str | None]] = set()
    for offset, (step, action) in enumerate(zip(record.steps, parsed, strict=True)):
        if action is None or action.skill_id is None:
            continue
        following = []
        for later_step, later_action in zip(
            record.steps[offset + 1 :], parsed[offset + 1 :], strict=True
        ):
            if later_action is not None and later_action.skill_id is not None:
                break
            following.append(later_step.index)
        later = tuple(
            later_step.index
            for later_step, later_action in zip(
                record.steps[offset + 1 :], parsed[offset + 1 :], strict=True
            )
            if later_action is not None and later_action.skill_id is None
        )
        body_visible = None
        if input_evidence is not None:
            forward = [
                row
                for row in input_evidence
                if row.get("phase") == "action" and row.get("step_index") in later
            ]
            if len(forward) == len(later) and all(
                isinstance(row.get("visible_skill_body_refs"), list) for row in forward
            ):
                body_visible = tuple(
                    cast(int, row["step_index"])
                    for row in forward
                    if any(
                        isinstance(ref, dict)
                        and ref.get("read_step_index") == step.index
                        and ref.get("skill_id") == action.skill_id
                        for ref in cast(list[JsonValue], row["visible_skill_body_refs"])
                    )
                )
        body_returned, returned_library, returned_version = catalog_read_result(
            record.initial_context.meta,
            action.skill_id,
            step.invoked_skill_ids,
            step.observation_status,
            step.observation_text,
        )
        identity = (action.skill_id, returned_version, returned_library)
        following_task = tuple(
            candidate.index
            for candidate, parsed_action in zip(
                record.steps[offset + 1 :], parsed[offset + 1 :], strict=True
            )
            if candidate.index in following
            and parsed_action is not None
            and parsed_action.skill_id is None
        )
        result.append(
            InvocationExecutionLink(
                step.index,
                action.skill_id,
                action.skill_id in step.invoked_skill_ids,
                step.observation_status,
                tuple(following),
                record.reward.success,
                body_returned,
                returned_library,
                returned_version,
                later,
                body_visible,
                len(result) + 1,
                identity in seen_versions if body_returned else None,
                record.steps[offset - 1].observation_status if offset else None,
                following_task,
            )
        )
        if body_returned:
            seen_versions.add(identity)
    return tuple(result)


def catalog_read_result(
    meta: Mapping[str, JsonValue],
    skill_id: str,
    invoked_ids: Sequence[str],
    status: str,
    observation_text: str,
) -> tuple[bool | None, str | None, str | None]:
    """Only the actual environment response establishes a returned body."""
    if meta.get("skill_exposure") not in CATALOG_EXPOSURES:
        return None, None, None
    try:
        observed = json.loads(observation_text)
    except (ValueError, TypeError):
        observed = None
    if not isinstance(observed, dict):
        return False, None, None
    library = observed.get("library_version")
    version = observed.get("version")
    received = (
        skill_id in invoked_ids
        and status == "success"
        and observed.get("status") == "skill-read"
        and observed.get("skill_id") == skill_id
        and library == meta.get("library_version")
        and isinstance(version, str)
        and bool(version)
        and isinstance(observed.get("content"), str)
        and bool(observed["content"].strip())
    )
    return (
        received,
        library if isinstance(library, str) else None,
        version if isinstance(version, str) else None,
    )
