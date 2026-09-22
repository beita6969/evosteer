"""Canonical skill-invocation admission for scientific trajectories.

The action text is the source of a skill invocation.  Environment adapters may
report execution observations, but they cannot independently create skill
credit. Legacy records retain their original projection. Versioned native
records use the same contract-bound codec as execution, rather than a second
approximate parser that can silently lose a real invocation at persistence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .action_text import action_payload
from .action_wire import NATIVE_CARRIER_WIRES
from .canonical import JsonValue, normalize_json
from .identity import validate_identifier


class SkillInvocationAdmissionError(ValueError):
    """A syntactically valid skill action is unavailable in this rollout H0."""


@dataclass(frozen=True, slots=True)
class ParsedActionInvocation:
    """The invocation-relevant projection of one structured action text."""

    kind: str
    skill_id: str | None


def _require_skill_id_tuple(value: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a tuple")
    if any(type(skill_id) is not str for skill_id in value):
        raise ValueError(f"{field} must contain text skill IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not repeat a skill ID")
    for skill_id in value:
        validate_identifier(skill_id)
    return value


def parse_action_invocation(
    action_text: str, *, initial_meta: Mapping[str, JsonValue] | None = None
) -> ParsedActionInvocation | None:
    """Parse exactly the structured-action wire subset relevant to invocation.

    ``None`` means the generated text is not a valid structured action.  The
    legacy projection mirrors the public JSON action codec rather than accepting
    a loose ``kind``/``skill_id`` fragment. Supplying persisted H0 metadata selects
    the declared native codec, including its public-signature disambiguation.
    """

    if type(action_text) is not str:
        raise TypeError("action_text must be text")
    if initial_meta is not None and initial_meta.get("action_wire") in NATIVE_CARRIER_WIRES:
        # Late import keeps contract definition/import free of rollout setup.
        # The persisted public signatures also disambiguate argument-only calls.
        from skillev.rollout.codec import codec_for_initial_meta

        action = codec_for_initial_meta(initial_meta).parse(action_text).action
        return (
            None
            if action is None
            else ParsedActionInvocation(kind=action.kind.value, skill_id=action.skill_id)
        )
    native = re.fullmatch(
        r"\s*<tool_call>\s*<function=read_skill>\s*<parameter=skill_id>\n?([^<\n]+)\n?</parameter>\s*</function>\s*</tool_call>\s*",
        action_text,
    )
    if native is not None:
        skill_id = native[1]
        try:
            validate_identifier(skill_id)
        except ValueError:
            return None
        return ParsedActionInvocation(kind="skill", skill_id=skill_id)
    try:
        raw = json.loads(action_payload(action_text))
        normalized = normalize_json(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(normalized, dict) or set(normalized) != {
        "arguments",
        "kind",
        "name",
        "resource_id",
        "skill_id",
    }:
        return None
    kind = normalized["kind"]
    name = normalized["name"]
    resource_id = normalized["resource_id"]
    skill_id = normalized["skill_id"]
    if kind not in {"tool", "skill", "complete"} or type(name) is not str or not name:
        return None
    if normalized["arguments"] != normalize_json(normalized["arguments"]):
        return None
    if kind in {"tool", "skill"} and (type(resource_id) is not str or not resource_id):
        return None
    if kind == "complete" and resource_id is not None:
        return None
    if kind == "skill":
        if type(skill_id) is not str or not skill_id:
            return None
        try:
            validate_identifier(skill_id)
        except ValueError:
            return None
        return ParsedActionInvocation(kind=kind, skill_id=skill_id)
    if skill_id is not None:
        return None
    return ParsedActionInvocation(kind=kind, skill_id=None)


def canonical_invoked_skill_ids(
    *,
    action_kind: str,
    action_skill_id: str | None,
    retrieved_skill_ids: tuple[str, ...],
    active_skill_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the only legal skill-credit tuple for one parsed action.

    A valid skill action may invoke exactly its own ID, and only when that
    document was both active in the pinned library and disclosed in this
    trajectory's H0.  Other action kinds have no skill credit.  An unavailable
    skill is agent data (handled as ``schema_invalid`` by rollout), not an
    infrastructure substitute or an opportunity for an environment adapter to
    inject a different skill ID.
    """

    _require_skill_id_tuple(retrieved_skill_ids, field="retrieved_skill_ids")
    _require_skill_id_tuple(active_skill_ids, field="active_skill_ids")
    if action_kind != "skill":
        if action_skill_id is not None:
            raise SkillInvocationAdmissionError("only skill actions may carry skill_id")
        return ()
    if type(action_skill_id) is not str:
        raise SkillInvocationAdmissionError("skill action is missing skill_id")
    try:
        validate_identifier(action_skill_id)
    except ValueError as error:
        raise SkillInvocationAdmissionError("skill action skill_id is invalid") from error
    if action_skill_id not in active_skill_ids:
        raise SkillInvocationAdmissionError("skill action targets an inactive skill")
    if action_skill_id not in retrieved_skill_ids:
        raise SkillInvocationAdmissionError("skill action targets a skill absent from H0")
    return (action_skill_id,)


def validate_trajectory_skill_invocations(
    *,
    retrieved_skill_ids: tuple[str, ...],
    active_skill_ids: tuple[str, ...],
    steps: tuple[object, ...],
    initial_meta: Mapping[str, JsonValue] | None = None,
) -> None:
    """Require every persisted edge to carry action-derived skill credit.

    ``steps`` is intentionally structural to avoid a circular import with the
    trajectory dataclass.  Each object must expose ``action_text``,
    ``invoked_skill_ids``, and ``observation_status``.
    """

    _require_skill_id_tuple(retrieved_skill_ids, field="retrieved_skill_ids")
    _require_skill_id_tuple(active_skill_ids, field="active_skill_ids")
    if not set(retrieved_skill_ids) <= set(active_skill_ids):
        raise ValueError("retrieved_skill_ids reference inactive skills")
    for position, step in enumerate(steps, start=1):
        action_text = getattr(step, "action_text", None)
        observed = getattr(step, "invoked_skill_ids", None)
        observation_status = getattr(step, "observation_status", None)
        if type(action_text) is not str or not isinstance(observed, tuple):
            raise TypeError("trajectory step lacks invocation admission fields")
        # Older codecs could reject a wrapper accepted by a later codec. Do not
        # reinterpret a recorded pre-dispatch rejection as a historical invocation.
        # Successful/dispatched edges still require exact action-derived identity.
        if observation_status in {"parse_error", "schema_invalid"} and observed == ():
            continue
        parsed = parse_action_invocation(action_text, initial_meta=initial_meta)
        if parsed is None:
            expected: tuple[str, ...] = ()
        else:
            try:
                expected = canonical_invoked_skill_ids(
                    action_kind=parsed.kind,
                    action_skill_id=parsed.skill_id,
                    retrieved_skill_ids=retrieved_skill_ids,
                    active_skill_ids=active_skill_ids,
                )
            except SkillInvocationAdmissionError as error:
                if observation_status != "schema_invalid" or observed != ():
                    raise ValueError(
                        f"trajectory step {position} contains unavailable skill credit"
                    ) from error
                continue
        if observed != expected:
            raise ValueError(f"trajectory step {position} invoked_skill_ids differ from its action")


__all__ = [
    "ParsedActionInvocation",
    "SkillInvocationAdmissionError",
    "canonical_invoked_skill_ids",
    "parse_action_invocation",
    "validate_trajectory_skill_invocations",
]
