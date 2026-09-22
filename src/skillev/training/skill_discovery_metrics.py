"""Aggregate-only catalog/read telemetry from committed original records."""

from typing import cast

from skillev.contracts import JsonValue
from skillev.contracts.skill_invocation import parse_action_invocation

from .invocation_evidence import catalog_read_result
from .metrics_contract import _object, _objects


def skill_discovery_facts(records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    visible: set[str] = set()
    invoked: set[str] = set()
    visibility_known = invocation_known = True
    reads: list[bool | None] = []
    followed = later = 0
    for record in records:
        context = _object(record.get("initial_context", {}))
        ids = context.get("retrieved_skill_ids")
        if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
            visible.update(str(item) for item in ids)
        else:
            visibility_known = False
        meta = _object(context.get("meta", {}))
        steps = _objects(record["steps"])
        actions = [
            parse_action_invocation(str(edge.get("action_text", "")), initial_meta=meta)
            for edge in steps
        ]
        for index, (edge, action) in enumerate(zip(steps, actions, strict=True)):
            calls = edge.get("invoked_skill_ids")
            call_ids = (
                [item for item in calls if isinstance(item, str)] if isinstance(calls, list) else []
            )
            invocation_known = invocation_known and isinstance(calls, list)
            invoked.update(call_ids)
            if action is None or action.skill_id is None:
                continue
            received, _, _ = catalog_read_result(
                meta,
                action.skill_id,
                call_ids,
                str(edge.get("observation_status", "")),
                str(edge.get("observation_text", "")),
            )
            reads.append(received)
            if received is True:
                later += int(
                    any(a is not None and a.skill_id is None for a in actions[index + 1 :])
                )
            if received is True and index + 1 < len(steps):
                next_action = actions[index + 1]
                followed += int(next_action is not None and next_action.skill_id is None)
    return {
        "skill_visible_ids": cast(JsonValue, sorted(visible)) if visibility_known else None,
        "skill_invoked_ids": cast(JsonValue, sorted(invoked)) if invocation_known else None,
        "skill_visible_id_count": len(visible) if visibility_known else None,
        "skill_body_read_count": sum(item is True for item in reads),
        "skill_body_read_failed_count": sum(item is False for item in reads),
        "skill_body_read_unknown_count": sum(item is None for item in reads),
        "skill_read_followed_by_action_count": followed,
        "skill_reads_with_later_action": later,
        "skill_visibility_interpretation": "legacy-retrieved-id-count-not-admitted-input-evidence",
        "skill_body_visibility_interpretation": (
            "returned-body-not-causal-use; actual-input-evidence-in-posterior-lifecycle"
        ),
    }
