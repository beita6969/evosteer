"""Committed exposure-to-invocation coverage, without changing evidence weights."""

from collections import defaultdict
from collections.abc import Iterable
from typing import cast

from skillev.contracts import JsonValue

from .evidence_context import TrajectoryEvidenceContext
from .posterior_state import PosteriorEvidenceBatch


def _skill_ids(values: Iterable[str]) -> list[JsonValue]:
    return cast(list[JsonValue], sorted(set(values)))


def batch_coverage(batch: PosteriorEvidenceBatch) -> dict[str, JsonValue]:
    groups: dict[str, list[TrajectoryEvidenceContext]] = defaultdict(list)
    for context in batch.trajectory_contexts:
        groups[context.benchmark_id or "unknown"].append(context)
    benchmarks: dict[str, JsonValue] = {}
    for name, contexts in sorted(groups.items()):
        ids = {context.trajectory_id for context in contexts}
        events = [event for event in batch.posterior.updates if event.trajectory_id in ids]
        edge_count = sum(context.executed_edge_count for context in contexts)
        invoking = sum(context.invoking_edge_count for context in contexts)
        links = [link for context in contexts for link in context.invocation_links or ()]
        links_known = all(context.invocation_links is not None for context in contexts)
        input_rows = [
            row
            for context in contexts
            for row in context.skill_input_evidence or ()
            if row.get("phase") == "action"
        ]
        input_known = all(context.skill_input_evidence is not None for context in contexts)
        visibility_known = all(c.visible_skill_ids is not None for c in contexts)
        body_visibility_known = all(c.body_visible_skill_ids is not None for c in contexts)
        body_receipts_known = links_known and all(link.body_returned is not None for link in links)
        benchmarks[name] = {
            "canonical_source_question_count": len(
                {c.canonical_source_key for c in contexts if c.canonical_source_key is not None}
            ),
            "canonical_source_unknown_trajectory_count": sum(
                c.canonical_source_key is None for c in contexts
            ),
            "reads_with_later_action": sum(
                link.body_returned is True and bool(link.later_execution_steps) for link in links
            )
            if links_known and all(link.later_execution_steps is not None for link in links)
            else None,
            "reads_visible_in_later_action": sum(
                bool(link.body_visible_execution_steps) for link in links
            )
            if links_known and all(link.body_visible_execution_steps is not None for link in links)
            else None,
            "body_not_fully_visible_input_count": sum(
                bool(row.get("not_fully_visible_skill_body_refs")) for row in input_rows
            )
            if input_known
            and all(
                isinstance(row.get("not_fully_visible_skill_body_refs"), list) for row in input_rows
            )
            else None,
            "read_source_question_count": len(
                {
                    c.canonical_source_key
                    for c in contexts
                    if any(link.body_returned is True for link in c.invocation_links or ())
                    and c.canonical_source_key is not None
                }
            )
            if links_known and all(c.canonical_source_key is not None for c in contexts)
            else None,
            "body_read_count": sum(link.body_returned is True for link in links)
            if body_receipts_known
            else None,
            "read_trajectory_count": sum(
                any(link.body_returned is True for link in c.invocation_links or ())
                for c in contexts
            )
            if links_known
            else None,
            "rejected_declaration_count": sum(
                not link.admitted for context in contexts for link in context.invocation_links or ()
            ),
            "admitted_without_following_execution_count": sum(
                link.admitted and not link.following_execution_steps
                for context in contexts
                for link in context.invocation_links or ()
            ),
            "admitted_terminal_failure_count": sum(
                link.admitted and not link.terminal_success
                for context in contexts
                for link in context.invocation_links or ()
            ),
            "exposed_but_uncalled_trajectory_count": sum(
                bool(context.visible_skill_ids) and context.invoking_edge_count == 0
                for context in contexts
            ),
            "invocation_link_unknown_trajectory_count": sum(
                context.invocation_links is None for context in contexts
            ),
            "trajectory_count": len(contexts),
            "distinct_source_question_count": len(
                {context.source_key for context in contexts if context.source_key is not None}
            ),
            "source_unknown_trajectory_count": sum(
                context.source_key is None for context in contexts
            ),
            "matched_skill_ids": _skill_ids(
                {skill for context in contexts for skill in context.matched_skill_ids or ()}
            ),
            "match_unknown_trajectory_count": sum(
                context.matched_skill_ids is None for context in contexts
            ),
            "retrieved_skill_ids": _skill_ids(
                {skill for context in contexts for skill in context.retrieved_skill_ids}
            ),
            "catalog_or_inline_visible_skill_ids": _skill_ids(
                skill for context in contexts for skill in context.visible_skill_ids or ()
            )
            if visibility_known
            else None,
            "catalog_or_inline_visible_trajectory_count": sum(
                bool(context.visible_skill_ids) for context in contexts
            )
            if visibility_known
            else None,
            "body_visible_skill_ids": _skill_ids(
                skill for context in contexts for skill in context.body_visible_skill_ids or ()
            )
            if body_visibility_known
            else None,
            "body_visible_trajectory_count": sum(
                bool(context.body_visible_skill_ids) for context in contexts
            )
            if body_visibility_known
            else None,
            "body_visibility_unknown_trajectory_count": sum(
                context.body_visible_skill_ids is None for context in contexts
            ),
            "body_read_failed_count": sum(
                link.body_returned is False
                for context in contexts
                for link in context.invocation_links or ()
            ),
            "visibility_unknown_trajectory_count": sum(
                context.visible_skill_ids is None for context in contexts
            ),
            "invoked_skill_ids": _skill_ids(event.skill_id for event in events),
            "invoking_trajectory_count": sum(
                context.invoking_edge_count > 0 for context in contexts
            ),
            "invoking_edge_count": invoking,
            "execution_edge_count": edge_count,
            "invocation_edge_fraction": invoking / edge_count if edge_count else 0.0,
            "posterior_update_event_count": len(events),
            "updated_cell_count": len({event.z.cell_key(event.skill_id) for event in events}),
            "posterior_weight_mass": sum(event.flow_weight for event in events),
        }
    return {
        "format": "skill-coverage-report@2",
        "visibility_interpretation": (
            "unknown-input-evidence-is-null; "
            "legacy-full-inline-projection-not-admitted-window-evidence"
        ),
        "batch_id": batch.posterior.batch_id,
        "optimizer_step": batch.optimizer_step,
        "policy_snapshot_id": batch.policy_snapshot_id,
        "library_version": batch.library_version,
        "trajectory_count": len(batch.trajectory_ids),
        "metadata_missing_trajectory_count": len(batch.trajectory_ids)
        - len(batch.trajectory_contexts),
        "benchmarks": benchmarks,
        "credit_unit": "actual-invocation-only-not-retrieval-or-exposure",
    }
