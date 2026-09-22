"""Read-only discovery/read/outcome chain; no call quotas or causal claims."""

import math

from skillev.contracts import JsonValue
from skillev.runtime import SkillLibrary

from .posterior_state import PosteriorEventProvenance
from .skill_coverage import skill_coverage_report


def skill_lifecycle_report(
    library: SkillLibrary, provenance: PosteriorEventProvenance
) -> dict[str, JsonValue]:
    """Private report joins immutable documents to actual per-batch evidence.

    Catalogs are attributable to the sampled library, not the current library.
    Summaries/applicability make relevance inspectable, not automatically proven.
    Bodies returned by read_skill are distinct from later task actions/success.
    """
    catalogs: list[JsonValue] = []
    by_skill: dict[str, list[JsonValue]] = {}
    documents = library.all_documents()
    for batch in provenance.batches:
        for context in batch.trajectory_contexts:
            catalogs.append(
                {
                    "batch_id": batch.posterior.batch_id,
                    "optimizer_step": batch.optimizer_step,
                    "trajectory_id": context.trajectory_id,
                    "policy_snapshot_id": batch.policy_snapshot_id,
                    "library_version": batch.library_version,
                    "task_family": context.task_family,
                    "skill_exposure": context.skill_exposure,
                    "input_evidence": list(context.skill_input_evidence)
                    if context.skill_input_evidence is not None
                    else None,
                    "terminal_verifier_version": context.terminal_verifier_version,
                    "source": list(context.canonical_source_key)
                    if context.canonical_source_key is not None
                    else None,
                    "visible_ids": list(context.visible_skill_ids)
                    if context.visible_skill_ids is not None
                    else None,
                    "matched_ids": list(context.matched_skill_ids)
                    if context.matched_skill_ids is not None
                    else None,
                    "reads": [link.to_value() for link in context.invocation_links]
                    if context.invocation_links is not None
                    else None,
                }
            )
        for document in documents:
            skill_id = document.manifest.skill_id
            contexts = [c for c in batch.trajectory_contexts if skill_id in c.active_skill_ids]
            if not contexts:
                continue
            links = [
                link
                for c in contexts
                for link in c.invocation_links or ()
                if link.declared_skill_id == skill_id
            ]
            events = [e for e in batch.posterior.updates if e.skill_id == skill_id]
            links_known = all(c.invocation_links is not None for c in contexts)
            by_skill.setdefault(skill_id, []).append(
                {
                    "optimizer_step": batch.optimizer_step,
                    "library_version": batch.library_version,
                    "matched_trajectories": sum(
                        skill_id in (c.matched_skill_ids or ()) for c in contexts
                    )
                    if all(c.matched_skill_ids is not None for c in contexts)
                    else None,
                    "visible_trajectories": sum(
                        skill_id in (c.visible_skill_ids or ()) for c in contexts
                    )
                    if all(c.visible_skill_ids is not None for c in contexts)
                    else None,
                    "read_trajectory_count": sum(
                        any(
                            link.declared_skill_id == skill_id and link.body_returned is True
                            for link in c.invocation_links or ()
                        )
                        for c in contexts
                    )
                    if links_known
                    else None,
                    "read_source_question_count": len(
                        {
                            c.canonical_source_key
                            for c in contexts
                            if c.canonical_source_key is not None
                            and any(
                                link.declared_skill_id == skill_id and link.body_returned is True
                                for link in c.invocation_links or ()
                            )
                        }
                    )
                    if links_known and all(c.canonical_source_key is not None for c in contexts)
                    else None,
                    "repeat_same_version_read_count": sum(
                        link.repeat_same_version is True for link in links
                    )
                    if links_known and all(link.read_ordinal is not None for link in links)
                    else None,
                    "read_followed_by_task_action": sum(
                        link.body_returned is True and bool(link.following_task_action_steps)
                        for link in links
                    )
                    if links_known
                    and all(link.following_task_action_steps is not None for link in links)
                    else None,
                    "success_weight_mass": math.fsum(e.flow_weight for e in events if e.outcome),
                    "failure_weight_mass": math.fsum(
                        e.flow_weight for e in events if not e.outcome
                    ),
                    "body_read_count": sum(link.body_returned is True for link in links)
                    if links_known
                    else None,
                    "body_read_failed_count": sum(link.body_returned is False for link in links)
                    if links_known
                    else None,
                    "body_read_unknown_count": sum(link.body_returned is None for link in links)
                    if links_known
                    else None,
                    "admitted_call_count": sum(link.admitted for link in links)
                    if links_known
                    else None,
                    "calls_with_following_actions": sum(
                        link.admitted and bool(link.following_execution_steps) for link in links
                    )
                    if links_known
                    else None,
                    "reads_with_later_action": sum(
                        link.body_returned is True and bool(link.later_execution_steps)
                        for link in links
                    )
                    if links_known and all(link.later_execution_steps is not None for link in links)
                    else None,
                    "reads_visible_in_later_action": sum(
                        bool(link.body_visible_execution_steps) for link in links
                    )
                    if links_known
                    and all(link.body_visible_execution_steps is not None for link in links)
                    else None,
                    "body_visible_trajectories": sum(
                        skill_id in (c.body_visible_skill_ids or ()) for c in contexts
                    )
                    if all(c.body_visible_skill_ids is not None for c in contexts)
                    else None,
                    "body_visible_source_count": len(
                        {
                            c.canonical_source_key
                            for c in contexts
                            if skill_id in (c.body_visible_skill_ids or ())
                            and c.canonical_source_key is not None
                        }
                    )
                    if all(
                        c.body_visible_skill_ids is not None and c.canonical_source_key is not None
                        for c in contexts
                    )
                    else None,
                    "posterior_update_count": len(events),
                    "success_labeled_events": sum(e.outcome for e in events),
                    "failure_labeled_events": sum(not e.outcome for e in events),
                }
            )
    return {
        "coverage_maturity": skill_coverage_report(provenance),
        "format": "skill-discovery-evidence-chain@2",
        "interpretation": (
            "observed-discovery-and-temporal-following-not-causal-efficacy; no-minimum-call-count"
        ),
        "credit_rule": (
            "admitted-explicit-read/declaration; terminal-success-label; "
            "full-batch-mean-one-invocation-flow; not-independent-execution-or-causal-use"
        ),
        "visibility_rule": (
            "catalog@1/@2 require persisted admitted-input evidence; "
            "historical missing evidence stays unknown; "
            "full-inline legacy projection is not window evidence"
        ),
        "catalogs": catalogs,
        "skills": [
            {
                "skill_id": document.manifest.skill_id,
                "version": document.manifest.version,
                "title": document.title,
                "summary": document.summary,
                "applicability": document.applicability.to_value(),
                "active_now": document.manifest.skill_id in library.active_skill_ids,
                "batches": by_skill.get(document.manifest.skill_id, []),
                "availability_status": "available-in-training-batch"
                if document.manifest.skill_id in by_skill
                else "not-yet-available-in-a-training-batch",
                "evidence_status": "actual-invocation-observed"
                if any(
                    e.skill_id == document.manifest.skill_id
                    for batch in provenance.batches
                    for e in batch.posterior.updates
                )
                else "no-actual-invocation-evidence",
            }
            for document in documents
        ],
    }
