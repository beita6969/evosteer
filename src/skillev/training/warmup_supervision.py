"""Read-only action supervision accounting and attributable public-behavior review.

Availability/temporal order are mechanical evidence, not proof of application or
causal benefit. The review remains attributed to its human/model reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillev.rollout import RolloutArtifact

from .invocation_evidence import invocation_execution_links

if TYPE_CHECKING:
    from .skill_use_warmup import WarmupCorpus, WarmupDemonstration

ANNOTATION_FORMAT = "skill-application-annotation@2"
CATEGORIES = (
    "actual_read_skill",
    "reviewed_body_visible_application",
    "reviewed_no_read_direct",
    "other_pre_read_or_unconfirmed",
)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_annotation(example: WarmupDemonstration) -> dict[str, Any]:
    """New inputs read their real reference; saved inputs reuse preserved contents."""
    value = example.annotation
    if value is None:
        value = json.loads(Path(example.annotation_ref).read_text())
    if not isinstance(value, dict):
        raise ValueError("application annotation must be a readable JSON object")
    return value


def validate_application_annotation(
    note: dict[str, Any], artifact: RolloutArtifact
) -> dict[str, Any]:
    """Shared @2 behavior contract. No native-success or warmup approval inference."""
    if (
        note.get("format") != ANNOTATION_FORMAT
        or note.get("review_kind") not in {"human", "model-assisted-public-behavior"}
        or any(not _text(note.get(k)) for k in ("reviewer", "reviewed_at", "rationale"))
        or note.get("trajectory_id") != artifact.record.trajectory_id
        or note.get("decision") not in {"applied", "not-called", "abandoned-after-read"}
    ):
        raise ValueError("structured attributable behavior annotation@2 is required")
    evidence = note.get("behavior_evidence")
    if not isinstance(evidence, dict) or type(evidence.get("ceremonial_read")) is not bool:
        raise ValueError("behavior review must explicitly address ceremonial reading")
    limits = evidence.get("attribution_limits")
    clauses, actions = evidence.get("method_clauses"), evidence.get("actions")
    if not isinstance(limits, list) or not limits or not all(_text(v) for v in limits):
        raise ValueError("review attribution limitations must be explicit")
    if not isinstance(clauses, list) or not isinstance(actions, list) or not actions:
        raise ValueError("method clauses and actual reviewed action steps are required")
    clause_map = {}
    for clause in clauses:
        if not isinstance(clause, dict) or not all(
            _text(clause.get(k)) for k in ("clause_id", "quote")
        ):
            raise ValueError("method clause needs a named quote from the returned body")
        if clause["clause_id"] in clause_map:
            raise ValueError("method clause identity repeats")
        clause_map[clause["clause_id"]] = clause["quote"]
    links = invocation_execution_links(artifact.record, artifact.skill_input_evidence)
    decision = note["decision"]
    visible: set[int] = set()
    if decision == "not-called":
        if (
            links
            or any(s.invoked_skill_ids for s in artifact.record.steps)
            or any(note.get(k) is not None for k in ("skill_id", "read_step"))
        ):
            raise ValueError("no-read review conflicts with actual invocation")
        if clauses or evidence["ceremonial_read"]:
            raise ValueError("no-read evidence cannot claim a read procedure")
    else:
        match = next(
            (
                v
                for v in links
                if v.step_index == note.get("read_step")
                and v.declared_skill_id == note.get("skill_id")
            ),
            None,
        )
        if match is None or not match.admitted or not match.body_returned:
            raise ValueError("method review needs the actual admitted body-returning read")
        visible = set(match.body_visible_execution_steps or ())
        read_step = next(s for s in artifact.record.steps if s.index == match.step_index)
        body = json.loads(read_step.observation_text)["content"]
        if not isinstance(body, str):
            raise ValueError("returned skill content must be public text")
        texts = [body]
        # Catalog bodies are themselves JSON: a multiline instruction is escaped
        # in that carrier, but reviews can quote its actual decoded public text.
        try:
            document = json.loads(body)
        except ValueError:
            document = None
        if isinstance(document, dict) and isinstance(document.get("instructions"), str):
            texts.append(document["instructions"])
        if not clause_map or any(
            not any(quote in text for text in texts) for quote in clause_map.values()
        ):
            raise ValueError("method quotes do not refer to the actual returned body")
    actual_steps = {s.index for s in artifact.record.steps}
    reviewed = set()
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("reviewed action evidence must be structured")
        step, refs = action.get("step_index"), action.get("clause_ids")
        if (
            type(step) is not int
            or step not in actual_steps
            or step in reviewed
            or not _text(action.get("behavior"))
        ):
            raise ValueError("review must identify distinct actual action steps and behavior")
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or ref not in clause_map for ref in refs
        ):
            raise ValueError("reviewed action cites an unknown method clause")
        if decision != "not-called" and (step not in visible or not refs):
            raise ValueError(
                "reviewed method action must follow a body visible in its actual input"
            )
        reviewed.add(step)
    if decision != "not-called" and note.get("decision_step") not in reviewed:
        raise ValueError("declared application/abandonment step lacks behavior evidence")
    return note


def supervision_report(corpus: WarmupCorpus) -> dict[str, Any]:
    """Partition ORIGINAL F action tokens only. No score, filtering or weighting."""
    total: dict[str, dict[str, Any]] = {
        name: {"actions": 0, "action_tokens": 0} for name in CATEGORIES
    }
    trajectories = []
    positives = set()
    complete_reviews = True
    for example in corpus.demonstrations:
        links = invocation_execution_links(
            example.artifact.record, example.artifact.skill_input_evidence
        )
        reads = {v.step_index for v in links if v.read_ordinal is not None}
        visible = set().union(*(set(v.body_visible_execution_steps or ()) for v in links))
        note, status, error = None, "structured-review", None
        try:
            loaded = load_annotation(example)
            if loaded.get("format") != ANNOTATION_FORMAT:
                status = "legacy-review-not-structured-application-evidence"
            else:
                note = validate_application_annotation(loaded, example.artifact)
        except (OSError, ValueError, TypeError, KeyError) as failure:
            status, error = "missing-or-invalid-review", type(failure).__name__
        complete_reviews &= note is not None
        applied, direct = set(), set()
        if note is not None:
            indices = {row["step_index"] for row in note["behavior_evidence"]["actions"]}
            if note["decision"] == "applied" and not note["behavior_evidence"]["ceremonial_read"]:
                applied = indices
                positives.add(corpus.canonical(example.source))
            elif note["decision"] == "not-called":
                direct = indices
        counts = {name: {"actions": 0, "action_tokens": 0} for name in CATEGORIES}
        for step in example.artifact.record.steps:
            category = (
                "actual_read_skill"
                if step.index in reads
                else "reviewed_body_visible_application"
                if step.index in applied
                else "reviewed_no_read_direct"
                if step.index in direct
                else "other_pre_read_or_unconfirmed"
            )
            counts[category]["actions"] += 1
            counts[category]["action_tokens"] += len(step.action_token_ids)
        for name in CATEGORIES:
            for key in ("actions", "action_tokens"):
                total[name][key] += counts[name][key]
        trajectories.append(
            {
                "trajectory_id": example.artifact.record.trajectory_id,
                "canonical_source": list(corpus.canonical(example.source)),
                "annotation_ref": example.annotation_ref,
                "annotation_status": status,
                "annotation_error_type": error,
                "counts": counts,
                "body_visible_following_actions": len(visible),
                "reviewer": note.get("reviewer") if note else None,
                "review_kind": note.get("review_kind") if note else None,
                "ceremonial_read": note["behavior_evidence"]["ceremonial_read"] if note else None,
            }
        )
    actions = sum(row["actions"] for row in total.values())
    tokens = sum(row["action_tokens"] for row in total.values())
    for row in total.values():
        row["action_fraction"] = row["actions"] / actions if actions else None
        row["action_token_fraction"] = row["action_tokens"] / tokens if tokens else None
    return {
        "format": "skill-warmup-supervision-composition@1",
        "categories": total,
        "original_action_count": actions,
        "original_action_tokens": tokens,
        "trajectories": trajectories,
        "complete_structured_reviews": complete_reviews,
        "reviewed_independent_positive_sources": len(positives),
        "multiple_independent_positives": len(positives) >= 2,
        "loss": "unchanged-original-action-token-mean-NLL",
        "weights_applied": False,
        "application_evidence": "attributed-review-plus-measured-body-visibility; not causal proof",
        "unreviewed_application_count": None,
        "read_or_success_implies_application": False,
    }
