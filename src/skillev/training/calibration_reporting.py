"""Read-only pre-batch predictions with post-hoc features, not causal credit.

All events of a cell in one batch use its first before-cell. Grouped errors and
weight concentration describe dependence; they do not change the Beta update,
assert independent samples, or reinterpret mean +/- k*sigma as a 95% interval.
"""

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean, pstdev

from skillev.contracts import JsonValue, PosteriorUpdateEvent

from .evidence_context import TrajectoryEvidenceContext
from .posterior_state import PosteriorEventProvenance


def _group_errors(groups: Sequence[Sequence[float]]) -> dict[str, JsonValue]:
    means = [fmean(values) for values in groups]
    return {
        "group_count": len(means),
        "equal_group_mean_brier": fmean(means) if means else None,
        "between_group_brier_std": pstdev(means) if means else None,
        "interpretation": "descriptive-group-dispersion-not-confidence-interval",
    }


def _summary(
    events: Sequence[PosteriorUpdateEvent],
    probabilities: dict[str, float],
    contexts: dict[str, TrajectoryEvidenceContext],
) -> dict[str, JsonValue]:
    errors = [
        (probabilities[event.z.cell_key(event.skill_id)] - event.outcome) ** 2 for event in events
    ]
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    by_source: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_canonical_source: dict[tuple[str, str], list[float]] = defaultdict(list)
    unknown_sources: set[str] = set()
    for event, error in zip(events, errors, strict=True):
        by_trajectory[event.trajectory_id].append(error)
        context = contexts.get(event.trajectory_id)
        key = context.source_key if context else None
        if key is None:
            unknown_sources.add(event.trajectory_id)
        else:
            by_source[key].append(error)
        if context is not None and context.canonical_source_key is not None:
            by_canonical_source[context.canonical_source_key].append(error)
    weights = [event.flow_weight for event in events]
    mass = sum(weights)
    shares = [weight / mass for weight in weights] if mass else []
    return {
        "invocation_event_count": len(events),
        "distinct_trajectory_count": len(by_trajectory),
        "distinct_source_question_count": len(by_source),
        "canonical_source_question_count": len(by_canonical_source),
        "canonical_source_grouped_error": _group_errors(list(by_canonical_source.values())),
        "prediction_available_event_count": len(events),
        "prediction_unavailable_event_count": 0,
        "source_unknown_trajectory_count": len(unknown_sources),
        "terminal_success_fraction": fmean(event.outcome for event in events) if events else None,
        "brier_score": fmean(errors) if errors else None,
        "success_weight": sum(event.flow_weight for event in events if event.outcome),
        "failure_weight": sum(event.flow_weight for event in events if not event.outcome),
        "weight_effective_event_count": 1 / sum(share**2 for share in shares) if shares else None,
        "effective_count_interpretation": "weight-concentration-only-not-independent-sample-count",
        "cumulative_flow_weight": mass,
        "largest_event_weight_share": max(shares) if shares else None,
        "event_weight_concentration": sum(share**2 for share in shares) if shares else None,
        "trajectory_grouped_error": _group_errors(list(by_trajectory.values())),
        "source_question_grouped_error": _group_errors(list(by_source.values())),
    }


def prequential_calibration(provenance: PosteriorEventProvenance) -> list[JsonValue]:
    reports: list[JsonValue] = []
    seen_sources: set[tuple[str, str]] = set()
    for batch in provenance.batches:
        grouped: dict[str, list[PosteriorUpdateEvent]] = defaultdict(list)
        by_benchmark: dict[str, list[PosteriorUpdateEvent]] = defaultdict(list)
        contexts = {item.trajectory_id: item for item in batch.trajectory_contexts}
        for event in batch.posterior.updates:
            grouped[event.z.cell_key(event.skill_id)].append(event)
            context = contexts.get(event.trajectory_id)
            by_benchmark[(context.benchmark_id if context else None) or "unknown"].append(event)
        probabilities = {
            key: events[0].alpha_before / (events[0].alpha_before + events[0].beta_count_before)
            for key, events in grouped.items()
        }
        first_source_events = [
            event
            for event in batch.posterior.updates
            if (context := contexts.get(event.trajectory_id)) is not None
            and context.canonical_source_key is not None
            and context.canonical_source_key not in seen_sources
        ]
        reliability: list[JsonValue] = []
        for bin_index in range(10):
            members = [
                event
                for event in first_source_events
                if min(9, int(probabilities[event.z.cell_key(event.skill_id)] * 10)) == bin_index
            ]
            if members:
                reliability.append(
                    {
                        "bin": bin_index,
                        "event_count": len(members),
                        "mean_prediction": fmean(
                            probabilities[event.z.cell_key(event.skill_id)] for event in members
                        ),
                        "success_fraction": fmean(event.outcome for event in members),
                    }
                )
        cells: list[JsonValue] = []
        for key, events in grouped.items():
            first = events[0]
            probability = probabilities[key]
            cells.append(
                {
                    "skill_id": first.skill_id,
                    "context": first.z.to_value(),
                    "pre_batch_mean": probability,
                    "probability_bin": min(9, int(probability * 10)),
                    **_summary(events, probabilities, contexts),
                }
            )
        reports.append(
            {
                "format": "prequential-calibration-report@2",
                "batch_id": batch.posterior.batch_id,
                "optimizer_step": batch.optimizer_step,
                "policy_snapshot_id": batch.policy_snapshot_id,
                "library_version": batch.library_version,
                "source_disjoint_from_prediction_history": {
                    "summary": _summary(first_source_events, probabilities, contexts),
                    "reliability": reliability,
                    "scope": "first-seen-known-canonical-source-prequential@2-not-fixed-holdout",
                    "excluded_event_count": len(batch.posterior.updates) - len(first_source_events),
                },
                "prediction_timing": "pre-batch-posterior-post-hoc-features",
                "unit": "invocation-event-not-independent-sample",
                "cells": cells,
                "benchmarks": {
                    name: _summary(events, probabilities, contexts)
                    for name, events in sorted(by_benchmark.items())
                },
            }
        )
        seen_sources.update(
            context.canonical_source_key
            for context in contexts.values()
            if context.canonical_source_key is not None
        )
    return reports


def posterior_evidence_composition(provenance: PosteriorEventProvenance) -> list[JsonValue]:
    """Cross-version support, not a claim of stationary current-policy accuracy."""
    groups: dict[
        str, list[tuple[PosteriorUpdateEvent, str, str, TrajectoryEvidenceContext | None]]
    ] = defaultdict(list)
    for batch in provenance.batches:
        contexts = {c.trajectory_id: c for c in batch.trajectory_contexts}
        for event in batch.posterior.updates:
            groups[event.z.cell_key(event.skill_id)].append(
                (
                    event,
                    batch.policy_snapshot_id,
                    batch.library_version,
                    contexts.get(event.trajectory_id),
                )
            )
    result: list[JsonValue] = []
    for rows in groups.values():
        weights = [row[0].flow_weight for row in rows]
        mass = sum(weights)
        versions: dict[tuple[str, str, str | None], list[PosteriorUpdateEvent]] = defaultdict(list)
        for event, policy, library, context in rows:
            versions[
                policy, library, context.terminal_verifier_version if context else None
            ].append(event)
        first = rows[0][0]
        result.append(
            {
                "skill_id": first.skill_id,
                "context": first.z.to_value(),
                "event_count": len(rows),
                "trajectory_count": len({row[0].trajectory_id for row in rows}),
                "canonical_source_question_count": len(
                    {
                        c.canonical_source_key
                        for _, _, _, c in rows
                        if c is not None and c.canonical_source_key is not None
                    }
                ),
                "source_unknown_event_count": sum(
                    c is None or c.canonical_source_key is None for _, _, _, c in rows
                ),
                "success_weight": sum(e.flow_weight for e, _, _, _ in rows if e.outcome),
                "failure_weight": sum(e.flow_weight for e, _, _, _ in rows if not e.outcome),
                "largest_event_weight_share": max(weights) / mass if mass else None,
                "versions": [
                    {
                        "policy_snapshot_id": policy,
                        "library_version": library,
                        "terminal_verifier_version": v,
                        "event_count": len(events),
                        "success_weight": sum(e.flow_weight for e in events if e.outcome),
                        "failure_weight": sum(e.flow_weight for e in events if not e.outcome),
                    }
                    for (policy, library, v), events in versions.items()
                ],
                "interpretation": (
                    "historical-version-mixture-not-current-policy-calibration-"
                    "or-independent-samples"
                ),
            }
        )
    return result
