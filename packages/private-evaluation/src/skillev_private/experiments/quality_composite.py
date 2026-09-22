"""Explicit owner-selected T0 baseline from complete per-domain evidence.

This is not a rollout batch or same-condition joint collection. Original sampled
coordinates and records are preserved; only read-only quality metrics are joined.
It cannot supply optimizer or posterior evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from skillev.rollout import RolloutArtifact
from skillev.training.quality_gate import ProtocolProbe

from .quality_panel import FixedQualityPanel, _artifact_probe


@dataclass(frozen=True)
class DomainQualityEvidence:
    benchmark_id: str
    source_condition_id: str
    artifacts: tuple[RolloutArtifact, ...]
    events_path: Path
    event_run_id: str


def composite_baseline(
    evidence: tuple[DomainQualityEvidence, ...],
    *,
    panel: FixedQualityPanel,
    policy_snapshot_id: str,
    declaration_id: str,
    declaration_reason: str,
) -> ProtocolProbe:
    """Require all predefined slots, not best-of candidates or partial domains."""
    if not declaration_id or not declaration_reason:
        raise ValueError("a composite baseline needs an explicit owner declaration")
    expected_domains = {slot.benchmark_id for slot in panel.slots}
    if (
        len(evidence) != len(expected_domains)
        or {e.benchmark_id for e in evidence} != expected_domains
    ):
        raise ValueError("composite evidence must contain each declared domain exactly once")
    by_task = {}
    origins = {}
    source_notes = []
    for part in evidence:
        expected = {slot.task_id for slot in panel.slots if slot.benchmark_id == part.benchmark_id}
        if (
            not part.source_condition_id
            or not part.event_run_id
            or len(part.artifacts) != len(expected)
            or {a.manifest.task_id for a in part.artifacts} != expected
            or len({a.record.trajectory_id for a in part.artifacts}) != len(part.artifacts)
        ):
            raise ValueError("a domain must preserve its complete fixed source slots")
        for artifact in part.artifacts:
            by_task[artifact.manifest.task_id] = artifact
            origins[artifact.manifest.task_id] = (part.events_path, part.event_run_id)
        source_notes.append(
            {
                "benchmark": part.benchmark_id,
                "condition": part.source_condition_id,
                "trajectory_count": len(part.artifacts),
                "sampling_coordinates": [
                    a.manifest.sampling_coordinate.to_value()
                    if a.manifest.sampling_coordinate is not None
                    else None
                    for a in part.artifacts
                ],
            }
        )
    artifacts = tuple(by_task[slot.task_id] for slot in panel.slots)
    if len({a.manifest.library_version for a in artifacts}) != 1:
        raise ValueError("composite baseline cannot mix skill libraries")
    probe = _artifact_probe(
        artifacts,
        panel=panel,
        policy_snapshot_id=policy_snapshot_id,
        policy_step=0,
        evidence_id=declaration_id,
        origins=tuple(origins[slot.task_id] for slot in panel.slots),
    )
    return replace(
        probe,
        metric_notes={
            **probe.metric_notes,
            "baseline_kind": "owner-declared-composite-not-joint-batch",
            "declaration_reason": declaration_reason,
            "source_conditions_and_coordinates": json.dumps(source_notes, sort_keys=True),
            "comparison_limit": (
                "Different source conditions and sampling coordinates are retained; "
                "not a paired T0 or untouched IID estimate."
            ),
        },
    )
