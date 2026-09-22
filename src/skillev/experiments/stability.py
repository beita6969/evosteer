"""Aggregate multi-cycle library stability measurements.

Skill identities and content hashes are method metadata, not benchmark
instances.  No task text, answer, reward payload, or trajectory identifier is
accepted by these records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class LibraryCycleObservation:
    """One ordered active-set snapshot, including the pre-evolution seed state."""

    cycle_index: int
    library_version: str
    active_skills: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        if not self.library_version.strip():
            raise ValueError("library_version must be non-empty")
        if self.active_skills != tuple(sorted(self.active_skills)):
            raise ValueError("active_skills must be sorted by skill identity")
        skill_ids = tuple(skill_id for skill_id, _ in self.active_skills)
        if len(set(skill_ids)) != len(skill_ids):
            raise ValueError("active skill identities must be unique")
        if any(
            not skill_id.strip() or not content_hash.strip()
            for skill_id, content_hash in self.active_skills
        ):
            raise ValueError("active skill identity and content hash must be non-empty")


@dataclass(frozen=True, slots=True)
class LibraryTransitionMetric:
    cycle_index: int
    before_size: int
    after_size: int
    additions: int
    removals: int
    retained: int
    content_replacements: int
    churn_rate: float


@dataclass(frozen=True, slots=True)
class MultiCycleStabilitySummary:
    """Library growth/churn/oscillation evidence over multiple Phi commits."""

    library_change_count: int
    initial_active_count: int
    final_active_count: int
    maximum_active_count: int
    total_additions: int
    total_removals: int
    total_content_replacements: int
    reappeared_skill_count: int
    duplicate_active_content_count: int
    maximum_growth_factor: float
    transitions: tuple[LibraryTransitionMetric, ...]


def multi_cycle_stability(
    observations: tuple[LibraryCycleObservation, ...],
) -> MultiCycleStabilitySummary:
    """Summarize at least one committed library transition in exact order."""

    if not isinstance(observations, tuple) or len(observations) < 2:
        raise ValueError("stability analysis requires a seed and at least one evolved snapshot")
    if any(not isinstance(item, LibraryCycleObservation) for item in observations):
        raise TypeError("stability observations contain an incompatible value")
    expected_indices = tuple(range(len(observations)))
    if tuple(item.cycle_index for item in observations) != expected_indices:
        raise ValueError("cycle snapshots must be contiguous and start at zero")
    versions = tuple(item.library_version for item in observations)
    if len(set(versions)) != len(versions):
        raise ValueError("each committed library change must produce a new version")

    seen_ids = {skill_id for skill_id, _ in observations[0].active_skills}
    absent_ids: set[str] = set()
    reappeared_ids: set[str] = set()
    transitions: list[LibraryTransitionMetric] = []
    total_additions = 0
    total_removals = 0
    total_replacements = 0
    for before, after in pairwise(observations):
        before_map = dict(before.active_skills)
        after_map = dict(after.active_skills)
        before_ids = set(before_map)
        after_ids = set(after_map)
        added = after_ids - before_ids
        removed = before_ids - after_ids
        retained = before_ids & after_ids
        replacements = sum(before_map[skill_id] != after_map[skill_id] for skill_id in retained)
        reappeared_ids.update(added & absent_ids)
        absent_ids.update(removed)
        absent_ids.difference_update(after_ids)
        seen_ids.update(added)
        total_additions += len(added)
        total_removals += len(removed)
        total_replacements += replacements
        denominator = max(1, len(before_ids | after_ids))
        transitions.append(
            LibraryTransitionMetric(
                cycle_index=after.cycle_index,
                before_size=len(before_ids),
                after_size=len(after_ids),
                additions=len(added),
                removals=len(removed),
                retained=len(retained),
                content_replacements=replacements,
                churn_rate=(len(added) + len(removed) + replacements) / denominator,
            )
        )

    sizes = tuple(len(item.active_skills) for item in observations)
    initial_denominator = max(1, sizes[0])
    duplicate_active_content_count = sum(
        len(snapshot.active_skills)
        - len({content_hash for _, content_hash in snapshot.active_skills})
        for snapshot in observations
    )
    growth = max(sizes) / initial_denominator
    if not math.isfinite(growth):
        raise ValueError("library growth factor must be finite")
    return MultiCycleStabilitySummary(
        library_change_count=len(observations) - 1,
        initial_active_count=sizes[0],
        final_active_count=sizes[-1],
        maximum_active_count=max(sizes),
        total_additions=total_additions,
        total_removals=total_removals,
        total_content_replacements=total_replacements,
        reappeared_skill_count=len(reappeared_ids),
        duplicate_active_content_count=duplicate_active_content_count,
        maximum_growth_factor=growth,
        transitions=tuple(transitions),
    )


__all__ = [
    "LibraryCycleObservation",
    "LibraryTransitionMetric",
    "MultiCycleStabilitySummary",
    "multi_cycle_stability",
]
