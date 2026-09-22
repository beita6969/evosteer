"""Natural-reference task anchors and a separate lagged structural channel.

Task baselines may use the current batch with task-level leave-one-out. Structure
buckets always come from committed earlier batches. Contexts are deliberately
not pooled: callers create a new store when a reference context changes. By
default the structural channel includes paired reference prefixes, including
forced prefixes, matching Appendix C.3's explicit implementation exception.
Those interventions are not unconditional reference samples: this pooling is
an approximation, not an unbiased estimate or independent task evidence.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from skillev.contracts.canonical import canonical_json
from skillev.scoring.anchor_tb import log_terminal_reward


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True, slots=True)
class ReferenceStatisticsConfig:
    beta: float = 1.0
    kappa_global: float = 2.0
    kappa_category: float = 4.0
    kappa_task: float = 0.5
    prior_count_cap: float = 20.0
    structure_minimum: int = 8
    structure_pseudocount: float = 8.0
    correction_bound: float = 0.25
    structure_source_mode: str = "reference_with_paired"

    def __post_init__(self) -> None:
        for name in (
            "beta",
            "kappa_global",
            "kappa_category",
            "kappa_task",
            "prior_count_cap",
            "structure_pseudocount",
            "correction_bound",
        ):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.structure_minimum) is not int or self.structure_minimum < 1:
            raise ValueError("structure_minimum must be a positive integer")
        if self.structure_source_mode not in {"reference_with_paired", "natural_only"}:
            raise ValueError("unknown structural reference source mode")


@dataclass(frozen=True, slots=True)
class StructureVisit:
    graph_key: str
    node_count: int
    edge_count: int
    stopped: bool = False

    def __post_init__(self) -> None:
        _text(self.graph_key, "graph_key")
        if any(type(v) is not int or v < 0 for v in (self.node_count, self.edge_count)):
            raise ValueError("graph counts must be nonnegative integers")
        if type(self.stopped) is not bool:
            raise ValueError("stopped must be boolean")

    @property
    def exact_key(self) -> tuple[str, str, int, int, bool]:
        return ("exact", self.graph_key, self.node_count, self.edge_count, self.stopped)

    @property
    def coarse_key(self) -> tuple[str, str, int, int, bool]:
        return ("coarse", "", self.node_count, self.edge_count, self.stopped)


@dataclass(frozen=True, slots=True)
class ReferenceObservation:
    observation_id: str
    task_id: str
    task_family: str
    reward: float
    context_version: str
    visits: tuple[StructureVisit, ...] = ()
    prior_rate: float | None = None
    prior_count: float = 0.0
    source: str = "natural_reference"
    healthy: bool = True
    legal: bool = True

    def __post_init__(self) -> None:
        for name in ("observation_id", "task_id", "task_family", "context_version"):
            _text(getattr(self, name), name)
        if not 0 <= _number(self.reward, "reward") <= 1:
            raise ValueError("reward must lie in [0, 1]")
        if self.source != "natural_reference" or self.healthy is not True or self.legal is not True:
            raise ValueError("anchors admit only healthy legal natural-reference observations")
        if not isinstance(self.visits, tuple) or any(
            not isinstance(visit, StructureVisit) for visit in self.visits
        ):
            raise ValueError("visits must be an immutable tuple of StructureVisit")
        _validate_prior(self.prior_rate, self.prior_count)


@dataclass(frozen=True, slots=True)
class StructureReferenceObservation:
    """One trajectory's prefix visits, isolated from all task shrinkage counts.

    Repeated structures within a trajectory are distinct prefix visits. Paired
    records explicitly retain their first forced prefix and intervention IDs.
    """

    observation_id: str
    task_id: str
    task_family: str
    reward: float
    context_version: str
    visits: tuple[StructureVisit, ...]
    source: str = "natural_reference"
    pair_id: str | None = None
    candidate_id: str | None = None
    forced_prefix_indices: tuple[int, ...] = ()
    healthy: bool = True
    legal: bool = True
    prior_rate: float | None = None
    prior_count: float = 0.0

    def __post_init__(self) -> None:
        for name in ("observation_id", "task_id", "task_family", "context_version"):
            _text(getattr(self, name), name)
        if not 0 <= _number(self.reward, "reward") <= 1:
            raise ValueError("structural reference reward must lie in [0, 1]")
        if self.source not in {"natural_reference", "paired_treatment", "paired_control"}:
            raise ValueError("structural observations require a reference continuation")
        if self.healthy is not True or self.legal is not True:
            raise ValueError("structural observations require healthy legal continuations")
        if not isinstance(self.visits, tuple) or any(
            not isinstance(v, StructureVisit) for v in self.visits
        ):
            raise ValueError("structural visits must be an immutable typed tuple")
        indices = self.forced_prefix_indices
        if (
            not isinstance(indices, tuple)
            or any(type(index) is not int or not 0 <= index < len(self.visits) for index in indices)
            or indices != tuple(sorted(set(indices)))
        ):
            raise ValueError("forced prefix indices must identify unique recorded visits")
        if self.source.startswith("paired_"):
            _text(self.pair_id, "pair_id")
            _text(self.candidate_id, "candidate_id")
            if indices != (0,):
                raise ValueError("paired structural evidence must retain its forced first prefix")
        elif self.pair_id is not None or self.candidate_id is not None or indices:
            raise ValueError("natural structural evidence cannot carry a paired intervention")
        _validate_prior(self.prior_rate, self.prior_count)


def _natural_structure(row: ReferenceObservation) -> StructureReferenceObservation:
    return StructureReferenceObservation(
        row.observation_id,
        row.task_id,
        row.task_family,
        row.reward,
        row.context_version,
        row.visits,
        prior_rate=row.prior_rate,
        prior_count=row.prior_count,
    )


def _validate_prior(rate: float | None, count: float) -> None:
    if _number(count, "prior_count") < 0:
        raise ValueError("prior_count cannot be negative")
    if rate is None:
        if count != 0:
            raise ValueError("positive prior_count requires prior_rate")
    elif not 0 <= _number(rate, "prior_rate") <= 1:
        raise ValueError("prior_rate must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class StructureBucket:
    key: tuple[str, str, int, int, bool]
    visits: int
    excess_ratio_sum: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, tuple)
            or len(self.key) != 5
            or self.key[0] not in {"exact", "coarse"}
            or not isinstance(self.key[1], str)
            or any(type(v) is not int or v < 0 for v in self.key[2:4])
            or type(self.key[4]) is not bool
            or type(self.visits) is not int
            or self.visits < 1
        ):
            raise ValueError("invalid structure bucket")
        if _number(self.excess_ratio_sum, "excess ratio") <= -self.visits:
            raise ValueError("reference ratios must be positive")


@dataclass(frozen=True, slots=True)
class ReferenceStatisticsSnapshot:
    context_version: str
    batch_id: str
    generation: int
    config: ReferenceStatisticsConfig
    observations: tuple[ReferenceObservation, ...]
    current_observation_ids: tuple[str, ...]
    structure_buckets: tuple[StructureBucket, ...]
    structure_observations: tuple[StructureReferenceObservation, ...] = ()

    def task_rate(
        self,
        task_id: str,
        task_family: str,
        *,
        exclude_observation_id: str | None = None,
        prior_rate: float | None = None,
        prior_count: float = 0.0,
    ) -> float:
        """Appendix B hierarchy; only the same-task empirical term is LOO."""
        _text(task_id, "task_id")
        _text(task_family, "task_family")
        _validate_prior(prior_rate, prior_count)
        all_values = self.observations
        same_family = tuple(row for row in all_values if row.task_family == task_family)
        same_task = tuple(row for row in all_values if row.task_id == task_id)
        if any(row.task_family != task_family for row in same_task):
            raise ValueError("task identity belongs to another category")
        if exclude_observation_id is not None:
            excluded = tuple(
                row for row in same_task if row.observation_id == exclude_observation_id
            )
            if len(excluded) != 1:
                raise ValueError("leave-one-out observation must belong to this task")
        if prior_rate is None:
            priors = {
                (row.prior_rate, row.prior_count) for row in same_task if row.prior_rate is not None
            }
            if len(priors) > 1:
                raise ValueError("same-task records disagree on the supplied prior")
            if priors:
                prior_rate, prior_count = next(iter(priors))
        global_mean = (
            math.fsum(row.reward for row in all_values) / len(all_values) if all_values else 0.5
        )
        category_mean = (
            math.fsum(row.reward for row in same_family) + self.config.kappa_global * global_mean
        ) / (len(same_family) + self.config.kappa_global)
        effective_count = min(prior_count, self.config.prior_count_cap)
        prior_mean = (
            effective_count * (prior_rate if prior_rate is not None else category_mean)
            + self.config.kappa_category * category_mean
        ) / (effective_count + self.config.kappa_category)
        retained = tuple(row for row in same_task if row.observation_id != exclude_observation_id)
        rate = (math.fsum(row.reward for row in retained) + self.config.kappa_task * prior_mean) / (
            len(retained) + self.config.kappa_task
        )
        return min(1.0, max(0.0, rate))

    def task_anchor(
        self,
        task_id: str,
        task_family: str,
        *,
        exclude_observation_id: str | None = None,
        prior_rate: float | None = None,
        prior_count: float = 0.0,
    ) -> float:
        return log_terminal_reward(
            self.task_rate(
                task_id,
                task_family,
                exclude_observation_id=exclude_observation_id,
                prior_rate=prior_rate,
                prior_count=prior_count,
            ),
            self.config.beta,
        )

    def structure_correction(self, visit: StructureVisit) -> float:
        buckets = {bucket.key: bucket for bucket in self.structure_buckets}
        bucket = buckets.get(visit.exact_key)
        if bucket is None or bucket.visits < self.config.structure_minimum:
            bucket = buckets.get(visit.coarse_key)
        if bucket is None or bucket.visits < self.config.structure_minimum:
            return 0.0
        value = math.log1p(
            bucket.excess_ratio_sum / (bucket.visits + self.config.structure_pseudocount)
        )
        return min(self.config.correction_bound, max(-self.config.correction_bound, value))

    def measured_flow(
        self, task_id: str, task_family: str, visit: StructureVisit, **kwargs: Any
    ) -> float:
        return min(
            self.config.beta,
            max(
                0.0,
                self.task_anchor(task_id, task_family, **kwargs) + self.structure_correction(visit),
            ),
        )


class ReferenceStatistics:
    """A context-specific store committed once after each successful batch."""

    def __init__(
        self, context_version: str, config: ReferenceStatisticsConfig | None = None
    ) -> None:
        _text(context_version, "context_version")
        self.context_version = context_version
        self.config = config or ReferenceStatisticsConfig()
        self._observations: tuple[ReferenceObservation, ...] = ()
        self._structure_buckets: tuple[StructureBucket, ...] = ()
        self._committed_batches: tuple[str, ...] = ()
        self._structure_observations: tuple[StructureReferenceObservation, ...] = ()
        self._structure_measurements: tuple[dict[str, Any], ...] = ()
        self._batch_events: tuple[dict[str, Any], ...] = ()

    def snapshot(
        self,
        batch_id: str,
        observations: Iterable[ReferenceObservation] = (),
        *,
        structure_observations: Iterable[StructureReferenceObservation] = (),
    ) -> ReferenceStatisticsSnapshot:
        """Read lagged buckets while staging isolated structure-only observations.

        Natural observation visits enter automatically. Additional paired rows
        never enter ``observations`` or task/category/global baseline counts.
        The natural-only control retains their provenance but excludes their
        contributions at commit, so callers can use the same rollout population.
        """
        _text(batch_id, "batch_id")
        if batch_id in self._committed_batches:
            raise ValueError("batch was already committed")
        current = tuple(observations)
        if any(not isinstance(row, ReferenceObservation) for row in current):
            raise ValueError("current batch requires ReferenceObservation records")
        combined = (*self._observations, *current)
        ids = tuple(row.observation_id for row in combined)
        if len(ids) != len(set(ids)):
            raise ValueError("reference observation IDs must be unique")
        if any(row.context_version != self.context_version for row in combined):
            raise ValueError("reference contexts cannot be pooled")
        extra_structure = tuple(structure_observations)
        if any(not isinstance(row, StructureReferenceObservation) for row in extra_structure):
            raise ValueError("structure channel requires StructureReferenceObservation records")
        structure = (*(_natural_structure(row) for row in current), *extra_structure)
        structure_ids = tuple(
            row.observation_id for row in (*self._structure_observations, *structure)
        )
        if len(structure_ids) != len(set(structure_ids)):
            raise ValueError("structural reference observation IDs must be unique")
        if any(row.context_version != self.context_version for row in structure):
            raise ValueError("structural reference contexts cannot be pooled")
        # The optional prior belongs to the immutable task, not a rollout arm.
        # Missing and supplied priors cannot be mixed silently: in a paired-only
        # context there are no natural records from which to recover an omission.
        task_priors: dict[str, tuple[float | None, float]] = {}
        all_evidence: tuple[ReferenceObservation | StructureReferenceObservation, ...] = (
            *combined,
            *self._structure_observations,
            *structure,
        )
        for evidence in all_evidence:
            prior = (evidence.prior_rate, evidence.prior_count)
            if task_priors.setdefault(evidence.task_id, prior) != prior:
                raise ValueError("same-task reference channels disagree on the supplied prior")
        result = ReferenceStatisticsSnapshot(
            self.context_version,
            batch_id,
            len(self._committed_batches),
            self.config,
            combined,
            tuple(row.observation_id for row in current),
            self._structure_buckets,
            structure,
        )
        # Reject inconsistent task/category/prior identities before any training.
        for row in current:
            result.task_rate(row.task_id, row.task_family)
        families = {row.task_id: row.task_family for row in combined}
        for structural_row in (*self._structure_observations, *structure):
            if (
                families.setdefault(structural_row.task_id, structural_row.task_family)
                != structural_row.task_family
            ):
                raise ValueError("structural task identity belongs to another category")
            result.task_rate(structural_row.task_id, structural_row.task_family)
        return result

    def commit_batch(self, snapshot: ReferenceStatisticsSnapshot) -> None:
        if (
            snapshot.context_version != self.context_version
            or snapshot.config != self.config
            or snapshot.generation != len(self._committed_batches)
            or snapshot.batch_id in self._committed_batches
            or snapshot.structure_buckets != self._structure_buckets
            or snapshot.observations[: len(self._observations)] != self._observations
        ):
            raise ValueError("snapshot is stale or belongs to another reference store")
        current = snapshot.observations[len(self._observations) :]
        if tuple(row.observation_id for row in current) != snapshot.current_observation_ids:
            raise ValueError("snapshot current-reference population differs")
        extras = snapshot.structure_observations[len(current) :]
        if self.snapshot(snapshot.batch_id, current, structure_observations=extras) != snapshot:
            raise ValueError("snapshot does not match the validated reference population")
        buckets = {
            bucket.key: (bucket.visits, bucket.excess_ratio_sum)
            for bucket in self._structure_buckets
        }
        natural_ids = {row.observation_id for row in current}
        measurements: list[dict[str, Any]] = []
        for row in snapshot.structure_observations:
            if (
                self.config.structure_source_mode == "natural_only"
                and row.source != "natural_reference"
            ):
                continue
            anchor = snapshot.task_anchor(
                row.task_id,
                row.task_family,
                exclude_observation_id=row.observation_id
                if row.observation_id in natural_ids
                else None,
                prior_rate=row.prior_rate,
                prior_count=row.prior_count,
            )
            try:
                excess = math.expm1(log_terminal_reward(row.reward, self.config.beta) - anchor)
            except OverflowError as error:
                raise ValueError("reference reward ratio overflowed") from error
            measurements.append(
                {
                    "observation_id": row.observation_id,
                    "batch_id": snapshot.batch_id,
                    "root_anchor": anchor,
                    "excess_ratio": excess,
                }
            )
            for visit in row.visits:
                for key in (visit.exact_key, visit.coarse_key):
                    count, total = buckets.get(key, (0, 0.0))
                    buckets[key] = (count + 1, math.fsum((total, excess)))
        updated = tuple(StructureBucket(key, *buckets[key]) for key in sorted(buckets))
        self._observations = snapshot.observations
        self._structure_buckets = updated
        self._committed_batches = (*self._committed_batches, snapshot.batch_id)
        self._structure_observations = (
            *self._structure_observations,
            *snapshot.structure_observations,
        )
        self._structure_measurements = (*self._structure_measurements, *measurements)
        self._batch_events = (
            *self._batch_events,
            {
                "batch_id": snapshot.batch_id,
                "natural_observation_ids": [row.observation_id for row in current],
                "structure_observation_ids": [
                    row.observation_id for row in snapshot.structure_observations
                ],
            },
        )

    def to_value(self) -> dict[str, Any]:
        return {
            "format": "evosteer-reference-statistics@2",
            "context_version": self.context_version,
            "config": asdict(self.config),
            "observations": [asdict(row) for row in self._observations],
            "structure_buckets": [asdict(bucket) for bucket in self._structure_buckets],
            "committed_batches": list(self._committed_batches),
            "structure_observations": [asdict(row) for row in self._structure_observations],
            "structure_measurements": copy.deepcopy(list(self._structure_measurements)),
            "batch_events": copy.deepcopy(list(self._batch_events)),
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> ReferenceStatistics:
        if set(value) != {
            "format",
            "context_version",
            "config",
            "observations",
            "structure_buckets",
            "committed_batches",
            "structure_observations",
            "structure_measurements",
            "batch_events",
        }:
            raise ValueError("reference statistics fields are incompatible")
        if value["format"] != "evosteer-reference-statistics@2":
            raise ValueError("unsupported reference statistics format")
        store = cls(value["context_version"], ReferenceStatisticsConfig(**value["config"]))
        observations = tuple(
            ReferenceObservation(
                **{
                    **row,
                    "visits": tuple(StructureVisit(**visit) for visit in row["visits"]),
                }
            )
            for row in value["observations"]
        )
        structural = tuple(
            StructureReferenceObservation(
                **{
                    **row,
                    "visits": tuple(StructureVisit(**visit) for visit in row["visits"]),
                    "forced_prefix_indices": tuple(row["forced_prefix_indices"]),
                }
            )
            for row in value["structure_observations"]
        )
        natural_by_id = {row.observation_id: row for row in observations}
        structural_by_id = {row.observation_id: row for row in structural}
        if len(natural_by_id) != len(observations) or len(structural_by_id) != len(structural):
            raise ValueError("restored reference evidence contains duplicate identities")
        for event in value["batch_events"]:
            if set(event) != {"batch_id", "natural_observation_ids", "structure_observation_ids"}:
                raise ValueError("invalid reference statistics batch event")
            natural_rows = tuple(natural_by_id[key] for key in event["natural_observation_ids"])
            structural_rows = tuple(
                structural_by_id[key] for key in event["structure_observation_ids"]
            )
            snapshot = store.snapshot(
                event["batch_id"],
                natural_rows,
                structure_observations=structural_rows[len(natural_rows) :],
            )
            if snapshot.structure_observations != structural_rows:
                raise ValueError("structural provenance differs from natural source visits")
            store.commit_batch(snapshot)
        if canonical_json(store.to_value()) != canonical_json(value):
            raise ValueError("reference statistics state differs from replayed batch evidence")
        return store
