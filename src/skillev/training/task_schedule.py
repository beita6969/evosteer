"""Deterministic source-balanced task batches for EvoSteer Algorithm 1.

The paper states 56 records balanced over sources and a static curriculum for
steps 1--12. It does not supply source ordering or stage proportions. Callers
must provide the stages explicitly; this module never supplies an invented
curriculum. Sampling is without replacement within a batch and can revisit a
record in later batches. Paired-evidence independence remains the admission
ledger's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from skillev.contracts.canonical import stable_hash

SCHEDULE_FORMAT = "evosteer-source-balanced-task-schedule@1"
CURRICULUM_STEPS = 12


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """Use these sources through the inclusive one-based training step."""

    through_step: int
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.through_step) is not int or not 1 <= self.through_step <= CURRICULUM_STEPS:
            raise ValueError("curriculum through_step must be between 1 and 12")
        if not isinstance(self.source_ids, tuple) or not self.source_ids:
            raise ValueError("curriculum source_ids must be a nonempty immutable tuple")
        for source in self.source_ids:
            _text(source, "source_id")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("curriculum source_ids must be unique")

    def to_value(self) -> dict[str, Any]:
        return {"through_step": self.through_step, "source_ids": list(self.source_ids)}

    @classmethod
    def from_value(cls, value: object) -> CurriculumStage:
        if not isinstance(value, dict) or set(value) != {"through_step", "source_ids"}:
            raise ValueError("invalid curriculum stage fields")
        if not isinstance(value["source_ids"], list):
            raise ValueError("curriculum source_ids must be an array")
        return cls(value["through_step"], tuple(value["source_ids"]))


@dataclass(frozen=True, slots=True)
class ScheduledTaskBatch:
    step_index: int
    task_ids: tuple[str, ...]
    source_counts: tuple[tuple[str, int], ...]
    active_sources: tuple[str, ...]
    configuration_id: str

    @property
    def identity(self) -> str:
        return str(stable_hash(self.to_value()))

    def to_value(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "task_ids": list(self.task_ids),
            "source_counts": dict(self.source_counts),
            "active_sources": list(self.active_sources),
            "configuration_id": self.configuration_id,
        }


class SourceBalancedTaskSchedule:
    """Plan without advancing; commit only after the training batch succeeds.

    Extra slots, when batch size is not divisible by source count, rotate across
    the sorted source IDs. Hash-ranked selection gives reproducible per-step
    sampling independent of Python RNG versions and mapping insertion order.
    Explicit source IDs are used; task families have no role in balancing.
    """

    def __init__(
        self,
        task_sources: Mapping[str, str],
        *,
        curriculum: Sequence[CurriculumStage],
        batch_size: int = 56,
        seed: int = 0,
    ) -> None:
        if not isinstance(task_sources, Mapping) or not task_sources:
            raise ValueError("task_sources must map task IDs to explicit source IDs")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        normalized = {
            _text(task, "task_id"): _text(source, "source_id")
            for task, source in task_sources.items()
        }
        stages = tuple(curriculum)
        if not stages or any(not isinstance(stage, CurriculumStage) for stage in stages):
            raise ValueError("an explicit static curriculum covering steps 1--12 is required")
        if stages[-1].through_step != CURRICULUM_STEPS or any(
            earlier.through_step >= later.through_step for earlier, later in pairwise(stages)
        ):
            raise ValueError("curriculum boundaries must increase and finish at step 12")
        sources = tuple(sorted(set(normalized.values())))
        if any(not set(stage.source_ids) <= set(sources) for stage in stages):
            raise ValueError("curriculum references an unknown source")
        self._task_sources = dict(sorted(normalized.items()))
        self._sources = sources
        self._pools = {
            source: tuple(task for task, origin in self._task_sources.items() if origin == source)
            for source in sources
        }
        self._curriculum = tuple(
            CurriculumStage(stage.through_step, tuple(sorted(stage.source_ids))) for stage in stages
        )
        self._batch_size = batch_size
        self._seed = seed
        self._committed_steps = 0
        # A source may receive the remainder slot on a later round. Reject an
        # undersized pool now, rather than silently repeating tasks or biasing quotas.
        for active in (*[stage.source_ids for stage in self._curriculum], self._sources):
            maximum_quota = (batch_size + len(active) - 1) // len(active)
            for source in active:
                if len(self._pools[source]) < maximum_quota:
                    raise ValueError(
                        f"source {source!r} has {len(self._pools[source])} records; "
                        f"balanced batch requires up to {maximum_quota} distinct records"
                    )
        self._configuration_id = str(
            stable_hash(
                {
                    "format": SCHEDULE_FORMAT,
                    "task_sources": self._task_sources,
                    "curriculum": [stage.to_value() for stage in self._curriculum],
                    "batch_size": batch_size,
                    "seed": seed,
                    "sampling": "per-step-hash-rank-without-replacement@1",
                }
            )
        )

    @property
    def configuration_id(self) -> str:
        return self._configuration_id

    @property
    def committed_steps(self) -> int:
        return self._committed_steps

    def _active_sources(self, step: int) -> tuple[str, ...]:
        for stage in self._curriculum:
            if step <= stage.through_step:
                return stage.source_ids
        return self._sources

    def plan_next(self) -> ScheduledTaskBatch:
        step = self._committed_steps + 1
        active = self._active_sources(step)
        base, remainder = divmod(self._batch_size, len(active))
        offset = ((step - 1) * remainder) % len(active)
        extras = {active[(offset + index) % len(active)] for index in range(remainder)}
        counts = tuple((source, base + (source in extras)) for source in active)
        selected: list[str] = []
        for source, count in counts:
            ranked = sorted(
                self._pools[source],
                key=lambda task: (
                    stable_hash({"seed": self._seed, "step": step, "source": source, "task": task}),
                    task,
                ),
            )
            selected.extend(ranked[:count])
        selected.sort(
            key=lambda task: (
                stable_hash({"seed": self._seed, "step": step, "batch-order": task}),
                task,
            )
        )
        return ScheduledTaskBatch(step, tuple(selected), counts, active, self.configuration_id)

    def commit(self, batch: ScheduledTaskBatch) -> None:
        if not isinstance(batch, ScheduledTaskBatch) or batch != self.plan_next():
            raise ValueError("cannot commit a stale batch or a plan from another schedule")
        self._committed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": SCHEDULE_FORMAT,
            "configuration_id": self.configuration_id,
            "committed_steps": self.committed_steps,
        }

    def load_state_dict(self, state: object) -> None:
        if self.committed_steps:
            raise RuntimeError("schedule restore requires a fresh instance")
        if not isinstance(state, dict) or set(state) != {
            "format",
            "configuration_id",
            "committed_steps",
        }:
            raise ValueError("invalid task schedule checkpoint fields")
        if state["format"] != SCHEDULE_FORMAT or state["configuration_id"] != self.configuration_id:
            raise ValueError("checkpoint task/source manifest, seed or curriculum differs")
        steps = state["committed_steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("committed_steps must be a nonnegative integer")
        self._committed_steps = steps


__all__ = ["SCHEDULE_FORMAT", "CurriculumStage", "ScheduledTaskBatch", "SourceBalancedTaskSchedule"]
