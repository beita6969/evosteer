"""Result-blind exact task schedules over a fully pinned private catalog.

The private schedule owns task IDs while the public protocol stores only its
hash, count, and benchmark composition.  A worker only reads an already
frozen sequence; it never shuffles, filters by reward, or replaces a task.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.experiments import (
    BENCHMARK_SPECS,
    Benchmark,
    BenchmarkRole,
    BenchmarkTaskCount,
    FrozenTaskSequenceIdentity,
    SchedulePurpose,
    TrainingUse,
    fixed_subsample_rank,
)
from skillev.rollout import RolloutTask

from .catalog import PrivateBenchmarkCatalog

PRIVATE_FROZEN_TASK_SEQUENCE_FORMAT = "skillev-private-frozen-task-sequence@1"
TRAINING_SCHEDULE_ALGORITHM = "codex-reviewed-domain-block-task-id@1"
EVALUATION_SCHEDULE_ALGORITHM = "protocol-selected-task-id-order@1"


def _task_benchmark(task: RolloutTask) -> Benchmark:
    context = task.public_context
    if not isinstance(context, dict):
        raise ValueError("benchmark rollout task public_context must be an object")
    raw = context.get("benchmark_id")
    if type(raw) is not str:
        raise ValueError("benchmark rollout task is missing benchmark_id")
    return Benchmark(raw)


def _benchmarks_for(purpose: SchedulePurpose) -> tuple[Benchmark, ...]:
    if purpose is SchedulePurpose.IID_TRAINING:
        return tuple(
            spec.benchmark
            for spec in BENCHMARK_SPECS
            if spec.training_use is TrainingUse.TRAINING_MIX
        )
    if purpose is SchedulePurpose.IID_PROGRESS:
        return ()
    if purpose is SchedulePurpose.IID_EVALUATION:
        return tuple(spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.IID)
    if purpose is SchedulePurpose.OOD_EVALUATION:
        return tuple(spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.OOD)
    from typing import assert_never

    assert_never(purpose)


def _ranked_tasks(tasks: Iterable[RolloutTask], benchmark: Benchmark) -> tuple[RolloutTask, ...]:
    return tuple(
        sorted(
            tasks,
            key=lambda task: (fixed_subsample_rank(benchmark.value, task.task_id), task.task_id),
        )
    )


@dataclass(frozen=True, slots=True)
class PrivateFrozenTaskSequence:
    """Private task IDs plus their public, answer-free frozen identity."""

    identity: FrozenTaskSequenceIdentity
    task_ids: tuple[str, ...]
    format: str = PRIVATE_FROZEN_TASK_SEQUENCE_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FrozenTaskSequenceIdentity):
            raise TypeError("private task sequence requires a frozen public identity")
        if not isinstance(self.task_ids, tuple) or not self.task_ids:
            raise ValueError("private task sequence requires task IDs")
        if any(type(item) is not str or not item for item in self.task_ids):
            raise ValueError("private task sequence task IDs must be non-empty text")
        if len(self.task_ids) != self.identity.task_count:
            raise ValueError("private task sequence length differs from public identity")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("one formal attempt cannot repeat a task identity")
        if stable_hash(list(self.task_ids)) != self.identity.ordered_task_ids_hash:
            raise ValueError("private task IDs differ from frozen public sequence hash")
        if self.format != PRIVATE_FROZEN_TASK_SEQUENCE_FORMAT:
            raise ValueError("unsupported private frozen task sequence format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "identity": self.identity.to_value(),
            "task_ids": list(self.task_ids),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateFrozenTaskSequence:
        data = normalize_json(value)
        fields = {"format", "identity", "task_ids"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("private frozen task sequence has incompatible fields")
        raw_task_ids = data["task_ids"]
        if not isinstance(raw_task_ids, list) or any(
            type(item) is not str for item in raw_task_ids
        ):
            raise ValueError("private frozen task IDs must be a text array")
        if type(data["format"]) is not str:
            raise ValueError("private frozen task sequence format must be text")
        return cls(
            identity=FrozenTaskSequenceIdentity.from_value(data["identity"]),
            task_ids=tuple(raw_task_ids),
            format=data["format"],
        )

    @classmethod
    def read(cls, path: Path) -> PrivateFrozenTaskSequence:
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        """Persist the schedule before execution without replacing a prior one."""

        encoded = canonical_json(self.to_value()).encode("utf-8") + b"\n"
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def resolve(self, catalog: PrivateBenchmarkCatalog) -> tuple[RolloutTask, ...]:
        by_id = {task.task_id: task for workload in catalog.workloads for task in workload.tasks}
        try:
            tasks = tuple(by_id[item] for item in self.task_ids)
        except KeyError as error:
            raise ValueError("frozen sequence references a task absent from catalog") from error
        counts = Counter(_task_benchmark(task) for task in tasks)
        expected = {item.benchmark: item.count for item in self.identity.benchmark_counts}
        if counts != expected:
            raise ValueError("resolved schedule benchmark counts differ from public identity")
        return tasks


def frozen_sequence_from_task_ids(
    catalog: PrivateBenchmarkCatalog,
    *,
    purpose: SchedulePurpose,
    task_ids: tuple[str, ...],
    schedule_algorithm: str,
) -> PrivateFrozenTaskSequence:
    """Freeze an already selected result-blind sequence against one catalog."""

    if not isinstance(catalog, PrivateBenchmarkCatalog):
        raise TypeError("catalog must be PrivateBenchmarkCatalog")
    if not isinstance(purpose, SchedulePurpose):
        raise TypeError("purpose must be SchedulePurpose")
    if type(schedule_algorithm) is not str or not schedule_algorithm:
        raise ValueError("schedule_algorithm must be non-empty text")
    by_id = {task.task_id: task for workload in catalog.workloads for task in workload.tasks}
    if len(by_id) != sum(len(workload.tasks) for workload in catalog.workloads):
        raise ValueError("private catalog task IDs must be globally unique")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("frozen sequence task IDs must be non-empty and unique")
    try:
        selected = tuple(by_id[task_id] for task_id in task_ids)
    except KeyError as error:
        raise ValueError("frozen sequence references a task absent from catalog") from error
    expected_benchmarks = set(_benchmarks_for(purpose))
    counts = Counter(_task_benchmark(task) for task in selected)
    if set(counts) != expected_benchmarks:
        raise ValueError("frozen sequence does not cover its declared benchmark role")
    benchmark_counts = tuple(
        BenchmarkTaskCount(benchmark=benchmark, count=counts[benchmark])
        for benchmark in sorted(counts, key=lambda item: item.value)
    )
    identity = FrozenTaskSequenceIdentity(
        purpose=purpose,
        ordered_task_ids_hash=stable_hash(list(task_ids)),
        task_count=len(task_ids),
        benchmark_counts=benchmark_counts,
        schedule_algorithm=schedule_algorithm,
    )
    return PrivateFrozenTaskSequence(identity=identity, task_ids=task_ids)


def build_result_blind_sequence(
    catalog: PrivateBenchmarkCatalog,
    *,
    purpose: SchedulePurpose,
    task_count: int | None = None,
) -> PrivateFrozenTaskSequence:
    """Build a deterministic sequence before result ingestion.

    Training concatenates the already reviewed per-domain episode workloads in
    protocol order.  Evaluation schedules use their protocol-selected task
    order.  The caller persists the returned sequence and workers only read it.
    """

    benchmarks = _benchmarks_for(purpose)
    missing = set(benchmarks) - {workload.benchmark for workload in catalog.workloads}
    if missing:
        raise ValueError("private catalog is missing a declared schedule benchmark")
    if purpose is SchedulePurpose.IID_TRAINING:
        if type(task_count) is not int or task_count < len(benchmarks):
            raise ValueError("training schedule task_count must cover every IID training domain")
        if task_count % len(benchmarks) != 0:
            raise ValueError("training schedule must allocate an equal block to every IID domain")
        per_benchmark = task_count // len(benchmarks)
        selected: list[str] = []
        for benchmark in benchmarks:
            tasks = catalog.workload(benchmark).tasks
            if len(tasks) != per_benchmark:
                raise ValueError("reviewed IID workload differs from its frozen domain-block size")
            selected.extend(task.task_id for task in tasks)
        return frozen_sequence_from_task_ids(
            catalog,
            purpose=purpose,
            task_ids=tuple(selected),
            schedule_algorithm=TRAINING_SCHEDULE_ALGORITHM,
        )

    if task_count is not None:
        raise ValueError("only IID training accepts an explicit task_count")
    selected_tasks: list[RolloutTask] = []
    for benchmark in benchmarks:
        spec = next(item for item in BENCHMARK_SPECS if item.benchmark is benchmark)
        selected_tasks.extend(catalog.workload(benchmark).selected_tasks(spec.evaluation_sampling))
    selected_task_ids = tuple(task.task_id for task in selected_tasks)
    return frozen_sequence_from_task_ids(
        catalog,
        purpose=purpose,
        task_ids=selected_task_ids,
        schedule_algorithm=EVALUATION_SCHEDULE_ALGORITHM,
    )


__all__ = [
    "EVALUATION_SCHEDULE_ALGORITHM",
    "PRIVATE_FROZEN_TASK_SEQUENCE_FORMAT",
    "TRAINING_SCHEDULE_ALGORITHM",
    "PrivateFrozenTaskSequence",
    "build_result_blind_sequence",
    "frozen_sequence_from_task_ids",
]
