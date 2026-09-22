"""Private workload routing over answer-free public rollout tasks.

The catalog contains deployment/session factories, but selection is based only
on public task identities and the preregistered seed.  No reward or verifier
truth participates in curriculum or evaluation-subsample construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from skillev.benchmarks.task_family import require_benchmark_task_family
from skillev.contracts import stable_hash
from skillev.experiments import (
    Benchmark,
    EvaluationSampling,
    FixedSubsampleEvaluation,
    FullEvaluationSampling,
    PopulationThresholdEvaluation,
    fixed_subsample_rank,
)
from skillev.rollout import RolloutTask
from skillev.runtime import OrderedTaskCursorState

if TYPE_CHECKING:
    from skillev.rollout import UnskilledRolloutSessionBundle
    from skillev.training import RolloutSessionBundle


class PrivateSessionFactory(Protocol):
    def create(self, task: RolloutTask) -> RolloutSessionBundle: ...


def _task_benchmark_id(task: RolloutTask) -> str:
    context = task.public_context
    if not isinstance(context, dict):
        raise ValueError("benchmark rollout task public_context must be an object")
    benchmark_id = context.get("benchmark_id")
    if type(benchmark_id) is not str or not benchmark_id:
        raise ValueError("benchmark rollout task is missing its public benchmark identity")
    return benchmark_id


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkWorkload:
    benchmark: Benchmark
    tasks: tuple[RolloutTask, ...]
    session_factory: PrivateSessionFactory

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("workload benchmark must be Benchmark")
        if not self.tasks or any(not isinstance(task, RolloutTask) for task in self.tasks):
            raise ValueError("workload tasks must be a non-empty RolloutTask tuple")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("workload task identities must be unique")
        if any(_task_benchmark_id(task) != self.benchmark.value for task in self.tasks):
            raise ValueError("workload contains a task from another benchmark")
        for task in self.tasks:
            require_benchmark_task_family(
                benchmark_id=self.benchmark.value,
                task_family=task.task_family,
            )
        if not callable(getattr(self.session_factory, "create", None)):
            raise TypeError("workload session_factory must implement create")

    def selected_tasks(
        self,
        sampling: EvaluationSampling,
    ) -> tuple[RolloutTask, ...]:
        """Apply the protocol's result-blind public-identity selection rule."""

        if not isinstance(
            sampling,
            FullEvaluationSampling | FixedSubsampleEvaluation | PopulationThresholdEvaluation,
        ):
            raise TypeError("sampling has an unsupported variant")
        if isinstance(sampling, FullEvaluationSampling):
            return tuple(sorted(self.tasks, key=lambda task: task.task_id))
        if (
            isinstance(sampling, PopulationThresholdEvaluation)
            and len(self.tasks) <= sampling.THRESHOLD
        ):
            return tuple(sorted(self.tasks, key=lambda task: task.task_id))
        if len(self.tasks) < sampling.SAMPLE_SIZE:
            raise ValueError("workload is smaller than the fixed evaluation subsample")
        ranked = sorted(
            self.tasks,
            key=lambda task: (
                fixed_subsample_rank(self.benchmark.value, task.task_id),
                task.task_id,
            ),
        )
        return tuple(ranked[: sampling.SAMPLE_SIZE])


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkCatalog:
    workloads: tuple[PrivateBenchmarkWorkload, ...]

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("benchmark catalog requires at least one workload")
        benchmarks = tuple(workload.benchmark for workload in self.workloads)
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError("benchmark catalog workloads must be unique")
        all_task_ids = tuple(task.task_id for workload in self.workloads for task in workload.tasks)
        if len(set(all_task_ids)) != len(all_task_ids):
            raise ValueError("benchmark task identities must be globally unique")

    def workload(self, benchmark: Benchmark) -> PrivateBenchmarkWorkload:
        matches = tuple(item for item in self.workloads if item.benchmark is benchmark)
        if len(matches) != 1:
            raise KeyError(benchmark.value)
        return matches[0]

    def route(self, tasks: tuple[RolloutTask, ...]) -> PrivateRoutedSessionFactory:
        if not tasks:
            raise ValueError("routed benchmark session factory requires tasks")
        task_ids = tuple(task.task_id for task in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("routed benchmark tasks must be unique")
        routes: list[tuple[str, PrivateSessionFactory]] = []
        for task in tasks:
            benchmark = Benchmark(_task_benchmark_id(task))
            workload = self.workload(benchmark)
            if task not in workload.tasks:
                raise ValueError("routed task does not equal its catalog projection")
            routes.append((task.task_id, workload.session_factory))
        return PrivateRoutedSessionFactory(tuple(routes))


@dataclass(slots=True)
class OrderedPublicTaskProvider:
    tasks: tuple[RolloutTask, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.tasks or any(not isinstance(task, RolloutTask) for task in self.tasks):
            raise ValueError("ordered task provider requires RolloutTask values")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("ordered task provider task identities must be unique")
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.tasks):
            raise ValueError("ordered task provider cursor is invalid")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.tasks):
            raise RuntimeError("ordered benchmark task provider is exhausted")
        task = self.tasks[self.cursor]
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id=stable_hash(
                {"kind": "private-routed", "task_ids": [task.task_id for task in self.tasks]}
            ),
            cursor=self.cursor,
        )


@dataclass(frozen=True, slots=True)
class PrivateRoutedSessionFactory:
    routes: tuple[tuple[str, PrivateSessionFactory], ...]

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("routed session factory requires routes")
        task_ids = tuple(task_id for task_id, _ in self.routes)
        if any(type(task_id) is not str or not task_id for task_id in task_ids):
            raise ValueError("routed session identities must be non-empty text")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("routed session identities must be unique")
        if any(not callable(getattr(factory, "create", None)) for _, factory in self.routes):
            raise TypeError("routed session factory contains an incompatible route")

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("routed session creation requires RolloutTask")
        matches = tuple(factory for task_id, factory in self.routes if task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("rollout task has no unique private session route")
        routed = matches[0].create(task)
        from skillev.rollout import UnskilledRolloutSessionBundle

        return UnskilledRolloutSessionBundle(
            environment=routed.environment,
            evaluator=routed.evaluator,
        )


__all__ = [
    "OrderedPublicTaskProvider",
    "PrivateBenchmarkCatalog",
    "PrivateBenchmarkWorkload",
    "PrivateRoutedSessionFactory",
    "PrivateSessionFactory",
]
