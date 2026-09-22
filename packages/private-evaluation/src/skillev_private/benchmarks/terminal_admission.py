"""Result-blind admission of every terminal-evaluator route in a training sequence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from skillev.contracts import JsonValue, stable_hash
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

from .catalog import PrivateBenchmarkCatalog, _task_benchmark_id


def _type_identity(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _case_task_id(case: object) -> str | None:
    public = getattr(case, "public", None)
    task_id = getattr(public, "task_id", None)
    if type(task_id) is str:
        return task_id
    task_id = getattr(case, "task_id", None)
    return task_id if type(task_id) is str else None


def _admit_factory_task(factory: object, task: RolloutTask) -> str:
    """Prove one exact private lookup without starting an official process."""

    nested_route = getattr(factory, "terminal_admission_route", None)
    if callable(nested_route):
        nested = nested_route(task)
        if (
            not isinstance(nested, tuple)
            or len(nested) != 2
            or not isinstance(nested[1], RolloutTask)
            or nested[0] is factory
        ):
            raise TypeError("terminal admission route is incompatible")
        return f"{_type_identity(factory)}:{_admit_factory_task(nested[0], nested[1])}"

    frozen_tasks = getattr(factory, "tasks", None)
    if isinstance(frozen_tasks, tuple):
        matches = tuple(
            item for item in frozen_tasks if getattr(item, "task_id", None) == task.task_id
        )
        if len(matches) != 1 or matches[0] != task:
            raise ValueError("terminal route has no unique frozen public task")

    cases = getattr(factory, "cases", None)
    if isinstance(cases, tuple):
        matches = tuple(case for case in cases if _case_task_id(case) == task.task_id)
        if len(matches) != 1:
            raise ValueError("terminal route has no unique private case")

    worker = getattr(factory, "worker", None)
    worker_cases = getattr(worker, "cases", None)
    if isinstance(worker_cases, tuple):
        matches = tuple(case for case in worker_cases if _case_task_id(case) == task.task_id)
        if len(matches) != 1:
            raise ValueError("terminal worker has no unique private case")

    runtime = getattr(factory, "runtime", None)
    if runtime is not None:
        benchmark = getattr(runtime, "benchmark", None)
        if not isinstance(benchmark, Benchmark) or benchmark.value != _task_benchmark_id(task):
            raise ValueError("terminal process runtime has another benchmark route")
        context = task.public_context
        source_id = context.get("source_id") if isinstance(context, dict) else None
        if type(source_id) is not str or not source_id or source_id.startswith("/"):
            raise ValueError("terminal process task has no normalized source identity")
        if ".." in source_id.split("/"):
            raise ValueError("terminal process task source identity escapes its source root")
        return f"{_type_identity(factory)}:{benchmark.value}"

    if not isinstance(cases, tuple) and not isinstance(frozen_tasks, tuple):
        raise TypeError("terminal session factory exposes no admission surface")
    return _type_identity(factory)


@dataclass(frozen=True, slots=True)
class TerminalEvaluatorRouteAdmission:
    """Sanitized proof that every ordered task has one frozen evaluator route."""

    task_count: int
    ordered_task_ids_hash: str
    benchmark_counts: tuple[tuple[str, int], ...]
    route_counts: tuple[tuple[str, int], ...]

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark_counts": dict(self.benchmark_counts),
            "ordered_task_ids_hash": self.ordered_task_ids_hash,
            "route_counts": dict(self.route_counts),
            "task_count": self.task_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


def admit_terminal_evaluator_routes(
    catalog: PrivateBenchmarkCatalog,
    tasks: tuple[RolloutTask, ...],
) -> TerminalEvaluatorRouteAdmission:
    """Admit all tasks without evaluating candidates or reading reward truth."""

    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("terminal route admission requires unique ordered tasks")
    routed = catalog.route(tasks)
    factories = dict(routed.routes)
    benchmark_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    for task in tasks:
        factory = factories[task.task_id]
        benchmark_counts[_task_benchmark_id(task)] += 1
        route_counts[_admit_factory_task(factory, task)] += 1
    return TerminalEvaluatorRouteAdmission(
        task_count=len(tasks),
        ordered_task_ids_hash=stable_hash([task.task_id for task in tasks]),
        benchmark_counts=tuple(sorted(benchmark_counts.items())),
        route_counts=tuple(sorted(route_counts.items())),
    )


__all__ = ["TerminalEvaluatorRouteAdmission", "admit_terminal_evaluator_routes"]
