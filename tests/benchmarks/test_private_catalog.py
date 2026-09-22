from __future__ import annotations

from dataclasses import dataclass

import pytest
from skillev_private.benchmarks import (
    OrderedPublicTaskProvider,
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
)

from skillev.experiments import FIXED_500_EVALUATION, Benchmark
from skillev.rollout import RolloutTask
from skillev.runtime import BudgetVector, EnvironmentObservation, StructuredAction
from skillev.training import RolloutSessionBundle


@dataclass(slots=True)
class _Environment:
    environment_id: str
    task_family: str

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        del action, step_index
        return EnvironmentObservation(
            public_value={"status": "fixture"},
            observation_status="success",
            budget_usage=BudgetVector(tool_calls=1),
        )

    def validate_completion(self, submission: object) -> bool:
        del submission
        return True


@dataclass(frozen=True, slots=True)
class _Evaluator:
    async def evaluate(self, request: object) -> object:
        del request
        raise AssertionError("fixture evaluator must not run during routing")


@dataclass(frozen=True, slots=True)
class _Factory:
    marker: str

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        return RolloutSessionBundle(
            environment=_Environment(task.environment_id, task.task_family),
            evaluator=_Evaluator(),
            retrieved_skills=(),
        )


def _task(benchmark: Benchmark, index: int) -> RolloutTask:
    return RolloutTask(
        task_id=f"{benchmark.value}/{index:04d}",
        environment_id=f"fixture:{benchmark.value}",
        task_family=f"{benchmark.value}/fixture",
        context_id=f"fixture/{benchmark.value}",
        query=f"Public task {index}",
        available_tools=(),
        public_context={"benchmark_id": benchmark.value},
    )


def test_catalog_selection_is_result_blind_deterministic_and_routable() -> None:
    webshop_tasks = tuple(_task(Benchmark.WEBSHOP, index) for index in range(520))
    medqa_tasks = tuple(_task(Benchmark.MED_QA, index) for index in range(500))
    catalog = PrivateBenchmarkCatalog(
        (
            PrivateBenchmarkWorkload(Benchmark.WEBSHOP, webshop_tasks, _Factory("webshop")),
            PrivateBenchmarkWorkload(Benchmark.MED_QA, medqa_tasks, _Factory("medqa")),
        )
    )

    selected = catalog.workload(Benchmark.WEBSHOP).selected_tasks(FIXED_500_EVALUATION)
    assert len(selected) == 500
    assert selected == catalog.workload(Benchmark.WEBSHOP).selected_tasks(FIXED_500_EVALUATION)
    assert {task.task_id for task in selected}.issubset({task.task_id for task in webshop_tasks})

    mixed = (selected[0], medqa_tasks[0])
    routed = catalog.route(mixed)
    assert routed.create(mixed[0]).environment.environment_id == "fixture:webshop"
    assert routed.create(mixed[1]).environment.environment_id == "fixture:medqa"
    provider = OrderedPublicTaskProvider(mixed)
    assert provider.next_task() is mixed[0]
    assert provider.next_task() is mixed[1]
    with pytest.raises(RuntimeError):
        provider.next_task()


def test_catalog_rejects_short_fixed_subsample_and_foreign_task() -> None:
    workload = PrivateBenchmarkWorkload(
        Benchmark.WEBSHOP,
        tuple(_task(Benchmark.WEBSHOP, index) for index in range(10)),
        _Factory("webshop"),
    )
    catalog = PrivateBenchmarkCatalog((workload,))
    with pytest.raises(ValueError):
        workload.selected_tasks(FIXED_500_EVALUATION)
    with pytest.raises(KeyError):
        catalog.workload(Benchmark.MED_QA)
