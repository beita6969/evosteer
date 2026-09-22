from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest
from skillev_private.benchmarks import (
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
    PrivateStaticBenchmarkCase,
    PrivateStaticBenchmarkSessionFactory,
    PrivateStaticTarget,
    StaticScoringRule,
    admit_terminal_evaluator_routes,
)
from skillev_private.experiments.terminal_evaluator_gate import run_terminal_evaluator_gate

from skillev.benchmarks import BenchmarkPublicItem
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask


def _task(task_id: str) -> RolloutTask:
    return RolloutTask(
        task_id=task_id,
        environment_id="terminal-admission-fixture",
        task_family="hotpotqa/bridge",
        context_id="hotpotqa/bridge",
        query="Answer the public multi-hop question.",
        available_tools=(),
        public_context={"benchmark_id": Benchmark.HOTPOT_QA.value},
    )


@dataclass(frozen=True, slots=True)
class _PublicCase:
    task: RolloutTask

    @property
    def task_id(self) -> str:
        return self.task.task_id

    def to_rollout_task(self) -> RolloutTask:
        return self.task


@dataclass(frozen=True, slots=True)
class _PrivateCase:
    public: _PublicCase


@dataclass(frozen=True, slots=True)
class _Factory:
    cases: tuple[_PrivateCase, ...]

    def create(self, task: RolloutTask) -> object:
        raise AssertionError("static admission must not create a rollout session")


def _catalog(
    tasks: tuple[RolloutTask, ...],
    cases: tuple[_PrivateCase, ...],
) -> PrivateBenchmarkCatalog:
    return PrivateBenchmarkCatalog(
        (
            PrivateBenchmarkWorkload(
                benchmark=Benchmark.HOTPOT_QA,
                tasks=tasks,
                session_factory=_Factory(cases),
            ),
        )
    )


def test_full_ordered_sequence_route_admission_is_result_blind_and_deterministic() -> None:
    tasks = (_task("hotpot-1"), _task("hotpot-2"))
    catalog = _catalog(tasks, tuple(_PrivateCase(_PublicCase(task)) for task in tasks))

    first = admit_terminal_evaluator_routes(catalog, tasks)
    second = admit_terminal_evaluator_routes(catalog, tasks)

    assert first == second
    assert first.task_count == 2
    assert first.benchmark_counts == ((Benchmark.HOTPOT_QA.value, 2),)
    assert first.route_counts == ((f"{__name__}._Factory", 2),)
    assert first.content_hash == second.content_hash


def test_route_admission_rejects_missing_or_duplicate_private_case() -> None:
    task = _task("hotpot-1")
    public = _PublicCase(task)

    with pytest.raises(ValueError):
        admit_terminal_evaluator_routes(_catalog((task,), ()), (task,))
    with pytest.raises(ValueError):
        admit_terminal_evaluator_routes(
            _catalog((task,), (_PrivateCase(public), _PrivateCase(public))),
            (task,),
        )


def test_route_admission_leaves_environment_specific_projection_to_its_factory() -> None:
    source = _task("hotpot-retrieval")
    routed = replace(source, environment_id="terminal-admission-retrieval-fixture")
    catalog = _catalog((routed,), (_PrivateCase(_PublicCase(source)),))

    report = admit_terminal_evaluator_routes(catalog, (routed,))

    assert report.task_count == 1


def test_cpu_gate_replays_exact_first_task_no_submission_as_zero() -> None:
    public = BenchmarkPublicItem(
        benchmark_id=Benchmark.HOTPOT_QA.value,
        dataset_revision="fixture@1",
        split="train",
        task_id="hotpot-gate-first",
        task_family="hotpotqa/bridge",
        query="Answer the public question.",
        public_context={"answer_format": "short-text"},
    )
    case = PrivateStaticBenchmarkCase(
        public,
        PrivateStaticTarget(public.task_id, StaticScoringRule.TOKEN_F1, ("private",)),
    )
    tasks = (public.to_rollout_task(),)
    catalog = PrivateBenchmarkCatalog(
        (
            PrivateBenchmarkWorkload(
                benchmark=Benchmark.HOTPOT_QA,
                tasks=tasks,
                session_factory=PrivateStaticBenchmarkSessionFactory((case,)),
            ),
        )
    )

    report = asyncio.run(run_terminal_evaluator_gate(catalog, tasks))

    assert report.route_admission.task_count == 1
    assert report.first_reward_value == 0.0
    assert report.first_reward_success is False
    assert report.first_verifier_version == "static-benchmark-evaluator@1"
    assert report.no_submission_replay_count == 1
    assert report.verifier_version_counts == (("static-benchmark-evaluator@1", 1),)


def test_cpu_gate_admits_every_task_but_replays_only_the_exact_first_boundary() -> None:
    items = tuple(
        BenchmarkPublicItem(
            benchmark_id=Benchmark.HOTPOT_QA.value,
            dataset_revision="fixture@1",
            split="train",
            task_id=f"hotpot-gate-{index}",
            task_family="hotpotqa/bridge",
            query="Answer the public question.",
            public_context={"answer_format": "short-text"},
        )
        for index in range(3)
    )
    cases = tuple(
        PrivateStaticBenchmarkCase(
            item,
            PrivateStaticTarget(item.task_id, StaticScoringRule.TOKEN_F1, ("private",)),
        )
        for item in items
    )
    tasks = tuple(item.to_rollout_task() for item in items)
    catalog = PrivateBenchmarkCatalog(
        (
            PrivateBenchmarkWorkload(
                benchmark=Benchmark.HOTPOT_QA,
                tasks=tasks,
                session_factory=PrivateStaticBenchmarkSessionFactory(cases),
            ),
        )
    )

    report = asyncio.run(run_terminal_evaluator_gate(catalog, tasks))

    assert report.route_admission.task_count == 3
    assert report.no_submission_replay_count == 1
    assert report.verifier_version_counts == (("static-benchmark-evaluator@1", 1),)
