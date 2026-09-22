from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from skillev_private.benchmarks.protocol_v10_population import (
    ProtocolV10PopulationCatalog,
    ProtocolV10PopulationSessionRegistry,
)
from skillev_private.experiments.protocol_v10_evaluation import (
    ProtocolV10EvaluationInfrastructureError,
    ProtocolV10EvaluationInput,
    ProtocolV10EvaluationMode,
    ProtocolV10EvaluationOutcome,
    ProtocolV10EvaluationRecord,
    ProtocolV10EvaluationStore,
    ProtocolV10EvaluationSupervisor,
    ProtocolV10EvaluationWorker,
    ProtocolV10ReadOnlyState,
    aggregate_protocol_v10_evaluation,
)

from skillev.contracts import SuccessRule, TerminalReward
from skillev.experiments import ACTIVE_BENCHMARKS_V10, BenchmarkV10, FormalMethodV10
from skillev.rollout import RolloutSessionBundle, RolloutTask


def _exact_input(
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    expected_record_count: int = len(ACTIVE_BENCHMARKS_V10),
) -> ProtocolV10EvaluationInput:
    return ProtocolV10EvaluationInput(
        run_id="formal-run",
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        mode=ProtocolV10EvaluationMode.FINAL,
        checkpoint_id="checkpoint-step-288",
        protocol_id="skillev-benchmark-protocol@10",
        population_manifest_id="private-populations-v10",
        evaluator_bundle_id="trusted-evaluators-v10",
        model_id="qwen3.5-9b",
        tokenizer_id="qwen3.5-tokenizer",
        generation_config_id="formal-generation-v10",
        expected_record_count=expected_record_count,
        shard_index=shard_index,
        shard_count=shard_count,
    )


@dataclass(slots=True)
class _Factory:
    cleanup_calls: list[str]

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        async def cleanup() -> None:
            self.cleanup_calls.append(task.task_id)

        return RolloutSessionBundle(
            environment=object(),  # type: ignore[arg-type]
            evaluator=object(),  # type: ignore[arg-type]
            retrieved_skills=(),
            cleanup=cleanup,
        )


def _catalog_and_sessions(
    cleanup_calls: list[str],
) -> tuple[ProtocolV10PopulationCatalog, ProtocolV10PopulationSessionRegistry]:
    populations: dict[BenchmarkV10, tuple[SimpleNamespace, ...]] = {}
    factories: dict[str, _Factory] = {}
    for benchmark in ACTIVE_BENCHMARKS_V10:
        population_id = f"{benchmark.value}/final"
        task = RolloutTask(
            task_id=f"{benchmark.value}/task",
            environment_id=f"fixture:{benchmark.value}",
            task_family=f"{benchmark.value}/fixture",
            context_id=f"fixture:{benchmark.value}",
            query="Public fixture query.",
            available_tools=(),
            public_context={"benchmark_id": benchmark.value},
        )
        item = SimpleNamespace(source_id=f"{benchmark.value}/source", task=task)
        populations[benchmark] = (
            SimpleNamespace(
                spec=SimpleNamespace(population_id=population_id),
                items=(item,),
            ),
        )
        factories[population_id] = _Factory(cleanup_calls)

    class _Catalog:
        def population(self, benchmark: BenchmarkV10, role: object) -> tuple[SimpleNamespace, ...]:
            del role
            return populations[benchmark]

    class _Sessions:
        def factory(self, population_id: str) -> _Factory:
            return factories[population_id]

    return (
        cast(ProtocolV10PopulationCatalog, _Catalog()),
        cast(ProtocolV10PopulationSessionRegistry, _Sessions()),
    )


@dataclass(slots=True)
class _Runner:
    calls: list[str]
    fail_task_id: str | None = None

    async def evaluate(
        self,
        task: RolloutTask,
        session: RolloutSessionBundle,
    ) -> ProtocolV10EvaluationOutcome:
        del session
        self.calls.append(task.task_id)
        if task.task_id == self.fail_task_id:
            raise RuntimeError("private evaluator detail")
        benchmark = BenchmarkV10(task.public_context["benchmark_id"])
        native_fields: dict[str, object] = {"score": 1.0}
        if benchmark is BenchmarkV10.APPWORLD:
            native_fields["scenario-id"] = "scenario-fixture"
            native_fields["task-goal-completion"] = 1.0
        return ProtocolV10EvaluationOutcome(
            reward=TerminalReward(
                value=1.0,
                success=True,
                success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
                success_threshold=None,
                native_metric_name="fixture",
                native_payload={"native_fields": native_fields},
                environment_id=task.environment_id,
                verifier_version="fixture-evaluator@1",
            ),
            episode_steps=2,
            tool_calls=1,
            model_requests=3,
            model_request_errors=0,
            parse_errors=0,
            wall_time_seconds=0.25,
            valid_actions=2,
            environment_admitted_actions=1,
            explicit_completions=1,
            scorer_invocations=1,
        )


def _state(step: int = 288) -> ProtocolV10ReadOnlyState:
    return ProtocolV10ReadOnlyState(
        policy_snapshot_id="policy-final",
        library_version="library-final",
        optimizer_step=step,
        projection_version="projection-final",
        detector_version="detector-final",
    )


def test_evaluation_input_and_record_are_strictly_round_trippable() -> None:
    exact = _exact_input()
    assert ProtocolV10EvaluationInput.from_value(exact.to_value()) == exact

    record = ProtocolV10EvaluationRecord(
        global_position=0,
        benchmark=BenchmarkV10.HOTPOT_QA,
        population_id="hotpot/final",
        source_id="private-source",
        reward=0.5,
        success=False,
        native_fields={"answer-f1": 0.5, "answer-exact-match": 0.0},
        environment_id="hotpot/private",
        verifier_version="hotpot-evaluator@1",
        episode_steps=2,
        tool_calls=0,
        model_requests=2,
        model_request_errors=0,
        parse_errors=0,
        wall_time_seconds=1.0,
    )
    assert ProtocolV10EvaluationRecord.from_value(record.to_value()) == record


def test_worker_resumes_records_without_repeating_evaluation(tmp_path: Path) -> None:
    cleanup_calls: list[str] = []
    catalog, sessions = _catalog_and_sessions(cleanup_calls)
    runner = _Runner([])
    exact = _exact_input()
    store = ProtocolV10EvaluationStore((tmp_path / "evaluation").resolve(), exact)
    worker = ProtocolV10EvaluationWorker(
        exact,
        catalog,
        sessions,
        runner,
        _state,
        store,
    )

    first = asyncio.run(worker.run())
    second = asyncio.run(worker.run())

    assert first == second
    assert len(first) == len(ACTIVE_BENCHMARKS_V10)
    assert len(runner.calls) == len(ACTIVE_BENCHMARKS_V10)
    assert cleanup_calls == runner.calls
    assert store.complete_path.is_file()
    assert store.input_path.stat().st_mode & 0o777 == 0o600
    assert store.records_path.stat().st_mode & 0o777 == 0o600


def test_worker_fails_closed_and_cleans_up_infrastructure_failure(tmp_path: Path) -> None:
    cleanup_calls: list[str] = []
    catalog, sessions = _catalog_and_sessions(cleanup_calls)
    failing = f"{ACTIVE_BENCHMARKS_V10[0].value}/task"
    runner = _Runner([], fail_task_id=failing)
    exact = _exact_input()
    store = ProtocolV10EvaluationStore((tmp_path / "evaluation").resolve(), exact)
    worker = ProtocolV10EvaluationWorker(exact, catalog, sessions, runner, _state, store)

    with pytest.raises(ProtocolV10EvaluationInfrastructureError) as raised:
        asyncio.run(worker.run())

    assert "private evaluator detail" not in str(raised.value)
    assert cleanup_calls == [failing]
    assert not store.complete_path.exists()
    assert store.open() == ()


def test_worker_rejects_training_state_mutation(tmp_path: Path) -> None:
    cleanup_calls: list[str] = []
    catalog, sessions = _catalog_and_sessions(cleanup_calls)
    reads = iter((_state(), _state(step=289)))
    exact = _exact_input()
    store = ProtocolV10EvaluationStore((tmp_path / "evaluation").resolve(), exact)
    worker = ProtocolV10EvaluationWorker(
        exact,
        catalog,
        sessions,
        _Runner([]),
        lambda: next(reads),
        store,
    )

    with pytest.raises(RuntimeError):
        asyncio.run(worker.run())

    assert not store.complete_path.exists()


def _record(
    position: int,
    *,
    benchmark: BenchmarkV10,
    success: bool,
    scenario: str | None = None,
) -> ProtocolV10EvaluationRecord:
    native: dict[str, object] = {"score": float(success)}
    if scenario is not None:
        native["scenario-id"] = scenario
        native["task-goal-completion"] = float(success)
    return ProtocolV10EvaluationRecord(
        global_position=position,
        benchmark=benchmark,
        population_id=f"{benchmark.value}/final",
        source_id=f"private-{position}",
        reward=float(success),
        success=success,
        native_fields=native,
        environment_id=f"{benchmark.value}/private",
        verifier_version="fixture@1",
        episode_steps=position + 1,
        tool_calls=1,
        model_requests=2,
        model_request_errors=int(not success),
        parse_errors=int(not success),
        wall_time_seconds=0.5,
        json_syntax_errors=int(not success),
        valid_actions=position + int(success),
        scorer_invocations=1,
    )


def test_aggregation_is_shard_order_independent_and_groups_appworld_sgc() -> None:
    inputs = (
        _exact_input(shard_index=0, shard_count=2, expected_record_count=4),
        _exact_input(shard_index=1, shard_count=2, expected_record_count=4),
    )
    records = (
        _record(0, benchmark=BenchmarkV10.HOTPOT_QA, success=True),
        _record(1, benchmark=BenchmarkV10.HOTPOT_QA, success=False),
        _record(2, benchmark=BenchmarkV10.APPWORLD, success=True, scenario="scenario-a"),
        _record(3, benchmark=BenchmarkV10.APPWORLD, success=False, scenario="scenario-a"),
    )

    report = aggregate_protocol_v10_evaluation(
        (inputs[1], inputs[0]),
        ((records[3], records[1]), (records[2], records[0])),
    )

    assert report.evaluated_records == 4
    assert report.candidate_failure_count == 2
    assert report.overall_reward == pytest.approx(0.5)
    assert report.overall_success_rate == pytest.approx(0.5)
    assert report.appworld_scenario_goal_completion is None
    assert report.parse_error_count == 2
    assert report.model_request_error_rate == pytest.approx(0.25)
    assert report.to_value()["checkpoint_id"] == "checkpoint-step-288"


def test_aggregation_rejects_a_missing_trailing_record() -> None:
    exact = _exact_input(expected_record_count=2)
    only = _record(0, benchmark=BenchmarkV10.HOTPOT_QA, success=True)

    with pytest.raises(ValueError):
        aggregate_protocol_v10_evaluation((exact,), ((only,),))


def test_reporting_distinguishes_zero_candidate_partial_and_no_submission() -> None:
    exact = _exact_input(expected_record_count=3)
    zero = _record(0, benchmark=BenchmarkV10.HOTPOT_QA, success=False)
    partial = replace(
        _record(1, benchmark=BenchmarkV10.HOTPOT_QA, success=False),
        reward=0.5,
    )
    no_submission = replace(
        _record(2, benchmark=BenchmarkV10.HOTPOT_QA, success=False),
        scorer_invocations=0,
        horizon_no_submissions=1,
    )

    report = aggregate_protocol_v10_evaluation(
        (exact,),
        ((zero, partial, no_submission),),
    )

    assert report.candidate_failure_count == 1
    assert report.posterior_failure_count == 2
    assert report.partial_reward_count == 1
    assert report.horizon_no_submission_count == 1
    assert report.submission_rate == pytest.approx(2 / 3)


def test_supervisor_merges_only_completed_shards_and_writes_private_report(
    tmp_path: Path,
) -> None:
    inputs = (
        _exact_input(shard_index=0, shard_count=2, expected_record_count=2),
        _exact_input(shard_index=1, shard_count=2, expected_record_count=2),
    )
    stores = tuple(
        ProtocolV10EvaluationStore((tmp_path / f"shard-{index}").resolve(), exact)
        for index, exact in enumerate(inputs)
    )
    for store, record in zip(
        stores,
        (
            _record(0, benchmark=BenchmarkV10.HOTPOT_QA, success=True),
            _record(1, benchmark=BenchmarkV10.HOTPOT_QA, success=False),
        ),
        strict=True,
    ):
        store.open()
        store.append(record)
        store.mark_complete()
    report_path = (tmp_path / "report" / "summary.json").resolve()
    supervisor = ProtocolV10EvaluationSupervisor(inputs, stores, report_path)

    first = supervisor.aggregate_completed()
    second = supervisor.aggregate_completed()

    assert first == second
    assert first.evaluated_records == 2
    assert report_path.is_file()
    assert report_path.stat().st_mode & 0o777 == 0o600
    assert "private-0" not in report_path.read_text(encoding="utf-8")


def test_supervisor_rejects_incomplete_shard(tmp_path: Path) -> None:
    exact = _exact_input(expected_record_count=1)
    store = ProtocolV10EvaluationStore((tmp_path / "shard").resolve(), exact)
    store.open()
    store.append(_record(0, benchmark=BenchmarkV10.HOTPOT_QA, success=True))
    supervisor = ProtocolV10EvaluationSupervisor(
        (exact,),
        (store,),
        (tmp_path / "report.json").resolve(),
    )

    with pytest.raises(RuntimeError):
        supervisor.aggregate_completed()
