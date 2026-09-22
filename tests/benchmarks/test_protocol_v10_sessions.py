from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from skillev_private.benchmarks.alfworld_official import (
    OfficialALFWorldResetResult,
    OfficialALFWorldStepResult,
    OfficialALFWorldTask,
)
from skillev_private.benchmarks.code_math import CodeExecutionResult, CodeExecutionStatus
from skillev_private.benchmarks.protocol_v10_materialization import (
    ProtocolV10SourceRecord,
    materialize_protocol_v10_population,
)
from skillev_private.benchmarks.protocol_v10_official import HealthBenchGrade
from skillev_private.benchmarks.protocol_v10_sessions import (
    ProtocolV10ALFWorldSessionBuilder,
    ProtocolV10AppWorldSessionBuilder,
    ProtocolV10CodeSessionBuilder,
    ProtocolV10HealthSessionBuilder,
    ProtocolV10PrivateRecord,
    ProtocolV10StaticSessionBuilder,
    ProtocolV10WebShopSessionBuilder,
    load_protocol_v10_private_record_file,
)
from skillev_private.benchmarks.webshop_official import (
    OfficialWebShopGoal,
    OfficialWebShopStepResult,
)

from skillev.contracts import SuccessRule, TerminalReward, stable_hash
from skillev.experiments import (
    BenchmarkProtocolV10,
    BenchmarkV10,
    PopulationRole,
    load_active_protocol_v10,
)
from skillev.rollout import (
    RolloutSessionBundle,
    RolloutTask,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
)
from skillev.runtime import ActionKind, EnvironmentObservation, StructuredAction

ROOT = Path(__file__).parents[2]


def _source_record() -> tuple[BenchmarkProtocolV10, ProtocolV10SourceRecord]:
    protocol = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    benchmark = protocol.benchmark(BenchmarkV10.HOTPOT_QA)
    spec = benchmark.population(PopulationRole.TRAINING)[0]
    task = RolloutTask(
        task_id="hotpotqa/source/1",
        environment_id="benchmark:hotpotqa@fixture:completion",
        task_family="hotpotqa/completion",
        context_id="hotpotqa:training",
        query="Which answer is supported by the evidence?",
        available_tools=(),
        public_context={
            "benchmark_id": "hotpotqa",
            "population_id": spec.population_id,
            "source_version": spec.source_version,
        },
    )
    return benchmark, ProtocolV10SourceRecord(
        task.task_id,
        task,
        {"accepted_answers": ["private answer"], "scoring_rule": "token-f1"},
    )


def test_private_record_loader_preserves_verifier_payload_only(tmp_path: Path) -> None:
    _, source = _source_record()
    path = tmp_path / "private.jsonl"
    path.write_text(json.dumps(source.private_value()) + "\n", encoding="utf-8")

    loaded = load_protocol_v10_private_record_file(path)

    assert loaded[0].source_id == source.source_id
    assert loaded[0].private_payload == source.private_payload
    assert "private answer" not in source.task.query


def test_static_session_uses_exact_materialized_environment_identity(tmp_path: Path) -> None:
    benchmark, source = _source_record()
    materialized = materialize_protocol_v10_population(
        benchmark.population(PopulationRole.TRAINING)[0],
        (source,),
    )
    path = tmp_path / "private.jsonl"
    path.write_text(json.dumps(source.private_value()) + "\n", encoding="utf-8")
    records = load_protocol_v10_private_record_file(path)
    factory = ProtocolV10StaticSessionBuilder().build(
        benchmark,
        materialized.population,
        records,
    )

    bundle = factory.create(source.task)
    reward = asyncio.run(
        bundle.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=source.task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue({"answer": "private answer"}),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )

    assert bundle.environment.environment_id == source.task.environment_id
    assert reward.environment_id == source.task.environment_id
    assert reward.value == 1.0
    assert reward.success is True


class _HealthGrader:
    verifier_version = "health-fixture@1"

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        assert task_id
        assert candidate_answer
        return HealthBenchGrade(0.75, 0)


class _PassingCodeExecutor:
    async def run(self, request: object) -> CodeExecutionResult:
        assert request is not None
        return CodeExecutionResult(CodeExecutionStatus.PASSED)


def _completion_source(
    benchmark: BenchmarkV10,
    private_payload: object,
) -> tuple[BenchmarkProtocolV10, ProtocolV10SourceRecord]:
    active = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    protocol = active.benchmark(benchmark)
    spec = protocol.population(PopulationRole.TRAINING)[0]
    public_context = {
        "benchmark_id": benchmark.value,
        "population_id": spec.population_id,
        "source_version": spec.source_version,
    }
    if benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
        public_context["code_signature"] = "solve()"
    task = RolloutTask(
        task_id=f"{benchmark.value}/source/1",
        environment_id=f"benchmark:{benchmark.value}@fixture:completion",
        task_family=f"{benchmark.value}/completion",
        context_id=f"{benchmark.value}:training",
        query="Return a candidate answer.",
        available_tools=(),
        public_context=public_context,
    )
    return protocol, ProtocolV10SourceRecord(task.task_id, task, private_payload)


def test_health_session_keeps_grader_payload_behind_terminal_boundary() -> None:
    protocol, source = _completion_source(
        BenchmarkV10.HEALTHBENCH,
        {"grader_kind": "fixture", "rubrics": ["private"]},
    )
    population = materialize_protocol_v10_population(
        protocol.population(PopulationRole.TRAINING)[0],
        (source,),
    ).population
    builder = ProtocolV10HealthSessionBuilder(lambda _population, _records: _HealthGrader())
    factory = builder.build(
        protocol,
        population,
        (ProtocolV10PrivateRecord(source.source_id, source.private_payload),),
    )

    reward = asyncio.run(
        factory.create(source.task).evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=source.task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue({"answer": "candidate"}),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )

    assert reward.value == 0.75
    assert reward.success is True
    assert "private" not in json.dumps(reward.native_payload)


def test_code_training_session_uses_isolated_executor_projection() -> None:
    protocol, source = _completion_source(
        BenchmarkV10.MBPP_PLUS_FIXED_100,
        {"entry_point": "solve", "evaluator_kind": "humaneval-sandbox", "tests": "pass"},
    )
    population = materialize_protocol_v10_population(
        protocol.population(PopulationRole.TRAINING)[0],
        (source,),
    ).population
    factory = ProtocolV10CodeSessionBuilder(_PassingCodeExecutor()).build(
        protocol,
        population,
        (ProtocolV10PrivateRecord(source.source_id, source.private_payload),),
    )
    reward = asyncio.run(
        factory.create(source.task).evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=source.task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue({"answer": "def solve(): return 1"}),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )

    assert reward.value == 1.0
    assert reward.success is True


@dataclass(slots=True)
class _WebShopEnvironment:
    goal: OfficialWebShopGoal
    query: str
    terminal: bool = False

    @property
    def goal_id(self) -> str:
        return self.goal.goal_id

    @property
    def session_id(self) -> str:
        return self.goal.session_id

    @property
    def instruction_text(self) -> str:
        return self.query

    def reset(self, session_id: str) -> str:
        assert session_id == self.goal.session_id
        return "public landing page"

    def step(self, action: str) -> OfficialWebShopStepResult:
        self.terminal = action == "click[buy now]"
        return OfficialWebShopStepResult(
            observation_text="public purchase result",
            reward=float(self.terminal),
            terminal=self.terminal,
        )

    def get_available_actions(self) -> tuple[str, ...]:
        return () if self.terminal else ("click[buy now]",)

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _WebShopFactory:
    query: str

    def create(self, goal: OfficialWebShopGoal) -> _WebShopEnvironment:
        return _WebShopEnvironment(goal, self.query)


def test_webshop_session_adapts_official_process_to_exact_v10_task() -> None:
    active = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    protocol = active.benchmark(BenchmarkV10.WEBSHOP)
    spec = protocol.population(PopulationRole.TRAINING)[0]
    task = RolloutTask(
        task_id="webshop/goal-00001",
        environment_id="benchmark:webshop@fixture:official-environment",
        task_family="webshop/shopping",
        context_id="webshop:training",
        query="Buy the requested public item.",
        available_tools=("click", "purchase", "search"),
        public_context={
            "benchmark_id": "webshop",
            "initial_available_actions": ["click[buy now]"],
            "initial_observation": "public landing page",
            "population_id": spec.population_id,
            "scenario_id": "webshop/goal-00001",
            "source_version": spec.source_version,
        },
    )
    source = ProtocolV10SourceRecord(task.task_id, task, {"goal_index": 1})
    population = materialize_protocol_v10_population(spec, (source,)).population
    factory = ProtocolV10WebShopSessionBuilder(_WebShopFactory(task.query)).build(
        protocol,
        population,
        (ProtocolV10PrivateRecord(source.source_id, source.private_payload),),
    )

    bundle = factory.create(task)
    observation = asyncio.run(
        bundle.environment.execute(
            StructuredAction(
                kind=ActionKind.TOOL,
                name="purchase",
                arguments={},
                resource_id="webshop",
            ),
            step_index=1,
        )
    )
    reward = asyncio.run(
        bundle.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue({"webshop_episode_terminal": True}),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )

    assert observation.terminal is True
    assert bundle.environment.environment_id == task.environment_id
    assert reward.environment_id == task.environment_id
    assert reward.value == 1.0
    assert reward.success is True
    assert bundle.cleanup is not None
    asyncio.run(bundle.cleanup())


@dataclass(frozen=True, slots=True)
class _AppWorldEnvironment:
    task: RolloutTask

    @property
    def environment_id(self) -> str:
        return self.task.environment_id

    @property
    def task_family(self) -> str:
        return self.task.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        raise AssertionError("AppWorld evaluation fixture does not execute tools")


@dataclass(frozen=True, slots=True)
class _AppWorldEvaluator:
    task: RolloutTask

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        assert request.task_id == self.task.task_id
        return TerminalReward(
            value=0.75,
            success=True,
            success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
            success_threshold=None,
            native_metric_name="appworld-test-pass-rate",
            native_payload={"benchmark_id": "appworld"},
            environment_id=self.task.environment_id,
            verifier_version="appworld-official-worker@fixture",
        )


@dataclass(slots=True)
class _AppWorldDeployment:
    created_task: RolloutTask | None = None

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        self.created_task = task
        return RolloutSessionBundle(
            environment=_AppWorldEnvironment(task),
            evaluator=_AppWorldEvaluator(task),
            retrieved_skills=(),
        )


def test_appworld_session_keeps_private_source_route_out_of_v10_task() -> None:
    active = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    protocol = active.benchmark(BenchmarkV10.APPWORLD)
    spec = protocol.population(PopulationRole.TRAINING)[0]
    task = RolloutTask(
        task_id="appworld/task-0001",
        environment_id="benchmark:appworld@fixture:official-environment",
        task_family="appworld/scenario",
        context_id="appworld:training",
        query="Complete the public scenario.",
        available_tools=("appworld.execute", "submit"),
        public_context={
            "benchmark_id": "appworld",
            "population_id": spec.population_id,
            "scenario_id": "appworld/scenario-0001",
            "source_version": spec.source_version,
        },
    )
    source = ProtocolV10SourceRecord(
        task.task_id,
        task,
        {"appworld_source_id": "task-0001", "split": "train"},
    )
    population = materialize_protocol_v10_population(spec, (source,)).population
    deployment = _AppWorldDeployment()
    factory = ProtocolV10AppWorldSessionBuilder(deployment).build(
        protocol,
        population,
        (ProtocolV10PrivateRecord(source.source_id, source.private_payload),),
    )

    bundle = factory.create(task)
    assert deployment.created_task is not None
    assert deployment.created_task.public_context["source_id"] == "task-0001"
    assert "source_id" not in task.public_context
    assert bundle.environment.environment_id == task.environment_id

    reward = asyncio.run(
        bundle.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue(
                    {"benchmark_id": "appworld", "task_id": task.task_id}
                ),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )
    assert reward.value == 0.75
    assert reward.success is True
    assert reward.native_payload["native_fields"] == {
        "official-task-success": 1.0,
        "scenario-expected-task-count": 1,
        "scenario-id": "appworld/scenario-0001",
        "task-goal-completion": 0.75,
    }


@dataclass(slots=True)
class _ALFWorldEnvironment:
    task: OfficialALFWorldTask

    @property
    def game_id(self) -> str:
        return self.task.game_id

    @property
    def seed(self) -> int:
        return self.task.seed

    @property
    def max_steps(self) -> int:
        return self.task.max_steps

    def reset(self, seed: int) -> OfficialALFWorldResetResult:
        assert seed == self.task.seed
        return OfficialALFWorldResetResult(
            observation_text="You are in a public room.",
            instruction_text="Place the public object on the table.",
            admissible_commands=("put object on table",),
        )

    def step(self, action: str) -> OfficialALFWorldStepResult:
        assert action == "put object on table"
        return OfficialALFWorldStepResult(
            observation_text="The public action completed.",
            admissible_commands=(),
            terminal=True,
            success=True,
        )

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _ALFWorldFactory:
    def create(self, task: OfficialALFWorldTask) -> _ALFWorldEnvironment:
        return _ALFWorldEnvironment(task)


def test_alfworld_session_uses_reset_pinned_public_context_and_private_game_route() -> None:
    active = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    protocol = active.benchmark(BenchmarkV10.ALFWORLD)
    spec = protocol.population(PopulationRole.TRAINING)[0]
    task = RolloutTask(
        task_id="alfworld/game-1",
        environment_id="benchmark:alfworld@fixture:official-environment",
        task_family="alfworld/pick_and_place",
        context_id="alfworld:training",
        query="Place the public object on the table.",
        available_tools=("act",),
        public_context={
            "admissible_commands": ["put object on table"],
            "benchmark_id": "alfworld",
            "initial_observation": "You are in a public room.",
            "population_id": spec.population_id,
            "scenario_id": "alfworld/game-1",
            "source_version": spec.source_version,
        },
    )
    source = ProtocolV10SourceRecord(
        task.task_id,
        task,
        {
            "game_id": "train/game-1",
            "max_steps": 8,
            "seed": 0,
            "trajectory_relative_path": "train/game-1",
        },
    )
    population = materialize_protocol_v10_population(spec, (source,)).population
    factory = ProtocolV10ALFWorldSessionBuilder(_ALFWorldFactory()).build(
        protocol,
        population,
        (ProtocolV10PrivateRecord(source.source_id, source.private_payload),),
    )

    bundle = factory.create(task)
    observation = asyncio.run(
        bundle.environment.execute(
            StructuredAction(
                kind=ActionKind.TOOL,
                name="act",
                arguments={"command": "put object on table"},
                resource_id="alfworld",
            ),
            step_index=1,
        )
    )
    reward = asyncio.run(
        bundle.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue(
                    {"benchmark_id": "alfworld", "episode_terminal": True}
                ),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )

    assert observation.terminal is True
    assert bundle.environment.environment_id == task.environment_id
    assert reward.environment_id == task.environment_id
    assert reward.value == 1.0
    assert reward.success is True
    assert bundle.cleanup is not None
    asyncio.run(bundle.cleanup())
