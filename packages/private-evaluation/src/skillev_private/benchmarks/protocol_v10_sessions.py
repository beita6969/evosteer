"""Load private Protocol 10 records and build fail-closed session routes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from skillev.benchmarks import (
    ALFWorldPublicItem,
    BenchmarkPublicItem,
    CompletionBenchmarkEnvironment,
)
from skillev.benchmarks.webshop import WebShopPublicItem
from skillev.contracts import JsonValue, normalize_json
from skillev.experiments import Benchmark
from skillev.experiments.protocol_v10 import BenchmarkProtocolV10, BenchmarkV10
from skillev.rollout import RolloutSessionBundle, RolloutTask
from skillev.runtime import EnvironmentObservation, RolloutEnvironmentSession, StructuredAction

from .alfworld import PrivateALFWorldCase, PrivateALFWorldSessionFactory
from .alfworld_official import (
    OfficialALFWorldEpisodeFactory,
    OfficialALFWorldTask,
    OfficialALFWorldTextEnvFactory,
)
from .catalog import PrivateSessionFactory
from .code_math import CodeExecutionBackend, CodeExecutionRequest
from .evalplus_official import EvalPlusEvaluationCase, EvalPlusOfficialWorker
from .humaneval_official import IsolatedHumanEvalExecutionBackend
from .protocol_v10_evaluator import (
    ExistingTerminalEvaluatorNativeBackend,
    ProtocolV10NativeBackend,
    ProtocolV10NativeResult,
    ProtocolV10TerminalEvaluator,
)
from .protocol_v10_materialization import PRIVATE_RECORD_FORMAT
from .protocol_v10_official import (
    ExistingAppWorldTerminalEvaluatorNativeBackend,
    HealthBenchNativeBackend,
    HealthBenchOfficialGrader,
    ProtocolV10StaticBackend,
)
from .protocol_v10_population import (
    PrivateBenchmarkPopulation,
    ProtocolV10PopulationCatalog,
    ProtocolV10PopulationSessionRegistry,
)
from .static import PrivateStaticBenchmarkCase, PrivateStaticTarget, StaticScoringRule
from .terminal_inputs import submitted_value
from .webshop import PrivateWebShopCase, PrivateWebShopSessionFactory
from .webshop_official import (
    OfficialWebShopEpisodeFactory,
    OfficialWebShopGoal,
    OfficialWebShopTextEnvFactory,
)


@dataclass(frozen=True, slots=True)
class ProtocolV10PrivateRecord:
    source_id: str
    private_payload: JsonValue

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Protocol 10 private source identity is empty")
        object.__setattr__(self, "private_payload", normalize_json(self.private_payload))


def load_protocol_v10_private_record_file(path: Path) -> tuple[ProtocolV10PrivateRecord, ...]:
    """Read one owner-only verifier file without projecting payloads to tasks."""

    records: list[ProtocolV10PrivateRecord] = []
    with path.open(encoding="utf-8", newline="\n") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = normalize_json(json.loads(line))
            if not isinstance(value, dict) or set(value) != {
                "format",
                "private_payload",
                "source_id",
            }:
                raise ValueError("Protocol 10 private record has incompatible fields")
            if value["format"] != PRIVATE_RECORD_FORMAT or type(value["source_id"]) is not str:
                raise ValueError("Protocol 10 private record format is unsupported")
            records.append(
                ProtocolV10PrivateRecord(
                    value["source_id"],
                    value["private_payload"],
                )
            )
    source_ids = tuple(record.source_id for record in records)
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Protocol 10 private record file is empty or repeats a source")
    return tuple(records)


class ProtocolV10PopulationSessionBuilder(Protocol):
    """Deployment-owned constructor for one trusted population route."""

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory: ...


HealthBenchGraderFactory = Callable[
    [PrivateBenchmarkPopulation, tuple[ProtocolV10PrivateRecord, ...]],
    HealthBenchOfficialGrader,
]


@dataclass(slots=True)
class _ExactCompletionEnvironment:
    task: RolloutTask
    delegate: RolloutEnvironmentSession

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
        return await self.delegate.execute(action, step_index=step_index)

    def validate_completion(self, submission: JsonValue) -> bool:
        return self.delegate.validate_completion(submission)


@dataclass(frozen=True, slots=True)
class _ProtocolV10CompletionSessionFactory:
    routes: tuple[tuple[RolloutTask, ProtocolV10TerminalEvaluator], ...]

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        matches = tuple(route for route in self.routes if route[0].task_id == task.task_id)
        if len(matches) != 1 or matches[0][0] != task:
            raise ValueError("Protocol 10 completion task has no exact private route")
        source_task, evaluator = matches[0]
        context = source_task.public_context
        if not isinstance(context, dict) or any(
            type(context.get(field)) is not str
            for field in ("benchmark_id", "source_version", "population_id")
        ):
            raise ValueError("Protocol 10 completion context is incomplete")
        public = BenchmarkPublicItem(
            benchmark_id=cast(str, context["benchmark_id"]),
            dataset_revision=cast(str, context["source_version"]),
            split=cast(str, context["population_id"]),
            task_id=source_task.task_id,
            task_family=source_task.task_family,
            query=source_task.query,
            public_context=source_task.public_context,
        )
        environment = _ExactCompletionEnvironment(
            source_task,
            CompletionBenchmarkEnvironment(public),
        )
        return RolloutSessionBundle(
            environment=environment, evaluator=evaluator, retrieved_skills=()
        )


@dataclass(frozen=True, slots=True)
class _AdaptedSessionRoute:
    task: RolloutTask
    delegate_task: RolloutTask
    delegate: PrivateSessionFactory
    protocol: BenchmarkProtocolV10
    reward_field: str
    success_field: str


@dataclass(frozen=True, slots=True)
class _ProtocolV10AdaptedSessionFactory:
    routes: tuple[_AdaptedSessionRoute, ...]

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        matches = tuple(route for route in self.routes if route.task.task_id == task.task_id)
        if len(matches) != 1 or matches[0].task != task:
            raise ValueError("Protocol 10 adapted task has no exact private route")
        route = matches[0]
        bundle = route.delegate.create(route.delegate_task)
        evaluator = ProtocolV10TerminalEvaluator(
            route.protocol,
            ExistingTerminalEvaluatorNativeBackend(
                bundle.evaluator,
                route.protocol.benchmark,
                route.reward_field,
                route.success_field,
                route.delegate_task.environment_id,
                task.environment_id,
            ),
        )
        return RolloutSessionBundle(
            environment=_ExactCompletionEnvironment(task, bundle.environment),
            evaluator=evaluator,
            retrieved_skills=bundle.retrieved_skills,
            cleanup=bundle.cleanup,
        )


@dataclass(frozen=True, slots=True)
class _ProtocolV10AppWorldRoute:
    task: RolloutTask
    delegate_task: RolloutTask
    delegate: PrivateSessionFactory
    protocol: BenchmarkProtocolV10
    scenario_id: str
    expected_scenario_task_count: int


@dataclass(frozen=True, slots=True)
class _ProtocolV10AppWorldSessionFactory:
    routes: tuple[_ProtocolV10AppWorldRoute, ...]

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        matches = tuple(route for route in self.routes if route.task.task_id == task.task_id)
        if len(matches) != 1 or matches[0].task != task:
            raise ValueError("Protocol 10 AppWorld task has no exact private route")
        route = matches[0]
        bundle = route.delegate.create(route.delegate_task)
        return RolloutSessionBundle(
            environment=_ExactCompletionEnvironment(task, bundle.environment),
            evaluator=ProtocolV10TerminalEvaluator(
                route.protocol,
                ExistingAppWorldTerminalEvaluatorNativeBackend(
                    task_id=task.task_id,
                    scenario_id=route.scenario_id,
                    expected_scenario_task_count=route.expected_scenario_task_count,
                    evaluator=bundle.evaluator,
                ),
            ),
            retrieved_skills=bundle.retrieved_skills,
            cleanup=bundle.cleanup,
        )


def _completion_factory(
    routes: list[tuple[RolloutTask, ProtocolV10TerminalEvaluator]],
) -> _ProtocolV10CompletionSessionFactory:
    if not routes or len({task.task_id for task, _ in routes}) != len(routes):
        raise ValueError("Protocol 10 completion routes must be non-empty and task-unique")
    return _ProtocolV10CompletionSessionFactory(tuple(routes))


@dataclass(frozen=True, slots=True)
class ProtocolV10StaticSessionBuilder:
    """Build HotpotQA, TriviaQA, and AIME sessions from private answer rows."""

    verifier_version: str = "protocol-v10-static-official@1"

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark not in {
            BenchmarkV10.HOTPOT_QA,
            BenchmarkV10.TRIVIA_QA,
            BenchmarkV10.AIME_2026,
        }:
            raise ValueError("static Protocol 10 builder received a non-static benchmark")
        by_source = {record.source_id: record for record in records}
        routes = []
        for item in population.items:
            payload = by_source[item.source_id].private_payload
            if not isinstance(payload, dict):
                raise ValueError("static Protocol 10 payload must be an object")
            accepted = payload.get("accepted_answers")
            rule = payload.get("scoring_rule")
            if not isinstance(accepted, list) or any(
                type(answer) is not str for answer in accepted
            ):
                raise ValueError("static Protocol 10 answers are incompatible")
            if type(rule) is not str:
                raise ValueError("static Protocol 10 scoring rule is incompatible")
            public = BenchmarkPublicItem(
                benchmark_id=protocol.benchmark.value,
                dataset_revision=population.spec.source_version,
                split=population.spec.population_id,
                task_id=item.task.task_id,
                task_family=item.task.task_family,
                query=item.task.query,
                public_context=item.task.public_context,
            )
            case = PrivateStaticBenchmarkCase(
                public,
                PrivateStaticTarget(
                    item.source_id,
                    StaticScoringRule(rule),
                    tuple(cast(str, answer) for answer in accepted),
                ),
            )
            routes.append(
                (
                    item.task,
                    ProtocolV10TerminalEvaluator(
                        protocol,
                        ProtocolV10StaticBackend(
                            case,
                            protocol.benchmark,
                            self.verifier_version,
                            environment_id=item.task.environment_id,
                        ),
                    ),
                )
            )
        return _completion_factory(routes)


@dataclass(frozen=True, slots=True)
class ProtocolV10HealthSessionBuilder:
    """Bind answer-free health prompts to a deployment-owned private grader.

    The factory receives the complete owner-only population once, so official
    HealthBench rubrics and independent reference responses remain outside the
    rollout task and can be loaded by separate pinned grader implementations.
    """

    grader_factory: HealthBenchGraderFactory

    def __post_init__(self) -> None:
        if not callable(self.grader_factory):
            raise TypeError("HealthBench grader factory must be callable")

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark is not BenchmarkV10.HEALTHBENCH:
            raise ValueError("HealthBench builder received another benchmark")
        grader = self.grader_factory(population, records)
        if not callable(getattr(grader, "grade", None)):
            raise TypeError("HealthBench grader does not implement grade")
        routes = [
            (
                item.task,
                ProtocolV10TerminalEvaluator(
                    protocol,
                    HealthBenchNativeBackend(
                        task_id=item.task.task_id,
                        environment_id=item.task.environment_id,
                        grader=grader,
                    ),
                ),
            )
            for item in population.items
        ]
        return _completion_factory(routes)


@dataclass(frozen=True, slots=True)
class ProtocolV10WebShopSessionBuilder:
    """Adapt the pinned official WebShop process without changing V10 tasks."""

    deployment: OfficialWebShopTextEnvFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.deployment, "create", None)):
            raise TypeError("WebShop builder requires an official environment factory")

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark is not BenchmarkV10.WEBSHOP:
            raise ValueError("WebShop builder received another benchmark")
        by_source = {record.source_id: record for record in records}
        cases: list[PrivateWebShopCase] = []
        delegate_tasks: dict[str, RolloutTask] = {}
        for item in population.items:
            payload = by_source[item.source_id].private_payload
            context = item.task.public_context
            if (
                not isinstance(payload, dict)
                or type(payload.get("goal_index")) is not int
                or not isinstance(context, dict)
                or type(context.get("initial_observation")) is not str
                or not isinstance(context.get("initial_available_actions"), list)
                or any(
                    type(action) is not str
                    for action in cast(list[JsonValue], context["initial_available_actions"])
                )
            ):
                raise ValueError("WebShop private goal route is incompatible")
            public = WebShopPublicItem(
                dataset_revision=population.spec.source_version,
                environment_snapshot_id=population.spec.population_id,
                split=population.spec.population_id,
                task_id=item.task.task_id,
                task_family=item.task.task_family,
                query=item.task.query,
                public_context={
                    "initial_available_actions": context["initial_available_actions"],
                    "initial_observation": context["initial_observation"],
                    "scenario_id": item.source_id,
                },
            )
            delegate_task = public.to_rollout_task()
            delegate_tasks[item.source_id] = delegate_task
            goal = OfficialWebShopGoal(
                task_id=item.task.task_id,
                environment_id=public.environment_id,
                goal_id=item.source_id,
                session_id=f"{population.spec.population_id}:{item.source_id}",
                payload=payload["goal_index"],
            )
            cases.append(PrivateWebShopCase(public, goal))
        delegate = PrivateWebShopSessionFactory(
            tuple(cases),
            OfficialWebShopEpisodeFactory(self.deployment),
        )
        routes = []
        for item in population.items:
            delegate_task = delegate_tasks[item.source_id]
            routes.append(
                _AdaptedSessionRoute(
                    item.task,
                    delegate_task,
                    delegate,
                    protocol,
                    "native-score",
                    "native-success",
                )
            )
        return _ProtocolV10AdaptedSessionFactory(tuple(routes))


@dataclass(frozen=True, slots=True)
class ProtocolV10AppWorldSessionBuilder:
    """Route exact V10 tasks through the official AppWorld process session."""

    deployment: PrivateSessionFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.deployment, "create", None)):
            raise TypeError("AppWorld builder requires an official process session factory")

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark is not BenchmarkV10.APPWORLD:
            raise ValueError("AppWorld builder received another benchmark")
        by_source = {record.source_id: record for record in records}
        scenario_counts: dict[str, int] = {}
        for item in population.items:
            context = item.task.public_context
            scenario = context.get("scenario_id") if isinstance(context, dict) else None
            if type(scenario) is not str or not scenario.strip():
                raise ValueError("AppWorld public scenario identity is unavailable")
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        routes = []
        for item in population.items:
            payload = by_source[item.source_id].private_payload
            if (
                not isinstance(payload, dict)
                or type(payload.get("appworld_source_id")) is not str
                or type(payload.get("split")) is not str
            ):
                raise ValueError("AppWorld private task route is incompatible")
            context = item.task.public_context
            if (
                not isinstance(context, dict)
                or type(context.get("scenario_id")) is not str
                or not cast(str, context["scenario_id"]).strip()
            ):
                raise ValueError("AppWorld public scenario identity is unavailable")
            delegate_task = RolloutTask(
                task_id=item.task.task_id,
                environment_id=item.task.environment_id,
                task_family=item.task.task_family,
                context_id=item.task.context_id,
                query=item.task.query,
                available_tools=item.task.available_tools,
                public_context={
                    **context,
                    "source_id": payload["appworld_source_id"],
                    "split": payload["split"],
                },
                action_surface=item.task.action_surface,
                budget_profile=item.task.budget_profile,
                model_visible_messages=item.task.model_visible_messages,
            )
            routes.append(
                _ProtocolV10AppWorldRoute(
                    task=item.task,
                    delegate_task=delegate_task,
                    delegate=self.deployment,
                    protocol=protocol,
                    scenario_id=cast(str, context["scenario_id"]),
                    expected_scenario_task_count=scenario_counts[cast(str, context["scenario_id"])],
                )
            )
        return _ProtocolV10AppWorldSessionFactory(tuple(routes))


@dataclass(frozen=True, slots=True)
class ProtocolV10ALFWorldSessionBuilder:
    """Adapt reset-pinned Protocol 10 tasks to the official ALFWorld bridge."""

    deployment: OfficialALFWorldTextEnvFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.deployment, "create", None)):
            raise TypeError("ALFWorld builder requires an official environment factory")

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark is not BenchmarkV10.ALFWORLD:
            raise ValueError("ALFWorld builder received another benchmark")
        by_source = {record.source_id: record for record in records}
        cases = []
        delegate_tasks: dict[str, RolloutTask] = {}
        for item in population.items:
            context = item.task.public_context
            payload = by_source[item.source_id].private_payload
            if (
                not isinstance(context, dict)
                or type(context.get("initial_observation")) is not str
                or not isinstance(context.get("admissible_commands"), list)
                or not isinstance(payload, dict)
                or type(payload.get("game_id")) is not str
                or type(payload.get("seed")) is not int
                or type(payload.get("max_steps")) is not int
                or type(payload.get("trajectory_relative_path")) is not str
            ):
                raise ValueError("ALFWorld reset-pinned route is incompatible")
            commands = cast(list[JsonValue], context["admissible_commands"])
            if any(type(command) is not str for command in commands):
                raise ValueError("ALFWorld admissible commands must be text")
            initial_observation = cast(str, context["initial_observation"])
            game_id = cast(str, payload["game_id"])
            seed = cast(int, payload["seed"])
            max_steps = cast(int, payload["max_steps"])
            trajectory_relative_path = cast(str, payload["trajectory_relative_path"])
            public = ALFWorldPublicItem(
                dataset_revision=population.spec.source_version,
                environment_snapshot_id=population.spec.population_id,
                split=population.spec.population_id,
                task_id=item.task.task_id,
                task_family=item.task.task_family,
                query=item.task.query,
                public_context={
                    "admissible_commands": commands,
                    "initial_observation": initial_observation,
                },
                seed=seed,
                max_steps=max_steps,
            )
            delegate_tasks[item.source_id] = public.to_rollout_task()
            private_task = OfficialALFWorldTask(
                task_id=public.task_id,
                environment_id=public.environment_id,
                game_id=game_id,
                seed=seed,
                max_steps=max_steps,
                payload={"trajectory_relative_path": trajectory_relative_path},
            )
            cases.append(PrivateALFWorldCase(public, private_task))
        delegate = PrivateALFWorldSessionFactory(
            tuple(cases),
            OfficialALFWorldEpisodeFactory(self.deployment),
        )
        routes = tuple(
            _AdaptedSessionRoute(
                item.task,
                delegate_tasks[item.source_id],
                delegate,
                protocol,
                "episode-success",
                "episode-success",
            )
            for item in population.items
        )
        return _ProtocolV10AdaptedSessionFactory(routes)


@dataclass(frozen=True, slots=True)
class _HumanEvalNativeBackend:
    task: RolloutTask
    payload: dict[str, JsonValue]
    executor: CodeExecutionBackend
    verifier_version: str = "humaneval-isolated-evaluator@1"

    async def evaluate_native(self, request: object) -> ProtocolV10NativeResult:
        from skillev.rollout import NoTerminalSubmission, TerminalEvaluationRequest

        if (
            not isinstance(request, TerminalEvaluationRequest)
            or request.task_id != self.task.task_id
        ):
            raise ValueError("HumanEval request belongs to another task")
        passed = False
        if not isinstance(request.evaluation_input, NoTerminalSubmission):
            submission = normalize_json(submitted_value(request))
            if not isinstance(submission, dict) or set(submission) != {"answer"}:
                raise ValueError("code completion has incompatible fields")
            answer = submission["answer"]
            test = self.payload.get("tests")
            entry_point = self.payload.get("entry_point")
            if type(answer) is not str or type(test) is not str or type(entry_point) is not str:
                raise ValueError("HumanEval private payload is incompatible")
            result = await self.executor.run(
                CodeExecutionRequest(
                    task_id=self.task.task_id,
                    prompt=self.task.query,
                    completion=answer,
                    test_source=test,
                    entry_point=entry_point,
                )
            )
            passed = result.passed
        return ProtocolV10NativeResult(
            task_id=self.task.task_id,
            benchmark=BenchmarkV10.MBPP_PLUS_FIXED_100,
            native_fields={"all-evalplus-tests-pass": float(passed)},
            environment_id=self.task.environment_id,
            verifier_version=self.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class _EvalPlusNativeBackend:
    task: RolloutTask
    worker: EvalPlusOfficialWorker

    async def evaluate_native(self, request: object) -> ProtocolV10NativeResult:
        from skillev.rollout import NoTerminalSubmission, TerminalEvaluationRequest

        if (
            not isinstance(request, TerminalEvaluationRequest)
            or request.task_id != self.task.task_id
        ):
            raise ValueError("EvalPlus request belongs to another task")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            result = self.worker.no_submission_result(task_id=self.task.task_id)
        else:
            submission = normalize_json(submitted_value(request))
            if not isinstance(submission, dict) or set(submission) != {"answer"}:
                raise ValueError("EvalPlus completion has incompatible fields")
            answer = submission["answer"]
            if type(answer) is not str or not answer.strip():
                raise ValueError("EvalPlus completion answer is empty")
            result = await self.worker.evaluate(task_id=self.task.task_id, submission=answer)
        return ProtocolV10NativeResult(
            task_id=self.task.task_id,
            benchmark=BenchmarkV10.MBPP_PLUS_FIXED_100,
            native_fields={"all-evalplus-tests-pass": float(result.success)},
            environment_id=self.task.environment_id,
            verifier_version=result.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10CodeSessionBuilder:
    """Use HumanEval for isolated training and official EvalPlus for fixed-100."""

    humaneval_executor: CodeExecutionBackend = field(
        default_factory=IsolatedHumanEvalExecutionBackend
    )

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark is not BenchmarkV10.MBPP_PLUS_FIXED_100:
            raise ValueError("code builder received another benchmark")
        by_source = {record.source_id: record for record in records}
        evalplus_cases: list[EvalPlusEvaluationCase] = []
        for item in population.items:
            payload = by_source[item.source_id].private_payload
            if not isinstance(payload, dict):
                raise ValueError("code private payload must be an object")
            official = payload.get("official_row")
            if official is not None:
                if not isinstance(official, dict):
                    raise ValueError("EvalPlus official row must be an object")
                evalplus_cases.append(
                    EvalPlusEvaluationCase(
                        benchmark=Benchmark.MBPP_PLUS,
                        task_id=item.task.task_id,
                        official_row=official,
                    )
                )
        worker = (
            EvalPlusOfficialWorker(Benchmark.MBPP_PLUS, tuple(evalplus_cases))
            if evalplus_cases
            else None
        )
        routes: list[tuple[RolloutTask, ProtocolV10TerminalEvaluator]] = []
        for item in population.items:
            payload = by_source[item.source_id].private_payload
            if not isinstance(payload, dict):  # pragma: no cover - checked above
                raise TypeError("code private payload must be an object")
            backend: ProtocolV10NativeBackend
            if payload.get("official_row") is not None:
                if worker is None:  # pragma: no cover - construction invariant
                    raise RuntimeError("EvalPlus worker is unavailable")
                backend = _EvalPlusNativeBackend(item.task, worker)
            else:
                backend = _HumanEvalNativeBackend(item.task, payload, self.humaneval_executor)
            routes.append((item.task, ProtocolV10TerminalEvaluator(protocol, backend)))
        return _completion_factory(routes)


def build_protocol_v10_population_sessions(
    catalog: ProtocolV10PopulationCatalog,
    private_record_files: Mapping[str, Path],
    builders: Mapping[BenchmarkV10, ProtocolV10PopulationSessionBuilder],
) -> ProtocolV10PopulationSessionRegistry:
    """Bind every public population to exactly one trusted deployment builder."""

    expected_ids = {population.spec.population_id for population in catalog.populations}
    if set(private_record_files) != expected_ids:
        raise ValueError("Protocol 10 private record files do not cover the catalog")
    if set(builders) != set(BenchmarkV10):
        raise ValueError("Protocol 10 session builders do not cover all nine benchmarks")
    benchmark_protocols = {protocol.benchmark: protocol for protocol in catalog.protocol.benchmarks}
    routes = []
    for population in catalog.populations:
        records = load_protocol_v10_private_record_file(
            private_record_files[population.spec.population_id]
        )
        if tuple(record.source_id for record in records) != tuple(
            item.source_id for item in population.items
        ):
            raise ValueError("Protocol 10 public and private population order differs")
        factory = builders[population.spec.benchmark].build(
            benchmark_protocols[population.spec.benchmark],
            population,
            records,
        )
        routes.append((population.spec.population_id, factory))
    return ProtocolV10PopulationSessionRegistry(tuple(routes))


__all__ = [
    "ProtocolV10ALFWorldSessionBuilder",
    "ProtocolV10AppWorldSessionBuilder",
    "ProtocolV10CodeSessionBuilder",
    "ProtocolV10HealthSessionBuilder",
    "ProtocolV10PopulationSessionBuilder",
    "ProtocolV10PrivateRecord",
    "ProtocolV10StaticSessionBuilder",
    "ProtocolV10WebShopSessionBuilder",
    "build_protocol_v10_population_sessions",
    "load_protocol_v10_private_record_file",
]
