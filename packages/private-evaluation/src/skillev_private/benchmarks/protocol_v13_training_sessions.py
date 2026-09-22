"""Trusted runtime sessions for the private Protocol 13 training stream.

The dataset deliberately stores environment reset state outside the model input.
This module hydrates that public state from the pinned official deployments and
keeps every answer, rubric, test, and native reward behind a terminal evaluator.
It is execution glue only; the BayesianImprove application and training method
remain unchanged.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from skillev.benchmarks import (
    ALFWorldPublicItem,
    BenchmarkPublicItem,
    CompletionBenchmarkEnvironment,
)
from skillev.benchmarks.alfworld import ALFWorldEnvironment
from skillev.benchmarks.reasoning_guidance import HOTPOT_DELIBERATION
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, canonical_json, normalize_json
from skillev.contracts.observed_reset import (
    ALFWORLD_RESET_KIND,
    RESET_BINDING_FORMAT,
    STATIC_RESET_KIND,
)
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.owner_final import parse_explicit_integer_payload, project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode
from skillev.rollout import (
    ActionSurface,
    NoTerminalSubmission,
    RolloutBudgetProfile,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluator,
    TerminalEvaluatorError,
    UnskilledRolloutSessionBundle,
)
from skillev.rollout.action_surface import ACTION_SURFACE_FORMAT_V3, PublicActionInstructions
from skillev.runtime import EnvironmentObservation, RolloutEnvironmentSession, StructuredAction
from skillev.training import AsyncResourceLimiter, RolloutWorkflowResources

from .alfworld import PrivateALFWorldCase, PrivateALFWorldTerminalEvaluator
from .alfworld_official import (
    OfficialALFWorldEpisodeFactory,
    OfficialALFWorldTask,
    bind_reset_public_item,
)
from .alfworld_public_goal import LEGACY_GOAL_BINDING
from .code_math import CodeExecutionRequest
from .evaluation_episode import EvaluationEpisodeRecord
from .healthbench_qwen_sglang import (
    HEALTHBENCH_QWEN_VERIFIER,
    HealthBenchQwenGraderConfig,
    HealthBenchQwenVerifierIdentity,
    QwenSGLangHealthBenchGrader,
)
from .humaneval_official import IsolatedHumanEvalExecutionBackend
from .mbpp_scoring import (
    MBPPScorerProfile,
    decode_mbpp_verdict,
    mbpp_failure_kind,
    mbpp_request,
    resolve_mbpp_profile,
)
from .official_process import (
    ALFWorldGameDeployment,
    OfficialALFWorldProcessFactory,
    PinnedOfficialProcess,
)
from .protocol_v10_official import HealthBenchOfficialGrader
from .protocol_v10_workers import _HEALTHBENCH_WORKER, PrivateJSONWorker
from .protocol_v13_session_deployments import Protocol13TrainingDeployments
from .protocol_v13_training import Protocol13TrainingRecord
from .qa_diagnostics import qa_answer_diagnostics
from .qa_metrics import (
    best_alias_metrics,
    normalize_triviaqa_answer,
    score_hotpotqa_answers,
)
from .static import parse_aime_answer
from .terminal_inputs import submitted_value

NativeEpisodeRecord = Protocol13TrainingRecord | EvaluationEpisodeRecord


_STATIC_VERIFIER = "protocol13-training-static@3"
_HUMANEVAL_VERIFIER = "humaneval-public-context@2-original-isolated@2"


def native_scorer_contracts(
    mbpp_profile: MBPPScorerProfile, healthbench_judge: str, *, domains: Iterable[str]
) -> dict[str, JsonValue]:
    """Native scorer contracts for the declared training/evaluation population."""
    from skillev.evaluation.healthbench_luna_profile import healthbench_condition

    from .alfworld import ALFWORLD_VERIFIER

    catalog: dict[str, JsonValue] = {
        "hotpotqa": {
            "verifier": _STATIC_VERIFIER,
            "metric": "answer-f1",
            "success": "answer-exact-match",
            "projection": StepZeroTerminalMode.SHORT_ANSWER.value,
        },
        "triviaqa": {
            "verifier": _STATIC_VERIFIER,
            "metric": "answer-f1",
            "success": "answer-exact-match",
            "projection": StepZeroTerminalMode.SHORT_ANSWER.value,
        },
        "aime-2026": {
            "verifier": _STATIC_VERIFIER,
            "metric": "accuracy",
            "success": "exact-integer",
            "projection": StepZeroTerminalMode.AIME_INTEGER.value,
        },
        "healthbench": healthbench_condition(healthbench_judge),
        "alfworld": {
            "verifier": ALFWORLD_VERIFIER,
            "metric": "alfworld-success",
            "success": "official-environment-terminal",
        },
        "mbpp-plus": mbpp_profile.to_value(),
        "humaneval": {
            "verifier": _HUMANEVAL_VERIFIER,
            "metric": "pass@1",
            "success": "original-tests",
            "projection": StepZeroTerminalMode.PYTHON_SOURCE.value,
        },
    }
    return {domain: catalog[domain] for domain in domains}


def _target(record: NativeEpisodeRecord) -> dict[str, JsonValue]:
    return record.output.target


def _answer(request: TerminalEvaluationRequest) -> str | None:
    if isinstance(request.evaluation_input, NoTerminalSubmission):
        return None
    value = normalize_json(submitted_value(request))
    if not isinstance(value, dict) or set(value) != {"answer"}:
        raise TerminalEvaluatorError("completion has incompatible fields")
    answer = value["answer"]
    if type(answer) is not str or not answer.strip():
        raise TerminalEvaluatorError("completion answer is empty")
    return answer


def _projected_answer(request: TerminalEvaluationRequest, mode: StepZeroTerminalMode) -> str | None:
    """Use the clean evaluator's answer-blind projection of this submission only.

    Neither a reference answer nor another model response enters this projection.
    The rollout artifact still contains the original action and sampled tokens.
    """
    answer = _answer(request)
    if answer is None:
        return None
    submission = project_owner_final(
        mode, answer, owner_id="rollout-policy", message_id=request.trajectory_id
    )
    return submission.payload if submission is not None else None


def _reward(
    *,
    task: RolloutTask,
    value: float,
    success: bool,
    metric: str,
    verifier: str,
    fields: dict[str, JsonValue],
) -> TerminalReward:
    return TerminalReward(
        value=value,
        success=success,
        success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
        success_threshold=None,
        native_metric_name=metric,
        native_payload={
            "benchmark_id": cast(dict[str, JsonValue], task.public_context)["benchmark_id"],
            **fields,
        },
        environment_id=task.environment_id,
        verifier_version=verifier,
    )


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

    async def execute(self, action: StructuredAction, *, step_index: int) -> EnvironmentObservation:
        return await self.delegate.execute(action, step_index=step_index)

    def validate_completion(self, submission: JsonValue) -> bool:
        return self.delegate.validate_completion(submission)


def _completion_environment(task: RolloutTask) -> _ExactCompletionEnvironment:
    context = cast(dict[str, JsonValue], task.public_context)
    public = BenchmarkPublicItem(
        benchmark_id=cast(str, context["benchmark_id"]),
        dataset_revision=cast(str, context["dataset_revision"]),
        split=cast(str, context["split"]),
        task_id=task.task_id,
        task_family=task.task_family,
        query=task.query,
        public_context=context.get("payload"),
    )
    return _ExactCompletionEnvironment(task, CompletionBenchmarkEnvironment(public))


@dataclass(frozen=True, slots=True)
class _StaticEvaluator:
    record: NativeEpisodeRecord
    task: RolloutTask

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("static request reached another task")
        benchmark = self.record.episode.benchmark
        answer = _projected_answer(
            request,
            StepZeroTerminalMode.AIME_INTEGER
            if benchmark is Protocol13Benchmark.AIME_2026
            else StepZeroTerminalMode.SHORT_ANSWER,
        )
        accepted = _target(self.record).get("accepted_answers")
        if not isinstance(accepted, list) or any(type(item) is not str for item in accepted):
            raise TerminalEvaluatorError("static private answers are incompatible")
        aliases = tuple(cast(list[str], accepted))
        if answer is None:
            value = exact = 0.0
        elif benchmark is Protocol13Benchmark.HOTPOT_QA:
            result = score_hotpotqa_answers(answer, aliases)
            value, exact = result.f1, result.em
        elif benchmark is Protocol13Benchmark.TRIVIA_QA:
            result = best_alias_metrics(answer, aliases, normalize=normalize_triviaqa_answer)
            value, exact = result.f1, result.em
        elif benchmark is Protocol13Benchmark.AIME_2026:
            parsed = parse_explicit_integer_payload(answer)
            expected = {parse_aime_answer(item) for item in aliases}
            value = exact = float(parsed is not None and str(parsed) in expected)
        else:  # pragma: no cover - construction closes this branch
            raise TerminalEvaluatorError("static evaluator received another benchmark")
        return _reward(
            task=self.task,
            value=value,
            success=bool(exact),
            metric="answer-f1" if benchmark is not Protocol13Benchmark.AIME_2026 else "accuracy",
            verifier=_STATIC_VERIFIER,
            fields={
                "answer-exact-match": exact,
                "qa_diagnostics": None
                if benchmark is Protocol13Benchmark.AIME_2026
                else qa_answer_diagnostics(
                    benchmark.value,
                    original_submission=_answer(request),
                    projected_answer=answer,
                    accepted_aliases=aliases,
                ),
                "public_metrics": (
                    {"accuracy": value}
                    if benchmark is Protocol13Benchmark.AIME_2026
                    else {"answer-exact-match": exact, "answer-f1": value}
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class _HealthEvaluator:
    task: RolloutTask
    grader: HealthBenchOfficialGrader
    judge_profile: str = "qwen-local@1"

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("HealthBench request reached another task")
        answer = _answer(request)
        if answer is None:
            raw, negative = 0.0, 0
        else:
            try:
                grade = await self.grader.grade(self.task.task_id, answer)
            except Exception as error:
                raise TerminalEvaluatorError("HealthBench judge infrastructure failed") from error
            raw = float(grade.official_rubric_score)
            negative = grade.triggered_negative_rubric_count
        clipped = min(1.0, max(0.0, raw))
        from skillev.evaluation.healthbench_luna_profile import (
            LEGACY_TRAINING_JUDGE,
            METRIC,
            healthbench_condition,
        )

        external = self.judge_profile != LEGACY_TRAINING_JUDGE
        metric = METRIC if external else "qwen-local-rubric-score"
        return _reward(
            task=self.task,
            value=clipped,
            success=raw >= 0.60 and negative == 0,
            metric=metric,
            verifier=self.grader.verifier_version,
            fields={
                "triggered-negative-rubric-count": negative,
                "native_raw_score": None if answer is None else raw,
                "learning_reward": clipped,
                "binary_success": raw >= 0.60 and negative == 0,
                "negative_criterion_count": None if answer is None else negative,
                "criterion_ledger": None if answer is None else grade.criterion_ledger,
                "public_metrics": {metric: raw},
                **(
                    {
                        "grader_profile": healthbench_condition(self.judge_profile),
                        "grader_cost": normalize_json(grade.grader_cost)
                        if answer is not None
                        else None,
                    }
                    if external
                    else {}
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class _HumanEvalEvaluator:
    record: NativeEpisodeRecord
    task: RolloutTask
    executor: IsolatedHumanEvalExecutionBackend

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("HumanEval request reached another task")
        answer = _projected_answer(request, StepZeroTerminalMode.PYTHON_SOURCE)
        passed = False
        if answer is not None:
            target = _target(self.record)
            entry = target.get("entry_point")
            tests = target.get("test")
            if type(entry) is not str or type(tests) is not str:
                raise TerminalEvaluatorError("HumanEval private target is incompatible")
            from .public_code_context import (
                CodeSubmission,
                PublicCodeContext,
                compose_code_candidate,
            )

            assembled = compose_code_candidate(
                PublicCodeContext(self.task.query, entry), CodeSubmission(answer)
            )
            result = await self.executor.run(
                CodeExecutionRequest(
                    task_id=self.task.task_id,
                    prompt=assembled.prefix,
                    completion=assembled.completion,
                    test_source=tests,
                    entry_point=entry,
                )
            )
            passed = result.passed
        return _reward(
            task=self.task,
            value=float(passed),
            success=passed,
            metric="pass@1",
            verifier=_HUMANEVAL_VERIFIER,
            fields={
                "passed": passed,
                "native_diagnostics": None
                if answer is None
                else {
                    **assembled.diagnostics(),
                    **result.diagnostics(),
                },
            },
        )


@dataclass(frozen=True, slots=True)
class _MBPPPlusEvaluator:
    record: NativeEpisodeRecord
    task: RolloutTask
    worker: PrivateJSONWorker
    scorer_profile: MBPPScorerProfile

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("MBPP+ request reached another task")
        answer = _projected_answer(request, StepZeroTerminalMode.PYTHON_SOURCE)
        passed = False
        base_passed = False
        plus_passed = False
        scorer_attempts = 0
        result = None
        if answer is not None:
            payload = mbpp_request(
                profile=self.scorer_profile,
                private_target=_target(self.record),
                prompt=self.task.query,
                source_task_id=self.record.episode.source_id,
                submission=answer,
                task_id=self.task.task_id,
            )
            infrastructure: tuple[str, str, str | None] | None = None
            for scorer_attempts in range(1, 3):
                try:
                    candidate_result = await self.worker.request(payload)
                except Exception as error:
                    if scorer_attempts == 1:
                        await asyncio.sleep(1.0)
                        continue
                    raise TerminalEvaluatorError(
                        "EvalPlus worker failed after two attempts"
                    ) from error
                if candidate_result.get("infrastructure_error"):
                    stage = str(candidate_result.get("error_stage"))
                    error_type = str(candidate_result.get("error_type"))
                    module = candidate_result.get("error_module")
                    infrastructure = stage, error_type, str(module) if module else None
                    if scorer_attempts == 1:
                        await asyncio.sleep(1.0)
                    continue
                try:
                    result = decode_mbpp_verdict(candidate_result, self.scorer_profile)
                except RuntimeError as error:
                    raise TerminalEvaluatorError("EvalPlus worker response differs") from error
                break
            if result is None:
                assert infrastructure is not None
                stage, error_type, error_module = infrastructure
                raise TerminalEvaluatorError(
                    f"EvalPlus worker infrastructure failure after two attempts at {stage}: "
                    f"{error_type} ({error_module})"
                )
            base_passed = result["base_passed"] is True
            plus_passed = result["plus_passed"] is True
            passed = base_passed and plus_passed
        return _reward(
            task=self.task,
            value=float(passed),
            success=passed,
            metric="base-plus-pass@1",
            verifier=self.scorer_profile.profile_id,
            fields={
                "base-passed": base_passed,
                "plus-passed": plus_passed,
                "scorer-attempts": scorer_attempts,
                "scorer_profile": self.scorer_profile.to_value(),
                "native_verdict": result,
                "failure_kind": None if result is None else mbpp_failure_kind(result),
                "public_metrics": {
                    "base-pass": float(base_passed),
                    "plus-pass": float(plus_passed),
                    "base-plus-pass@1": float(passed),
                },
            },
        )


@dataclass(frozen=True, slots=True)
class _SourceBoundEvaluator:
    delegate: TerminalEvaluator
    record: NativeEpisodeRecord
    public_task: RolloutTask | None = None
    _reset_binding_json: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # Capture before the first action; later mutable public-state updates
        # must not turn the final state into purported initial-state evidence.
        binding = self._reset_binding()
        if binding is not None:
            object.__setattr__(self, "_reset_binding_json", canonical_json(binding))

    def _reset_binding(self) -> dict[str, JsonValue] | None:
        if self.public_task is None:
            return None
        observed: JsonValue = None
        if self.record.episode.benchmark is Protocol13Benchmark.ALF_WORLD:
            if not isinstance(self.delegate, PrivateALFWorldTerminalEvaluator):
                return None
            captured = self.delegate.observed_reset_json
            if captured is None:
                return None
            observed = normalize_json(json.loads(captured))
            kind = ALFWORLD_RESET_KIND
        else:
            kind = STATIC_RESET_KIND
        # All actor-visible source material/scaffolds, without the per-occurrence
        # address. Never read record.output or replace source text with an ID.
        public = self.public_task.to_value()
        public.pop("task_id")
        return {
            "format": RESET_BINDING_FORMAT,
            "kind": kind,
            "public_task": public,
            "observed_reset": observed,
        }

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        reward = await self.delegate.evaluate(request)
        # Private statistical coordinates only. Never modify reward/success or
        # add the evaluator target/source identity to public task context.
        coordinates: dict[str, JsonValue] = {
            "benchmark_id": self.record.episode.benchmark.value,
            "population_id": self.record.episode.population_id,
            "source_question_id": self.record.episode.source_id,
            "occurrence_id": self.record.episode.episode_id,
        }
        if self._reset_binding_json is not None:
            coordinates["reset_binding"] = normalize_json(json.loads(self._reset_binding_json))
        return replace(
            reward,
            native_payload={
                **reward.native_payload,
                (
                    "evaluation_evidence_source"
                    if isinstance(self.record, EvaluationEpisodeRecord)
                    else "training_evidence_source"
                ): coordinates,
            },
        )


@dataclass(frozen=True, slots=True)
class _Route:
    task: RolloutTask
    create: Callable[[], UnskilledRolloutSessionBundle]


@dataclass(frozen=True, slots=True)
class Protocol13TrainingSessionFactory:
    routes: dict[str, _Route]
    mbpp_profile: MBPPScorerProfile
    prepare_pending: (
        Callable[[tuple[RolloutTask, ...]], Awaitable[tuple[RolloutTask, ...]]] | None
    ) = None
    healthbench_judge: str = "qwen-local@1"
    alfworld_goal_binding: str = LEGACY_GOAL_BINDING
    format_review_from_step: int | None = None

    async def prepare_tasks(self, tasks: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        return tasks if self.prepare_pending is None else await self.prepare_pending(tasks)

    @property
    def task_feature_mapping_version(self) -> str:
        from skillev.evolution.task_features import TASK_FEATURE_MAPPING_VERSION

        return TASK_FEATURE_MAPPING_VERSION

    @property
    def terminal_evaluation_conditions_json(self) -> str:
        from skillev.evaluation.healthbench_luna_profile import (
            LEGACY_TRAINING_JUDGE,
            healthbench_condition,
        )

        conditions: dict[str, JsonValue] = {"mbpp-plus": self.mbpp_profile.to_value()}
        if self.healthbench_judge != LEGACY_TRAINING_JUDGE:
            conditions["healthbench"] = healthbench_condition(self.healthbench_judge)
        if self.alfworld_goal_binding != LEGACY_GOAL_BINDING:
            conditions["alfworld_goal_binding"] = self.alfworld_goal_binding
        if self.format_review_from_step is not None:
            from skillev.evaluation.format_content_review import format_review_condition

            conditions["format_content_review"] = format_review_condition(
                self.format_review_from_step
            )
        return canonical_json(conditions)

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        route = self.routes.get(task.task_id)
        if route is None or route.task != task or not callable(route.create):
            raise ValueError("Protocol 13 task has no exact trusted session route")
        bundle = route.create()
        if not isinstance(bundle, UnskilledRolloutSessionBundle):
            raise TypeError("Protocol 13 session route returned an incompatible bundle")
        return bundle


async def _alfworld_route(
    record: NativeEpisodeRecord,
    deployments: Protocol13TrainingDeployments,
) -> tuple[RolloutTask, Callable[[], UnskilledRolloutSessionBundle]]:
    target = _target(record)
    raw_route = target.get("environment_route")
    if not isinstance(raw_route, dict):
        raise ValueError("Protocol 13 ALFWorld route is absent")
    raw_game_file = raw_route.get("game_file")
    raw_config_file = raw_route.get("config_file")
    if type(raw_game_file) is not str or type(raw_config_file) is not str:
        raise ValueError("Protocol 13 ALFWorld path route is incompatible")
    # The private dataset may be mirrored between GPU hosts.  Its absolute path is
    # an acquisition-time binding, not a scientific identity.  Rebase only the
    # canonical path below json_2.1.1 onto the active pinned deployment root.
    source_game_file = Path(raw_game_file)
    try:
        marker = source_game_file.parts.index("json_2.1.1")
    except ValueError as error:
        raise ValueError("Protocol 13 ALFWorld game has no canonical dataset route") from error
    relative_game_file = Path(*source_game_file.parts[marker + 1 :])
    dataset_root = (deployments.alfworld.dataset_root / "json_2.1.1").resolve()
    game_file = (dataset_root / relative_game_file).resolve()
    if not game_file.is_relative_to(dataset_root) or not game_file.is_file():
        raise ValueError("Protocol 13 ALFWorld game is absent from the active deployment")
    config_file = deployments.alfworld.config_path
    seed = cast(int, raw_route["seed"])
    max_steps = cast(int, raw_route["max_steps"])
    mode = cast(str, raw_route["mode"])
    source = record.input
    context = cast(dict[str, JsonValue], source.public_context)
    revision = cast(str, context["dataset_revision"])
    snapshot = record.episode.population_id
    public = ALFWorldPublicItem(
        dataset_revision=revision,
        environment_snapshot_id=snapshot,
        split=cast(str, context["split"]),
        task_id=source.task_id,
        task_family=source.task_family,
        query=source.query,
        public_context={"admissible_commands": ["look"], "initial_observation": "pending"},
        seed=seed,
        max_steps=max_steps,
    )
    official_task = OfficialALFWorldTask(
        task_id=source.task_id,
        environment_id=public.environment_id,
        game_id=record.episode.source_id,
        seed=seed,
        max_steps=max_steps,
        payload={"game_file": str(game_file)},
    )
    deployment = deployments.alfworld
    process_factory = OfficialALFWorldProcessFactory(
        PinnedOfficialProcess(
            deployment.interpreter,
            deployment.source_root,
            deployment.source_revision,
            deployment.timeout_seconds,
        ),
        config_file,
        {record.episode.source_id: ALFWorldGameDeployment(game_file.parent, mode, source.query)},
        seed,
        simulator_max_steps=max_steps,
        goal_binding=deployments.alfworld_goal_binding,
    )
    env = await asyncio.to_thread(process_factory.create, official_task)
    try:
        reset = await asyncio.to_thread(env.reset, seed)
    finally:
        await env.close()
    public = bind_reset_public_item(public, reset, deployments.alfworld_goal_binding)
    surface, profile = _action_contract(Protocol13Benchmark.ALF_WORLD, max_steps)
    task = replace(public.to_rollout_task(), action_surface=surface, budget_profile=profile)
    if isinstance(record, EvaluationEpisodeRecord):
        if Path(raw_config_file).resolve() != config_file.resolve():
            raise ValueError("IID and deployed ALFWorld configuration must match")
        task = replace(
            task,
            public_context={
                **cast(dict[str, JsonValue], task.public_context),
                "input_profile": context["input_profile"],
            },
        )
    case = PrivateALFWorldCase(public, official_task)
    episode_factory = OfficialALFWorldEpisodeFactory(
        process_factory, deployments.alfworld_goal_binding
    )

    def create() -> UnskilledRolloutSessionBundle:
        session = episode_factory.create(case)
        return UnskilledRolloutSessionBundle(
            ALFWorldEnvironment(public, session.episode),
            PrivateALFWorldTerminalEvaluator(
                public, session.outcome_view, session.observed_reset_json
            ),
            session.cleanup,
        )

    return task, create


def _action_contract(
    benchmark: Protocol13Benchmark,
    max_steps: int | None,
    *,
    hotpot_deliberation: bool = False,
) -> tuple[ActionSurface, RolloutBudgetProfile]:
    from skillev.benchmarks.protocol_v10_action import (
        COMPLETION_WIRE_INSTRUCTION,
        protocol_v10_action_contract,
    )

    route = (
        "mbpp-plus-fixed-100"
        if benchmark in {Protocol13Benchmark.MBPP_PLUS, Protocol13Benchmark.HUMAN_EVAL}
        else benchmark.value
    )
    surface, profile = protocol_v10_action_contract(route, max_steps=max_steps)
    instruction = {
        Protocol13Benchmark.AIME_2026: (
            "State one final integer from 0 through 999 in answer. "
            "Plain decimal or boxed notation is accepted."
        ),
        Protocol13Benchmark.HOTPOT_QA: "Submit the final answer in answer.",
        Protocol13Benchmark.TRIVIA_QA: "Submit the final answer in answer.",
        Protocol13Benchmark.HEALTHBENCH: (
            "Place your complete response to the conversation in answer."
        ),
        Protocol13Benchmark.HUMAN_EVAL: (
            "Return the Python completion of the supplied function or a complete Python "
            "module in answer. Preserve indentation."
        ),
        Protocol13Benchmark.MBPP_PLUS: (
            "Return the Python implementation requested by the public task in answer."
        ),
    }.get(benchmark)
    if instruction is not None:
        if hotpot_deliberation and benchmark is Protocol13Benchmark.HOTPOT_QA:
            instruction = HOTPOT_DELIBERATION + instruction
        # Domain answer semantics may override the old protocol (e.g. boxed AIME),
        # but must not discard the executable JSON wire/escaping instructions.
        surface = replace(
            surface,
            format=ACTION_SURFACE_FORMAT_V3,
            public_instructions=(),
            instructions=PublicActionInstructions((instruction,), (COMPLETION_WIRE_INSTRUCTION,)),
        )
    return surface, profile


async def build_protocol13_training_sessions(
    records: tuple[NativeEpisodeRecord, ...],
    *,
    deployments_path: Path,
    endpoint_base: str,
    base_model: str,
    resources: RolloutWorkflowResources,
    mbpp_interpreter: Path,
    mbpp_source_root: Path,
    mbpp_profile: MBPPScorerProfile,
    hotpot_deliberation: bool = False,
    rollout_budget: RolloutBudgetProfile | None = None,
    static_rollout_budget: RolloutBudgetProfile | None = None,
    domain_rollout_budgets: Mapping[str, RolloutBudgetProfile] | None = None,
    lazy_environments: bool = False,
    judge_endpoints: tuple[str, ...] = (),
    request_journal_path: Path | None = None,
    healthbench_judge: str = "qwen-local@1",
    format_review_from_step: int | None = None,
) -> tuple[tuple[RolloutTask, ...], Protocol13TrainingSessionFactory]:
    """Hydrate public reset state and bind all verifier-only routes."""

    if not records or len({record.input.task_id for record in records}) != len(records):
        raise ValueError("Protocol 13 mini records must be non-empty and task-unique")
    if any(record.episode.benchmark is Protocol13Benchmark.WEB_SHOP for record in records):
        raise ValueError("new training excludes WebShop; select the seven-domain condition")
    resolved = resolve_mbpp_profile(
        {"source_root": str(mbpp_source_root), "profile": mbpp_profile.to_value()}
    )
    if resolved != mbpp_profile or not mbpp_source_root.is_absolute():
        raise ValueError("training MBPP source differs from the frozen scoring condition")
    deployments = Protocol13TrainingDeployments.read(deployments_path)
    health_cases = {
        record.input.task_id: record.output.target
        for record in records
        if record.episode.benchmark is Protocol13Benchmark.HEALTHBENCH
    }
    health_config = deployments.healthbench
    if healthbench_judge == "qwen-local@1" and health_config.model_revision != base_model:
        raise ValueError("HealthBench grader route differs from the base model")

    from skillev.runtime.request_journal import DurableRequestJournal

    request_journal = (
        DurableRequestJournal(request_journal_path) if request_journal_path is not None else None
    )
    if format_review_from_step is not None:
        from skillev.evaluation.format_content_review import format_review_condition

        format_review_condition(format_review_from_step)
        if request_journal is None or any(isinstance(r, EvaluationEpisodeRecord) for r in records):
            raise ValueError("format review is training-only and requires durable Judge requests")

    def make_health_grader(endpoint_base: str) -> QwenSGLangHealthBenchGrader:
        return QwenSGLangHealthBenchGrader(
            worker=PrivateJSONWorker(
                command=(
                    str(health_config.interpreter),
                    str(_HEALTHBENCH_WORKER),
                    "--official-source-root",
                    str(health_config.source_root),
                    "--grader-model",
                    base_model,
                    "--api-base-url",
                    endpoint_base.rstrip("/") + "/v1",
                    "--request-timeout-seconds",
                    str(health_config.request_timeout_seconds),
                    "--repair-max-output-tokens",
                    str(deployments.healthbench_repair_max_output_tokens),
                ),
                working_directory=health_config.source_root.parent,
                timeout_seconds=health_config.worker_timeout_seconds,
                process_limiter=resources.process_graders,
            ),
            private_cases=health_cases,
            verifier_version=(
                HEALTHBENCH_QWEN_VERIFIER
                + (
                    "+repair-output-4096@1"
                    if deployments.healthbench_repair_max_output_tokens == 4096
                    else ""
                )
            ),
            model_request_limiter=resources.model_limiter(endpoint_base),
            request_journal=request_journal,
            broker_config=HealthBenchQwenGraderConfig(
                endpoint_base=endpoint_base,
                base_model=base_model,
                identity=HealthBenchQwenVerifierIdentity(
                    model_revision=health_config.model_revision,
                    tokenizer_revision=health_config.tokenizer_revision,
                    sglang_version=health_config.sglang_version,
                ),
                request_timeout_seconds=health_config.request_timeout_seconds,
                worker_timeout_seconds=health_config.worker_timeout_seconds,
            ),
        )

    endpoints = judge_endpoints or (endpoint_base,)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("judge endpoints must be distinct")
    from skillev.evaluation.external_judge_policy import (
        DIRECT_HEALTHBENCH_PROFILE,
        GATEWAY_HEALTHBENCH_PROFILE,
    )
    from skillev.evaluation.healthbench_luna_profile import (
        LEGACY_TRAINING_JUDGE,
        PROFILE_ID,
        healthbench_condition,
    )

    from .healthbench_api import OpenAIHealthBenchGrader

    if healthbench_judge not in {
        LEGACY_TRAINING_JUDGE,
        DIRECT_HEALTHBENCH_PROFILE,
        GATEWAY_HEALTHBENCH_PROFILE,
        PROFILE_ID,
    }:
        raise ValueError("unknown HealthBench judge profile")
    health_graders: tuple[HealthBenchOfficialGrader, ...] = (
        (
            OpenAIHealthBenchGrader(
                health_cases,
                health_config.source_root,
                AsyncResourceLimiter(4),
                verifier_version=str(healthbench_condition(healthbench_judge)["verifier"]),
                judge_profile=healthbench_judge,
                criterion_ledger_root=None
                if request_journal_path is None
                else request_journal_path.with_name(
                    request_journal_path.stem + "-healthbench-criteria"
                ),
            ),
        )
        if healthbench_judge != LEGACY_TRAINING_JUDGE
        else tuple(make_health_grader(endpoint) for endpoint in endpoints)
    )
    # Frozen case assignment uses only declared task order, never reward/answers.
    health_routes = {
        task_id: health_graders[index % len(health_graders)]
        for index, task_id in enumerate(health_cases)
    }
    mbpp_worker = PrivateJSONWorker(
        command=(
            str(mbpp_interpreter),
            str(Path(__file__).with_name("protocol_v13_mbpp_worker.py")),
            "--official-source-root",
            str(mbpp_source_root),
        ),
        working_directory=deployments_path.parent,
        timeout_seconds=mbpp_profile.outer_timeout_seconds,
        process_limiter=resources.process_graders,
    )
    humaneval = IsolatedHumanEvalExecutionBackend()

    async def hydrate(record: NativeEpisodeRecord) -> _Route:
        benchmark = record.episode.benchmark
        from skillev.evolution.task_features import public_task_features

        features = public_task_features(benchmark.value, task_family=record.input.task_family)
        record = replace(
            record,
            input=replace(
                record.input, task_family=features.task_family, context_id=features.context_id
            ),
        )
        if benchmark is Protocol13Benchmark.ALF_WORLD:
            async with resources.process_graders.lease():
                task, create = await _alfworld_route(record, deployments)
        else:
            surface, profile = _action_contract(
                benchmark, None, hotpot_deliberation=hotpot_deliberation
            )
            task = replace(record.input, action_surface=surface, budget_profile=profile)
            evaluator: TerminalEvaluator
            if benchmark in {
                Protocol13Benchmark.HOTPOT_QA,
                Protocol13Benchmark.TRIVIA_QA,
                Protocol13Benchmark.AIME_2026,
            }:
                evaluator = _StaticEvaluator(record, task)
            elif benchmark is Protocol13Benchmark.HEALTHBENCH:
                evaluator = _HealthEvaluator(task, health_routes[task.task_id], healthbench_judge)
            elif benchmark is Protocol13Benchmark.MBPP_PLUS:
                evaluator = _MBPPPlusEvaluator(record, task, mbpp_worker, mbpp_profile)
            elif benchmark is Protocol13Benchmark.HUMAN_EVAL:
                evaluator = _HumanEvalEvaluator(record, task, humaneval)
            else:  # pragma: no cover
                raise ValueError("unsupported Protocol 13 training benchmark")

            def create(
                task: RolloutTask = task,
                evaluator: TerminalEvaluator = evaluator,
            ) -> UnskilledRolloutSessionBundle:
                return UnskilledRolloutSessionBundle(_completion_environment(task), evaluator)

        def source_bound_create(
            create: Callable[[], UnskilledRolloutSessionBundle] = create,
            record: NativeEpisodeRecord = record,
        ) -> UnskilledRolloutSessionBundle:
            bundle = create()
            if format_review_from_step is not None:
                from .format_content_review import wrap_training_evaluator

                assert request_journal is not None
                assert isinstance(record, Protocol13TrainingRecord)
                bundle = replace(
                    bundle,
                    evaluator=wrap_training_evaluator(
                        bundle.evaluator,
                        task,
                        record,
                        format_review_from_step,
                        request_journal,
                    ),
                )
            return replace(bundle, evaluator=_SourceBoundEvaluator(bundle.evaluator, record, task))

        # Native hydration may carry a split-specific legacy context. Retrieval
        # uses the same public coordinate as completion tasks and IID.
        override = (
            static_rollout_budget
            if benchmark is not Protocol13Benchmark.ALF_WORLD and static_rollout_budget is not None
            else rollout_budget
        )
        override = (domain_rollout_budgets or {}).get(benchmark.value, override)
        task = replace(
            task,
            context_id=features.context_id,
            budget_profile=override if override is not None else task.budget_profile,
        )
        return _Route(task, source_bound_create)

    from .session_hydration import hydrate_ordered

    pending: dict[str, NativeEpisodeRecord] = {}

    async def describe(record: NativeEpisodeRecord) -> _Route:
        if not lazy_environments or record.episode.benchmark is not Protocol13Benchmark.ALF_WORLD:
            return await hydrate(record)
        from skillev.evolution.task_features import public_task_features

        features = public_task_features(
            record.episode.benchmark.value, task_family=record.input.task_family
        )
        raw = _target(record)["environment_route"]
        assert isinstance(raw, dict)
        surface, profile = _action_contract(
            Protocol13Benchmark.ALF_WORLD, cast(int, raw["max_steps"])
        )
        task = replace(
            record.input,
            task_family=features.task_family,
            context_id=features.context_id,
            action_surface=surface,
            budget_profile=(domain_rollout_budgets or {}).get(
                "alfworld", rollout_budget or profile
            ),
        )
        pending[task.task_id] = record

        def not_prepared() -> UnskilledRolloutSessionBundle:
            raise RuntimeError("environment description must be hydrated before rollout")

        return _Route(task, not_prepared)

    hydrated = await hydrate_ordered(records, describe, limiter=resources.session_setups)
    routes = {route.task.task_id: route for route in hydrated}
    tasks = tuple(route.task for route in hydrated)

    async def prepare(selected: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        # Only the imminent batch is hydrated. No future-policy generation and
        # no sharing of mutable native episodes between independent rollouts.
        wanted = tuple(pending[task.task_id] for task in selected if task.task_id in pending)
        ready = await hydrate_ordered(wanted, hydrate, limiter=resources.session_setups)
        for route in ready:
            routes[route.task.task_id] = route
            del pending[route.task.task_id]
        return tuple(routes[task.task_id].task for task in selected)

    return tasks, Protocol13TrainingSessionFactory(
        routes,
        mbpp_profile,
        prepare if lazy_environments else None,
        healthbench_judge,
        deployments.alfworld_goal_binding,
        format_review_from_step,
    )


__all__ = ["Protocol13TrainingSessionFactory", "build_protocol13_training_sessions"]
