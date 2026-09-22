"""Native scoring domain: only read submitted finals, never call an actor.

The evaluated single owner is in a separate filesystem/network namespace.
This module may read evaluator targets; only the HealthBench rubric grader is
allowed a model call, and it receives no experiment-arm identity.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.healthbench_luna_profile import (
    external_healthbench_profile,
    healthbench_condition,
    luna_profile,
)
from skillev.evaluation.healthbench_official import (
    HealthBenchExternalJudgeProfile,
    HealthBenchQwenJudgeProfile,
)
from skillev.evaluation.input_metric_contracts import CONTRACTS
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_results import EvaluationStatus, NativeScore
from skillev.evaluation.sealed_candidates import CandidateReader, EventOrigin
from skillev.evaluation.step0_completion import StepZeroTerminalMode, project_terminal_candidate
from skillev_private.benchmarks.code_math import (
    CodeExecutionInfrastructureError,
    CodeExecutionRequest,
    CodeExecutionStatus,
)
from skillev_private.benchmarks.humaneval_official import IsolatedHumanEvalExecutionBackend
from skillev_private.benchmarks.public_code_context import HUMANEVAL_VERIFIER
from skillev_private.evaluation.humaneval_source import (
    humaneval_source_parts as humaneval_source_parts,
)
from skillev_private.evaluation.integrity_grader_usage import (
    IncompleteNativeGradingError,
    MeteredCompletions,
)
from skillev_private.evaluation.integrity_mbpp import grade_mbpp as _grade_mbpp
from skillev_private.evaluation.integrity_mbpp import mbpp_failure_kind
from skillev_private.evaluation.integrity_scoring import SealedQAScorer
from skillev_private.evaluation.result_contracts import public_native_metric_values


def resolve_healthbench_profile(
    settings: dict[str, Any],
) -> HealthBenchQwenJudgeProfile | HealthBenchExternalJudgeProfile:
    """Resolve once before freezing controls; grading consumes that exact profile."""
    import yaml

    from skillev.evaluation.healthbench_official import (
        load_external_judge_profile,
        load_qwen_judge_profile,
    )

    value = settings.get("effective_profile")
    if value is not None:
        profile_type = (
            HealthBenchExternalJudgeProfile
            if value.get("backend") == "openai-chat-completions"
            else HealthBenchQwenJudgeProfile
        )
        profile = profile_type(**value)
    elif "profile" not in settings:
        profile = luna_profile()
    else:
        path = Path(settings["profile"])
        value = yaml.safe_load(path.read_text())
        loader = (
            load_external_judge_profile
            if value.get("backend") == "openai-chat-completions"
            else load_qwen_judge_profile
        )
        profile = loader(path)
    if isinstance(
        profile, HealthBenchExternalJudgeProfile
    ) and profile != external_healthbench_profile(profile.profile_id):
        raise ValueError("native HealthBench requires a declared external API profile")
    return profile


def _grade_health(
    candidate: str,
    target: dict[str, Any],
    settings: dict[str, Any],
    *,
    diagnostics: dict[str, Any] | None = None,
    record_requests: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[float, int, dict[str, float]]:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )

    from skillev.evaluation.healthbench_official import (
        ExternalHealthBenchRubricSampler,
        QwenChatTransport,
        QwenHealthBenchRubricSampler,
        RubricAttemptRegistry,
        TransportRetryPolicy,
    )
    from skillev_private.benchmarks.protocol_v10_healthbench_worker import _load_official, _rubrics

    profile = resolve_healthbench_profile(settings)
    health_type, rubric_type, response_type = _load_official(Path(settings["official_source"]))
    external = isinstance(profile, HealthBenchExternalJudgeProfile)
    from skillev_private.benchmarks.healthbench_api import BoundedAPICompletions, make_client

    client = (
        make_client(profile_id=profile.profile_id)
        if external
        else OpenAI(
            api_key="EMPTY",
            base_url=settings["endpoint_base"].rstrip("/") + "/v1",
            timeout=profile.request_timeout_seconds,
            max_retries=0,
        )
    )
    delegate = (
        BoundedAPICompletions(client.chat.completions) if external else client.chat.completions
    )
    meter = MeteredCompletions(
        delegate, retain_evidence=diagnostics is not None, record_evidence=record_requests
    )
    attempts = RubricAttemptRegistry()
    started = time.monotonic()
    grades: list[dict[str, Any]] = []
    rubrics: list[Any] | None = None
    completed = False
    try:
        sampler: Any
        if isinstance(profile, HealthBenchExternalJudgeProfile):
            from skillev.evaluation.healthbench_transport import ExternalJudgeTransport

            sampler = ExternalHealthBenchRubricSampler(
                ExternalJudgeTransport(
                    client=cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=meter))),
                    model=profile.model,
                    max_completion_tokens=profile.max_completion_tokens,
                    reasoning_effort=profile.reasoning_effort,
                    retry_policy=TransportRetryPolicy(
                        maximum_attempts=profile.maximum_attempts,
                        request_timeout_seconds=profile.request_timeout_seconds,
                        total_deadline_seconds=300,
                    ),
                    transient_error_types=(
                        APIConnectionError,
                        APITimeoutError,
                        InternalServerError,
                        RateLimitError,
                    ),
                ),
                response_type,
                attempts,
                profile_id=profile.profile_id,
            )
        else:
            transport = QwenChatTransport(
                cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=meter))),
                settings["model_route"],
                TransportRetryPolicy(
                    maximum_attempts=profile.maximum_attempts,
                    request_timeout_seconds=profile.request_timeout_seconds,
                    total_deadline_seconds=profile.total_deadline_seconds,
                ),
                (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, OSError),
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                top_p=profile.top_p,
                top_k=profile.top_k,
                seed=profile.seed,
                enable_thinking=profile.enable_thinking,
            )
            # One registry per candidate, never shared across arms.
            sampler = QwenHealthBenchRubricSampler((transport,), response_type, attempts)
        evaluator = object.__new__(health_type)
        evaluator.grader_model = sampler
        evaluator.length_adjustment_center = None
        evaluator.length_adjustment_penalty_per_500_chars = None
        rubrics = (
            _rubrics(target, rubric_type)
            if "grader_kind" in target
            else [rubric_type.from_dict(row) for row in target["rubrics"]]
        )
        from skillev_private.benchmarks.healthbench_ledger import official_criterion_messages

        meter.bind_criterion_messages(
            official_criterion_messages(
                health_type.grade_sample,
                target["prompt"],
                candidate,
                rubrics,
                system_message=profile.system_message
                if isinstance(profile, HealthBenchQwenJudgeProfile)
                else None,
            )
        )
        _, _, grades = evaluator.grade_sample(
            prompt=target["prompt"],
            response_text=candidate,
            example_tags=[],
            rubric_items=rubrics,
        )
        if len(grades) != len(rubrics) or any(
            type(row.get("criteria_met")) is not bool for row in grades
        ):
            raise RuntimeError("HealthBench rubric grading is incomplete")
        positive = sum(float(row["points"]) for row in grades if row["points"] > 0)
        if positive <= 0:
            raise ValueError("HealthBench rubric has no positive denominator")
        earned = sum(float(row["points"]) for row in grades if row["criteria_met"])
        negative = sum(row["points"] < 0 and row["criteria_met"] for row in grades)
        cost = meter.snapshot()
        cost["semantic_repairs"] = float(sampler.semantic_attempts.semantic_repair_count)
        cost["wall_seconds"] = time.monotonic() - started
        completed = True
        return earned / positive, negative, cost
    except Exception as error:
        # A failed rubric is not a false verdict. Preserve partial usage even
        # though the caller must leave this candidate's score absent.
        cost = meter.snapshot()
        cost["semantic_repairs"] = float(attempts.semantic_repair_count)
        cost["wall_seconds"] = time.monotonic() - started
        raise IncompleteNativeGradingError(cost) from error
    finally:
        if diagnostics is not None:
            diagnostics.update(
                status="completed" if completed else "incomplete-grading",
                effective_profile=asdict(profile),
                model_route=profile.model
                if isinstance(profile, HealthBenchExternalJudgeProfile)
                else settings["model_route"],
                rubric_grades=grades,
                criterion_results=[
                    {"criterion_index": index, **row} for index, row in enumerate(grades)
                ],
                criterion_count=None if rubrics is None else len(rubrics),
                candidate_answer=candidate,
                native_raw_score=None if not completed else earned / positive,
                learning_reward=None if not completed else min(1.0, max(0.0, earned / positive)),
                binary_success=None
                if not completed
                else earned / positive >= 0.60 and negative == 0,
                negative_criterion_count=None if not completed else negative,
                requests=meter.evidence(),
            )
        client.close()


async def score_native(
    reader: CandidateReader,
    scope: tuple[str, str, str],
    benchmark: str,
    target: dict[str, Any],
    *,
    settings: dict[str, Any],
    sandbox: ActorSandbox,
    mbpp_sandbox: ActorSandbox,
    record_diagnostics: Callable[[dict[str, Any]], None] | None = None,
) -> NativeScore:
    started = time.monotonic()
    candidate = reader.get(*scope)
    contract = CONTRACTS[benchmark]
    metric, verifier = contract.metric, NATIVE_VERIFIER_VERSIONS[benchmark]
    if benchmark == "humaneval":
        verifier = HUMANEVAL_VERIFIER
    if benchmark == "healthbench":
        profile = resolve_healthbench_profile(settings.get("healthbench", {}))
        if isinstance(profile, HealthBenchExternalJudgeProfile):
            condition = healthbench_condition(profile.profile_id)
            metric, verifier = str(condition["metric"]), str(condition["verifier"])
    if candidate.parser_id != contract.parser:
        raise ValueError("candidate parser does not match the frozen native metric contract")
    secondary: dict[str, float] = {}
    scorer_cost: dict[str, float] = {}
    status = EvaluationStatus.SCORED
    grader_used: bool | None = None
    failure_kind: str | None = None
    if benchmark in {"webshop", "alfworld"}:
        outcomes = reader.traces(scope, "native-outcome", origin=EventOrigin.ENVIRONMENT)
        if len(outcomes) != 1 or not isinstance(outcomes[0], dict):
            raise RuntimeError("native simulator outcome is unavailable or ambiguous")
        outcome = outcomes[0]
        value = float(outcome["reward"]) if benchmark == "webshop" else float(outcome["success"])
        secondary = {"success": float(outcome["success"])}
    elif not candidate.text.strip():
        value, status = 0.0, EvaluationStatus.CANDIDATE_FAILURE
        failure_kind = "empty-submission"
        if benchmark in {"hotpotqa", "triviaqa"}:
            secondary = {"answer-exact-match": 0.0, "answer-f1": 0.0}
        elif benchmark == "healthbench":
            secondary = {"triggered-negative-rubric-count": 0.0}
            grader_used = False
        elif benchmark == "mbpp-plus":
            secondary = {"base-pass": 0.0, "plus-pass": 0.0}
            if record_diagnostics is not None:
                record_diagnostics(
                    {
                        "status": "empty-submission",
                        "lanes": {lane: {"native_status": "not-run"} for lane in ("base", "plus")},
                    }
                )
    elif benchmark in {"hotpotqa", "triviaqa"}:
        reward = await SealedQAScorer(
            reader, benchmark, tuple(target["accepted_answers"]), f"native:{benchmark}"
        ).score(*scope)
        secondary = {item.metric_name: item.value for item in public_native_metric_values(reward)}
        value = secondary["answer-f1"]
        if record_diagnostics is not None:
            record_diagnostics(cast(dict[str, Any], reward.native_payload["qa_diagnostics"]))
    elif benchmark == "aime-2026":
        parsed = project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, candidate.text)
        expected = rf"\boxed{{{int(target['answer'])}}}"
        value = float(parsed == expected)
        if parsed is None:
            status = EvaluationStatus.CANDIDATE_FAILURE
            failure_kind = "unparseable-integer"
    elif benchmark == "healthbench":
        diagnostics: dict[str, Any] = {}
        try:
            value, negative, scorer_cost = await asyncio.to_thread(
                _grade_health,
                candidate.text,
                target,
                settings["healthbench"],
                diagnostics=diagnostics,
            )
        finally:
            # SQLite belongs to the coordinator, not the parallel rubric workers.
            if record_diagnostics is not None and diagnostics:
                record_diagnostics(diagnostics)
        secondary = {"triggered-negative-rubric-count": float(negative)}
        grader_used = True
    elif benchmark == "humaneval":
        from skillev_private.benchmarks.public_code_context import (
            CodeSubmission,
            PublicCodeContext,
            compose_code_candidate,
        )

        assembled = compose_code_candidate(
            PublicCodeContext(target["prompt"], target["entry_point"]),
            CodeSubmission(candidate.text),
        )
        prompt, source = assembled.prefix, assembled.completion
        request = CodeExecutionRequest(
            candidate.episode_id, prompt, source, target["test"], target["entry_point"]
        )
        backend = IsolatedHumanEvalExecutionBackend(command_prefix=sandbox.command()[:-1])
        try:
            result = await asyncio.to_thread(lambda: asyncio.run(backend.run(request)))
        except CodeExecutionInfrastructureError as error:
            if record_diagnostics is not None:
                record_diagnostics(
                    {
                        **assembled.diagnostics(),
                        "stage": "infrastructure",
                        "exception_type": type(error).__name__,
                        "native_result": str(error),
                        "passed": None,
                    }
                )
            raise
        if record_diagnostics is not None:
            record_diagnostics({**assembled.diagnostics(), **result.diagnostics()})
        value = float(result.status is CodeExecutionStatus.PASSED)
        if not value:
            status, failure_kind = EvaluationStatus.CANDIDATE_FAILURE, result.status.value
    elif benchmark == "mbpp-plus":
        try:
            result_value = await asyncio.to_thread(
                _grade_mbpp, candidate.text, target, settings["mbpp-plus"], mbpp_sandbox
            )
        except RuntimeError as error:
            if record_diagnostics is not None:
                record_diagnostics({"status": "infrastructure-failure", "error": str(error)})
            raise
        if record_diagnostics is not None:
            record_diagnostics(result_value)
        if any(type(result_value.get(key)) is not bool for key in ("base_passed", "plus_passed")):
            raise TypeError("EvalPlus Base and Plus verdicts must be native booleans")
        value = float(result_value["base_passed"] and result_value["plus_passed"])
        if not value:
            status = EvaluationStatus.CANDIDATE_FAILURE
            failure_kind = mbpp_failure_kind(result_value)
        secondary = {
            "base-pass": float(result_value["base_passed"]),
            "plus-pass": float(result_value["plus_passed"]),
        }
    else:
        raise ValueError("unsupported native IID scorer")
    return NativeScore(
        task_id=candidate.episode_id,
        benchmark=benchmark,
        metric=metric,
        value=value,
        status=status,
        secondary_metrics=secondary,
        verifier_version=verifier,
        scorer_cost={**scorer_cost, "wall_seconds": time.monotonic() - started},
        grader_used=grader_used,
        failure_kind=failure_kind,
    )
