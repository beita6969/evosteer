"""Terminal-only HealthBench API resources, separate from actor/author inference."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue
from skillev.evaluation.healthbench_luna_profile import (
    PROFILE_ID,
    VERIFIER,
    external_healthbench_profile,
)
from skillev.training import AsyncResourceLimiter

from .protocol_v10_official import HealthBenchGrade

# Actual API calls share capacity across concurrent rubric pools, never an actor
# semaphore. No key, rubric, explanation or private path enters task prompts.
_API_CAPACITY = threading.BoundedSemaphore(4)


def make_client(*, profile_id: str = PROFILE_ID) -> Any:
    from skillev.evaluation.external_judge_policy import (
        DIRECT_HEALTHBENCH_PROFILE,
        HEALTHBENCH_EXTERNAL_PROFILE,
    )
    from skillev_private.evaluation.external_judge_api import make_external_judge_client

    external_healthbench_profile(profile_id)
    if profile_id == DIRECT_HEALTHBENCH_PROFILE:
        from skillev_private.evaluation.external_judge_failover import make_official_client

        return make_official_client()
    return make_external_judge_client(
        allow_official_fallback=profile_id == HEALTHBENCH_EXTERNAL_PROFILE
    )


class IncompleteHealthBenchResponseError(RuntimeError):
    def __init__(self, response: Any) -> None:
        super().__init__("HealthBench judge response was incomplete")
        self.response = response


class BoundedAPICompletions:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def create(self, **kwargs: Any) -> Any:
        from skillev.evaluation.judge_spool import JudgeSpoolClient

        started = time.monotonic()
        timeout = float(kwargs.get("timeout", 120.0))
        durable_delivery = isinstance(self.delegate, JudgeSpoolClient)
        # An outage may keep the previous four operations awaiting delivery.
        # Waiting for that infrastructure is not another API execution attempt.
        acquired = (
            _API_CAPACITY.acquire() if durable_delivery else _API_CAPACITY.acquire(timeout=timeout)
        )
        if not acquired:
            raise TimeoutError("HealthBench API queue deadline reached")
        try:
            remaining = timeout if durable_delivery else timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("HealthBench API request deadline reached")
            response = self.delegate.create(**{**kwargs, "timeout": remaining, "store": False})
            if getattr(response.choices[0], "finish_reason", None) in {"length", "content_filter"}:
                raise IncompleteHealthBenchResponseError(response)
            return response
        finally:
            _API_CAPACITY.release()


@dataclass(frozen=True, slots=True)
class OpenAIHealthBenchGrader:
    private_cases: Mapping[str, JsonValue]
    official_source: Path
    case_limiter: AsyncResourceLimiter
    verifier_version: str = VERIFIER
    criterion_ledger_root: Path | None = None
    judge_profile: str = PROFILE_ID

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        from skillev_private.evaluation.integrity_native_scoring import _grade_health

        case = self.private_cases[task_id]
        if not isinstance(case, dict):
            raise ValueError("HealthBench private rubric case is unavailable")
        settings = {
            "official_source": str(self.official_source),
            "effective_profile": asdict(external_healthbench_profile(self.judge_profile)),
        }

        def grade_once() -> HealthBenchGrade:
            from .healthbench_ledger import HealthBenchCriterionLedger

            ledger = (
                None
                if self.criterion_ledger_root is None
                else HealthBenchCriterionLedger(
                    self.criterion_ledger_root,
                    task_id,
                    {
                        "candidate_answer": candidate_answer,
                        "private_case": case,
                        "settings": settings,
                    },
                )
            )
            if ledger is not None and ledger.completed:
                saved = ledger.state["result"]
                return HealthBenchGrade(
                    saved["native_raw_score"],
                    saved["negative_criterion_count"],
                    saved["grader_cost"],
                    ledger.reference(),
                )
            diagnostics: dict[str, Any] = {}
            try:
                score, negative, cost = _grade_health(
                    candidate_answer,
                    case,
                    settings,
                    diagnostics=diagnostics,
                    record_requests=None if ledger is None else ledger.record_requests,
                )
            except BaseException as error:
                if ledger is not None:
                    ledger.update(
                        status="incomplete-grading",
                        diagnostics=diagnostics,
                        error_type=type(error).__name__,
                    )
                raise
            if ledger is not None:
                ledger.update(
                    status="completed",
                    diagnostics=diagnostics,
                    result={
                        "native_raw_score": score,
                        "negative_criterion_count": negative,
                        "learning_reward": min(1.0, max(0.0, score)),
                        "binary_success": score >= 0.60 and negative == 0,
                        "grader_cost": cost,
                    },
                )
            return HealthBenchGrade(
                score, negative, cost, None if ledger is None else ledger.reference()
            )

        async with self.case_limiter.lease():
            task = asyncio.create_task(asyncio.to_thread(grade_once))
            try:
                grade = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Drain the owned request before releasing its capacity. Partial
                # grading never becomes a false task label or a partial batch.
                await asyncio.gather(task, return_exceptions=True)
                raise
        return grade
