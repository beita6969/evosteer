from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from skillev.evaluation.direct_baseline.client import (
    DirectGenerationError,
    DirectGenerationRequest,
    DirectGenerationResult,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark, DirectDecodingProfile
from skillev.evaluation.direct_baseline.parsing import ParsedResponse, parse_short_answer
from skillev.evaluation.direct_baseline.runner import (
    DirectTask,
    RawGenerationRecord,
    profile_for_seed,
    run_direct_tasks,
)


def _profile() -> DirectDecodingProfile:
    return DirectDecodingProfile("test", False, 0, 1, 1, 0, 0, 1, 10, sampling_mode="greedy")


@dataclass
class _FakeClient:
    fail: frozenset[str] = frozenset()

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        if request.request_id in self.fail:
            raise DirectGenerationError("unavailable")
        return DirectGenerationResult(request.request_id, "Final answer: ok", "stop", 2, 3)


def test_runner_preserves_panel_order_and_separates_infrastructure_failure() -> None:
    tasks = tuple(
        DirectTask(
            task_id=str(index),
            benchmark=DirectBenchmark.HOTPOT_QA,
            messages=({"role": "user", "content": "question"},),
            profile=_profile(),
            parser=parse_short_answer,
            prompt_profile_id="qa@1",
            parser_profile_id="short@1",
            population_id="panel@1",
            run_seed=42,
        )
        for index in range(3)
    )
    attempts = asyncio.run(run_direct_tasks(_FakeClient(frozenset({"1"})), tasks, concurrency=2))
    assert tuple(item.task_id for item in attempts) == ("0", "1", "2")
    assert attempts[0].parsed is not None
    assert attempts[0].parsed.value == "ok"
    assert attempts[1].parsed is None
    assert attempts[1].infrastructure_error == "DirectGenerationError"


def test_runner_rejects_duplicate_panel_ids() -> None:
    task = DirectTask(
        task_id="same",
        benchmark=DirectBenchmark.HOTPOT_QA,
        messages=({"role": "user", "content": "question"},),
        profile=_profile(),
        parser=parse_short_answer,
        prompt_profile_id="qa@1",
        parser_profile_id="short@1",
        population_id="panel@1",
        run_seed=42,
    )
    with pytest.raises(ValueError):
        asyncio.run(run_direct_tasks(_FakeClient(), (task, task), concurrency=1))


def test_generation_observer_receives_raw_and_contract_ids() -> None:
    observed: list[RawGenerationRecord] = []

    async def observer(attempt: RawGenerationRecord) -> None:
        observed.append(attempt)

    task = DirectTask(
        task_id="task-1",
        benchmark=DirectBenchmark.HOTPOT_QA,
        messages=({"role": "user", "content": "question"},),
        profile=profile_for_seed(_profile(), 7),
        parser=parse_short_answer,
        prompt_profile_id="qa@1",
        parser_profile_id="short@1",
        population_id="panel@1",
        run_seed=7,
    )
    attempts = asyncio.run(
        run_direct_tasks(_FakeClient(), (task,), concurrency=1, generation_observer=observer)
    )
    assert observed[0].task_id == attempts[0].task_id
    assert observed[0].raw_text == "Final answer: ok"
    assert observed[0].prompt_profile_id == "qa@1"
    assert observed[0].parser_profile_id == "short@1"
    assert observed[0].decoding_profile_id == "test/seed-7"
    assert observed[0].population_id == "panel@1"
    assert observed[0].run_seed == 7


def test_generation_is_observed_before_parser_failure() -> None:
    observed: list[RawGenerationRecord] = []

    async def observer(record: RawGenerationRecord) -> None:
        observed.append(record)

    def exploding_parser(_: str) -> ParsedResponse:
        raise RuntimeError("parser bug")

    task = DirectTask(
        task_id="parser-failure",
        benchmark=DirectBenchmark.HOTPOT_QA,
        messages=({"role": "user", "content": "question"},),
        profile=_profile(),
        parser=exploding_parser,
        prompt_profile_id="qa@1",
        parser_profile_id="broken@1",
        population_id="panel@1",
        run_seed=42,
    )
    with pytest.raises(RuntimeError, match="parser bug"):
        asyncio.run(
            run_direct_tasks(_FakeClient(), (task,), concurrency=1, generation_observer=observer)
        )
    assert len(observed) == 1
    assert observed[0].raw_text == "Final answer: ok"
