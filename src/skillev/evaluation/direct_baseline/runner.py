"""Bounded asynchronous generation runner for direct benchmark tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from .client import (
    DirectGenerationClient,
    DirectGenerationError,
    DirectGenerationRequest,
    DirectGenerationResult,
)
from .config import DirectBenchmark, DirectDecodingProfile
from .parsing import ParsedResponse


@dataclass(frozen=True, slots=True)
class DirectTask:
    task_id: str
    benchmark: DirectBenchmark
    messages: tuple[dict[str, str], ...]
    profile: DirectDecodingProfile
    parser: Callable[[str], ParsedResponse]
    prompt_profile_id: str
    parser_profile_id: str
    population_id: str
    run_seed: int
    panel_index: int = 0
    service_slot: int = 0

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        for value in (
            self.prompt_profile_id,
            self.parser_profile_id,
            self.population_id,
        ):
            if not value.strip():
                raise ValueError("direct task contract IDs must be non-empty")
        if self.run_seed != self.profile.seed:
            raise ValueError("task run_seed must equal request profile seed")
        if type(self.panel_index) is not int or self.panel_index < 0:
            raise ValueError("panel_index must be a non-negative integer")
        if type(self.service_slot) is not int or self.service_slot < 0:
            raise ValueError("service_slot must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RawGenerationRecord:
    task_id: str
    benchmark: DirectBenchmark
    raw_text: str | None
    reasoning_text: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    response_model: str | None
    response_id: str | None
    service_instance_id: str | None
    panel_index: int
    service_slot: int
    prompt_profile_id: str
    parser_profile_id: str
    decoding_profile_id: str
    population_id: str
    run_seed: int
    request_attempt: int
    infrastructure_error: str | None


@dataclass(frozen=True, slots=True)
class DirectAttempt:
    task_id: str
    benchmark: DirectBenchmark
    raw_text: str | None
    reasoning_text: str | None
    parsed: ParsedResponse | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    response_model: str | None
    response_id: str | None
    service_instance_id: str | None
    panel_index: int
    service_slot: int
    prompt_profile_id: str
    parser_profile_id: str
    decoding_profile_id: str
    population_id: str
    run_seed: int
    request_attempt: int
    infrastructure_error: str | None


GenerationObserver = Callable[[RawGenerationRecord], Awaitable[None]]


def profile_for_seed(profile: DirectDecodingProfile, seed: int) -> DirectDecodingProfile:
    if seed < 0:
        raise ValueError("generation seed must be non-negative")
    return replace(profile, profile_id=f"{profile.profile_id}/seed-{seed}", seed=seed)


async def run_direct_tasks(
    client: DirectGenerationClient,
    tasks: tuple[DirectTask, ...],
    *,
    concurrency: int,
    generation_observer: GenerationObserver | None = None,
    request_attempt: int = 1,
) -> tuple[DirectAttempt, ...]:
    if concurrency <= 0 or request_attempt <= 0:
        raise ValueError("concurrency and request_attempt must be positive")
    identities = tuple(task.task_id for task in tasks)
    if len(set(identities)) != len(identities):
        raise ValueError("task ids must be unique")
    semaphore = asyncio.Semaphore(concurrency)

    async def observe(record: RawGenerationRecord) -> None:
        if generation_observer is not None:
            await generation_observer(record)

    async def run_one(task: DirectTask) -> DirectAttempt:
        async with semaphore:
            try:
                result: DirectGenerationResult = await client.generate(
                    DirectGenerationRequest(
                        request_id=task.task_id,
                        messages=task.messages,
                        profile=task.profile,
                        service_slot=task.service_slot,
                    )
                )
            except DirectGenerationError as exc:
                record = _raw_record(task, request_attempt, infrastructure_error=type(exc).__name__)
                await observe(record)
                return _attempt_from_raw(record)
            record = _raw_record(
                task,
                request_attempt,
                raw_text=result.text,
                reasoning_text=result.reasoning_text,
                finish_reason=result.finish_reason,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                response_model=result.response_model,
                response_id=result.response_id,
                service_instance_id=result.service_instance_id,
            )
            await observe(record)
            parsed = task.parser(result.text)
            return _attempt_from_raw(record, parsed=parsed)

    return tuple(await asyncio.gather(*(run_one(task) for task in tasks)))


def _raw_record(
    task: DirectTask,
    request_attempt: int,
    *,
    raw_text: str | None = None,
    reasoning_text: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    response_model: str | None = None,
    response_id: str | None = None,
    service_instance_id: str | None = None,
    infrastructure_error: str | None = None,
) -> RawGenerationRecord:
    return RawGenerationRecord(
        task_id=task.task_id,
        benchmark=task.benchmark,
        raw_text=raw_text,
        reasoning_text=reasoning_text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_model=response_model,
        response_id=response_id,
        service_instance_id=service_instance_id,
        panel_index=task.panel_index,
        service_slot=task.service_slot,
        prompt_profile_id=task.prompt_profile_id,
        parser_profile_id=task.parser_profile_id,
        decoding_profile_id=task.profile.profile_id,
        population_id=task.population_id,
        run_seed=task.run_seed,
        request_attempt=request_attempt,
        infrastructure_error=infrastructure_error,
    )


def _attempt_from_raw(
    record: RawGenerationRecord, *, parsed: ParsedResponse | None = None
) -> DirectAttempt:
    return DirectAttempt(
        task_id=record.task_id,
        benchmark=record.benchmark,
        raw_text=record.raw_text,
        reasoning_text=record.reasoning_text,
        parsed=parsed,
        finish_reason=record.finish_reason,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        response_model=record.response_model,
        response_id=record.response_id,
        service_instance_id=record.service_instance_id,
        panel_index=record.panel_index,
        service_slot=record.service_slot,
        prompt_profile_id=record.prompt_profile_id,
        parser_profile_id=record.parser_profile_id,
        decoding_profile_id=record.decoding_profile_id,
        population_id=record.population_id,
        run_seed=record.run_seed,
        request_attempt=record.request_attempt,
        infrastructure_error=record.infrastructure_error,
    )
