"""Bounded HealthBench model I/O, separate from rubric and aggregation semantics."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol, cast

from .external_judge_policy import (
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_MAX_TOKENS,
    EXTERNAL_JUDGE_MODEL,
)

CANDIDATE_TEMPERATURE = 0.5
CANDIDATE_MAX_TOKENS = 2048


class _Completions(Protocol):
    def create(self, **kwargs: object) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class _Client(Protocol):
    chat: _Chat


@dataclass(frozen=True, slots=True)
class TransportRetryPolicy:
    maximum_attempts: int = 4
    request_timeout_seconds: float = 120.0
    total_deadline_seconds: float = 300.0
    initial_backoff_seconds: float = 0.5
    maximum_backoff_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.maximum_attempts < 1:
            raise ValueError("transport maximum_attempts must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.total_deadline_seconds <= self.request_timeout_seconds:
            raise ValueError("total deadline must exceed request timeout")
        if self.initial_backoff_seconds < 0 or self.maximum_backoff_seconds < 0:
            raise ValueError("transport backoff must be non-negative")


class HealthBenchTransportError(RuntimeError):
    """A HealthBench request failed at the serving boundary."""


class HealthBenchTransportExhausted(HealthBenchTransportError):  # noqa: N818
    """All bounded attempts for one semantic request were exhausted."""


@dataclass(slots=True)
class QwenChatTransport:
    client: _Client
    model: str
    retry_policy: TransportRetryPolicy
    transient_error_types: tuple[type[BaseException], ...]
    sleeper: Callable[[float], None] = time.sleep
    temperature: float = CANDIDATE_TEMPERATURE
    max_tokens: int = CANDIDATE_MAX_TOKENS
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    enable_thinking: bool = False
    transport_retry_count: int = 0
    clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("transport max_tokens must be positive")

    def create(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None,
    ) -> Any:
        deadline = self.clock() + self.retry_policy.total_deadline_seconds
        delay = self.retry_policy.initial_backoff_seconds
        last_error: BaseException | None = None
        for attempt in range(1, self.retry_policy.maximum_attempts + 1):
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise HealthBenchTransportExhausted(
                    "HealthBench transport reached its request deadline"
                ) from last_error
            timeout = min(self.retry_policy.request_timeout_seconds, remaining)
            self.transport_retry_count += int(attempt > 1)
            request: dict[str, object] = {
                "timeout": timeout,
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            }
            if self.top_p is not None:
                request["top_p"] = self.top_p
            if self.top_k is not None or self.seed is not None:
                extra_body = cast(dict[str, object], request["extra_body"])
                if self.top_k is not None:
                    extra_body["top_k"] = -1 if self.top_k == 0 else self.top_k
                if self.seed is not None:
                    extra_body["seed"] = self.seed
            if response_format is not None:
                request["response_format"] = response_format
            try:
                return self.client.chat.completions.create(**request)
            except self.transient_error_types as exc:
                last_error = exc
                remaining = deadline - self.clock()
                if attempt == self.retry_policy.maximum_attempts or remaining <= 0:
                    raise HealthBenchTransportExhausted(
                        "Qwen HealthBench transport exhausted its bounded attempts"
                    ) from exc
                self.sleeper(min(delay, remaining))
                delay = min(delay * 2, self.retry_policy.maximum_backoff_seconds)
        raise AssertionError("bounded transport loop did not return or raise")


@dataclass(slots=True)
class ExternalJudgeTransport:
    """Bounded external rubric transport, defaulting to the new medium condition.

    HealthBench uses one semantic call per rubric item.  Transport retries keep
    the exact request invariant and never turn a missing judge response into a
    negative rubric verdict.
    """

    client: _Client
    model: str
    retry_policy: TransportRetryPolicy
    transient_error_types: tuple[type[BaseException], ...]
    sleeper: Callable[[float], None] = time.sleep
    max_completion_tokens: int = EXTERNAL_JUDGE_MAX_TOKENS
    reasoning_effort: str = EXTERNAL_JUDGE_EFFORT
    transport_retry_count: int = 0
    clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("external judge model must be non-empty")
        if self.model == EXTERNAL_JUDGE_MODEL and self.retry_policy.maximum_attempts != 1:
            raise ValueError("third-party Luna judging must not automatically retry paid requests")
        if self.max_completion_tokens <= 0:
            raise ValueError("external judge token limit must be positive")
        if self.reasoning_effort not in {"low", "medium"}:
            raise ValueError("external judge effort must be current medium or explicit legacy low")

    def create(self, *, messages: list[dict[str, str]]) -> Any:
        deadline = self.clock() + self.retry_policy.total_deadline_seconds
        delay = self.retry_policy.initial_backoff_seconds
        request: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "store": False,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        last_error: BaseException | None = None
        for attempt in range(1, self.retry_policy.maximum_attempts + 1):
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise HealthBenchTransportExhausted(
                    "HealthBench transport reached its request deadline"
                ) from last_error
            timeout = min(self.retry_policy.request_timeout_seconds, remaining)
            self.transport_retry_count += int(attempt > 1)
            try:
                response = self.client.chat.completions.create(**request, timeout=timeout)
                content = response.choices[0].message.content
                if content is None:
                    raise HealthBenchTransportError(
                        "external HealthBench judge returned no assistant content"
                    )
                if response.choices[0].finish_reason != "stop":
                    raise HealthBenchTransportError(
                        "external HealthBench judge did not complete its response"
                    )
                return response
            except self.transient_error_types as exc:
                last_error = exc
                remaining = deadline - self.clock()
                if attempt == self.retry_policy.maximum_attempts or remaining <= 0:
                    raise HealthBenchTransportExhausted(
                        "external HealthBench judge exhausted its bounded attempts"
                    ) from exc
                self.sleeper(min(delay, remaining))
                delay = min(delay * 2, self.retry_policy.maximum_backoff_seconds)
        raise AssertionError("bounded external judge loop did not return or raise")
