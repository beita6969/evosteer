"""Shared token budgeting and retry policy for OpenAI-compatible requests."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from openai import APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatTokenBudget:
    prompt_tokens: int
    completion_tokens: int
    safety_tokens: int
    context_limit: int

    @property
    def required_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.safety_tokens

    @property
    def overflow_tokens(self) -> int:
        return max(0, self.required_tokens - self.context_limit)


class ContextBudgetExceeded(RuntimeError):
    def __init__(
        self,
        budget: ChatTokenBudget,
        *,
        request_kind: str,
        source: str,
    ) -> None:
        self.budget = budget
        self.request_kind = request_kind
        self.source = source
        super().__init__(
            "OpenAI request context budget exceeded: "
            f"request_kind={request_kind} source={source} "
            f"prompt_tokens={budget.prompt_tokens} "
            f"completion_tokens={budget.completion_tokens} "
            f"context_limit={budget.context_limit}"
        )


class PermanentRequestRejected(RuntimeError):
    def __init__(self, *, request_kind: str, status_code: int | None) -> None:
        self.request_kind = request_kind
        self.status_code = status_code
        super().__init__(
            "Permanent OpenAI request rejection: "
            f"request_kind={request_kind} status_code={status_code}"
        )


class TransientRequestExhausted(RuntimeError):
    def __init__(
        self,
        *,
        request_kind: str,
        attempts: int,
        status_code: int | None,
    ) -> None:
        self.request_kind = request_kind
        self.attempts = attempts
        self.status_code = status_code
        super().__init__(
            "Transient OpenAI request retries exhausted: "
            f"request_kind={request_kind} attempts={attempts} "
            f"status_code={status_code}"
        )


class OpenAIRequestPolicy:
    def __init__(
        self,
        *,
        tokenizer: Any,
        context_limit: int,
        safety_tokens: int = 64,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if context_limit <= 0:
            raise ValueError("context_limit must be positive")
        if safety_tokens < 0 or max_retries < 0:
            raise ValueError("safety_tokens and max_retries must be non-negative")
        self.tokenizer = tokenizer
        self.context_limit = int(context_limit)
        self.safety_tokens = int(safety_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self._sleep = sleep

    def budget(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        completion_tokens: int,
        enable_thinking: bool,
    ) -> ChatTokenBudget:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = list(tools)
        tokenizer_messages = _messages_for_chat_template(messages)
        try:
            encoded = self.tokenizer.apply_chat_template(
                tokenizer_messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(tokenizer_messages, **kwargs)
        return ChatTokenBudget(
            prompt_tokens=_encoded_token_count(encoded),
            completion_tokens=int(completion_tokens),
            safety_tokens=self.safety_tokens,
            context_limit=self.context_limit,
        )

    def create_chat_completion(
        self,
        client: Any,
        *,
        request_kind: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Any:
        budget = self.budget(
            messages=messages,
            tools=tools,
            completion_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        if budget.required_tokens > budget.context_limit:
            raise ContextBudgetExceeded(
                budget, request_kind=request_kind, source="local_preflight"
            )

        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": list(messages),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = list(tools)
                if extra_body:
                    kwargs["extra_body"] = dict(extra_body)
                return client.chat.completions.create(**kwargs)
            except Exception as error:
                status_code = getattr(error, "status_code", None)
                if status_code == 400 and _is_server_context_rejection(error):
                    raise ContextBudgetExceeded(
                        budget,
                        request_kind=request_kind,
                        source="server_rejection",
                    ) from error
                transient = isinstance(error, (APIConnectionError, APITimeoutError)) or (
                    status_code in {408, 409, 429}
                    or isinstance(status_code, int) and status_code >= 500
                )
                logger.warning(
                    "OpenAI request failed: request_kind=%s attempt=%s/%s "
                    "error_class=%s http_status=%s prompt_tokens=%s "
                    "completion_tokens=%s context_limit=%s",
                    request_kind,
                    attempt,
                    attempts,
                    type(error).__name__,
                    status_code,
                    budget.prompt_tokens,
                    budget.completion_tokens,
                    budget.context_limit,
                )
                if not transient:
                    raise PermanentRequestRejected(
                        request_kind=request_kind,
                        status_code=status_code if isinstance(status_code, int) else None,
                    ) from error
                if attempt == attempts:
                    raise TransientRequestExhausted(
                        request_kind=request_kind,
                        attempts=attempts,
                        status_code=status_code if isinstance(status_code, int) else None,
                    ) from error
                self._sleep(self.retry_delay_seconds * attempt)
        raise AssertionError("unreachable")


def _is_server_context_rejection(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token count exceeds",
            "input is too long",
        )
    )


def _encoded_token_count(encoded: Any) -> int:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise TypeError("chat template mapping has no input_ids")
        return _encoded_token_count(encoded["input_ids"])
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is not None:
        return _encoded_token_count(input_ids)
    shape = getattr(encoded, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
        if encoded and isinstance(encoded[0], Sequence):
            return len(encoded[0])
        return len(encoded)
    raise TypeError("unsupported chat template token container")


def _messages_for_chat_template(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt OpenAI tool-call arguments to the Qwen tokenizer template.

    OpenAI's wire format stores ``function.arguments`` as a JSON string, while
    the pinned Qwen tokenizer iterates over it as a mapping.  SGLang performs
    this conversion before rendering its template; mirror that conversion for
    local token counting without changing the actual HTTP payload.
    """

    normalized = [dict(message) for message in messages]
    for message in normalized:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            continue
        normalized_calls: list[Any] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                normalized_calls.append(tool_call)
                continue
            normalized_call = dict(tool_call)
            function = normalized_call.get("function")
            if isinstance(function, Mapping):
                normalized_function = dict(function)
                arguments = normalized_function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, Mapping):
                        normalized_function["arguments"] = dict(parsed)
                normalized_call["function"] = normalized_function
            normalized_calls.append(normalized_call)
        message["tool_calls"] = normalized_calls
    return normalized
