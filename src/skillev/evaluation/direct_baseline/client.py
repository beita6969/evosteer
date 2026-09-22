"""OpenAI-compatible, adapter-free client for direct Qwen evaluation."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, cast

from skillev.policy.token_counting import load_qwen_chat_tokenizer

from .config import DirectDecodingProfile


@dataclass(frozen=True, slots=True)
class DirectGenerationRequest:
    request_id: str
    messages: tuple[dict[str, str], ...]
    profile: DirectDecodingProfile
    service_slot: int = 0

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.messages:
            raise ValueError("request_id and messages are required")
        if type(self.service_slot) is not int or self.service_slot < 0:
            raise ValueError("service_slot must be a non-negative integer")
        for message in self.messages:
            if set(message) != {"role", "content"}:
                raise ValueError("messages require exactly role and content")
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError("unsupported message role")
            if not message["content"].strip():
                raise ValueError("message content must be non-empty")


@dataclass(frozen=True, slots=True)
class DirectGenerationResult:
    request_id: str
    text: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_text: str | None = None
    response_model: str | None = None
    response_id: str | None = None
    service_instance_id: str | None = None


class DirectGenerationError(RuntimeError):
    """An infrastructure or model-request failure, never a candidate failure."""


@dataclass(frozen=True, slots=True)
class DirectTokenBudget:
    input_tokens: int
    max_new_tokens: int
    context_length: int

    def validate(self) -> None:
        if min(self.input_tokens, self.max_new_tokens, self.context_length) < 0:
            raise ValueError("direct token budget fields must be non-negative")
        if self.input_tokens + self.max_new_tokens > self.context_length:
            raise DirectGenerationError("direct request exceeds frozen context budget")


class DirectGenerationClient(Protocol):
    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult: ...


class ChatTokenCounter(Protocol):
    tokenizer_id: str

    def count(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class QwenChatTokenCounter:
    tokenizer: Any
    tokenizer_id: str

    @classmethod
    def from_pretrained(cls, path: str) -> QwenChatTokenCounter:
        if not path.strip():
            raise ValueError("tokenizer path must be non-empty")
        tokenizer = load_qwen_chat_tokenizer(path)
        return cls(tokenizer=tokenizer, tokenizer_id=path)

    def count(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
    ) -> int:
        values = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if hasattr(values, "get"):
            input_ids = values.get("input_ids")
            if input_ids is not None:
                values = input_ids
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise TypeError("Qwen chat template did not return token IDs")
        return len(values)


class OpenAICompatibleDirectClient:
    """Minimal direct chat client that cannot address adapters by construction."""

    def __init__(
        self,
        *,
        endpoint_base: str,
        served_model_name: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 600.0,
        context_length: int,
        token_counter: ChatTokenCounter,
        service_instance_id: str | None = None,
    ) -> None:
        endpoint = endpoint_base.rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint_base must be HTTP(S)")
        if not served_model_name.strip():
            raise ValueError("served_model_name must be non-empty")
        lowered = served_model_name.lower()
        if any(marker in lowered for marker in ("adapter", "lora", "skill")):
            raise ValueError("direct lane rejects adapter-like served model names")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not token_counter.tokenizer_id.strip():
            raise ValueError("token counter identity must be non-empty")
        if service_instance_id is not None and not service_instance_id.strip():
            raise ValueError("service_instance_id must be non-empty or null")
        self._url = endpoint + "/chat/completions"
        self._models_url = endpoint + "/models"
        self._model = served_model_name
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._context_length = context_length
        self._token_counter = token_counter
        self._service_instance_id = service_instance_id

    async def model_routes(self) -> tuple[str, ...]:
        return await asyncio.to_thread(self._model_routes_sync)

    def _model_routes_sync(self) -> tuple[str, ...]:
        request = urllib.request.Request(  # noqa: S310 -- endpoint is operator supplied
            self._models_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 -- endpoint is operator supplied
                request, timeout=self._timeout_seconds
            ) as response:
                value = json.loads(response.read())
            data = cast(list[object], cast(dict[str, object], value)["data"])
            routes = tuple(cast(str, cast(dict[str, object], item)["id"]) for item in data)
        except (OSError, TimeoutError, KeyError, TypeError, urllib.error.HTTPError) as exc:
            raise DirectGenerationError("direct model-route probe failed") from exc
        if routes != (self._model,):
            raise DirectGenerationError("direct endpoint exposes an unexpected model route set")
        return routes

    def _payload(self, request: DirectGenerationRequest) -> dict[str, object]:
        profile = request.profile
        return {
            "model": self._model,
            "messages": list(request.messages),
            "max_tokens": profile.max_new_tokens,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "presence_penalty": profile.presence_penalty,
            "seed": profile.seed,
            "stop": list(profile.stop) or None,
            # Protocol profiles use zero for disabled top-k, while SGLang's
            # OpenAI-compatible wire contract uses -1 for the same setting.
            "top_k": -1 if profile.top_k == 0 else profile.top_k,
            "min_p": profile.min_p,
            "repetition_penalty": profile.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": profile.enable_thinking},
            "n": 1,
            "stream": False,
        }

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        input_tokens = self._token_counter.count(
            request.messages,
            enable_thinking=request.profile.enable_thinking,
        )
        DirectTokenBudget(
            input_tokens,
            request.profile.max_new_tokens,
            self._context_length,
        ).validate()
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        body = json.dumps(self._payload(request)).encode("utf-8")
        http_request = urllib.request.Request(  # noqa: S310 -- endpoint is operator supplied
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 -- endpoint is operator supplied
                http_request, timeout=self._timeout_seconds
            ) as response:
                value = json.loads(response.read())
        except (OSError, TimeoutError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise DirectGenerationError(
                f"direct generation request failed: {type(exc).__name__}"
            ) from exc
        try:
            choice = cast(dict[str, object], cast(list[object], value["choices"])[0])
            message = cast(dict[str, object], choice["message"])
            text = cast(str, message["content"])
            reasoning_value = message.get("reasoning_content")
            reasoning_text = reasoning_value if type(reasoning_value) is str else None
            finish_reason = cast(str, choice.get("finish_reason", "unknown"))
            usage = cast(dict[str, object], value.get("usage", {}))
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if type(text) is not str or type(finish_reason) is not str:
                raise TypeError
            if prompt_tokens is not None and type(prompt_tokens) is not int:
                raise TypeError
            if completion_tokens is not None and type(completion_tokens) is not int:
                raise TypeError
            response_model_value = value.get("model")
            response_model = response_model_value if type(response_model_value) is str else None
            response_id_value = value.get("id")
            response_id = response_id_value if type(response_id_value) is str else None
        except (KeyError, IndexError, TypeError) as exc:
            raise DirectGenerationError(
                "direct generation response has incompatible shape"
            ) from exc
        if response_model != self._model:
            raise DirectGenerationError("direct response model differs from requested base route")
        return DirectGenerationResult(
            request_id=request.request_id,
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_text=reasoning_text,
            response_model=response_model,
            response_id=response_id,
            service_instance_id=self._service_instance_id,
        )


@dataclass(frozen=True, slots=True)
class DeterministicReplicaClient:
    """Route each request to its manifest-derived replica slot."""

    clients: tuple[DirectGenerationClient, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("at least one replica client is required")

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        if request.service_slot >= len(self.clients):
            raise DirectGenerationError("invalid deterministic service slot")
        return await self.clients[request.service_slot].generate(request)


__all__ = [
    "ChatTokenCounter",
    "DeterministicReplicaClient",
    "DirectGenerationClient",
    "DirectGenerationError",
    "DirectGenerationRequest",
    "DirectGenerationResult",
    "DirectTokenBudget",
    "OpenAICompatibleDirectClient",
    "QwenChatTokenCounter",
]
