from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from skillev.evaluation.direct_baseline.client import (
    DeterministicReplicaClient,
    DirectGenerationRequest,
    DirectGenerationResult,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from skillev.evaluation.direct_baseline.config import DirectDecodingProfile


class _TokenCounter:
    tokenizer_id = "unit-test-tokenizer"

    def count(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
    ) -> int:
        del enable_thinking
        return sum(len(message["content"].split()) + 3 for message in messages)


class _BatchEncodingLike(dict[str, list[int]]):
    pass


class _QwenTokenizer:
    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> _BatchEncodingLike:
        self.enable_thinking_values.append(bool(_kwargs["enable_thinking"]))
        return _BatchEncodingLike(input_ids=[1, 2, 3, 4])


def test_qwen_counter_accepts_transformers_batch_encoding() -> None:
    tokenizer = _QwenTokenizer()
    counter = QwenChatTokenCounter(tokenizer, "qwen-test")
    assert counter.count(({"role": "user", "content": "hello"},), enable_thinking=False) == 4
    assert counter.count(({"role": "user", "content": "hello"},), enable_thinking=True) == 4
    assert tokenizer.enable_thinking_values == [False, True]


def _client(**changes: object) -> OpenAICompatibleDirectClient:
    values: dict[str, object] = {
        "endpoint_base": "http://localhost:8000/v1",
        "served_model_name": "qwen35-direct-base",
        "context_length": 4096,
        "token_counter": _TokenCounter(),
    }
    values.update(changes)
    return OpenAICompatibleDirectClient(**values)  # type: ignore[arg-type]


def _profile() -> DirectDecodingProfile:
    return DirectDecodingProfile(
        profile_id="test@1",
        enable_thinking=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_new_tokens=2048,
        stop=("STOP",),
        seed=42,
    )


@pytest.mark.parametrize("name", ["my-lora", "adapter-v1", "learned-skill-model"])
def test_base_route_rejects_adapter_like_model_names(name: str) -> None:
    with pytest.raises(ValueError):
        _client(served_model_name=name)


def test_payload_sends_every_decoding_control_at_wire_level() -> None:
    client = _client()
    payload = client._payload(
        DirectGenerationRequest(
            request_id="one",
            messages=({"role": "user", "content": "question"},),
            profile=_profile(),
        )
    )
    assert payload["model"] == "qwen35-direct-base"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repetition_penalty"] == 1.0
    assert payload["max_tokens"] == 2048
    assert payload["stop"] == ["STOP"]
    assert payload["seed"] == 42
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in payload


def test_payload_maps_disabled_top_k_to_sglang_wire_sentinel() -> None:
    profile = replace(_profile(), top_k=0)
    payload = _client()._payload(
        DirectGenerationRequest(
            request_id="disabled-top-k",
            messages=({"role": "user", "content": "question"},),
            profile=profile,
        )
    )

    assert profile.top_k == 0
    assert payload["top_k"] == -1


class _Replica:
    def __init__(self, instance: str) -> None:
        self.instance = instance
        self.request_ids: list[str] = []

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        self.request_ids.append(request.request_id)
        return DirectGenerationResult(
            request.request_id,
            "answer",
            "stop",
            1,
            1,
            service_instance_id=self.instance,
        )


def test_replica_routing_uses_manifest_slot_not_coroutine_arrival_order() -> None:
    replicas = (_Replica("service-0"), _Replica("service-1"))
    client = DeterministicReplicaClient(replicas)
    requests = tuple(
        DirectGenerationRequest(
            request_id=f"task-{index}",
            messages=({"role": "user", "content": "question"},),
            profile=_profile(),
            service_slot=index % 2,
        )
        for index in range(6)
    )

    async def run_all() -> tuple[DirectGenerationResult, ...]:
        return tuple(await asyncio.gather(*(client.generate(item) for item in requests)))

    results = asyncio.run(run_all())
    assert [item.service_instance_id for item in results] == [
        "service-0",
        "service-1",
        "service-0",
        "service-1",
        "service-0",
        "service-1",
    ]
