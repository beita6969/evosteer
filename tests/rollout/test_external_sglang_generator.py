from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import pytest

from skillev.contracts import JsonValue
from skillev.rollout import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
    GenerationPhase,
    PolicySnapshot,
    RolloutGenerationRequest,
)
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayConfig


@dataclass(slots=True)
class _Transport:
    response: JsonValue
    calls: list[tuple[str, str, object]] = field(default_factory=list)

    def request(
        self,
        *,
        method: str,
        url: str,
        payload: object,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        del timeout_seconds, max_response_bytes
        self.calls.append((method, url, payload))
        if url.endswith("/load_lora_adapter"):
            return 200, {"success": True}
        if url.endswith("/health"):
            return 200, None
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "base"}, {"id": "theta_step_step-0"}]}
        if url.endswith("/v1/chat/completions"):
            return 200, {"choices": [{"message": {"content": "OK"}}]}
        return 200, self.response


@dataclass(slots=True)
class _BlockingTransport:
    response: JsonValue
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def request(
        self,
        *,
        method: str,
        url: str,
        payload: object,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        del method, payload, timeout_seconds, max_response_bytes
        if url.endswith("/generate"):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test transport was not released")
        return 200, self.response


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    tokenizer_id: str = "tokenizer-fixture"

    def encode(self, text: str) -> list[int]:
        del text
        return [1]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join({10: "A", 11: "B"}[token_id] for token_id in token_ids)


def _snapshot() -> PolicySnapshot:
    return PolicySnapshot.create(
        backbone_id="qwen-fixture",
        forward_adapter_version="adapter@0",
        tokenizer_id="tokenizer-fixture",
        backend_id="sglang-native-exact-token",
        initial_trainable_state_hash="a" * 64,
    )


def _generator(response: JsonValue) -> tuple[ExternalSGLangRolloutGenerator, _Transport]:
    transport = _Transport(response)
    gateway = SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base="http://127.0.0.1:30000",
            base_model="base",
            supervisor_adapter="theta_step_",
        ),
        _control_transport=transport,
    )
    asyncio.run(
        gateway.swap_supervisor_adapter(
            adapter_path="/private/adapter",
            adapter_revision="step-0",
        )
    )
    return (
        ExternalSGLangRolloutGenerator(
            config=ExternalSGLangRolloutConfig("http://127.0.0.1:30000"),
            tokenizer=_Tokenizer(),
            gateway=gateway,
            snapshot_provider=_snapshot,
            transport=transport,
        ),
        transport,
    )


def _request() -> RolloutGenerationRequest:
    return RolloutGenerationRequest(
        phase=GenerationPhase.REASONING,
        input_ids=(1, 2, 3),
        max_new_tokens=8,
        seed=0,
        decoding_snapshot_id="decode@1",
        expected_policy_snapshot_id=_snapshot().snapshot_id,
    )


def test_native_generator_keeps_exact_completion_suffix_and_raw_sampling() -> None:
    generator, transport = _generator(
        {
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "length"},
                "prompt_tokens": 3,
            },
            "output_ids": [91, 92, 10, 11],
            "text": "AB",
        }
    )

    result = asyncio.run(generator.generate(_request()))

    assert result.content_token_ids == (10, 11)
    assert result.stop_token_ids == ()
    assert result.usage.output_tokens == 2
    payload = transport.calls[-1][2]
    assert isinstance(payload, dict)
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["sampling_params"] == {
        "max_new_tokens": 8,
        "sampling_seed": 0,
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
    }
    assert "logprobs" not in payload


def test_native_generator_can_call_a_frozen_base_without_an_adapter() -> None:
    transport = _Transport(
        {
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "length"},
                "prompt_tokens": 3,
            },
            "output_ids": [10, 11],
            "text": "AB",
        }
    )
    generator = ExternalSGLangRolloutGenerator(
        config=ExternalSGLangRolloutConfig("http://127.0.0.1:30000"),
        tokenizer=_Tokenizer(),
        gateway=None,
        snapshot_provider=_snapshot,
        transport=transport,
    )

    result = asyncio.run(generator.generate(_request()))

    assert result.content_token_ids == (10, 11)
    payload = transport.calls[-1][2]
    assert isinstance(payload, dict)
    assert "lora_path" not in payload
    generator.close()


def test_native_generator_uses_exact_ids_when_detokenized_text_differs() -> None:
    generator, _ = _generator(
        {
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "length"},
                "prompt_tokens": 3,
            },
            "output_ids": [10, 11],
            "text": "different",
        }
    )

    result = asyncio.run(generator.generate(_request()))

    assert result.content_token_ids == (10, 11)


def test_native_generator_separates_matched_stop_token() -> None:
    generator, _ = _generator(
        {
            "meta_info": {
                "completion_tokens": 3,
                "finish_reason": {"matched": 99, "type": "stop"},
                "prompt_tokens": 3,
            },
            "output_ids": [10, 11, 99],
            "text": "AB",
        }
    )

    result = asyncio.run(generator.generate(_request()))
    assert result.content_token_ids == (10, 11)
    assert result.stop_token_ids == (99,)


def test_blocking_transport_does_not_block_the_event_loop() -> None:
    response: JsonValue = {
        "meta_info": {
            "completion_tokens": 2,
            "finish_reason": {"type": "length"},
            "prompt_tokens": 3,
        },
        "output_ids": [10, 11],
        "text": "AB",
    }
    generator, _ = _generator(response)
    transport = _BlockingTransport(response)
    generator.transport = transport

    async def scenario() -> None:
        generation = asyncio.create_task(generator.generate(_request()))
        while not transport.started.is_set():
            await asyncio.sleep(0)
        heartbeat_completed = False

        async def heartbeat() -> None:
            nonlocal heartbeat_completed
            await asyncio.sleep(0.01)
            heartbeat_completed = True

        await heartbeat()
        assert heartbeat_completed
        assert not generation.done()
        transport.release.set()
        assert (await generation).content_token_ids == (10, 11)

    asyncio.run(scenario())
    generator.close()


def test_cancellation_drains_transport_before_releasing_adapter_lease() -> None:
    response: JsonValue = {
        "meta_info": {
            "completion_tokens": 2,
            "finish_reason": {"type": "length"},
            "prompt_tokens": 3,
        },
        "output_ids": [10, 11],
        "text": "AB",
    }
    generator, _ = _generator(response)
    transport = _BlockingTransport(response)
    generator.transport = transport

    async def scenario() -> None:
        generation = asyncio.create_task(generator.generate(_request()))
        while not transport.started.is_set():
            await asyncio.sleep(0)
        generation.cancel()
        await asyncio.sleep(0.01)
        assert generator.gateway.supervisor_inflight == 1
        transport.release.set()
        with pytest.raises(asyncio.CancelledError):
            await generation
        assert generator.gateway.supervisor_inflight == 0

    asyncio.run(scenario())
    generator.close()
