from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from skillev.contracts import JsonValue
from skillev.rollout import (
    EvaluationGenerationConstraint,
    EvaluationGenerationProfile,
    EvaluationProfileRequiredError,
    EvaluationSGLangRolloutGenerator,
    ExternalSGLangRolloutConfig,
    GenerationPhase,
    PolicySnapshot,
    RolloutGenerationRequest,
)
from skillev.runtime.sglang_gateway import SGLangGatewayError


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    tokenizer_id: str = "tokenizer-fixture"

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def encode_rollout_prompt(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return bytes(token_ids).decode()


@dataclass(slots=True)
class _Transport:
    calls: list[object] = field(default_factory=list)
    failures_remaining: int = 0

    async def request(
        self,
        *,
        method: str,
        url: str,
        payload: object,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        del method, timeout_seconds, max_response_bytes
        if url.endswith("/abort_request"):
            return 200, None
        self.calls.append(payload)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise SGLangGatewayError("transient fixture transport failure")
        return 200, {
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 3,
            },
            "output_ids": [65, 66],
        }


def _snapshot() -> PolicySnapshot:
    return PolicySnapshot.create(
        backbone_id="base",
        forward_adapter_version="adapter-free",
        tokenizer_id="tokenizer-fixture",
        backend_id="sglang-native-evaluation-exact-token",
        initial_trainable_state_hash="frozen-base",
    )


def _generator() -> tuple[EvaluationSGLangRolloutGenerator, _Transport]:
    transport = _Transport()
    return (
        EvaluationSGLangRolloutGenerator(
            ExternalSGLangRolloutConfig("http://127.0.0.1:30000"),
            _Tokenizer(),
            _snapshot,
            transport,
        ),
        transport,
    )


def test_evaluation_generator_cannot_implicitly_use_training_raw_softmax() -> None:
    generator, _ = _generator()
    request = RolloutGenerationRequest(
        GenerationPhase.REASONING,
        (1, 2, 3),
        8,
        42,
        "matched@1",
        _snapshot().snapshot_id,
    )
    with pytest.raises(EvaluationProfileRequiredError):
        asyncio.run(generator.generate(request))
    generator.close()


def test_evaluation_generator_sends_the_explicit_profile_and_json_grammar() -> None:
    generator, transport = _generator()
    profile = EvaluationGenerationProfile(
        "matched@1",
        False,
        0.7,
        0.8,
        20,
        0.0,
        0.0,
        1.0,
        8,
        42,
    )
    request = RolloutGenerationRequest(
        GenerationPhase.ACTION,
        (1, 2, 3),
        8,
        42,
        profile.profile_id,
        _snapshot().snapshot_id,
    )
    constraint = EvaluationGenerationConstraint(
        json_schema={"type": "object"},
        apply_json_root_boundary=True,
    )

    result = asyncio.run(
        generator.generate_evaluation(request, profile=profile, constraint=constraint)
    )

    assert result.content_token_ids == (65, 66)
    payload = transport.calls[0]
    assert isinstance(payload, dict)
    assert payload["sampling_params"] == {
        "json_schema": '{"type":"object"}',
        "max_new_tokens": 8,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "sampling_seed": 42,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }
    assert "lora_path" not in payload
    generator.close()


def test_evaluation_generator_routes_every_call_through_loaded_adapter() -> None:
    generator, transport = _generator()
    generator.adapter_name = "trained-forward-step-16"
    profile = EvaluationGenerationProfile(
        "matched@1",
        False,
        0.7,
        0.8,
        20,
        0.0,
        0.0,
        1.0,
        8,
        42,
    )
    request = RolloutGenerationRequest(
        GenerationPhase.REASONING,
        (1, 2, 3),
        8,
        42,
        profile.profile_id,
        _snapshot().snapshot_id,
    )

    asyncio.run(generator.generate_evaluation(request, profile=profile))

    payload = transport.calls[0]
    assert isinstance(payload, dict)
    assert payload["lora_path"] == "trained-forward-step-16"
    generator.close()


def test_evaluation_generator_retries_only_transport_failures() -> None:
    generator, transport = _generator()
    generator.transport_retry_delay_seconds = 0.0
    transport.failures_remaining = 1
    profile = EvaluationGenerationProfile(
        "matched@1",
        False,
        0.7,
        0.8,
        20,
        0.0,
        0.0,
        1.0,
        8,
        42,
    )
    request = RolloutGenerationRequest(
        GenerationPhase.REASONING,
        (1, 2, 3),
        8,
        42,
        profile.profile_id,
        _snapshot().snapshot_id,
    )

    asyncio.run(generator.generate_evaluation(request, profile=profile))

    assert len(transport.calls) == 2
    generator.close()
