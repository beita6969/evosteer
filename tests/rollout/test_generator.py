from __future__ import annotations

import asyncio
from dataclasses import fields
from typing import Any, cast

import pytest

from skillev.policy.interface import (
    AdapterRole,
    GenerationResult,
    PolicyBackbone,
    PolicyGenerationRequest,
)
from skillev.rollout import (
    GenerationPhase,
    LocalPolicyGenerator,
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
)
from skillev.runtime import BudgetVector


class _Tokenizer:
    tokenizer_id = "tokenizer-v1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _Backbone:
    """Dependency-free behavioral fake for the narrow local-generator adapter."""

    def __init__(self, result: GenerationResult) -> None:
        self._result = result
        self._backbone_id = "base-v1"
        self._adapter_version = "adapter-v1@0"
        self._initial_trainable_state_hash = "initial-trainable-state-generator-v1"
        self._tokenizer = _Tokenizer()
        self.requests: list[PolicyGenerationRequest] = []
        self.mutate_during_generate = False

    @property
    def backbone_id(self) -> str:
        return self._backbone_id

    @property
    def tokenizer(self) -> _Tokenizer:
        return self._tokenizer

    @property
    def initial_trainable_state_hash(self) -> str:
        return self._initial_trainable_state_hash

    def adapter_version(self, role: AdapterRole) -> str:
        assert role is AdapterRole.FORWARD_POLICY
        return self._adapter_version

    def generate_policy(self, request: PolicyGenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.mutate_during_generate:
            self._adapter_version = "adapter-v1@1"
        return self._result


def _local_generator(
    *,
    content_token_ids: tuple[int, ...] = (10, 11),
    stop_token_ids: tuple[int, ...] = (2,),
    finish_reason: str = "stop",
) -> tuple[LocalPolicyGenerator, _Backbone]:
    backbone = _Backbone(
        GenerationResult(
            content_token_ids=content_token_ids,
            stop_token_ids=stop_token_ids,
            finish_reason=finish_reason,
        )
    )
    return LocalPolicyGenerator(cast(PolicyBackbone, backbone)), backbone


def _request(snapshot_id: str, **changes: Any) -> RolloutGenerationRequest:
    values: dict[str, Any] = {
        "phase": GenerationPhase.ACTION,
        "input_ids": (4, 5, 6),
        "max_new_tokens": 8,
        "seed": 17,
        "decoding_snapshot_id": "decoding-v1",
        "expected_policy_snapshot_id": snapshot_id,
    }
    values.update(changes)
    return RolloutGenerationRequest(**values)


def test_local_generator_splits_only_the_explicit_stop_suffix() -> None:
    generator, backbone = _local_generator()
    pinned = generator.snapshot()
    request = _request(pinned.snapshot_id)

    result = asyncio.run(generator.generate(request))

    assert result.content_token_ids == (10, 11)
    assert result.stop_token_ids == (2,)
    assert result.finish_reason == "stop"
    assert result.policy_snapshot_id == pinned.snapshot_id
    assert result.backend_id == pinned.backend_id == "hf-local"
    assert result.usage == BudgetVector(
        input_tokens=len(request.input_ids),
        output_tokens=3,
        model_calls=1,
    )
    assert backbone.requests == [
        PolicyGenerationRequest(
            input_ids=request.input_ids,
            max_new_tokens=request.max_new_tokens,
            seed=request.seed,
            decoding_snapshot_id=request.decoding_snapshot_id,
        )
    ]


def test_local_generator_preserves_all_output_as_content_without_stop_suffix() -> None:
    generator, _ = _local_generator(
        content_token_ids=(10, 2, 11),
        stop_token_ids=(),
        finish_reason="length",
    )
    pinned = generator.snapshot()

    result = asyncio.run(generator.generate(_request(pinned.snapshot_id)))

    # Token identity, not token value guessing, determines the content boundary.
    assert result.content_token_ids == (10, 2, 11)
    assert result.stop_token_ids == ()


def test_local_generator_rejects_a_stale_expected_snapshot_before_generation() -> None:
    generator, backbone = _local_generator()

    with pytest.raises(PolicySnapshotMismatchError):
        asyncio.run(generator.generate(_request("stale-snapshot")))

    assert backbone.requests == []


def test_local_generator_rejects_snapshot_change_during_generation() -> None:
    generator, backbone = _local_generator()
    pinned = generator.snapshot()
    backbone.mutate_during_generate = True

    with pytest.raises(PolicySnapshotMismatchError):
        asyncio.run(generator.generate(_request(pinned.snapshot_id)))

    assert len(backbone.requests) == 1
    assert generator.snapshot().snapshot_id != pinned.snapshot_id


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "action"},
        {"input_ids": ()},
        {"input_ids": [1]},
        {"input_ids": (True,)},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"seed": -1},
        {"seed": 2**64},
        {"decoding_snapshot_id": " "},
        {"decoding_snapshot_id": 1},
        {"expected_policy_snapshot_id": " "},
        {"expected_policy_snapshot_id": 1},
    ],
)
def test_rollout_generation_request_rejects_invalid_values(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _request("snapshot-v1", **changes)


def test_rollout_generation_result_round_trips_usage_without_a_logprob_channel() -> None:
    result = RolloutGenerationResult(
        content_token_ids=(7, 8),
        stop_token_ids=(2,),
        finish_reason="stop",
        policy_snapshot_id="snapshot-v1",
        backend_id="fake-local",
        usage=BudgetVector(input_tokens=4, output_tokens=3, model_calls=1),
    )

    assert RolloutGenerationResult.from_value(result.to_value()) == result
    assert all("logprob" not in item.name for item in fields(result))
    assert "logprob" not in repr(result.to_value()).lower()


@pytest.mark.parametrize(
    "changes",
    [
        {"content_token_ids": [7]},
        {"content_token_ids": (-1,)},
        {"stop_token_ids": [2]},
        {"stop_token_ids": (True,)},
        {"finish_reason": " "},
        {"policy_snapshot_id": " "},
        {"backend_id": 1},
        {"usage": BudgetVector(output_tokens=1)},
    ],
)
def test_rollout_generation_result_rejects_invalid_values(
    changes: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "content_token_ids": (7, 8),
        "stop_token_ids": (2,),
        "finish_reason": "stop",
        "policy_snapshot_id": "snapshot-v1",
        "backend_id": "fake-local",
        "usage": BudgetVector(output_tokens=3),
    }
    values.update(changes)
    with pytest.raises(ValueError):
        RolloutGenerationResult(**values)
