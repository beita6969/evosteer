"""Condition-matched SGLang generation for evaluation.

This module is deliberately separate from :mod:`external_sglang`.  Training
rollouts use raw categorical sampling so that the sampled distribution is the
one scored by teacher forcing.  Evaluation instead receives an explicit
benchmark condition for every call and never silently falls back to the
training sampler.
"""

from __future__ import annotations

import logging
import math
from asyncio import CancelledError, Semaphore, sleep
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable
from uuid import uuid4

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.runtime import BudgetVector
from skillev.runtime.async_sglang_transport import (
    AiohttpSGLangControlTransport,
    AsyncSGLangControlTransport,
)
from skillev.runtime.sglang_gateway import SGLangGatewayError

from .action_root_boundary import (
    ACTION_JSON_ROOT_BOUNDARY_VERSION,
    apply_action_json_root_boundary,
)
from .external_sglang import ExternalSGLangRolloutConfig
from .generator import (
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutTokenizerProtocol,
)
from .types import PolicySnapshot


class RolloutSamplingMode(StrEnum):
    """Sampling authorities that must never be implicitly interchanged."""

    TRAINING_RAW_SOFTMAX = "training-raw-softmax"
    EVALUATION_MATCHED = "evaluation-matched"


@dataclass(frozen=True, slots=True)
class EvaluationGenerationProfile:
    """Complete decoding condition for one evaluation generation call."""

    profile_id: str
    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    stop: tuple[str, ...] = ()
    sampling_mode: str = "sampling"
    thinking_boundary: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("evaluation generation profile identity is required")
        if type(self.enable_thinking) is not bool:
            raise TypeError("enable_thinking must be boolean")
        if self.thinking_boundary not in (None, "\n", "</think>", "\n\n"):
            raise ValueError("only native thinking boundary syntax may be constrained")
        if self.thinking_boundary is not None and (
            not self.enable_thinking or self.max_new_tokens != 1
        ):
            raise ValueError("thinking closure is a single delimiter token in a thinking-on stream")
        for name in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("evaluation sampling probabilities are invalid")
        if not -2 <= self.presence_penalty <= 2 or self.repetition_penalty <= 0:
            raise ValueError("evaluation sampling penalties are invalid")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if any(type(item) is not str or not item for item in self.stop):
            raise ValueError("stop values must be non-empty text")
        if self.sampling_mode not in {"sampling", "greedy"}:
            raise ValueError("sampling_mode must be sampling or greedy")
        if self.sampling_mode == "greedy" and (
            self.temperature != 0.0 or self.top_p != 1.0 or self.top_k != 1
        ):
            raise ValueError("greedy evaluation requires temperature/top_p/top_k = 0/1/1")
        if self.sampling_mode == "sampling" and self.temperature <= 0:
            raise ValueError("sampling evaluation requires positive temperature")

    @classmethod
    def frozen_reasoner(cls, *, max_new_tokens: int, seed: int) -> EvaluationGenerationProfile:
        """Return the deterministic frozen-base reasoner required by the TTB contract."""

        return cls(
            profile_id="qwen35-frozen-deterministic-reasoner@1",
            enable_thinking=True,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            seed=seed,
            sampling_mode="greedy",
        )


@dataclass(frozen=True, slots=True)
class EvaluationGenerationConstraint:
    """Syntax-only constraint fixed before generation; never a semantic repair."""

    json_schema: JsonValue | None = None
    regex: str | None = None
    apply_json_root_boundary: bool = False

    def __post_init__(self) -> None:
        if self.json_schema is not None and self.regex is not None:
            raise ValueError("evaluation generation accepts one grammar constraint")
        if self.json_schema is not None:
            normalized = normalize_json(self.json_schema)
            if not isinstance(normalized, dict) or normalized != self.json_schema:
                raise ValueError("json_schema must be a normalized JSON object")
        if self.regex is not None and (type(self.regex) is not str or not self.regex):
            raise ValueError("regex must be non-empty text")
        if self.apply_json_root_boundary and self.json_schema is None:
            raise ValueError("JSON root boundary requires a JSON schema")


class EvaluationProfileRequiredError(RuntimeError):
    """Raised when evaluation would otherwise inherit the training sampler."""


class EvaluationSGLangGenerationError(RuntimeError):
    """The adapter-free evaluation service returned an invalid native response."""


@dataclass(frozen=True, slots=True)
class EvaluationPolicyDescriptor:
    """Runtime identity for adapter-free or explicitly routed forward evaluation.

    The paired executor records and compares expanded service/model conditions.
    The coordinator binds trained descriptors to the selected checkpoint and
    observed serving route; this lightweight descriptor never loads training state.
    """

    snapshot_id: str
    backbone_id: str
    tokenizer_id: str
    forward_adapter_version: str = "adapter-free"

    def __post_init__(self) -> None:
        if not all((self.snapshot_id, self.backbone_id, self.tokenizer_id)):
            raise ValueError("evaluation policy descriptor is incomplete")


@runtime_checkable
class EvaluationRolloutGenerator(Protocol):
    """Rollout-compatible generator with an explicit per-call evaluation condition."""

    @property
    def tokenizer(self) -> RolloutTokenizerProtocol: ...

    def snapshot(self) -> PolicySnapshot | EvaluationPolicyDescriptor: ...

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None: ...

    def end_episode(self, episode_id: str) -> None: ...

    async def generate_evaluation(
        self,
        request: RolloutGenerationRequest,
        *,
        profile: EvaluationGenerationProfile,
        constraint: EvaluationGenerationConstraint | None = None,
    ) -> RolloutGenerationResult: ...


@dataclass(slots=True)
class EvaluationSGLangRolloutGenerator:
    """Exact-token SGLang client that requires a benchmark decoding profile."""

    config: ExternalSGLangRolloutConfig
    tokenizer: RolloutTokenizerProtocol
    snapshot_provider: Callable[[], PolicySnapshot | EvaluationPolicyDescriptor]
    transport: AsyncSGLangControlTransport = field(
        default_factory=AiohttpSGLangControlTransport,
        repr=False,
    )
    adapter_name: str | None = None
    transport_maximum_attempts: int = 3
    transport_retry_delay_seconds: float = 1.0
    _episodes: set[str] = field(default_factory=set, init=False, repr=False)
    _transport_slots: Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.adapter_name is not None and not self.adapter_name.strip():
            raise ValueError("evaluation adapter name must be non-empty or null")
        if type(self.transport_maximum_attempts) is not int or self.transport_maximum_attempts < 1:
            raise ValueError("evaluation transport attempts must be positive")
        if (
            isinstance(self.transport_retry_delay_seconds, bool)
            or not isinstance(self.transport_retry_delay_seconds, int | float)
            or not math.isfinite(float(self.transport_retry_delay_seconds))
            or self.transport_retry_delay_seconds < 0
        ):
            raise ValueError("evaluation transport retry delay must be finite and non-negative")
        self._transport_slots = Semaphore(self.config.transport_worker_threads)

    def snapshot(self) -> PolicySnapshot | EvaluationPolicyDescriptor:
        snapshot = self.snapshot_provider()
        if not isinstance(snapshot, PolicySnapshot | EvaluationPolicyDescriptor):
            raise TypeError("evaluation snapshot provider returned an incompatible value")
        return snapshot

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        if not episode_id.strip() or episode_id in self._episodes:
            raise ValueError("evaluation episode identity is empty or already active")
        if self.snapshot().snapshot_id != expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("evaluation policy changed before episode")
        self._episodes.add(episode_id)

    def end_episode(self, episode_id: str) -> None:
        if episode_id not in self._episodes:
            raise ValueError("evaluation episode is not active")
        self._episodes.remove(episode_id)

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        del request
        raise EvaluationProfileRequiredError(
            "evaluation generation requires an explicit condition-matched profile"
        )

    async def generate_evaluation(
        self,
        request: RolloutGenerationRequest,
        *,
        profile: EvaluationGenerationProfile,
        constraint: EvaluationGenerationConstraint | None = None,
    ) -> RolloutGenerationResult:
        if not isinstance(profile, EvaluationGenerationProfile):
            raise TypeError("evaluation generation profile is required")
        constraint = constraint or EvaluationGenerationConstraint()
        before = self.snapshot()
        if before.snapshot_id != request.expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("evaluation policy changed before generation")
        if request.max_new_tokens != profile.max_new_tokens:
            raise ValueError("evaluation request token budget differs from its profile")
        if request.seed != profile.seed:
            raise ValueError("evaluation request seed differs from its profile")
        if request.decoding_snapshot_id != profile.profile_id:
            raise ValueError("evaluation request decoding identity differs from its profile")

        top_k = -1 if profile.top_k == 0 else profile.top_k
        sampling: dict[str, JsonValue] = {
            "max_new_tokens": profile.max_new_tokens,
            "min_p": float(profile.min_p),
            "presence_penalty": float(profile.presence_penalty),
            "repetition_penalty": float(profile.repetition_penalty),
            "sampling_seed": profile.seed,
            "temperature": float(profile.temperature),
            "top_k": top_k,
            "top_p": float(profile.top_p),
        }
        if profile.stop:
            sampling["stop"] = list(profile.stop)
        closing_tokens: tuple[int, ...] = ()
        if profile.thinking_boundary is not None:
            closing_tokens = tuple(self.tokenizer.encode(profile.thinking_boundary))
            if len(closing_tokens) != 1 or constraint != EvaluationGenerationConstraint():
                raise ValueError("bounded thinking requires one native delimiter and no grammar")
            # Evaluation-only channel control. The delimiter is generated by the
            # server, counted in usage and persisted like every other token.
            # No answer tokens are forced or copied out of the reasoning draft.
            sampling["logit_bias"] = {str(closing_tokens[0]): 1_000_000.0}
        if constraint.json_schema is not None:
            sampling["json_schema"] = canonical_json(constraint.json_schema)
        if constraint.regex is not None:
            sampling["regex"] = constraint.regex
        payload = cast(
            dict[str, JsonValue],
            normalize_json(
                {
                    "input_ids": list(request.input_ids),
                    "sampling_params": sampling,
                    "stream": False,
                }
            ),
        )
        if self.adapter_name is not None:
            # SGLang's native endpoint calls this field ``lora_path`` even
            # when a previously loaded adapter is selected by its route name.
            # Keeping the route explicit prevents a trained evaluation from
            # silently falling back to the base model.
            payload["lora_path"] = self.adapter_name
        try:
            status, raw = await self._request_with_retry(payload)
        except SGLangGatewayError as exc:
            raise EvaluationSGLangGenerationError("SGLang evaluation request failed") from exc
        if status != 200:
            raise EvaluationSGLangGenerationError("SGLang evaluation returned failure")
        content_ids, stop_ids, finish_reason, prompt_tokens = self._parse(raw)
        if closing_tokens and (content_ids != closing_tokens or stop_ids):
            raise EvaluationSGLangGenerationError(
                "serving did not honor the channel-close boundary"
            )
        if constraint.apply_json_root_boundary:
            boundary = apply_action_json_root_boundary(self.tokenizer, content_ids)
            if boundary.matched:
                content_ids = boundary.content_token_ids
                stop_ids = ()
                finish_reason = ACTION_JSON_ROOT_BOUNDARY_VERSION
        after = self.snapshot()
        if after != before:
            raise PolicySnapshotMismatchError("evaluation policy changed during generation")
        return RolloutGenerationResult(
            content_token_ids=content_ids,
            stop_token_ids=stop_ids,
            finish_reason=finish_reason,
            policy_snapshot_id=after.snapshot_id,
            backend_id="sglang-native-evaluation-exact-token",
            usage=BudgetVector(
                input_tokens=prompt_tokens,
                output_tokens=len(content_ids) + len(stop_ids),
                model_calls=1,
            ),
        )

    async def _request_with_retry(self, payload: dict[str, JsonValue]) -> tuple[int, JsonValue]:
        """Retry only failures that produced no usable SGLang response.

        Candidate parse failures and non-2xx model responses remain definitive;
        this narrow retry covers transport resets and truncated/invalid JSON
        responses, which otherwise turn an interactive episode into an
        infrastructure failure before the model action can be observed.
        """

        for attempt in range(1, self.transport_maximum_attempts + 1):
            try:
                return await self._request(payload)
            except SGLangGatewayError:
                if attempt == self.transport_maximum_attempts:
                    raise
                await sleep(float(self.transport_retry_delay_seconds) * attempt)
        raise AssertionError("positive retry budget must return or raise")

    async def _request(self, payload: dict[str, JsonValue]) -> tuple[int, JsonValue]:
        async with self._transport_slots:
            request_id = f"skillev-eval-{uuid4().hex}"
            try:
                return await self.transport.request(
                    method="POST",
                    url=self.config.generate_url,
                    payload={**payload, "rid": request_id},
                    timeout_seconds=float(self.config.request_timeout_seconds),
                    max_response_bytes=self.config.max_response_bytes,
                )
            except (CancelledError, SGLangGatewayError):
                # Closing HTTP triggers SGLang's disconnect handling. Also
                # request an explicit, bounded abort of this exact request;
                # never abort a replica's other trajectories or adapter users.
                await self._abort_request(request_id)
                raise

    async def _abort_request(self, request_id: str) -> None:
        try:
            status, _ = await self.transport.request(
                method="POST",
                url=self.config.generate_url.removesuffix("/generate") + "/abort_request",
                payload={"rid": request_id, "abort_all": False},
                timeout_seconds=min(3.0, float(self.config.request_timeout_seconds)),
                max_response_bytes=self.config.max_response_bytes,
            )
            if not 200 <= status < 300:
                raise SGLangGatewayError("request-scoped abort was not acknowledged")
        except SGLangGatewayError:
            logging.getLogger(__name__).warning(
                "SGLang request-scoped abort failed; HTTP was closed, usage remains unknown"
            )

    @staticmethod
    def _parse(raw: JsonValue) -> tuple[tuple[int, ...], tuple[int, ...], str, int]:
        if not isinstance(raw, dict):
            raise EvaluationSGLangGenerationError("SGLang response must be an object")
        output_ids = raw.get("output_ids")
        meta = raw.get("meta_info")
        if not isinstance(output_ids, list) or not isinstance(meta, dict):
            raise EvaluationSGLangGenerationError("SGLang response fields are incomplete")
        completion_tokens = meta.get("completion_tokens")
        prompt_tokens = meta.get("prompt_tokens")
        finish = meta.get("finish_reason")
        if (
            type(completion_tokens) is not int
            or completion_tokens < 0
            or type(prompt_tokens) is not int
            or prompt_tokens < 0
            or not isinstance(finish, dict)
        ):
            raise EvaluationSGLangGenerationError("SGLang usage fields are invalid")
        if completion_tokens > len(output_ids):
            raise EvaluationSGLangGenerationError("SGLang completion count exceeds output IDs")
        generated = output_ids[-completion_tokens:] if completion_tokens else []
        if any(type(token_id) is not int or token_id < 0 for token_id in generated):
            raise EvaluationSGLangGenerationError("SGLang output IDs are invalid")
        generated_ids = tuple(cast(list[int], generated))
        finish_type = finish.get("type")
        if type(finish_type) is not str or not finish_type:
            raise EvaluationSGLangGenerationError("SGLang finish type is invalid")
        matched = finish.get("matched")
        stop_ids: tuple[int, ...] = ()
        content_ids = generated_ids
        if type(matched) is int and generated_ids and generated_ids[-1] == matched:
            content_ids = generated_ids[:-1]
            stop_ids = (matched,)
        return content_ids, stop_ids, finish_type, prompt_tokens

    def close(self) -> None:
        """Awaited requests own and close their sockets; no worker pool remains."""


__all__ = [
    "EvaluationGenerationConstraint",
    "EvaluationGenerationProfile",
    "EvaluationProfileRequiredError",
    "EvaluationRolloutGenerator",
    "EvaluationSGLangGenerationError",
    "EvaluationSGLangRolloutGenerator",
    "RolloutSamplingMode",
]
