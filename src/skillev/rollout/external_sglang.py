"""Exact-token rollout generation through SGLang's native ``/generate`` API."""

from __future__ import annotations

import math
import time
from asyncio import CancelledError, sleep
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import urlsplit

from skillev.contracts import JsonValue, normalize_json
from skillev.diagnostics.rollout_progress import current_progress, server_metrics
from skillev.runtime import BudgetVector
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.sglang_gateway import (
    SGLangControlTransport,
    SGLangGateway,
    SGLangGatewayError,
    UrllibSGLangControlTransport,
)

from .action_root_boundary import (
    ACTION_JSON_ROOT_BOUNDARY_VERSION,
    apply_action_json_root_boundary,
)
from .generator import (
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutTokenizerProtocol,
)
from .types import GenerationPhase, PolicySnapshot


class ExternalSGLangGenerationError(RuntimeError):
    """SGLang returned no exact, self-consistent generated-token sequence."""


@dataclass(frozen=True, slots=True)
class ExternalSGLangRolloutConfig:
    endpoint_base: str
    request_timeout_seconds: float = 300.0
    max_response_bytes: int = 16 * 1024 * 1024
    transport_worker_threads: int = 1
    server_action_boundary: bool = True

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint_base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("SGLang rollout endpoint must be an HTTP(S) URL without credentials")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, int | float)
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("SGLang rollout timeout must be finite and positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise ValueError("SGLang rollout response limit must be positive")
        if type(self.transport_worker_threads) is not int or self.transport_worker_threads < 1:
            raise ValueError("SGLang transport worker count must be positive")
        if type(self.server_action_boundary) is not bool:
            raise TypeError("SGLang action boundary switch must be boolean")

    @property
    def generate_url(self) -> str:
        return self.endpoint_base.rstrip("/").removesuffix("/v1") + "/generate"

    def to_value(self) -> dict[str, JsonValue]:
        """Return the private execution binding without transport state."""

        return {
            "endpoint_base": self.endpoint_base,
            "max_response_bytes": self.max_response_bytes,
            "request_timeout_seconds": float(self.request_timeout_seconds),
            "transport_worker_threads": self.transport_worker_threads,
            "server_action_boundary": self.server_action_boundary,
        }

    @classmethod
    def from_value(cls, value: object) -> ExternalSGLangRolloutConfig:
        normalized = normalize_json(value)
        fields = {
            "endpoint_base",
            "max_response_bytes",
            "request_timeout_seconds",
            "transport_worker_threads",
        }
        if not isinstance(normalized, dict) or set(normalized) not in (
            fields,
            fields | {"server_action_boundary"},
        ):
            raise ValueError("SGLang rollout binding has incompatible fields")
        endpoint = normalized["endpoint_base"]
        timeout = normalized["request_timeout_seconds"]
        response_limit = normalized["max_response_bytes"]
        worker_threads = normalized["transport_worker_threads"]
        if type(endpoint) is not str:
            raise TypeError("SGLang rollout endpoint must be text")
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise TypeError("SGLang rollout timeout must be numeric")
        if type(response_limit) is not int or type(worker_threads) is not int:
            raise TypeError("SGLang rollout integer fields are invalid")
        return cls(
            endpoint_base=endpoint,
            request_timeout_seconds=float(timeout),
            max_response_bytes=response_limit,
            transport_worker_threads=worker_threads,
            server_action_boundary=cast(bool, normalized.get("server_action_boundary", True)),
        )


@dataclass(slots=True)
class ExternalSGLangRolloutGenerator:
    """Use remote forward-policy sampling while retaining exact output IDs.

    The request fixes raw categorical sampling (temperature 1, unrestricted
    nucleus/top-k) so GPU-side teacher forcing remains the scientific logprob
    source.  Serving logprobs are deliberately not requested or consumed.
    """

    config: ExternalSGLangRolloutConfig
    tokenizer: RolloutTokenizerProtocol
    gateway: SGLangGateway | None
    snapshot_provider: Callable[[], PolicySnapshot]
    transport: SGLangControlTransport = field(
        default_factory=UrllibSGLangControlTransport,
        repr=False,
    )
    request_journal: DurableRequestJournal | None = None
    physical_usage: dict[str, int] = field(
        default_factory=lambda: {
            "server_generated_tokens": 0,
            "admitted_content_tokens": 0,
            "admitted_stop_tokens": 0,
            "discarded_suffix_tokens": 0,
            "server_action_root_stops": 0,
            "action_boundary_cpu_microseconds": 0,
        },
        init=False,
    )
    _episodes: set[str] = field(default_factory=set, init=False, repr=False)
    _episode_gateways: dict[str, SGLangGateway] = field(
        default_factory=dict, init=False, repr=False
    )
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.transport_worker_threads,
            thread_name_prefix="skillev-sglang-http",
        )

    def snapshot(self) -> PolicySnapshot:
        snapshot = self.snapshot_provider()
        if not isinstance(snapshot, PolicySnapshot):
            raise TypeError("external SGLang snapshot provider returned an incompatible value")
        return snapshot

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        if not episode_id.strip() or episode_id in self._episodes:
            raise ValueError("rollout episode identity is empty or already active")
        if self.snapshot().snapshot_id != expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("remote rollout policy changed before episode")
        from skillev.runtime.sglang_pool import SGLangActorPool

        if isinstance(self.gateway, SGLangActorPool):
            endpoint = (
                self.request_journal.episode_route(episode_id, expected_policy_snapshot_id)
                if self.request_journal is not None
                else None
            )
            row = current_progress()
            horizon = 1 if row is None else row.request_priority[0]
            member = self.gateway.acquire_episode(
                episode_id,
                expected_policy_snapshot_id,
                endpoint=endpoint,
                estimated_work=horizon * self.gateway.config.max_output_tokens,
            )
            try:
                if self.request_journal is not None:
                    self.request_journal.save_episode_route(
                        episode_id, expected_policy_snapshot_id, member.config.api_root
                    )
            except Exception:
                self.gateway.release_episode(episode_id)
                raise
            self._episode_gateways[episode_id] = member
        self._episodes.add(episode_id)

    def end_episode(self, episode_id: str) -> None:
        if episode_id not in self._episodes:
            raise ValueError("rollout episode is not active")
        from skillev.runtime.sglang_pool import SGLangActorPool

        if isinstance(self.gateway, SGLangActorPool):
            self.gateway.release_episode(episode_id)
            del self._episode_gateways[episode_id]
        self._episodes.remove(episode_id)

    def execution_endpoint(self, request: RolloutGenerationRequest) -> str:
        gateway = self._episode_gateways.get(request.episode_id or "", self.gateway)
        return self.config.endpoint_base if gateway is None else gateway.config.api_root

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        from skillev.runtime.sglang_pool import SGLangActorPool

        if isinstance(self.gateway, SGLangActorPool) and request.episode_id is not None:
            row = current_progress()
            horizon = 1 if row is None else row.request_priority[0]
            remaining = max(1, horizon - (request.turn_index or 1) + 1)
            self.gateway.update_episode_work(
                request.episode_id, remaining * (len(request.input_ids) + request.max_new_tokens)
            )
        before = self.snapshot()
        if before.snapshot_id != request.expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("remote rollout policy changed before generation")
        gateway = self._episode_gateways.get(request.episode_id or "", self.gateway)
        generation = None if gateway is None else gateway.begin_supervisor_rollout()
        endpoint = (
            self.config.generate_url if gateway is None else gateway.config.api_root + "/generate"
        )
        payload: dict[str, JsonValue] = cast(
            dict[str, JsonValue],
            normalize_json(
                {
                    "input_ids": list(request.input_ids),
                    "sampling_params": {
                        "max_new_tokens": request.max_new_tokens,
                        "sampling_seed": request.seed,
                        "temperature": 1.0,
                        "top_k": -1,
                        "top_p": 1.0,
                    },
                    "stream": False,
                }
            ),
        )
        if generation is not None:
            payload["lora_path"] = generation.adapter_name
        if (
            request.phase is GenerationPhase.ACTION
            and request.action_boundary_version == ACTION_JSON_ROOT_BOUNDARY_VERSION
            and self.config.server_action_boundary
        ):
            sampling = cast(dict[str, JsonValue], payload["sampling_params"])
            sampling["custom_params"] = {
                "skillev_action_boundary": ACTION_JSON_ROOT_BOUNDARY_VERSION
            }
            sampling["no_stop_trim"] = True
        try:
            status, raw = await self._request_without_early_lease_release(
                payload=payload, endpoint=endpoint, request=request
            )
        except SGLangGatewayError as error:
            raise ExternalSGLangGenerationError("SGLang native rollout request failed") from error
        finally:
            if generation is not None:
                assert gateway is not None
                gateway.end_supervisor_rollout()
        if status != 200:
            raise ExternalSGLangGenerationError("SGLang native rollout returned failure")
        content_ids, stop_ids, finish_reason, prompt_tokens = self._parse(raw)
        physical_tokens = len(content_ids) + len(stop_ids)
        restored = isinstance(raw, dict) and raw.get("skillev_restored_response") is True
        row = current_progress()
        if row is not None:
            meta = cast(dict[str, JsonValue], cast(dict[str, JsonValue], raw)["meta_info"])
            row.phase_metrics(
                **({} if restored else server_metrics(meta)),
                restored_response=restored,
                serving_endpoint=endpoint,
                serving_adapter_name=None if generation is None else generation.adapter_name,
                serving_adapter_revision=None
                if generation is None
                else generation.adapter_revision,
                server_action_root_stop=(
                    cast(dict[str, JsonValue], meta["finish_reason"]).get("matched")
                    == ACTION_JSON_ROOT_BOUNDARY_VERSION
                ),
            )
        if (
            request.phase is GenerationPhase.ACTION
            and request.action_boundary_version == ACTION_JSON_ROOT_BOUNDARY_VERSION
        ):
            boundary_started = time.perf_counter()
            boundary = apply_action_json_root_boundary(self.tokenizer, content_ids)
            self.physical_usage["action_boundary_cpu_microseconds"] += round(
                (time.perf_counter() - boundary_started) * 1_000_000
            )
            meta = cast(dict[str, JsonValue], cast(dict[str, JsonValue], raw)["meta_info"])
            finish = cast(dict[str, JsonValue], meta["finish_reason"])
            if not restored and finish.get("matched") == ACTION_JSON_ROOT_BOUNDARY_VERSION:
                self.physical_usage["server_action_root_stops"] += 1
            if boundary.matched:
                content_ids = boundary.content_token_ids
                stop_ids = ()
                finish_reason = ACTION_JSON_ROOT_BOUNDARY_VERSION
        if not restored:
            self.physical_usage["server_generated_tokens"] += physical_tokens
            self.physical_usage["admitted_content_tokens"] += len(content_ids)
            self.physical_usage["admitted_stop_tokens"] += len(stop_ids)
            self.physical_usage["discarded_suffix_tokens"] += (
                physical_tokens - len(content_ids) - len(stop_ids)
            )
        after = self.snapshot()
        if after != before:
            raise PolicySnapshotMismatchError("remote rollout policy changed during generation")
        return RolloutGenerationResult(
            content_token_ids=content_ids,
            stop_token_ids=stop_ids,
            finish_reason=finish_reason,
            policy_snapshot_id=after.snapshot_id,
            backend_id="sglang-native-exact-token",
            usage=BudgetVector(
                input_tokens=prompt_tokens,
                output_tokens=len(content_ids) + len(stop_ids),
                model_calls=1,
            ),
        )

    async def _request_without_early_lease_release(
        self,
        *,
        payload: dict[str, JsonValue],
        endpoint: str | None = None,
        request: RolloutGenerationRequest | None = None,
    ) -> tuple[int, JsonValue]:
        """Drain the real HTTP call before a cancellation may release its adapter lease."""

        row = current_progress()
        queued_at = time.perf_counter()
        if row is not None:
            row.stage(f"{row.phase or 'model'}-transport-queue")

        def send() -> tuple[int, JsonValue]:
            started = time.perf_counter()
            if row is not None:
                row.phase_metrics(client_transport_queue_seconds=started - queued_at)
                row.stage(f"{row.phase or 'model'}-awaiting-response")
            try:

                def dispatch() -> tuple[int, JsonValue]:
                    return self.transport.request(
                        method="POST",
                        url=endpoint or self.config.generate_url,
                        payload=payload,
                        timeout_seconds=float(self.config.request_timeout_seconds),
                        max_response_bytes=self.config.max_response_bytes,
                    )

                if self.request_journal is None:
                    return dispatch()
                if request is None or request.episode_id is None or request.turn_index is None:
                    raise ValueError("durable rollout requests need episode and turn coordinates")
                return self.request_journal.request(
                    identity=(
                        request.episode_id,
                        str(request.turn_index),
                        request.phase.value,
                        request.expected_policy_snapshot_id,
                        request.library_version or "unbound-library",
                        request.decoding_snapshot_id,
                    ),
                    endpoint=endpoint or self.config.generate_url,
                    payload=payload,
                    send=dispatch,
                )
            finally:
                if row is not None:
                    row.phase_metrics(client_response_seconds=time.perf_counter() - started)

        future: Future[tuple[int, JsonValue]] = self._executor.submit(send)
        try:
            return await _await_thread_result(future)
        except CancelledError:
            # ThreadPoolExecutor work cannot be cancelled once running.  Wait
            # for the transport to finish so the gateway's finally block does
            # not announce zero in-flight requests while SGLang is still using
            # the adapter.  The original cancellation remains the outcome.
            while not future.done():
                await sleep(0.001)
            raise

    def close(self) -> None:
        """Release idle transport threads after the owning runtime has drained."""

        self._executor.shutdown(wait=True, cancel_futures=False)

    def _parse(
        self,
        raw: JsonValue,
    ) -> tuple[tuple[int, ...], tuple[int, ...], str, int]:
        if not isinstance(raw, dict):
            raise ExternalSGLangGenerationError("SGLang native response must be an object")
        output_ids = raw.get("output_ids")
        meta = raw.get("meta_info")
        if not isinstance(output_ids, list) or not isinstance(meta, dict):
            raise ExternalSGLangGenerationError("SGLang native response fields are incomplete")
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
            raise ExternalSGLangGenerationError("SGLang native usage or finish reason is invalid")
        if completion_tokens > len(output_ids):
            raise ExternalSGLangGenerationError("SGLang completion count exceeds output IDs")
        # Pinned SGLang builds have returned a detokenizer prefix in output_ids.
        # completion_tokens remains authoritative, so retain only that suffix.
        generated = output_ids[-completion_tokens:] if completion_tokens else []
        if any(type(token_id) is not int or token_id < 0 for token_id in generated):
            raise ExternalSGLangGenerationError("SGLang output IDs are invalid")
        generated_ids = tuple(cast(list[int], generated))
        finish_type = finish.get("type")
        if not isinstance(finish_type, str) or not finish_type:
            raise ExternalSGLangGenerationError("SGLang finish type is invalid")
        matched = finish.get("matched")
        stop_ids: tuple[int, ...] = ()
        content_ids = generated_ids
        if type(matched) is int and generated_ids and generated_ids[-1] == matched:
            content_ids = generated_ids[:-1]
            stop_ids = (matched,)
        # SGLang's detokenized ``text`` is a convenience field and may differ
        # at tokenizer boundaries. Exact output IDs remain the sole identity
        # source and are decoded locally by the rollout engine.
        return content_ids, stop_ids, finish_type, prompt_tokens


async def _await_thread_result(
    future: Future[tuple[int, JsonValue]],
) -> tuple[int, JsonValue]:
    """Await executor work without depending on an event-loop callback bridge."""

    while not future.done():
        await sleep(0.001)
    return future.result()


__all__ = [
    "ExternalSGLangGenerationError",
    "ExternalSGLangRolloutConfig",
    "ExternalSGLangRolloutGenerator",
]
