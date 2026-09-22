"""Sticky, coordinate-assigned replicas for adapter-free readonly episodes only.

This delegates unchanged requests and journal semantics to the existing external
SGLang generator. It does not publish adapters, load-balance retries, or sample.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, cast

from skillev.diagnostics.rollout_progress import current_progress
from skillev.rollout import PolicySnapshot
from skillev.rollout.external_sglang import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
)
from skillev.rollout.generator import RolloutGenerationRequest, RolloutGenerationResult
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training.rollout_workflow import RolloutWorkflowResources


def replica_declaration(request: dict[str, Any]) -> dict[str, Any]:
    """Absent means exactly the historical single-endpoint condition."""
    if "actor_endpoints" not in request:
        if "replica_actor_requests" in request:
            raise ValueError("replica_actor_requests requires an explicit actor_endpoints pool")
        return {}
    values = request["actor_endpoints"]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
        raise ValueError("actor_endpoints must be an explicit nonempty ordered endpoint list")
    endpoints = [
        ExternalSGLangRolloutConfig(v).generate_url.removesuffix("/generate") for v in values
    ]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("readonly actor replicas must be distinct endpoints")
    admission = {}
    if "replica_actor_requests" in request:
        limit = request["replica_actor_requests"]
        if type(limit) is not int or limit < 1:
            raise ValueError("replica_actor_requests must be a positive integer")
        admission = {
            "admission": {
                "format": "readonly-replica-admission@1",
                "actor_requests_per_endpoint": limit,
                "scope": "per-collector-process",
                "cross_process_shared_limiter": False,
            }
        }
    return {
        "actor_replica_pool": {
            "format": "adapter-free-readonly-replicas@1",
            "actor_endpoints": endpoints,
            "allocation": "canonical-position-modulo@1",
            "episode_sticky": True,
            "failure_reassignment": False,
            **admission,
        }
    }


def configure_replica_admission(
    resources: RolloutWorkflowResources, declaration: dict[str, Any]
) -> None:
    """Opt-in independent actor permits; aggregate timing is not a global gate.

    These permits belong to one collector process. Concurrent arms are still
    bounded jointly by the actual SGLang scheduler, not a shared client limiter.
    External judge admission is unchanged.
    """
    pool = declaration.get("actor_replica_pool", {})
    admission = pool.get("admission")
    if admission is not None:
        for endpoint in pool["actor_endpoints"]:
            resources.configure_model_endpoint(
                endpoint, capacity=admission["actor_requests_per_endpoint"]
            )


def measured_capacity(
    server_info: object, *, include_request_limit: bool = False
) -> dict[str, Any]:
    """Actual token-arena limits, not server_args options or context_length.

    A deployed service accepted an 81k context declaration but rejected a 45k
    input because its runtime token arena was only 40k. Missing measurements
    therefore cannot establish capacity for an explicit replica pool.
    """
    data = server_info if isinstance(server_info, dict) else {}
    states = data.get("internal_states")
    workers = states if isinstance(states, list) and states else None
    result: dict[str, Any] = {"format": "sglang-measured-request-capacity@1", "sources": {}}
    fields: tuple[str, ...] = ("max_req_input_len", "max_total_num_tokens")
    if include_request_limit:
        fields += ("max_running_requests",)
    for field in fields:
        top = data.get(field)
        if type(top) is int and top >= 0:
            result[field], result["sources"][field] = top, "server_info.top-level"
            continue
        values = (
            [worker.get(field) if isinstance(worker, dict) else None for worker in workers]
            if workers is not None
            else []
        )
        complete = bool(values) and all(type(v) is int and v >= 0 for v in values)
        result[field] = min(cast(list[int], values)) if complete else None
        result["sources"][field] = "internal_states.worker-minimum" if complete else None
    return result


def require_capacity(
    capacity: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
    actor_requests: int | None = None,
) -> None:
    requirements: tuple[tuple[str, int], ...] = (
        ("max_req_input_len", input_tokens),
        ("max_total_num_tokens", input_tokens + output_tokens),
    )
    if actor_requests is not None:
        requirements += (("max_running_requests", actor_requests),)
    for field, required in requirements:
        observed = capacity.get(field)
        if type(observed) is not int or observed < required:
            raise ValueError(f"readonly replica has unmeasured or insufficient actual {field}")


class ReadonlyReplicaGenerator:
    def __init__(
        self, members: tuple[ExternalSGLangRolloutGenerator, ...], journal: DurableRequestJournal
    ) -> None:
        if not members or any(
            m.gateway is not None or m.request_journal is not journal for m in members
        ):
            raise ValueError("readonly replicas require adapter-free members and the same journal")
        self.members, self.journal = members, journal
        self.endpoints = tuple(m.config.endpoint_base.rstrip("/") for m in members)
        if len(set(self.endpoints)) != len(members):
            raise ValueError("replica endpoints must be distinct")
        self.tokenizer = members[0].tokenizer
        self._active: dict[str, int] = {}
        self.snapshot()

    def snapshot(self) -> PolicySnapshot:
        policy = self.members[0].snapshot()
        if policy.forward_adapter_version != "adapter-free" or any(
            m.snapshot() != policy or m.tokenizer.tokenizer_id != policy.tokenizer_id
            for m in self.members
        ):
            raise ValueError("readonly replica policy/tokenizer differs or is not adapter-free")
        return policy

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        if episode_id in self._active or self.snapshot().snapshot_id != expected_policy_snapshot_id:
            raise ValueError("readonly episode is already active or policy changed")
        progress = current_progress()
        position = None if progress is None else progress.snapshot().get("canonical_position")
        if type(position) is not int or position < 0:
            raise ValueError("readonly replica routing requires the original canonical position")
        selected = position % len(self.members)
        endpoint = self.endpoints[selected]
        previous = self.journal.episode_route(episode_id, expected_policy_snapshot_id)
        if previous is not None and previous != endpoint:
            raise ValueError("original episode route conflicts with this ordered replica condition")
        self.journal.save_episode_route(episode_id, expected_policy_snapshot_id, endpoint)
        self.members[selected].begin_episode(episode_id, expected_policy_snapshot_id)
        self._active[episode_id] = selected

    def end_episode(self, episode_id: str) -> None:
        member = self.members[self._active[episode_id]]
        member.end_episode(episode_id)
        del self._active[episode_id]

    def _member(self, request: RolloutGenerationRequest) -> ExternalSGLangRolloutGenerator:
        if request.episode_id is None or request.episode_id not in self._active:
            raise ValueError("readonly generation requires an active pinned episode")
        return self.members[self._active[request.episode_id]]

    def execution_endpoint(self, request: RolloutGenerationRequest) -> str:
        return self._member(request).execution_endpoint(request)

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        return await self._member(request).generate(request)

    @property
    def physical_usage(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for member in self.members:
            for key, value in member.physical_usage.items():
                values[key] = values.get(key, 0) + value
        return values

    def close(self) -> None:
        # ExitStack runs every transport close even if one close raises.
        with ExitStack() as stack:
            for member in self.members:
                stack.callback(member.close)
