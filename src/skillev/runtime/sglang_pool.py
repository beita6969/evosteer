"""All-replica adapter publication and stable same-version episode routing.

The SGLang HTTP/control implementation stays in SGLangGateway. This coordinator
adds the training transaction boundary that a generic retrying router lacks.
No generation fallback, health-based resampling, or implicit request retry.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from .sglang_gateway import (
    AdapterGeneration,
    PreparedAdapterSwap,
    SGLangGateway,
    SGLangGatewayError,
    SGLangRole,
)


class SGLangActorPool(SGLangGateway):
    def __init__(self, members: Sequence[SGLangGateway]) -> None:
        if not members:
            raise ValueError("actor pool cannot be empty")
        super().__init__(members[0].config)
        self.members = tuple(members)
        for member in self.members:
            if (member.config.base_model, member.config.supervisor_adapter) != (
                self.config.base_model,
                self.config.supervisor_adapter,
            ):
                raise ValueError("actor replicas require the same model and adapter namespace")
        if len({v.config.api_root for v in self.members}) != len(self.members):
            raise ValueError("actor endpoints must be distinct")
        self._pool_lock = threading.RLock()
        self._sessions: dict[str, SGLangGateway] = {}
        self._remaining_work: dict[str, int] = {}
        self._pool_swaps: tuple[PreparedAdapterSwap, ...] = ()
        self._available = False
        self._publishing = False
        self._policy_snapshot_id: str | None = None
        self._previous_policy_snapshot_id: str | None = None

    @property
    def adapter_generation(self) -> AdapterGeneration:
        return self.members[0].adapter_generation

    def _require_consistent(self) -> None:
        revisions = {
            (v.adapter_generation.adapter_name, v.adapter_generation.adapter_revision)
            for v in self.members
        }
        if len(revisions) != 1 or self.adapter_generation.adapter_revision == "not-loaded":
            raise SGLangGatewayError("actor replicas do not have one published policy")

    def bind_policy_snapshot(self, policy_snapshot_id: str) -> None:
        with self._pool_lock:
            if self._publishing or self._sessions or not self._available:
                raise SGLangGatewayError("policy identity requires a fully published idle pool")
            self._require_consistent()
            self._policy_snapshot_id = policy_snapshot_id

    def acquire_episode(
        self,
        episode_id: str,
        expected_policy_snapshot_id: str | None = None,
        *,
        endpoint: str | None = None,
        estimated_work: int = 1,
    ) -> SGLangGateway:
        if type(estimated_work) is not int or estimated_work < 1:
            raise ValueError("episode work estimate must be positive")
        with self._pool_lock:
            if not self._available or self._publishing:
                raise SGLangGatewayError("actor publication is incomplete")
            self._require_consistent()
            if (
                expected_policy_snapshot_id is not None
                and expected_policy_snapshot_id != self._policy_snapshot_id
            ):
                raise SGLangGatewayError("episode policy differs from the published actor pool")
            if episode_id in self._sessions:
                raise ValueError("episode already has an actor lease")
            # Initial placement only. Subsequent reasoning/action requests retain
            # this replica and its prefix cache, regardless of completion order.
            loads = [
                sum(self._remaining_work[k] for k, v in self._sessions.items() if v is m)
                for m in self.members
            ]
            member = self.members[min(range(len(loads)), key=lambda i: (loads[i], i))]
            if endpoint is not None:
                matches = [m for m in self.members if m.config.api_root == endpoint]
                if len(matches) != 1:
                    raise SGLangGatewayError("original episode replica is not in the ready pool")
                member = matches[0]
            member.begin_supervisor_rollout()
            self._sessions[episode_id] = member
            self._remaining_work[episode_id] = estimated_work
            return member

    def update_episode_work(self, episode_id: str, remaining_work: int) -> None:
        """Answer-free scheduling estimate; never migrate an active session."""
        if type(remaining_work) is not int or remaining_work < 1:
            raise ValueError("remaining work must be positive")
        with self._pool_lock:
            if episode_id not in self._sessions:
                raise ValueError("episode has no serving lease")
            self._remaining_work[episode_id] = remaining_work

    def release_episode(self, episode_id: str) -> None:
        with self._pool_lock:
            member = self._sessions.pop(episode_id)
            del self._remaining_work[episode_id]
            member.end_supervisor_rollout()

    def _begin_publication(self) -> None:
        with self._pool_lock:
            if self._publishing or self._sessions:
                raise SGLangGatewayError("cannot publish across an active episode or transaction")
            self._publishing = True
            self._available = False
            self._previous_policy_snapshot_id = self._policy_snapshot_id
            self._policy_snapshot_id = None

    def prepare_supervisor_adapter(
        self, *, adapter_path: str, adapter_revision: str
    ) -> PreparedAdapterSwap:
        self._begin_publication()
        swaps = []
        try:
            for member in self.members:
                swaps.append(
                    member.prepare_supervisor_adapter(
                        adapter_path=adapter_path, adapter_revision=adapter_revision
                    )
                )
        except Exception:
            # A failed rollback intentionally leaves the pool unavailable. A
            # recovered controller must restore every member before sampling.
            for member, swap in reversed(list(zip(self.members, swaps, strict=False))):
                member.rollback_supervisor_adapter(swap)
            self._publishing = False
            raise
        self._pool_swaps = tuple(swaps)
        return swaps[0]

    def _require_pool_swap(self, prepared: PreparedAdapterSwap) -> None:
        if not self._pool_swaps or prepared is not self._pool_swaps[0]:
            raise ValueError("publication does not belong to this pool transaction")

    def commit_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> AdapterGeneration:
        self._require_pool_swap(prepared)
        # Keep the OUTER barrier closed even after an individual member's commit.
        # A partial failure cannot serve the candidate; recovery restores all.
        for member, swap in zip(self.members, self._pool_swaps, strict=True):
            member.commit_supervisor_adapter(swap)
        self._require_consistent()
        self._pool_swaps = ()
        self._publishing = False
        self._available = True
        return self.adapter_generation

    def rollback_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> None:
        self._require_pool_swap(prepared)
        for member, swap in reversed(list(zip(self.members, self._pool_swaps, strict=True))):
            member.rollback_supervisor_adapter(swap)
        self._pool_swaps = ()
        self._publishing = False
        self._policy_snapshot_id = self._previous_policy_snapshot_id
        # Initial rollback has no usable prior policy.
        self._available = self.adapter_generation.adapter_revision != "not-loaded"
        if self._available:
            self._require_consistent()

    def restore_supervisor_adapter(
        self, *, adapter_path: str, adapter_revision: str
    ) -> AdapterGeneration:
        self._begin_publication()
        for member in self.members:
            member.restore_supervisor_adapter(
                adapter_path=adapter_path, adapter_revision=adapter_revision
            )
        self._require_consistent()
        self._publishing = False
        self._available = True
        return self.adapter_generation

    def bind_existing_supervisor_adapter(self, *, adapter_revision: str) -> AdapterGeneration:
        self._begin_publication()
        for member in self.members:
            member.bind_existing_supervisor_adapter(adapter_revision=adapter_revision)
        self._require_consistent()
        self._publishing = False
        self._available = True
        return self.adapter_generation

    async def health(self) -> tuple[str, ...]:
        models = [await member.health() for member in self.members]
        return tuple(sorted(set.intersection(*(set(v) for v in models))))

    def _begin_request(self, role: SGLangRole) -> None:
        if role is SGLangRole.SUPERVISOR:
            raise SGLangGatewayError("actor-pool requests require an episode lease")
        super()._begin_request(role)

    def begin_supervisor_rollout(self) -> AdapterGeneration:
        raise SGLangGatewayError("actor-pool requests require an episode lease")
