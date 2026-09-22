"""One serialized environment owner, with durable intent before external effects."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace

from .direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveEnvironment,
    NativePublicState,
)
from .environment_transition import EnvironmentTransitionReceipt
from .sealed_candidates import CandidateJournal, EventOrigin


class JournaledEnvironment:
    def __init__(
        self,
        delegate: NativeInteractiveEnvironment,
        journal: CandidateJournal,
        scope: tuple[str, str, str],
        *,
        owner: str,
        maximum_steps: int | None = None,
    ) -> None:
        self.delegate, self.journal, self.scope, self.owner = delegate, journal, scope, owner
        self.maximum_steps = maximum_steps
        self.revision = 0
        self._lock = asyncio.Lock()

    async def reset(self) -> str | NativePublicState:
        if self.revision:
            raise RuntimeError("an active environment cannot be silently reset")
        observation = await self.delegate.reset()
        self.revision = 1
        self.journal.record(
            self.scope,
            "environment-reset",
            asdict(observation) if isinstance(observation, NativePublicState) else observation,
            origin=EventOrigin.ENVIRONMENT,
        )
        return observation

    async def execute(
        self, actor: str, decision_id: str, action: str, revision: int
    ) -> NativeEnvironmentStep:
        if actor != self.owner:
            raise PermissionError("only the environment owner may execute side effects")
        async with self._lock:
            if revision != self.revision:
                raise ValueError("decision was made on a stale public state")
            self.journal.intent(self.scope, decision_id, action, revision)
            # If this call crashes or loses its acknowledgement, UNKNOWN persists.
            # Do not retry a possible purchase or other side effect automatically.
            result = await self.delegate.step(action)
            status = (
                "confirmed"
                if result.action_valid is True
                else "rejected"
                if result.action_valid is False
                else "unknown"
            )
            self.journal.acknowledge(
                self.scope, decision_id, status=status, observation=result.observation
            )
            self.revision += 1
            self.journal.record(
                self.scope,
                "environment-result",
                asdict(result),
                origin=EventOrigin.ENVIRONMENT,
            )
            transition = getattr(self.delegate, "last_transition", None)
            if isinstance(transition, EnvironmentTransitionReceipt):
                self.journal.record(
                    self.scope,
                    "environment-transition",
                    {
                        "decision_id": decision_id,
                        "source_revision": revision,
                        "result_revision": self.revision,
                        "category": transition.category,
                        **asdict(transition),
                    },
                    origin=EventOrigin.ENVIRONMENT,
                )
            return result

    async def step(self, action: str) -> NativeEnvironmentStep:
        return await self.execute(self.owner, f"action-{self.revision}", action, self.revision)

    async def outcome(self) -> NativeEnvironmentOutcome:
        result = await self.delegate.outcome()
        if (
            self.maximum_steps is not None
            and self.revision - 1 >= self.maximum_steps
            and not result.success
            and not result.terminal_reached
        ):
            # Count actual acknowledged native steps, not actor stop messages.
            # A terminal purchase can coincide with the limit; do not infer its
            # termination cause. Preserve any horizon flag supplied natively.
            result = replace(result, terminated_by_horizon=True)
        self.journal.record(
            self.scope,
            "native-outcome",
            asdict(result),
            origin=EventOrigin.ENVIRONMENT,
        )
        return result

    async def close(self) -> None:
        await self.delegate.close()
