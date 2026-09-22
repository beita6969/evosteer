"""Opt-in ScienceWorld stopping and bounded drain, not extra task-solving time."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from skillev.evaluation.interactive_termination import (
    LEGACY_TERMINATION_PROFILE,
    OWNER_FINISH_PROFILE,
    owner_finish_requested,
)
from skillev.evaluation.journaled_environment import JournaledEnvironment
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin


def termination_controls(config: dict[str, Any]) -> dict[str, object]:
    profile = config.get("scienceworld_termination_profile", LEGACY_TERMINATION_PROFILE)
    if profile not in {LEGACY_TERMINATION_PROFILE, OWNER_FINISH_PROFILE}:
        raise ValueError("unknown ScienceWorld termination condition")
    return {
        "profile": profile,
        "admission_deadline_seconds": config["episode_timeout_seconds"],
        "inflight_drain_seconds": float(config["request_timeout_seconds"]) + 30
        if profile == OWNER_FINISH_PROFILE
        else 0.0,
        "outcome_capture_timeout_seconds": 10.0 if profile == OWNER_FINISH_PROFILE else None,
    }


@dataclass
class InteractiveLifecycle:
    deadline: float
    hard_timeout: float
    journal: CandidateJournal
    scope: tuple[str, str, str]
    owner_finished: bool = False
    outcome_attempted: bool = False

    @classmethod
    def for_episode(
        cls,
        benchmark: str,
        config: dict[str, Any],
        started: float,
        journal: CandidateJournal,
        scope: tuple[str, str, str],
    ) -> InteractiveLifecycle | None:
        if benchmark != "scienceworld":
            return None
        controls = termination_controls(config)
        if controls["profile"] != OWNER_FINISH_PROFILE:
            return None
        admission = float(config["episode_timeout_seconds"])
        return cls(
            started + admission,
            admission + float(config["request_timeout_seconds"]) + 30,
            journal,
            scope,
        )

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def admit(self, operation: str) -> bool:
        if self.owner_finished:
            raise ValueError("owner finished; no later calls or actions are allowed")
        if not self.expired():
            return True
        self.journal.record(
            self.scope,
            "deadline-admission",
            {"operation": operation, "admitted": False},
            origin=EventOrigin.MODEL_TRANSPORT,
        )
        return False

    def finish(self, participant: str, response: str, call_id: str) -> None:
        if participant != "owner" or not owner_finish_requested(response) or self.owner_finished:
            raise ValueError("finish must come from the unique current owner output")
        self.owner_finished = True
        self.journal.record(
            self.scope,
            "owner-finish",
            {"call_id": call_id, "native_steps": 0},
            origin=EventOrigin.MODEL_TRANSPORT,
        )

    async def capture_before_close(self, environment: JournaledEnvironment) -> None:
        """Persist the current native state even on failure, never fabricate a candidate."""
        if self.outcome_attempted or self.journal.traces(
            self.scope, "native-outcome", origin=EventOrigin.ENVIRONMENT
        ):
            return
        self.outcome_attempted = True
        try:
            async with asyncio.timeout(10):
                await environment.outcome()
        except Exception as exc:
            self.journal.record(
                self.scope,
                "native-outcome-incomplete",
                {"failure_type": type(exc).__name__},
                origin=EventOrigin.ENVIRONMENT,
            )
