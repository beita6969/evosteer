"""Deterministic environment-side evidence checks for Algorithm 1's risk gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from skillev.contracts.canonical import stable_hash
from skillev.contracts.evosteer import EvoTrajectory
from skillev.contracts.evosteer_risk import TrajectoryRiskAssessment


class EvoSteerRiskError(RuntimeError):
    """The complete batch cannot be admitted for an update or skill authoring."""


def require_assessed(trajectory: EvoTrajectory) -> TrajectoryRiskAssessment:
    risk = trajectory.risk
    if risk is None or risk.evidence_id != trajectory.risk_evidence_id:
        raise EvoSteerRiskError("a complete trajectory needs a matching trusted risk assessment")
    if not risk.accepted:
        raise EvoSteerRiskError("trajectory rejected by risk gate: " + "; ".join(risk.reasons))
    return risk


@dataclass(frozen=True, slots=True)
class ExecutionRiskPolicy:
    """A deterministic assessor explicitly bound to trusted execution capability.

    ``text_only`` verifies there were no world/tool calls. ``isolated`` requires
    an external adapter's isolation attestation; it permits changes inside that
    disposable world. This interface never claims to establish OS isolation.
    Environment adapters with richer hazards can implement their own assessor.
    Neither reward zero nor a recoverable tool/parse error is itself a risk.
    """

    executor_id: str
    environment_config_id: str
    scope: str
    capability_id: str

    def __post_init__(self) -> None:
        if self.scope not in {"text_only", "isolated"}:
            raise ValueError("an explicit text-only or isolated execution capability is required")
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.executor_id, self.environment_config_id, self.capability_id)
        ):
            raise ValueError("risk policy requires immutable executor/environment/capability IDs")

    def __call__(self, trajectory: EvoTrajectory) -> TrajectoryRiskAssessment:
        state = json.loads(trajectory.terminal_state_json)
        usage = json.loads(trajectory.usage_json)
        reasons: list[str] = []
        if trajectory.executor_id != self.executor_id:
            reasons.append("executor identity differs from risk capability")
        if trajectory.task.environment_config_id != self.environment_config_id:
            reasons.append("environment identity differs from risk capability")
        if state.get("stopped") is not True or state.get("poisoned", False) is not False:
            reasons.append("execution did not reach a healthy complete stop")
        if self.scope == "text_only" and usage.get("tool_calls", 0) != 0:
            reasons.append("text-only capability executed a tool")

        def inspect(value: Any) -> None:
            if isinstance(value, dict):
                # These are structured evaluator fields, not substring matches
                # inside public questions, model outputs, or quoted documents.
                if set(value) & {"answer_key", "ground_truth", "hidden_tests", "private_payload"}:
                    reasons.append("structured evaluator-private data reached execution state")
                if value.get("status") == "infrastructure-failure":
                    reasons.append("execution includes an unresolved infrastructure failure")
                events = value.get("risk_events", ())
                if not isinstance(events, list | tuple):
                    reasons.append("environment reported malformed execution risk evidence")
                    events = ()
                for event in events:
                    if not isinstance(event, dict) or event.get("severity") not in {
                        "informational"
                    }:
                        reasons.append("environment reported unresolved execution risk")
                if self.scope == "text_only" and "environment_step_index" in value:
                    reasons.append("text-only capability includes a world dispatch")
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(state)
        accepted = not reasons
        return TrajectoryRiskAssessment(
            assessor_id=str(
                stable_hash(
                    {
                        "format": "evosteer-execution-risk@1",
                        "executor": self.executor_id,
                        "environment": self.environment_config_id,
                        "scope": self.scope,
                        "capability": self.capability_id,
                    }
                )
            ),
            evidence_id=trajectory.risk_evidence_id,
            accepted=accepted,
            side_effect_free=accepted,
            reasons=tuple(sorted(set(reasons))),
        )
