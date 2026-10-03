from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from skillev.contracts.canonical import stable_hash
from skillev.contracts.evosteer import EvoTrajectory
from skillev.contracts.evosteer_risk import TrajectoryRiskAssessment


class EvoSteerRiskError(RuntimeError):
    pass


def require_assessed(trajectory: EvoTrajectory) -> TrajectoryRiskAssessment:
    risk = trajectory.risk
    if risk is None or risk.evidence_id != trajectory.risk_evidence_id:
        raise EvoSteerRiskError("a complete trajectory needs a matching trusted risk assessment")
    if not risk.accepted:
        raise EvoSteerRiskError("trajectory rejected by risk gate: " + "; ".join(risk.reasons))
    return risk


@dataclass(frozen=True, slots=True)
class ExecutionRiskPolicy:
    executor_id: str
    environment_config_id: str
    scope: str
    capability_id: str
    tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in {"text_only", "isolated", "sandboxed_tools"}:
            raise ValueError("an explicit text-only or isolated execution capability is required")
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.executor_id, self.environment_config_id, self.capability_id)
        ):
            raise ValueError("risk policy requires immutable executor/environment/capability IDs")
        if (
            not isinstance(self.tools, tuple)
            or any(type(tool) is not str or not tool.strip() for tool in self.tools)
            or list(self.tools) != sorted(set(self.tools))
        ):
            raise ValueError("risk policy tools must be a sorted tuple of distinct names")
        if (self.scope == "sandboxed_tools") != bool(self.tools):
            raise ValueError("exactly the sandboxed_tools capability lists its tools")

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
        if self.scope == "sandboxed_tools":
            self._check_tool_use(state, usage, reasons)

        def inspect(value: Any) -> None:
            if isinstance(value, dict):
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
                if self.scope in {"text_only", "sandboxed_tools"} and (
                    "environment_step_index" in value
                ):
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
                        **({"tools": list(self.tools)} if self.tools else {}),
                    }
                )
            ),
            evidence_id=trajectory.risk_evidence_id,
            accepted=accepted,
            side_effect_free=accepted,
            reasons=tuple(sorted(set(reasons))),
        )

    def _check_tool_use(self, state: Any, usage: Any, reasons: list[str]) -> None:
        history = state.get("history", []) if isinstance(state, dict) else []
        reported = 0
        for event in history if isinstance(history, list) else ():
            observation = event.get("observation") if isinstance(event, dict) else None
            result = observation.get("node_result") if isinstance(observation, dict) else None
            metadata = result.get("metadata") if isinstance(result, dict) else None
            if not isinstance(metadata, dict):
                continue
            tool_use = metadata.get("tool_use")
            if tool_use is not None:
                calls = tool_use.get("calls") if isinstance(tool_use, dict) else None
                if type(calls) is not int or calls < 0:
                    reasons.append("a node reported malformed tool use")
                else:
                    reported += calls
            transcript = metadata.get("tool_transcript", [])
            for entry in transcript if isinstance(transcript, list) else ():
                if not isinstance(entry, dict) or entry.get("executed") is not True:
                    continue
                if entry.get("tool") not in self.tools:
                    reasons.append("a node ran a tool outside the sandboxed capability")
                sandbox = entry.get("sandbox")
                if isinstance(sandbox, dict) and sandbox.get("contained") is False:
                    reasons.append("a sandboxed tool process outlived its call")
        if not isinstance(usage, dict) or usage.get("tool_calls", 0) != reported:
            reasons.append("booked tool calls differ from the calls nodes report")
