"""Read-only CGF/Jensen explanation of observed invocation flow.

No Z term, clipping, sentinel ability, EMA, trigger, posterior update or mutation.
One value per invoking trajectory is the logsumexp of its matching log-flow
edges; uninvoked trajectories are missing, not low-ability observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import BatchDiagnostics

from skillev.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class SkillFlowDiagnosticView:
    skill_id: str
    invoked_trajectory_count: int
    g_at_one: float | None
    lambda_at_zero: float | None
    jensen_gap: float | None
    lambda_at_one: float | None = None
    diagnostic_only: bool = True

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": "observed-skill-flow-cgf@1",
            "diagnostic_only": True,
            "skill_id": self.skill_id,
            "invoked_trajectory_count": self.invoked_trajectory_count,
            "G_1": self.g_at_one,
            "lambda_0": self.lambda_at_zero,
            "lambda_1": self.lambda_at_one,
            "jensen_gap": self.jensen_gap,
            "missing_reason": "no-observed-invocation" if self.g_at_one is None else None,
        }


def skill_flow_diagnostic(
    skill_id: str, trajectory_log_flows: tuple[float, ...]
) -> SkillFlowDiagnosticView:
    if not skill_id.strip() or not all(math.isfinite(value) for value in trajectory_log_flows):
        raise ValueError("diagnostic requires a skill identity and finite observed log flows")
    if not trajectory_log_flows:
        return SkillFlowDiagnosticView(skill_id, 0, None, None, None)
    maximum = max(trajectory_log_flows)
    mean = math.fsum(trajectory_log_flows) / len(trajectory_log_flows)
    g = (
        maximum
        + math.log(math.fsum(math.exp(value - maximum) for value in trajectory_log_flows))
        - math.log(len(trajectory_log_flows))
    )
    weights = tuple(math.exp(value - maximum) for value in trajectory_log_flows)
    tilted = math.fsum(
        value * weight for value, weight in zip(trajectory_log_flows, weights, strict=True)
    ) / math.fsum(weights)
    return SkillFlowDiagnosticView(skill_id, len(trajectory_log_flows), g, mean, g - mean, tilted)


def skillflow_diagnostic_view(batch: BatchDiagnostics) -> tuple[SkillFlowDiagnosticView, ...]:
    """Observe one complete frozen batch without writing to its method state."""
    result = []
    for skill in batch.skill_flows:
        observed = []
        for trajectory in batch.trajectories:
            values = tuple(
                edge.sample_log_state_weight
                for edge in trajectory.edges
                if skill.skill_id in edge.invoked_skill_ids
            )
            if values:
                maximum = max(values)
                observed.append(
                    maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
                )
        result.append(skill_flow_diagnostic(skill.skill_id, tuple(observed)))
    return tuple(result)
