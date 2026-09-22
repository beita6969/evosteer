"""The text-skill invocation acknowledgement used by rollout environments."""

from .contracts import BudgetVector
from .execution import EnvironmentObservation


def skill_invocation_observation(skill_id: str) -> EnvironmentObservation:
    """Record explicit adoption of a text skill, not execution of a hidden solver."""
    return EnvironmentObservation(
        public_value={"status": "skill-invoked"},
        observation_status="success",
        invoked_skill_ids=(skill_id,),
        budget_usage=BudgetVector(tool_calls=1),
    )
