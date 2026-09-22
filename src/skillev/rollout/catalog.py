"""Opt-in catalog exposure and deterministic read of a pinned skill body."""

from __future__ import annotations

import json

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.skill_exposure import CATALOG_EXPOSURE_VERSION as CATALOG_EXPOSURE_VERSION
from skillev.runtime import ActionKind, BudgetVector, FullRetrievedSkillContext, StructuredAction
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession

TWO_SKILL_GUIDANCE = (
    "During this trajectory, read at least two DIFFERENT skills from the visible catalog "
    "before submitting the final answer or completing the environment task. Choose the skills "
    "yourself, read their returned bodies, and apply relevant guidance to the task. "
    "Mentioning a skill or rereading the same ID does not satisfy the two-skill target. "
    "Skill reads count as actions and consume the existing turn and tool-call budgets; "
    "reserve enough turns for the task. If fewer than two skills are available, read the "
    "available skills and continue the task without inventing skill IDs."
)

PROACTIVE_SKILL_GUIDANCE = (
    "Use relevant skills proactively rather than defaulting to solving every task unaided. "
    "Before committing to a long solution or when you encounter difficulty, inspect the "
    "catalog and prefer reading a skill whose applicability matches the task. Choose its "
    "exact visible ID yourself, call read_skill, and apply useful methods from the returned "
    "body to your subsequent reasoning and actions. Reading only supplies advice: it does "
    "not execute a task action, verify the environment, or certify success. Do not read "
    "irrelevant skills or repeat a still-visible same-version body merely to increase calls. "
    "If no skill applies, continue solving directly. There is no required number of reads; "
    "each read consumes the existing controller-turn and tool-call budgets."
)
PROACTIVE_SKILL_TOOL_DESCRIPTION = (
    "Read a relevant skill by exact visible ID; proactively use applicable methods rather "
    "than always solving unaided. The returned body is advice, not an executed task action, "
    "environment check or success certificate. Reads consume controller turns and tool calls; "
    "there is no call quota."
)


def catalog_entry_value(skill: FullRetrievedSkillContext) -> dict[str, JsonValue]:
    """Single projection shared by rendering and read-only visibility reporting."""
    value = json.loads(skill.content)
    if not isinstance(value, dict) or any(
        key not in value for key in ("title", "summary", "applicability")
    ):
        raise ValueError("catalog requires a complete typed skill document projection")
    return {
        "skill_id": skill.metadata.skill_id,
        "version": skill.metadata.version,
        "title": value["title"],
        "summary": value["summary"],
        "applicability": value["applicability"],
    }


def render_skill_catalog_entry(skill: FullRetrievedSkillContext, *, position: int) -> str:
    return f"[{position}] " + canonical_json(catalog_entry_value(skill)) + "\n"


class CatalogReadEnvironment:
    """One pinned retrieval set; no second model or inferred skill invocation."""

    def __init__(
        self,
        environment: RolloutEnvironmentSession,
        skills: tuple[FullRetrievedSkillContext, ...],
        library_version: str,
        *,
        advisory_semantics: bool = False,
    ) -> None:
        self.advisory_semantics = advisory_semantics
        self.environment = environment
        self.skills = {skill.metadata.skill_id: skill for skill in skills}
        if len(self.skills) != len(skills) or not library_version:
            raise ValueError("catalog must have unique skills and a library identity")
        self.library_version = library_version

    @property
    def environment_id(self) -> str:
        return self.environment.environment_id

    @property
    def task_family(self) -> str:
        return self.environment.task_family

    def validate_completion(self, submission: JsonValue) -> bool:
        return self.environment.validate_completion(submission)

    async def execute(self, action: StructuredAction, *, step_index: int) -> EnvironmentObservation:
        if action.kind is not ActionKind.SKILL:
            return await self.environment.execute(action, step_index=step_index)
        if (
            action.skill_id not in self.skills
            or action.resource_id != "skill-runtime"
            or action.name != "invoke"
            or action.arguments != {}
        ):
            return EnvironmentObservation({"error": "skill_not_available"}, "schema_invalid")
        skill = self.skills[action.skill_id]
        return EnvironmentObservation(
            {
                **(
                    {
                        "read_kind": "advisory_document",
                        "environment_action_executed": False,
                        "environment_state_checked": False,
                    }
                    if self.advisory_semantics
                    else {}
                ),
                "status": "skill-read",
                "skill_id": skill.metadata.skill_id,
                "version": skill.metadata.version,
                "library_version": self.library_version,
                "content": skill.content,
            },
            "success",
            (skill.metadata.skill_id,),
            budget_usage=BudgetVector(tool_calls=1),
        )
