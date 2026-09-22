"""Read-only discovery and invocation over the production active skill library."""

from __future__ import annotations

from skillev.runtime import FullRetrievedSkillContext, SkillLibrary, SkillLibraryState
from skillev.runtime.skill_invocation import skill_invocation_observation

from .retriever import (
    TaskConditionedSkillRetriever,
    TaskRetrievalFeatures,
    applicability_mismatch_reasons,
)

ACTIVE_APPLICABILITY_RULE = "active-applicability@1"


class ReadOnlySkillAccess:
    """No calibration updates, learned ranker, optimizer, or evolution side effects.

    The production retriever currently uses document applicability only. Its
    entire inference state is the active library; posterior/detector state is
    training evidence, not a hidden inference dependency.
    """

    def __init__(self, state: SkillLibraryState) -> None:
        self._library = SkillLibrary(state)
        self._retriever = TaskConditionedSkillRetriever(library=self._library)

    def retrieve(self, features: TaskRetrievalFeatures) -> tuple[FullRetrievedSkillContext, ...]:
        return self._retriever.retrieve_features(features)

    def applicability_report(self, features: TaskRetrievalFeatures) -> list[dict[str, object]]:
        return [
            {
                "skill_id": document.manifest.skill_id,
                "mismatch_reasons": list(
                    applicability_mismatch_reasons(document.applicability, features)
                ),
                "missing_tools": sorted(
                    set(document.applicability.required_tools) - set(features.available_tools)
                ),
            }
            for document in self._library.active_documents()
        ]

    def discover(self, *, cursor: int = 0, limit: int = 8) -> dict[str, object]:
        ids = self._library.active_skill_ids
        if type(cursor) is not int or not 0 <= cursor <= len(ids):
            raise ValueError("skill cursor is outside the active catalog")
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("skill page size must be between 1 and 32")
        selected = ids[cursor : cursor + limit]
        return {
            "skills": [
                {
                    "skill_id": skill_id,
                    "title": self._library.document(skill_id).title,
                    "summary": self._library.document(skill_id).summary,
                    "applicability": self._library.document(skill_id).applicability.to_value(),
                }
                for skill_id in selected
            ],
            "next_cursor": cursor + len(selected) if cursor + len(selected) < len(ids) else None,
            "total": len(ids),
        }

    def read(self, skill_id: str) -> str:
        from skillev.runtime import model_visible_skill_content

        if skill_id not in self._library.active_skill_ids:
            raise ValueError("skill is not in the active library")
        return model_visible_skill_content(self._library.document(skill_id))

    def invoke(self, skill_id: str) -> dict[str, object]:
        self.read(skill_id)  # Invocation is of an actual active document, never just a label.
        observation = skill_invocation_observation(skill_id)
        return {
            "observation": observation.public_value,
            "invoked_skill_ids": list(observation.invoked_skill_ids),
        }
