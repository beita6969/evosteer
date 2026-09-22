"""Frozen, model-accessible library projection; no checkpoint filesystem access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skillev.evolution.skill_access import ACTIVE_APPLICABILITY_RULE
from skillev.runtime import SkillLibraryState


@dataclass(frozen=True, slots=True)
class FrozenSkillLibrary:
    library_id: str
    kind: str
    state: SkillLibraryState
    source_optimizer_step: int = 0
    retrieval_rule: str = ACTIVE_APPLICABILITY_RULE
    # Coordinator-only provenance. It is deliberately absent from to_value(),
    # the public actor projection, like optimizer/posterior state and paths.
    source_checkpoint: str | None = None
    actual_mutation_count: int | None = None
    committed_proposal_count: int | None = None
    posterior_update_count: int | None = None
    initial_library_version: str | None = None

    def training_provenance(self) -> dict[str, Any]:
        return {
            "source_checkpoint": self.source_checkpoint,
            "source_optimizer_step": self.source_optimizer_step,
            "actual_library_mutation_count": self.actual_mutation_count,
            "committed_proposal_count": self.committed_proposal_count,
            "posterior_update_count": self.posterior_update_count,
            "initial_library_version": self.initial_library_version,
            "current_library_version": self.state.current_version,
            "active_skill_ids": list(self.state.active_skill_ids),
            "source_kind_semantics": "checkpoint-library-not-proof-of-mutation-or-use",
        }

    def __post_init__(self) -> None:
        if not self.library_id.strip() or any(char in self.library_id for char in "/\\\n"):
            raise ValueError("skill library identity must be a public label, not a path")
        if self.kind not in {"initial", "evolved"}:
            raise ValueError("skill source must be initial or evolved")
        if type(self.source_optimizer_step) is not int or self.source_optimizer_step < 0:
            raise ValueError("skill source optimizer step must be nonnegative")
        if (self.kind == "evolved") != (self.source_optimizer_step > 0):
            raise ValueError("evolved skills require a nonzero checkpoint step")
        if self.retrieval_rule != ACTIVE_APPLICABILITY_RULE:
            raise ValueError("unsupported skill retrieval rule; do not silently replace it")

    def to_value(self) -> dict[str, Any]:
        # Only active documents cross the actor boundary. Training examples,
        # scores, detector/projection state and private paths do not cross it.
        state = self.state.to_value()
        state["documents"] = [
            self.state.documents[skill_id].to_value() for skill_id in self.state.active_skill_ids
        ]
        return {
            "library_id": self.library_id,
            "kind": self.kind,
            "source_optimizer_step": self.source_optimizer_step,
            "retrieval_rule": self.retrieval_rule,
            "state": state,
        }

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> FrozenSkillLibrary:
        return cls(
            library_id=value["library_id"],
            kind=value["kind"],
            state=SkillLibraryState.from_value(value["state"]),
            source_optimizer_step=value["source_optimizer_step"],
            retrieval_rule=value["retrieval_rule"],
        )


def initial_skill_library() -> FrozenSkillLibrary:
    from skillev.experiments._evolution_preflight_seed import planned_seed_documents

    return FrozenSkillLibrary(
        "planned-advisory-seeds@1",
        "initial",
        SkillLibraryState.from_seed_documents(planned_seed_documents()),
    )
