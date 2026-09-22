"""Opt-in method extension, separate from idea.tex's joint phase criterion."""

from dataclasses import dataclass

from skillev.contracts import JsonValue

COLD_START_VERSION = "zero-coverage-generate@1"


@dataclass(frozen=True, slots=True)
class ColdStartConfig:
    """Generate only, after the unchanged two-window residual plateau.

    A scope must have no executed skill invocation in that complete window.
    Support counts distinct trusted source questions, not repeated rollouts.
    The normal Phi always has priority; global cycle budgets still apply.
    """

    version: str = COLD_START_VERSION
    min_source_questions: int = 2
    min_batches: int = 2
    max_new_skills: int = 1
    max_evidence_edges: int = 8

    def __post_init__(self) -> None:
        if self.version != COLD_START_VERSION:
            raise ValueError("unsupported cold-start method extension")
        for value in (
            self.min_source_questions,
            self.min_batches,
            self.max_new_skills,
            self.max_evidence_edges,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("cold-start support and capacity must be positive integers")

        if self.max_evidence_edges < self.min_source_questions + self.min_batches:
            raise ValueError(
                "cold-start witness capacity must fit its declared source/batch support"
            )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "min_source_questions": self.min_source_questions,
            "min_batches": self.min_batches,
            "max_new_skills": self.max_new_skills,
            "max_evidence_edges": self.max_evidence_edges,
        }

    @classmethod
    def from_value(cls, value: object) -> "ColdStartConfig":
        if not isinstance(value, dict) or set(value) != {
            "version",
            "min_source_questions",
            "min_batches",
            "max_new_skills",
            "max_evidence_edges",
        }:
            raise ValueError("cold-start configuration is incomplete")
        return cls(**value)
