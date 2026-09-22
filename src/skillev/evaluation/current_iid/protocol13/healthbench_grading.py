"""Project-owned HealthBench rubric and grader contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    criterion: str
    points: Decimal
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RubricVerdict:
    criteria_met: bool
    explanation: str


@dataclass(frozen=True, slots=True)
class HealthBenchGraderContract:
    profile_id: str
    backend: str
    model: str
    rubric_call_mode: str
    response_schema: str
    max_tokens: int
    reasoning_effort: str | None
    maximum_attempts: int
    aggregation_profile: str

    def __post_init__(self) -> None:
        required = (
            self.profile_id,
            self.backend,
            self.model,
            self.rubric_call_mode,
            self.response_schema,
            self.aggregation_profile,
        )
        if any(not item.strip() for item in required):
            raise ValueError("HealthBench grader contract is incomplete")
        if self.max_tokens <= 0 or self.maximum_attempts <= 0:
            raise ValueError("HealthBench grader limits must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "backend": self.backend,
            "model": self.model,
            "rubric_call_mode": self.rubric_call_mode,
            "response_schema": self.response_schema,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "maximum_attempts": self.maximum_attempts,
            "aggregation_profile": self.aggregation_profile,
        }


def calculate_healthbench_score(
    rubrics: tuple[RubricCriterion, ...],
    verdicts: tuple[RubricVerdict, ...],
) -> Decimal:
    """Apply the simple-evals numerator/positive-denominator formula."""

    if len(rubrics) != len(verdicts):
        raise ValueError("HealthBench rubric/verdict length differs")
    positive = sum((item.points for item in rubrics if item.points > 0), Decimal(0))
    if positive <= 0:
        raise ValueError("HealthBench record has no positive rubric points")
    achieved = sum(
        (
            rubric.points
            for rubric, verdict in zip(rubrics, verdicts, strict=True)
            if verdict.criteria_met
        ),
        Decimal(0),
    )
    return achieved / positive


def aggregate_healthbench_scores(
    scores: tuple[Decimal, ...], *, expected_count: int = 128
) -> Decimal:
    if len(scores) != expected_count:
        raise ValueError("HealthBench score coverage differs")
    mean = sum(scores, Decimal(0)) / Decimal(expected_count)
    return min(Decimal(1), max(Decimal(0), mean))


__all__ = [
    "HealthBenchGraderContract",
    "RubricCriterion",
    "RubricVerdict",
    "aggregate_healthbench_scores",
    "calculate_healthbench_score",
]
