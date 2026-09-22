"""Result-blind AIME decoding-profile selection on pre-final populations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class ProtocolEvidence(StrEnum):
    AUTHOR_CONFIG = "author-config"
    MODEL_CARD = "model-card"
    PRE_FINAL_CALIBRATION_ONLY = "pre-final-calibration-only"


class AIMEPopulationRole(StrEnum):
    CALIBRATION = "calibration"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class AIMEPopulationSpec:
    population_id: str
    role: AIMEPopulationRole
    maximum_problem_year: int

    def require_calibration_safe(self) -> None:
        if self.role is not AIMEPopulationRole.CALIBRATION:
            raise ValueError("AIME profile selection requires a calibration population")
        if self.maximum_problem_year >= 2026:
            raise ValueError("AIME 2026 cannot select its own evaluation profile")


@dataclass(frozen=True, slots=True)
class AIMEProfileCandidate:
    profile_id: str
    evidence: ProtocolEvidence
    evidence_source: str
    calibration_population: AIMEPopulationSpec
    prompt_profile: str
    parser_profile: str
    decoding_profile: str
    formal_reference_eligible: bool

    def __post_init__(self) -> None:
        fields = (
            self.profile_id,
            self.evidence_source,
            self.prompt_profile,
            self.parser_profile,
            self.decoding_profile,
        )
        if any(not value.strip() for value in fields):
            raise ValueError("AIME profile identity is incomplete")
        if type(self.formal_reference_eligible) is not bool:
            raise ValueError("AIME formal eligibility must be boolean")
        self.calibration_population.require_calibration_safe()


def select_formal_aime_profile(
    candidates: tuple[AIMEProfileCandidate, ...],
) -> AIMEProfileCandidate:
    formal = tuple(item for item in candidates if item.formal_reference_eligible)
    if len(formal) != 1:
        raise ValueError("AIME formal profile requires one unambiguous external protocol source")
    return formal[0]


@dataclass(frozen=True, slots=True)
class AIMECalibrationResult:
    profile_id: str
    calibration_population: str
    problem_count: int
    correct_count: int
    parse_invalid_count: int
    infrastructure_failure_count: int
    mean_reasoning_tokens: Decimal
    explicit_final_rate: Decimal

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.calibration_population.strip():
            raise ValueError("AIME calibration identity is incomplete")
        if self.problem_count <= 0:
            raise ValueError("AIME calibration population must be non-empty")
        counts = (self.correct_count, self.parse_invalid_count, self.infrastructure_failure_count)
        if any(value < 0 or value > self.problem_count for value in counts):
            raise ValueError("AIME calibration counts are invalid")

    @property
    def accuracy(self) -> Decimal:
        return Decimal(self.correct_count) / Decimal(self.problem_count)


@dataclass(frozen=True, slots=True)
class AIMEProfileSelectionPolicy:
    primary_metric: str = "accuracy"
    tie_breakers: tuple[str, ...] = (
        "lower-parse-invalid",
        "lower-mean-output-tokens",
        "lexicographic-profile-id",
    )


def select_best_pre_final_calibration_profile(
    results: tuple[AIMECalibrationResult, ...],
    policy: AIMEProfileSelectionPolicy | None = None,
) -> AIMECalibrationResult:
    if not results:
        raise ValueError("AIME selection requires declared calibration results")
    active_policy = policy or AIMEProfileSelectionPolicy()
    if active_policy.primary_metric != "accuracy":
        raise ValueError("unsupported AIME selection policy")
    if any(item.infrastructure_failure_count for item in results):
        raise ValueError("AIME calibration contains infrastructure failures")
    populations = {item.calibration_population for item in results}
    if any("2026" in item.casefold() or "final" in item.casefold() for item in populations):
        raise ValueError("final AIME population cannot select decoding")
    return min(
        results,
        key=lambda item: (
            -item.accuracy,
            item.parse_invalid_count,
            item.mean_reasoning_tokens,
            item.profile_id,
        ),
    )


select_aime_profile = select_best_pre_final_calibration_profile


__all__ = [
    "AIMECalibrationResult",
    "AIMEPopulationRole",
    "AIMEPopulationSpec",
    "AIMEProfileCandidate",
    "AIMEProfileSelectionPolicy",
    "ProtocolEvidence",
    "select_aime_profile",
    "select_best_pre_final_calibration_profile",
    "select_formal_aime_profile",
]
