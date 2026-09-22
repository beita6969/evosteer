"""Runtime-facing task and submission types.

This module deliberately contains no task generator, seed, policy, protected
truth, or evaluator.  Rollout workers may import it without making any hidden
evaluation state reachable from the model/tool sandbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

Vector2 = tuple[float, float]
Matrix2x2 = tuple[Vector2, Vector2]
Matrix2x1 = tuple[tuple[float], tuple[float]]


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector2(value: Vector2, label: str) -> None:
    if len(value) != 2:
        raise ValueError(f"{label} must contain two coordinates")
    for coordinate in value:
        _finite(coordinate, label)


def _matrix2x2(value: Matrix2x2, label: str) -> None:
    if len(value) != 2:
        raise ValueError(f"{label} must contain two rows")
    for row in value:
        _vector2(row, label)


def _matrix2x1(value: Matrix2x1, label: str) -> None:
    if len(value) != 2 or any(len(row) != 1 for row in value):
        raise ValueError(f"{label} must have shape 2x1")
    for row in value:
        _finite(row[0], label)


class SystemIdentificationFamily(StrEnum):
    """Public mechanism label for the six reusable system-ID families."""

    WELL_EXCITED_NOISE_FREE = "WELL_EXCITED_NOISE_FREE"
    PROCESS_NOISE = "PROCESS_NOISE"
    MEASUREMENT_NOISE = "MEASUREMENT_NOISE"
    MISSING_OBSERVATIONS = "MISSING_OBSERVATIONS"
    PIECEWISE_DRIFT = "PIECEWISE_DRIFT"
    INSUFFICIENT_EXCITATION = "INSUFFICIENT_EXCITATION"


class IdentificationAction(StrEnum):
    ESTIMATE = "ESTIMATE"
    WITHHOLD = "WITHHOLD"


class IdentificationEstimator(StrEnum):
    ORDINARY_LEAST_SQUARES = "ORDINARY_LEAST_SQUARES"
    COMPLETE_CASE_OLS = "COMPLETE_CASE_OLS"
    PIECEWISE_OLS = "PIECEWISE_OLS"
    VALIDATION_FUSED_STATE_SPACE = "VALIDATION_FUSED_STATE_SPACE"
    WITHHOLD = "WITHHOLD"


@dataclass(frozen=True, slots=True)
class PublicValidationTransition:
    """A released, independent measurement of one state transition."""

    step: int
    current_state: Vector2
    next_state: Vector2

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("validation step must be non-negative")
        _vector2(self.current_state, "validation current state")
        _vector2(self.next_state, "validation next state")

    def to_public_value(self) -> dict[str, object]:
        return {
            "current_state": list(self.current_state),
            "next_state": list(self.next_state),
            "step": self.step,
        }


@dataclass(frozen=True, slots=True)
class PublicSystemIdentificationObservations:
    """Public trajectory projection; missing coordinates are represented by a mask."""

    states: tuple[Vector2, ...]
    observed_mask: tuple[tuple[bool, bool], ...]
    inputs: tuple[float, ...]
    time_step: float
    declared_change_point: int | None = None
    measurement_noise_variance: float | None = None
    validation_noise_variance: float | None = None
    validation_transitions: tuple[PublicValidationTransition, ...] = ()

    def __post_init__(self) -> None:
        if len(self.states) != len(self.inputs) + 1:
            raise ValueError("states must contain one more row than inputs")
        if len(self.observed_mask) != len(self.states):
            raise ValueError("observation mask must align with states")
        if not self.inputs:
            raise ValueError("system-identification observations cannot be empty")
        for state in self.states:
            _vector2(state, "observed state")
        for mask in self.observed_mask:
            if len(mask) != 2 or any(type(item) is not bool for item in mask):
                raise TypeError("each observation mask row must contain two booleans")
        for item in self.inputs:
            _finite(item, "input")
        if _finite(self.time_step, "time step") <= 0:
            raise ValueError("time step must be positive")
        if self.declared_change_point is not None and not (
            0 < self.declared_change_point < len(self.inputs)
        ):
            raise ValueError("declared change point must lie inside the trajectory")
        for value, label in (
            (self.measurement_noise_variance, "measurement noise variance"),
            (self.validation_noise_variance, "validation noise variance"),
        ):
            if value is not None and _finite(value, label) <= 0:
                raise ValueError(f"{label} must be positive")
        if self.validation_transitions and (
            self.measurement_noise_variance is None or self.validation_noise_variance is None
        ):
            raise ValueError("validation transitions require both released noise variances")
        if any(item.step >= len(self.inputs) for item in self.validation_transitions):
            raise ValueError("validation transition lies outside the public trajectory")

    @property
    def horizon(self) -> int:
        return len(self.inputs)

    def to_public_value(self) -> dict[str, object]:
        return {
            "declared_change_point": self.declared_change_point,
            "inputs": list(self.inputs),
            "measurement_noise_variance": self.measurement_noise_variance,
            "observed_mask": [list(row) for row in self.observed_mask],
            "states": [list(row) for row in self.states],
            "time_step": self.time_step,
            "validation_noise_variance": self.validation_noise_variance,
            "validation_transitions": [
                item.to_public_value() for item in self.validation_transitions
            ],
        }


@dataclass(frozen=True, slots=True)
class PublicSystemIdentificationTask:
    task_id: str
    family: SystemIdentificationFamily
    observations: PublicSystemIdentificationObservations
    public_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.isascii():
            raise ValueError("task identity must be non-empty ASCII")
        if not self.public_conditions or any(not item for item in self.public_conditions):
            raise ValueError("a task must declare its public conditions")

    def to_public_value(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "observations": self.observations.to_public_value(),
            "public_conditions": list(self.public_conditions),
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class SystemIdentificationSegmentEstimate:
    start_step: int
    end_step: int
    state_matrix: Matrix2x2
    input_matrix: Matrix2x1

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("estimate segment bounds are invalid")
        _matrix2x2(self.state_matrix, "estimated state matrix")
        _matrix2x1(self.input_matrix, "estimated input matrix")


@dataclass(frozen=True, slots=True)
class SystemIdentificationSubmission:
    """Structured candidate output; executable source is intentionally unsupported."""

    action: IdentificationAction
    estimator: IdentificationEstimator
    segments: tuple[SystemIdentificationSegmentEstimate, ...] = ()
    withhold_reason: str | None = None

    def __post_init__(self) -> None:
        if self.action is IdentificationAction.ESTIMATE:
            if not self.segments or self.estimator is IdentificationEstimator.WITHHOLD:
                raise ValueError("ESTIMATE requires at least one numerical segment")
            if self.withhold_reason is not None:
                raise ValueError("ESTIMATE cannot include a withholding reason")
            ordered = tuple(sorted(self.segments, key=lambda item: item.start_step))
            if ordered != self.segments or ordered[0].start_step != 0:
                raise ValueError("estimate segments must be ordered from step zero")
            if any(left.end_step != right.start_step for left, right in pairwise(ordered)):
                raise ValueError("estimate segments must be contiguous")
        else:
            if (
                self.estimator is not IdentificationEstimator.WITHHOLD
                or self.segments
                or not self.withhold_reason
            ):
                raise ValueError("WITHHOLD requires only a non-empty reason")


class SocialTaskClass(StrEnum):
    RANDOMIZED = "RANDOMIZED"
    OBSERVED_CONFOUNDER = "OBSERVED_CONFOUNDER"
    MAR_OUTCOME = "MAR_OUTCOME"
    KNOWN_SELECTION_PROBABILITY = "KNOWN_SELECTION_PROBABILITY"
    VALIDATION_MEASUREMENT = "VALIDATION_MEASUREMENT"
    UNOBSERVED_COMMON_CAUSE = "UNOBSERVED_COMMON_CAUSE"


@dataclass(frozen=True, slots=True)
class PublicSocialRow:
    row_id: str
    environment: str
    treatment: int
    observed_outcome: float | None
    selected: bool
    observed_confounder: float | None = None
    missingness_predictor: float | None = None
    sampling_probability: float | None = None
    validation_member: bool | None = None
    validation_outcome: float | None = None

    def __post_init__(self) -> None:
        if not self.row_id or not self.environment or self.treatment not in {0, 1}:
            raise ValueError("social row identity or treatment is invalid")
        for value, label in (
            (self.observed_outcome, "observed outcome"),
            (self.observed_confounder, "observed confounder"),
            (self.missingness_predictor, "missingness predictor"),
            (self.sampling_probability, "sampling probability"),
            (self.validation_outcome, "validation outcome"),
        ):
            if value is not None:
                _finite(value, label)
        if self.sampling_probability is not None and not 0 < self.sampling_probability <= 1:
            raise ValueError("sampling probability must lie in (0, 1]")
        if self.validation_member not in {None, False, True}:
            raise TypeError("validation membership must be boolean or absent")
        if self.validation_outcome is not None and self.validation_member is not True:
            raise ValueError("validation outcome requires validation membership")

    def to_public_value(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "missingness_predictor": self.missingness_predictor,
            "observed_confounder": self.observed_confounder,
            "observed_outcome": self.observed_outcome,
            "row_id": self.row_id,
            "sampling_probability": self.sampling_probability,
            "selected": self.selected,
            "treatment": self.treatment,
            "validation_member": self.validation_member,
            "validation_outcome": self.validation_outcome,
        }


@dataclass(frozen=True, slots=True)
class PublicSocialTask:
    task_id: str
    task_class: SocialTaskClass
    rows: tuple[PublicSocialRow, ...]
    estimand: str = "population_average_treatment_effect"

    def __post_init__(self) -> None:
        if not self.task_id or len(self.rows) < 2:
            raise ValueError("social task requires an identity and at least two rows")
        if len({row.row_id for row in self.rows}) != len(self.rows):
            raise ValueError("social row identities must be unique")
        if not self.estimand:
            raise ValueError("estimand must be declared")

    def to_public_value(self) -> dict[str, object]:
        return {
            "estimand": self.estimand,
            "rows": [row.to_public_value() for row in self.rows],
            "task_class": self.task_class.value,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class SocialEffectSubmission:
    action: IdentificationAction
    effect_estimate: float | None = None
    withhold_reason: str | None = None

    def __post_init__(self) -> None:
        if self.action is IdentificationAction.ESTIMATE:
            if self.effect_estimate is None or self.withhold_reason is not None:
                raise ValueError("ESTIMATE requires one effect estimate")
            _finite(self.effect_estimate, "effect estimate")
        elif self.effect_estimate is not None or not self.withhold_reason:
            raise ValueError("WITHHOLD requires a reason and no numerical estimate")


__all__ = [
    "IdentificationAction",
    "IdentificationEstimator",
    "Matrix2x1",
    "Matrix2x2",
    "PublicSocialRow",
    "PublicSocialTask",
    "PublicSystemIdentificationObservations",
    "PublicSystemIdentificationTask",
    "PublicValidationTransition",
    "SocialEffectSubmission",
    "SocialTaskClass",
    "SystemIdentificationFamily",
    "SystemIdentificationSegmentEstimate",
    "SystemIdentificationSubmission",
    "Vector2",
]
