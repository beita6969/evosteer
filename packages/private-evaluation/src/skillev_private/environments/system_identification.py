"""Private builder for the six-family system-identification environment.

Only the public projection returned in ``BuiltSystemIdentificationTask.public``
may cross into a rollout worker.  The policy, seed, plant matrices, latent
states, and conditional targets remain evaluator-private.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skillev.environments.public import (
    IdentificationAction,
    IdentificationEstimator,
    Matrix2x1,
    Matrix2x2,
    PublicSystemIdentificationObservations,
    PublicSystemIdentificationTask,
    PublicValidationTransition,
    SystemIdentificationFamily,
    SystemIdentificationSegmentEstimate,
    SystemIdentificationSubmission,
    Vector2,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PrivateSystemIdentificationPolicy:
    """One formal seed and the non-secret mechanism parameters for a task build."""

    seed: int
    namespace: str
    horizon: int = 96
    time_step: float = 1.0
    process_noise_standard_deviation: float = 0.025
    measurement_noise_standard_deviation: float = 0.08
    validation_noise_standard_deviation: float = 0.015
    missing_fraction: float = 0.15
    validation_fraction: float = 0.20
    change_fraction: float = 0.50

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("private task seed must be an integer")
        if not self.namespace:
            raise ValueError("private task namespace is required")
        if self.horizon < 24:
            raise ValueError("system-identification horizon is too short")
        positive = (
            self.time_step,
            self.process_noise_standard_deviation,
            self.measurement_noise_standard_deviation,
            self.validation_noise_standard_deviation,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("system-identification scales must be finite and positive")
        fractions = (self.missing_fraction, self.validation_fraction, self.change_fraction)
        if not all(math.isfinite(value) and 0 < value < 1 for value in fractions):
            raise ValueError("system-identification fractions must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class PrivateSystemIdentificationSegment:
    start_step: int
    end_step: int
    state_matrix: Matrix2x2
    input_matrix: Matrix2x1


@dataclass(frozen=True, slots=True)
class PrivateSystemIdentificationTruth:
    task_id: str
    family: SystemIdentificationFamily
    required_action: IdentificationAction
    required_estimator: IdentificationEstimator
    required_withhold_reason: str | None
    segments: tuple[PrivateSystemIdentificationSegment, ...]
    latent_states: tuple[Vector2, ...]
    conditional_targets: tuple[Vector2, ...]
    parameter_absolute_bound: float = 2.0


@dataclass(frozen=True, slots=True)
class BuiltSystemIdentificationTask:
    public: PublicSystemIdentificationTask
    truth: PrivateSystemIdentificationTruth


def frozen_persistent_excitation(horizon: int) -> FloatArray:
    """Deterministic, bounded multi-frequency input used by every estimable family."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    index = np.arange(horizon, dtype=np.float64)
    values = (
        0.5 * np.sin(2.0 * math.pi * (index + 1.0) / 17.0)
        + 0.3 * np.cos(2.0 * math.pi * (index + 1.0) / 29.0)
        + 0.2 * np.sin(2.0 * math.pi * (index + 1.0) / 7.0)
    )
    result = values.astype(np.float64, copy=False)
    result.setflags(write=False)
    return result


def _private_rng(
    policy: PrivateSystemIdentificationPolicy,
    family: SystemIdentificationFamily,
    ordinal: int,
) -> np.random.Generator:
    if ordinal < 0:
        raise ValueError("task ordinal must be non-negative")
    material = f"{policy.namespace}\0{policy.seed}\0{family.value}\0{ordinal}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(seed)


def _task_id(
    policy: PrivateSystemIdentificationPolicy,
    family: SystemIdentificationFamily,
    ordinal: int,
) -> str:
    material = f"{policy.namespace}\0{policy.seed}\0{family.value}\0{ordinal}".encode()
    suffix = hashlib.sha256(material).hexdigest()[:20]
    return f"system-id-{family.value.lower().replace('_', '-')}-{suffix}"


def _stable_matrix(matrix: FloatArray, target_radius: float) -> FloatArray:
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("generated state matrix has invalid spectral radius")
    return matrix * (target_radius / radius)


def _plant(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    raw = np.array(
        [
            [rng.uniform(0.55, 0.85), rng.uniform(-0.18, 0.18)],
            [rng.uniform(-0.18, 0.18), rng.uniform(0.45, 0.78)],
        ],
        dtype=np.float64,
    )
    state = _stable_matrix(raw, rng.uniform(0.72, 0.90))
    inputs = rng.uniform(-0.65, 0.65, size=(2, 1)).astype(np.float64)
    if float(np.linalg.norm(inputs)) < 0.25:
        inputs[0, 0] += 0.35
    return state, inputs


def _matrix2x2(value: FloatArray) -> Matrix2x2:
    return (
        (float(value[0, 0]), float(value[0, 1])),
        (float(value[1, 0]), float(value[1, 1])),
    )


def _matrix2x1(value: FloatArray) -> Matrix2x1:
    return ((float(value[0, 0]),), (float(value[1, 0]),))


def _vector2(value: FloatArray) -> Vector2:
    return (float(value[0]), float(value[1]))


def _family_conditions(family: SystemIdentificationFamily) -> tuple[str, ...]:
    common = ("discrete-time", "state-dimension-2", "input-dimension-1")
    additions = {
        SystemIdentificationFamily.WELL_EXCITED_NOISE_FREE: ("well-excited", "noise-free"),
        SystemIdentificationFamily.PROCESS_NOISE: ("well-excited", "process-noise"),
        SystemIdentificationFamily.MEASUREMENT_NOISE: (
            "measurement-noise",
            "independent-validation-transitions",
            "released-noise-variances",
        ),
        SystemIdentificationFamily.MISSING_OBSERVATIONS: (
            "well-excited",
            "missing-observations",
        ),
        SystemIdentificationFamily.PIECEWISE_DRIFT: (
            "well-excited",
            "declared-change-point",
        ),
        SystemIdentificationFamily.INSUFFICIENT_EXCITATION: (
            "insufficient-excitation",
            "withholding-permitted",
        ),
    }
    return common + additions[family]


def _required_estimator(family: SystemIdentificationFamily) -> IdentificationEstimator:
    return {
        SystemIdentificationFamily.WELL_EXCITED_NOISE_FREE: (
            IdentificationEstimator.ORDINARY_LEAST_SQUARES
        ),
        SystemIdentificationFamily.PROCESS_NOISE: (IdentificationEstimator.ORDINARY_LEAST_SQUARES),
        SystemIdentificationFamily.MEASUREMENT_NOISE: (
            IdentificationEstimator.VALIDATION_FUSED_STATE_SPACE
        ),
        SystemIdentificationFamily.MISSING_OBSERVATIONS: (
            IdentificationEstimator.COMPLETE_CASE_OLS
        ),
        SystemIdentificationFamily.PIECEWISE_DRIFT: IdentificationEstimator.PIECEWISE_OLS,
        SystemIdentificationFamily.INSUFFICIENT_EXCITATION: IdentificationEstimator.WITHHOLD,
    }[family]


def build_system_identification_task(
    policy: PrivateSystemIdentificationPolicy,
    family: SystemIdentificationFamily,
    *,
    ordinal: int = 0,
) -> BuiltSystemIdentificationTask:
    """Build one task while retaining every reconstructive value in this module."""

    rng = _private_rng(policy, family, ordinal)
    task_id = _task_id(policy, family, ordinal)
    state, inputs_matrix = _plant(rng)
    state_matrices = [state]
    input_matrices = [inputs_matrix]
    change_point: int | None = None
    if family is SystemIdentificationFamily.PIECEWISE_DRIFT:
        change_point = round(policy.horizon * policy.change_fraction)
        drifted = state + np.array([[0.025, -0.035], [0.02, -0.02]], dtype=np.float64)
        state_matrices.append(_stable_matrix(drifted, min(0.94, _spectral_radius(state) + 0.03)))
        input_matrices.append(inputs_matrix * 1.08)

    if family is SystemIdentificationFamily.INSUFFICIENT_EXCITATION:
        inputs = np.zeros(policy.horizon, dtype=np.float64)
        initial = np.array([0.25, -0.15], dtype=np.float64)
    else:
        inputs = np.asarray(frozen_persistent_excitation(policy.horizon), dtype=np.float64)
        initial = np.zeros(2, dtype=np.float64)

    latent = np.empty((policy.horizon + 1, 2), dtype=np.float64)
    targets = np.empty((policy.horizon, 2), dtype=np.float64)
    latent[0] = initial
    for step in range(policy.horizon):
        regime = int(change_point is not None and step >= change_point)
        targets[step] = (
            state_matrices[regime] @ latent[step] + input_matrices[regime][:, 0] * inputs[step]
        )
        latent[step + 1] = targets[step]
        if family is SystemIdentificationFamily.PROCESS_NOISE:
            latent[step + 1] += rng.normal(0.0, policy.process_noise_standard_deviation, size=2)

    observed = np.array(latent, copy=True)
    mask = np.ones_like(observed, dtype=np.bool_)
    measurement_variance: float | None = None
    validation_variance: float | None = None
    validation: tuple[PublicValidationTransition, ...] = ()

    if family is SystemIdentificationFamily.MEASUREMENT_NOISE:
        observed += rng.normal(
            0.0, policy.measurement_noise_standard_deviation, size=observed.shape
        )
        measurement_variance = policy.measurement_noise_standard_deviation**2
        validation_variance = policy.validation_noise_standard_deviation**2
        count = max(3, round(policy.validation_fraction * policy.horizon))
        indices = np.sort(rng.choice(policy.horizon, size=count, replace=False))
        validation = tuple(
            PublicValidationTransition(
                step=int(step),
                current_state=_vector2(
                    latent[step]
                    + rng.normal(0.0, policy.validation_noise_standard_deviation, size=2)
                ),
                next_state=_vector2(
                    latent[step + 1]
                    + rng.normal(0.0, policy.validation_noise_standard_deviation, size=2)
                ),
            )
            for step in indices
        )
    elif family is SystemIdentificationFamily.MISSING_OBSERVATIONS:
        count = max(1, round(policy.missing_fraction * (policy.horizon - 1)))
        missing = rng.choice(np.arange(1, policy.horizon), size=count, replace=False)
        mask[missing] = False
        observed[~mask] = 0.0

    observations = PublicSystemIdentificationObservations(
        states=tuple(_vector2(row) for row in observed),
        observed_mask=tuple((bool(row[0]), bool(row[1])) for row in mask),
        inputs=tuple(float(item) for item in inputs),
        time_step=policy.time_step,
        declared_change_point=change_point,
        measurement_noise_variance=measurement_variance,
        validation_noise_variance=validation_variance,
        validation_transitions=validation,
    )
    public = PublicSystemIdentificationTask(
        task_id=task_id,
        family=family,
        observations=observations,
        public_conditions=_family_conditions(family),
    )

    bounds = (
        ((0, policy.horizon),)
        if change_point is None
        else ((0, change_point), (change_point, policy.horizon))
    )
    segments = tuple(
        PrivateSystemIdentificationSegment(
            start_step=start,
            end_step=end,
            state_matrix=_matrix2x2(state_matrices[index]),
            input_matrix=_matrix2x1(input_matrices[index]),
        )
        for index, (start, end) in enumerate(bounds)
    )
    required = _required_estimator(family)
    truth = PrivateSystemIdentificationTruth(
        task_id=task_id,
        family=family,
        required_action=(
            IdentificationAction.WITHHOLD
            if family is SystemIdentificationFamily.INSUFFICIENT_EXCITATION
            else IdentificationAction.ESTIMATE
        ),
        required_estimator=required,
        required_withhold_reason=(
            "insufficient-excitation"
            if family is SystemIdentificationFamily.INSUFFICIENT_EXCITATION
            else None
        ),
        segments=segments,
        latent_states=tuple(_vector2(row) for row in latent),
        conditional_targets=tuple(_vector2(row) for row in targets),
    )
    return BuiltSystemIdentificationTask(public, truth)


def _spectral_radius(matrix: FloatArray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def trusted_reference_submission(
    truth: PrivateSystemIdentificationTruth,
) -> SystemIdentificationSubmission:
    """Private-only exact reference used to test the evaluator boundary."""

    if truth.required_action is IdentificationAction.WITHHOLD:
        return SystemIdentificationSubmission(
            action=IdentificationAction.WITHHOLD,
            estimator=IdentificationEstimator.WITHHOLD,
            withhold_reason=truth.required_withhold_reason,
        )
    return SystemIdentificationSubmission(
        action=IdentificationAction.ESTIMATE,
        estimator=truth.required_estimator,
        segments=tuple(
            SystemIdentificationSegmentEstimate(
                start_step=item.start_step,
                end_step=item.end_step,
                state_matrix=item.state_matrix,
                input_matrix=item.input_matrix,
            )
            for item in truth.segments
        ),
    )


__all__ = [
    "BuiltSystemIdentificationTask",
    "PrivateSystemIdentificationPolicy",
    "PrivateSystemIdentificationSegment",
    "PrivateSystemIdentificationTruth",
    "build_system_identification_task",
    "frozen_persistent_excitation",
    "trusted_reference_submission",
]
