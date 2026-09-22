"""Trusted numerical evaluator for structured system-identification outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skillev.environments.public import (
    IdentificationAction,
    PublicSystemIdentificationTask,
    SystemIdentificationSegmentEstimate,
    SystemIdentificationSubmission,
)
from skillev.evaluation.outcomes import EvaluationStatus, TrustedEvaluatorOutcome
from skillev_private.environments.system_identification import (
    PrivateSystemIdentificationSegment,
    PrivateSystemIdentificationTruth,
)

from .result import PrivateEvaluationResult


@dataclass(frozen=True, slots=True)
class PrivateSystemIdentificationEvaluationPolicy:
    maximum_parameter_relative_error: float = 0.10
    maximum_normalized_forecast_error: float = 0.10
    parameter_absolute_bound: float = 2.0
    maximum_spectral_radius: float = 1.0

    def __post_init__(self) -> None:
        for value in (
            self.maximum_parameter_relative_error,
            self.maximum_normalized_forecast_error,
            self.parameter_absolute_bound,
            self.maximum_spectral_radius,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("system-identification evaluator bounds must be positive")


def _matrix_values(
    segment: SystemIdentificationSegmentEstimate | PrivateSystemIdentificationSegment,
) -> tuple[float, ...]:
    return (
        *segment.state_matrix[0],
        *segment.state_matrix[1],
        segment.input_matrix[0][0],
        segment.input_matrix[1][0],
    )


def _relative_error(values: tuple[float, ...], expected: tuple[float, ...]) -> float:
    numerator = math.sqrt(
        sum((value - target) ** 2 for value, target in zip(values, expected, strict=True))
    )
    denominator = math.sqrt(sum(target**2 for target in expected))
    return numerator if denominator == 0 else numerator / denominator


def _segment_bounds_match(
    submission: SystemIdentificationSubmission,
    truth: PrivateSystemIdentificationTruth,
) -> bool:
    if len(submission.segments) != len(truth.segments):
        return False
    return all(
        proposed.start_step == expected.start_step and proposed.end_step == expected.end_step
        for proposed, expected in zip(submission.segments, truth.segments, strict=True)
    )


def _parameter_error(
    submission: SystemIdentificationSubmission,
    truth: PrivateSystemIdentificationTruth,
) -> float:
    if not _segment_bounds_match(submission, truth):
        return math.inf
    return max(
        _relative_error(_matrix_values(proposed), _matrix_values(expected))
        for proposed, expected in zip(submission.segments, truth.segments, strict=True)
    )


def _forecast_error(
    task: PublicSystemIdentificationTask,
    submission: SystemIdentificationSubmission,
    truth: PrivateSystemIdentificationTruth,
) -> float:
    if not _segment_bounds_match(submission, truth):
        return math.inf
    predictions: list[tuple[float, float]] = []
    for step, input_value in enumerate(task.observations.inputs):
        segment = next(
            item for item in submission.segments if item.start_step <= step < item.end_step
        )
        state = np.asarray(truth.latent_states[step], dtype=np.float64)
        state_matrix = np.asarray(segment.state_matrix, dtype=np.float64)
        input_matrix = np.asarray(segment.input_matrix, dtype=np.float64)
        predicted = state_matrix @ state + input_matrix[:, 0] * input_value
        predictions.append((float(predicted[0]), float(predicted[1])))
    proposed = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(truth.conditional_targets, dtype=np.float64)
    denominator = float(np.linalg.norm(expected))
    numerator = float(np.linalg.norm(proposed - expected))
    return numerator if denominator == 0 else numerator / denominator


def _maximum_spectral_radius(submission: SystemIdentificationSubmission) -> float:
    return max(
        float(np.max(np.abs(np.linalg.eigvals(np.asarray(item.state_matrix)))))
        for item in submission.segments
    )


def _parameters_bounded(
    submission: SystemIdentificationSubmission,
    bound: float,
) -> bool:
    return bool(submission.segments) and all(
        abs(value) <= bound for segment in submission.segments for value in _matrix_values(segment)
    )


def evaluate_system_identification(
    *,
    trajectory_id: str,
    task: PublicSystemIdentificationTask,
    submission: SystemIdentificationSubmission,
    truth: PrivateSystemIdentificationTruth,
    policy: PrivateSystemIdentificationEvaluationPolicy | None = None,
) -> PrivateEvaluationResult:
    """Evaluate data only; candidate source is neither accepted nor executed."""

    selected_policy = policy or PrivateSystemIdentificationEvaluationPolicy()
    if task.task_id != truth.task_id or task.family is not truth.family:
        return PrivateEvaluationResult(
            TrustedEvaluatorOutcome(
                trajectory_id,
                EvaluationStatus.INVALID_SUBMISSION,
                False,
                0.0,
            ),
            (("binding_valid", False),),
        )

    diagnostics: dict[str, object] = {"binding_valid": True}
    if truth.required_action is IdentificationAction.WITHHOLD:
        success = (
            submission.action is IdentificationAction.WITHHOLD
            and submission.withhold_reason == truth.required_withhold_reason
        )
        diagnostics.update(
            {
                "action_conformant": success,
                "withholding_correct": success,
            }
        )
    elif submission.action is not IdentificationAction.ESTIMATE:
        success = False
        diagnostics["action_conformant"] = False
    else:
        parameter_error = _parameter_error(submission, truth)
        forecast_error = _forecast_error(task, submission, truth)
        maximum_radius = _maximum_spectral_radius(submission)
        gates = {
            "action_conformant": True,
            "estimator_conformant": submission.estimator is truth.required_estimator,
            "forecast_accurate": (
                math.isfinite(forecast_error)
                and forecast_error <= selected_policy.maximum_normalized_forecast_error
            ),
            "parameter_accurate": (
                math.isfinite(parameter_error)
                and parameter_error <= selected_policy.maximum_parameter_relative_error
            ),
            "parameters_bounded": _parameters_bounded(
                submission, selected_policy.parameter_absolute_bound
            ),
            "segment_bounds_conformant": _segment_bounds_match(submission, truth),
            "stable": (
                math.isfinite(maximum_radius)
                and maximum_radius < selected_policy.maximum_spectral_radius
            ),
        }
        diagnostics.update(gates)
        diagnostics.update(
            {
                "maximum_parameter_relative_error": parameter_error,
                "maximum_spectral_radius": maximum_radius,
                "normalized_forecast_error": forecast_error,
            }
        )
        success = all(gates.values())

    outcome = TrustedEvaluatorOutcome(
        trajectory_id=trajectory_id,
        status=EvaluationStatus.COMPLETED,
        scientific_success=success,
        score=float(success),
    )
    return PrivateEvaluationResult(outcome, tuple(sorted(diagnostics.items())))


__all__ = [
    "PrivateSystemIdentificationEvaluationPolicy",
    "evaluate_system_identification",
]
