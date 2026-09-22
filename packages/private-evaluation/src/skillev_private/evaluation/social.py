"""Protected evaluator for semi-synthetic social-effect submissions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from skillev.environments.public import (
    IdentificationAction,
    PublicSocialTask,
    SocialEffectSubmission,
)
from skillev.evaluation.outcomes import EvaluationStatus, TrustedEvaluatorOutcome
from skillev_private.environments.social_dgp import PrivateSocialTruth

from .result import PrivateEvaluationResult


@dataclass(frozen=True, slots=True)
class PrivateSocialEvaluationPolicy:
    maximum_absolute_effect_error: float = 0.15

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_absolute_effect_error)
            or self.maximum_absolute_effect_error <= 0
        ):
            raise ValueError("social evaluator tolerance must be finite and positive")


def evaluate_social_effect(
    *,
    trajectory_id: str,
    task: PublicSocialTask,
    submission: SocialEffectSubmission,
    truth: PrivateSocialTruth,
    policy: PrivateSocialEvaluationPolicy | None = None,
) -> PrivateEvaluationResult:
    selected_policy = policy or PrivateSocialEvaluationPolicy()
    if task.task_id != truth.task_id or task.task_class is not truth.task_class:
        return PrivateEvaluationResult(
            TrustedEvaluatorOutcome(
                trajectory_id,
                EvaluationStatus.INVALID_SUBMISSION,
                False,
                0.0,
            ),
            (("binding_valid", False),),
        )

    if truth.identifiable:
        error = (
            math.inf
            if submission.effect_estimate is None
            else abs(submission.effect_estimate - truth.treatment_effect)
        )
        success = (
            submission.action is IdentificationAction.ESTIMATE
            and math.isfinite(error)
            and error <= selected_policy.maximum_absolute_effect_error
        )
        diagnostics: tuple[tuple[str, object], ...] = (
            ("absolute_effect_error", error),
            ("action_conformant", submission.action is IdentificationAction.ESTIMATE),
            ("binding_valid", True),
        )
    else:
        success = submission.action is IdentificationAction.WITHHOLD
        diagnostics = (
            ("action_conformant", success),
            ("binding_valid", True),
            ("non_identifiability_respected", success),
        )

    return PrivateEvaluationResult(
        TrustedEvaluatorOutcome(
            trajectory_id,
            EvaluationStatus.COMPLETED,
            success,
            float(success),
        ),
        diagnostics,
    )


__all__ = ["PrivateSocialEvaluationPolicy", "evaluate_social_effect"]
