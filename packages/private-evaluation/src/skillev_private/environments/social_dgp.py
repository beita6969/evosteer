"""Private semi-synthetic social-science task builder.

The builder retains all latent variables and the planted estimand.  Its public
return type contains only variables needed by the declared identification
strategy for the selected task class.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from skillev.environments.public import (
    IdentificationAction,
    PublicSocialRow,
    PublicSocialTask,
    SocialEffectSubmission,
    SocialTaskClass,
)


@dataclass(frozen=True, slots=True)
class PrivateSocialPolicy:
    seed: int
    namespace: str
    task_class: SocialTaskClass
    treatment_effect: float
    sample_size_per_environment: int = 80
    distribution_shift: float = 0.35
    validation_fraction: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("private social seed must be an integer")
        if not self.namespace:
            raise ValueError("private social namespace is required")
        if self.sample_size_per_environment < 50:
            raise ValueError("social tasks require at least 50 rows per environment")
        if not math.isfinite(self.treatment_effect):
            raise ValueError("planted treatment effect must be finite")
        if not math.isfinite(self.distribution_shift):
            raise ValueError("distribution shift must be finite")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation fraction must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class PrivateSocialRow:
    row_id: str
    latent_common_cause: float
    true_outcome: float


@dataclass(frozen=True, slots=True)
class PrivateSocialTruth:
    task_id: str
    task_class: SocialTaskClass
    treatment_effect: float
    identifiable: bool
    rows: tuple[PrivateSocialRow, ...]


@dataclass(frozen=True, slots=True)
class BuiltSocialTask:
    public: PublicSocialTask
    truth: PrivateSocialTruth


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _seed(policy: PrivateSocialPolicy, environment: str, index: int) -> int:
    material = (
        f"{policy.namespace}\0{policy.seed}\0{policy.task_class.value}\0{environment}\0{index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _task_id(policy: PrivateSocialPolicy) -> str:
    material = (
        f"{policy.namespace}\0{policy.seed}\0{policy.task_class.value}\0"
        f"{policy.sample_size_per_environment}"
    ).encode()
    return (
        f"social-{policy.task_class.value.lower().replace('_', '-')}-"
        + hashlib.sha256(material).hexdigest()[:20]
    )


def build_social_task(policy: PrivateSocialPolicy) -> BuiltSocialTask:
    """Build one public dataset and retain its causal target in private memory."""

    task_id = _task_id(policy)
    public_rows: list[PublicSocialRow] = []
    private_rows: list[PrivateSocialRow] = []
    for environment, shift in (
        ("source", 0.0),
        ("target", policy.distribution_shift),
    ):
        for index in range(policy.sample_size_per_environment):
            rng = random.Random(_seed(policy, environment, index))  # noqa: S311
            common_cause = rng.gauss(shift, 1.0)

            treatment_probability = 0.5
            if policy.task_class in {
                SocialTaskClass.OBSERVED_CONFOUNDER,
                SocialTaskClass.UNOBSERVED_COMMON_CAUSE,
            }:
                treatment_probability = _sigmoid(1.5 * common_cause)
            treatment = int(rng.random() < treatment_probability)
            true_outcome = (
                policy.treatment_effect * treatment
                + 1.6 * common_cause
                + 0.3 * shift
                + rng.gauss(0.0, 0.5)
            )

            observed_confounder = (
                common_cause if policy.task_class is SocialTaskClass.OBSERVED_CONFOUNDER else None
            )
            missingness_predictor = (
                common_cause if policy.task_class is SocialTaskClass.MAR_OUTCOME else None
            )

            measured_outcome = true_outcome
            validation_member: bool | None = None
            validation_outcome: float | None = None
            if policy.task_class is SocialTaskClass.VALIDATION_MEASUREMENT:
                measured_outcome = true_outcome + 0.55 * treatment + rng.gauss(0.0, 0.2)
                validation_member = rng.random() < policy.validation_fraction
                if validation_member:
                    validation_outcome = true_outcome + rng.gauss(0.0, 0.06)

            observation_probability = 0.96
            if policy.task_class is SocialTaskClass.MAR_OUTCOME:
                observation_probability = _sigmoid(0.9 - 2.2 * treatment * common_cause)
            observed_outcome = measured_outcome if rng.random() < observation_probability else None

            sampling_probability = 0.94
            released_probability: float | None = None
            if policy.task_class is SocialTaskClass.KNOWN_SELECTION_PROBABILITY:
                sampling_probability = _sigmoid(1.0 - 1.3 * treatment - 0.8 * common_cause)
                released_probability = sampling_probability
            selected = rng.random() < sampling_probability
            if not selected:
                observed_outcome = None

            row_id = f"{environment}-{index:06d}"
            public_rows.append(
                PublicSocialRow(
                    row_id=row_id,
                    environment=environment,
                    treatment=treatment,
                    observed_outcome=observed_outcome,
                    selected=selected,
                    observed_confounder=observed_confounder,
                    missingness_predictor=missingness_predictor,
                    sampling_probability=released_probability,
                    validation_member=validation_member,
                    validation_outcome=validation_outcome,
                )
            )
            private_rows.append(
                PrivateSocialRow(
                    row_id=row_id,
                    latent_common_cause=common_cause,
                    true_outcome=true_outcome,
                )
            )

    public = PublicSocialTask(
        task_id=task_id,
        task_class=policy.task_class,
        rows=tuple(public_rows),
    )
    truth = PrivateSocialTruth(
        task_id=task_id,
        task_class=policy.task_class,
        treatment_effect=policy.treatment_effect,
        identifiable=policy.task_class is not SocialTaskClass.UNOBSERVED_COMMON_CAUSE,
        rows=tuple(private_rows),
    )
    return BuiltSocialTask(public, truth)


def trusted_reference_submission(truth: PrivateSocialTruth) -> SocialEffectSubmission:
    """Private-only exact reference for evaluator tests."""

    if truth.identifiable:
        return SocialEffectSubmission(
            action=IdentificationAction.ESTIMATE,
            effect_estimate=truth.treatment_effect,
        )
    return SocialEffectSubmission(
        action=IdentificationAction.WITHHOLD,
        withhold_reason="unobserved-common-cause",
    )


__all__ = [
    "BuiltSocialTask",
    "PrivateSocialPolicy",
    "PrivateSocialRow",
    "PrivateSocialTruth",
    "build_social_task",
    "trusted_reference_submission",
]
