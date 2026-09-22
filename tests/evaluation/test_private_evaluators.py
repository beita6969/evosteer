from __future__ import annotations

from dataclasses import fields, replace

import pytest
from skillev_private.environments.social_dgp import (
    PrivateSocialPolicy,
    build_social_task,
)
from skillev_private.environments.social_dgp import (
    trusted_reference_submission as social_reference,
)
from skillev_private.environments.system_identification import (
    PrivateSystemIdentificationPolicy,
    build_system_identification_task,
)
from skillev_private.environments.system_identification import (
    trusted_reference_submission as system_reference,
)
from skillev_private.evaluation.exact_match import (
    PrivateAnswerEntry,
    PrivateAnswerKey,
    evaluate_exact_match,
)
from skillev_private.evaluation.social import evaluate_social_effect
from skillev_private.evaluation.system_identification import (
    evaluate_system_identification,
)

from skillev.environments.public import (
    SocialTaskClass,
    SystemIdentificationFamily,
    SystemIdentificationSegmentEstimate,
    SystemIdentificationSubmission,
)
from skillev.evaluation.outcomes import EvaluationStatus, TrustedEvaluatorOutcome


@pytest.mark.parametrize("family", tuple(SystemIdentificationFamily))
def test_private_system_evaluator_accepts_trusted_structured_reference(
    family: SystemIdentificationFamily,
) -> None:
    built = build_system_identification_task(
        PrivateSystemIdentificationPolicy(seed=13, namespace="evaluation"),
        family,
    )
    result = evaluate_system_identification(
        trajectory_id=f"trajectory-{family.value}",
        task=built.public,
        submission=system_reference(built.truth),
        truth=built.truth,
    )

    assert result.outcome.status is EvaluationStatus.COMPLETED
    assert result.outcome.scientific_success
    assert result.outcome.score == 1
    assert not any(
        fragment in field.name
        for field in fields(TrustedEvaluatorOutcome)
        for fragment in ("answer", "diagnostic", "truth")
    )


def test_private_system_evaluator_rejects_inaccurate_parameters() -> None:
    built = build_system_identification_task(
        PrivateSystemIdentificationPolicy(seed=14, namespace="evaluation"),
        SystemIdentificationFamily.WELL_EXCITED_NOISE_FREE,
    )
    reference = system_reference(built.truth)
    segment = reference.segments[0]
    changed = SystemIdentificationSegmentEstimate(
        segment.start_step,
        segment.end_step,
        (
            (segment.state_matrix[0][0] + 0.5, segment.state_matrix[0][1]),
            segment.state_matrix[1],
        ),
        segment.input_matrix,
    )
    submission = SystemIdentificationSubmission(
        reference.action,
        reference.estimator,
        (changed,),
    )
    result = evaluate_system_identification(
        trajectory_id="trajectory-inaccurate",
        task=built.public,
        submission=submission,
        truth=built.truth,
    )

    assert not result.outcome.scientific_success
    assert result.outcome.score == 0
    assert result.diagnostics


@pytest.mark.parametrize("task_class", tuple(SocialTaskClass))
def test_private_social_evaluator_handles_identification_and_withholding(
    task_class: SocialTaskClass,
) -> None:
    built = build_social_task(
        PrivateSocialPolicy(
            seed=15,
            namespace="evaluation",
            task_class=task_class,
            treatment_effect=0.3,
            sample_size_per_environment=50,
        )
    )
    result = evaluate_social_effect(
        trajectory_id=f"trajectory-{task_class.value}",
        task=built.public,
        submission=social_reference(built.truth),
        truth=built.truth,
    )
    assert result.outcome.scientific_success
    assert result.outcome.score == 1


def test_exact_match_key_and_diagnostic_remain_private() -> None:
    key = PrivateAnswerKey((PrivateAnswerEntry("synthetic-instance", frozenset({"x", "y"}), "y"),))
    result = evaluate_exact_match(
        trajectory_id="trajectory-choice",
        instance_id="synthetic-instance",
        selected_option="y",
        key=key,
    )

    assert result.outcome.scientific_success
    assert not hasattr(result.public_projection(), "correct_option")
    assert result.diagnostics


def test_public_task_binding_mismatch_is_invalid_without_credit() -> None:
    built = build_system_identification_task(
        PrivateSystemIdentificationPolicy(seed=16, namespace="evaluation"),
        SystemIdentificationFamily.PROCESS_NOISE,
    )
    result = evaluate_system_identification(
        trajectory_id="trajectory-mismatch",
        task=replace(built.public, task_id="different-task"),
        submission=system_reference(built.truth),
        truth=built.truth,
    )
    assert result.outcome.status is EvaluationStatus.INVALID_SUBMISSION
    assert result.outcome.score == 0
