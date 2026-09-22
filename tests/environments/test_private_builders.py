from __future__ import annotations

import pytest
from skillev_private.environments.social_dgp import (
    PrivateSocialPolicy,
    build_social_task,
)
from skillev_private.environments.system_identification import (
    PrivateSystemIdentificationPolicy,
    build_system_identification_task,
)

from skillev.environments.public import SocialTaskClass, SystemIdentificationFamily


def _all_mapping_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(str(key) for key in value) + tuple(
            key for child in value.values() for key in _all_mapping_keys(child)
        )
    if isinstance(value, list | tuple):
        return tuple(key for child in value for key in _all_mapping_keys(child))
    return ()


@pytest.mark.parametrize("family", tuple(SystemIdentificationFamily))
def test_system_identification_builder_is_replayable_but_projects_no_truth(
    family: SystemIdentificationFamily,
) -> None:
    policy = PrivateSystemIdentificationPolicy(seed=7, namespace="unit")
    first = build_system_identification_task(policy, family)
    replay = build_system_identification_task(policy, family)

    assert first == replay
    public_value = first.public.to_public_value()
    forbidden = ("latent", "seed", "truth", "true_")
    assert all(
        fragment not in key.lower()
        for key in _all_mapping_keys(public_value)
        for fragment in forbidden
    )
    assert first.public.task_id == first.truth.task_id
    assert len(first.truth.conditional_targets) == first.public.observations.horizon


def test_system_identification_mechanism_specific_public_channels() -> None:
    policy = PrivateSystemIdentificationPolicy(seed=8, namespace="channels")
    measurement = build_system_identification_task(
        policy, SystemIdentificationFamily.MEASUREMENT_NOISE
    ).public
    missing = build_system_identification_task(
        policy, SystemIdentificationFamily.MISSING_OBSERVATIONS
    ).public
    drift = build_system_identification_task(
        policy, SystemIdentificationFamily.PIECEWISE_DRIFT
    ).public

    assert measurement.observations.validation_transitions
    assert any(not all(row) for row in missing.observations.observed_mask)
    assert drift.observations.declared_change_point is not None


@pytest.mark.parametrize("task_class", tuple(SocialTaskClass))
def test_social_builder_keeps_latent_rows_out_of_public_projection(
    task_class: SocialTaskClass,
) -> None:
    policy = PrivateSocialPolicy(
        seed=9,
        namespace="social-unit",
        task_class=task_class,
        treatment_effect=0.4,
        sample_size_per_environment=50,
    )
    built = build_social_task(policy)
    replay = build_social_task(policy)

    assert built == replay
    public_keys = _all_mapping_keys(built.public.to_public_value())
    assert all(
        fragment not in key.lower()
        for key in public_keys
        for fragment in ("latent", "seed", "true_", "truth")
    )
    assert len(built.public.rows) == len(built.truth.rows)
    assert built.truth.identifiable is (task_class is not SocialTaskClass.UNOBSERVED_COMMON_CAUSE)
