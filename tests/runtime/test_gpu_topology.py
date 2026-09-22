from dataclasses import replace
from pathlib import Path

import pytest

from skillev.runtime.gpu_topology import (
    FourGPURoleAssignment,
    FourGPURolePolicy,
    GPUObservation,
    ThreeGPURolePolicy,
    select_four_idle_gpus,
    validate_assignment_is_idle,
    validate_four_gpu_roles,
    validate_three_gpu_roles,
)


def _gpu(
    index: int,
    *,
    used: int = 4,
    pids: tuple[int, ...] = (),
    utilization: int = 0,
) -> GPUObservation:
    return GPUObservation(
        physical_index=index,
        name="NVIDIA H800",
        uuid=f"GPU-{index}",
        memory_total_mib=81559,
        memory_used_mib=used,
        utilization_percent=utilization,
        compute_pids=pids,
    )


def _policy() -> FourGPURolePolicy:
    return FourGPURolePolicy(
        inference_physical_index=5,
        coordinator_physical_index=4,
        gradient_primary_physical_index=7,
        gradient_standby_physical_index=6,
        forbidden_physical_indices=(),
        minimum_free_memory_mib=70000,
        standby_minimum_free_memory_mib=70000,
        standby_may_share=False,
        standby_must_be_cold=True,
    )


def test_policy_loads_the_tracked_dynamic_role_config() -> None:
    policy = FourGPURolePolicy.read_yaml(Path("configs/hardware/four_gpu_roles.yaml"))

    assert policy == _policy()


def test_protocol_v10_policy_loads_three_distinct_roles_without_standby() -> None:
    policy = ThreeGPURolePolicy.read_yaml(Path("configs/hardware/three_gpu_roles.yaml"))

    assert policy.assigned_physical_indices == (5, 4, 6)
    assert not policy.forbidden_physical_indices
    assert "gradient_standby_physical_index" not in policy.to_value()


def test_protocol_v10_roles_require_serving_and_idle_training_gpus() -> None:
    policy = ThreeGPURolePolicy.read_yaml(Path("configs/hardware/three_gpu_roles.yaml"))
    observations = tuple(
        _gpu(index, pids=(61104,) if index == policy.inference_physical_index else ())
        for index in range(8)
    )

    validate_three_gpu_roles(policy, observations)

    occupied = tuple(
        _gpu(
            index,
            pids=(61104,)
            if index == policy.inference_physical_index
            else (70001,)
            if index == policy.gradient_primary_physical_index
            else (),
        )
        for index in range(8)
    )
    with pytest.raises(RuntimeError):
        validate_three_gpu_roles(policy, occupied)


@pytest.mark.parametrize("inference_index", range(8))
def test_all_gpu_indices_can_host_each_configured_role(inference_index: int) -> None:
    indices = tuple((inference_index + offset) % 8 for offset in range(4))
    three = ThreeGPURolePolicy.read_yaml(Path("configs/hardware/three_gpu_roles.yaml"))
    three = replace(
        three,
        inference_physical_index=indices[0],
        coordinator_physical_index=indices[1],
        gradient_primary_physical_index=indices[2],
    )
    four = replace(
        _policy(),
        inference_physical_index=indices[0],
        coordinator_physical_index=indices[1],
        gradient_primary_physical_index=indices[2],
        gradient_standby_physical_index=indices[3],
    )
    observations = tuple(
        _gpu(index, pids=(61104,) if index == inference_index else ()) for index in range(8)
    )

    validate_three_gpu_roles(three, observations)
    validate_four_gpu_roles(four, observations)


def test_protocol_v10_roles_allow_explicit_training_colocation_with_memory_reserve() -> None:
    policy = ThreeGPURolePolicy(
        inference_physical_index=5,
        coordinator_physical_index=4,
        gradient_primary_physical_index=6,
        forbidden_physical_indices=(0, 1, 2, 3, 7),
        minimum_free_memory_mib=12_000,
        training_may_share=True,
    )
    observations = tuple(
        _gpu(
            index,
            used=65_000 if index == 4 else 18_000 if index == 6 else 4,
            pids=(61104,) if index == 5 else (70001,) if index in {4, 6} else (),
            utilization=100 if index == 6 else 0,
        )
        for index in range(8)
    )

    validate_three_gpu_roles(policy, observations)


def test_protocol_v10_shared_training_still_requires_memory_reserve() -> None:
    policy = ThreeGPURolePolicy(
        inference_physical_index=5,
        coordinator_physical_index=4,
        gradient_primary_physical_index=6,
        forbidden_physical_indices=(0, 1, 2, 3, 7),
        minimum_free_memory_mib=12_000,
        training_may_share=True,
    )
    observations = tuple(
        _gpu(
            index,
            used=70_000 if index == 4 else 18_000 if index == 6 else 4,
            pids=(61104,) if index == 5 else (70001,) if index in {4, 6} else (),
        )
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_three_gpu_roles(policy, observations)


def test_protocol_v10_shared_training_rejects_unavailable_idle_device() -> None:
    policy = ThreeGPURolePolicy(
        inference_physical_index=5,
        coordinator_physical_index=4,
        gradient_primary_physical_index=6,
        forbidden_physical_indices=(0, 1, 2, 3, 7),
        minimum_free_memory_mib=12_000,
        training_may_share=True,
    )
    observations = tuple(
        _gpu(
            index,
            pids=(61104,) if index == 5 else (70001,) if index == 4 else (),
            utilization=100 if index == 6 else 0,
        )
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_three_gpu_roles(policy, observations)


def test_unassigned_gpu_can_remain_eligible_without_being_forbidden() -> None:
    policy = ThreeGPURolePolicy(
        inference_physical_index=5,
        coordinator_physical_index=4,
        gradient_primary_physical_index=6,
        forbidden_physical_indices=(0, 1, 2, 3),
        minimum_free_memory_mib=70_000,
    )
    observations = tuple(_gpu(index, pids=(61104,) if index == 5 else ()) for index in range(8))

    validate_three_gpu_roles(policy, observations)


def test_selector_accepts_external_inference_and_three_idle_training_roles() -> None:
    observations = tuple(
        _gpu(index, pids=(44325,) if index == 0 else (61104,) if index == 5 else ())
        for index in range(8)
    )

    assignment = select_four_idle_gpus(_policy(), observations)

    assert assignment == FourGPURoleAssignment(5, 4, 7, 6)
    assert assignment.visible_devices_for_role("inference") == "5"
    assert assignment.visible_devices_for_role("gradient_primary", expanded=True) == "7,6"
    with pytest.raises(ValueError):
        assignment.visible_devices_for_role("gradient_standby")


def test_selector_fails_instead_of_reusing_an_occupied_or_low_memory_gpu() -> None:
    observations = tuple(
        _gpu(
            index,
            used=60_000 if index == 6 else 4,
            pids=(100 + index,) if index in {0, 5, 6} else (),
        )
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        select_four_idle_gpus(_policy(), observations)


def test_explicit_assignment_is_rechecked_immediately_before_launch() -> None:
    assignment = FourGPURoleAssignment(5, 4, 7, 6)
    observations = tuple(_gpu(index, pids=(777,) if index == 7 else ()) for index in range(8))

    with pytest.raises(RuntimeError):
        validate_assignment_is_idle(
            assignment,
            observations,
            minimum_free_memory_mib=70000,
            standby_minimum_free_memory_mib=70000,
            standby_may_share=False,
        )


def test_duplicate_role_assignment_is_rejected() -> None:
    with pytest.raises(ValueError):
        FourGPURoleAssignment(5, 4, 7, 7)


def test_policy_rejects_forbidden_or_repurposed_gpu() -> None:
    with pytest.raises(ValueError):
        FourGPURolePolicy(
            inference_physical_index=5,
            coordinator_physical_index=0,
            gradient_primary_physical_index=7,
            gradient_standby_physical_index=6,
            forbidden_physical_indices=(0, 1, 2, 3),
            minimum_free_memory_mib=70000,
            standby_minimum_free_memory_mib=70000,
            standby_may_share=False,
            standby_must_be_cold=True,
        )


def test_selector_rejects_an_occupied_cold_standby() -> None:
    observations = tuple(
        _gpu(index, used=20_000 if index == 6 else 4, pids=(900,) if index in {5, 6} else ())
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        select_four_idle_gpus(_policy(), observations)


@pytest.mark.parametrize("unavailable_index", [4, 7, 6])
def test_formal_roles_reject_a_driver_unavailable_training_gpu(
    unavailable_index: int,
) -> None:
    observations = tuple(
        _gpu(
            index,
            pids=(61104,) if index == 5 else (),
            utilization=100 if index == unavailable_index else 0,
        )
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_four_gpu_roles(_policy(), observations)


def test_shared_standby_still_rejects_a_driver_unavailable_gpu() -> None:
    assignment = FourGPURoleAssignment(5, 4, 7, 6)
    observations = tuple(
        _gpu(index, pids=(900,) if index == 6 else (), utilization=100 if index == 6 else 0)
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_assignment_is_idle(
            assignment,
            observations,
            minimum_free_memory_mib=70000,
            standby_minimum_free_memory_mib=70000,
            standby_may_share=True,
        )


def test_formal_four_roles_require_all_gpus_and_explicit_forbidden_set() -> None:
    observations = tuple(_gpu(index, pids=(61104,) if index == 5 else ()) for index in range(8))

    validate_four_gpu_roles(_policy(), observations)
    with pytest.raises(RuntimeError):
        validate_four_gpu_roles(_policy(), observations[:-1])
