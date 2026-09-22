"""Explicit physical GPU role policies without touching occupied devices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

FOUR_GPU_ROLE_FORMAT: Final = "skillev-four-gpu-roles@4"
THREE_GPU_ROLE_FORMAT: Final = "skillev-three-gpu-roles@1"


@dataclass(frozen=True, slots=True)
class GPUObservation:
    """One physical GPU and the compute processes observed during preflight."""

    physical_index: int
    name: str
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.physical_index) is not int or self.physical_index < 0:
            raise ValueError("physical_index must be a non-negative integer")
        if not self.name.strip() or not self.uuid.strip():
            raise ValueError("GPU name and UUID must be non-empty")
        for field_name in ("memory_total_mib", "memory_used_mib", "utilization_percent"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.memory_used_mib > self.memory_total_mib:
            raise ValueError("memory_used_mib cannot exceed memory_total_mib")
        if self.utilization_percent > 100:
            raise ValueError("utilization_percent cannot exceed 100")
        if any(type(pid) is not int or pid <= 0 for pid in self.compute_pids):
            raise ValueError("compute_pids must contain positive integers")
        if len(set(self.compute_pids)) != len(self.compute_pids):
            raise ValueError("compute_pids must be unique")

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib

    @property
    def is_process_idle(self) -> bool:
        """An idle device has no compute process, regardless of utilization sampling."""

        return not self.compute_pids

    @property
    def is_launchable_idle(self) -> bool:
        """A launch candidate must be process-idle and available to the driver."""

        return self.is_process_idle and self.is_driver_available

    @property
    def is_driver_available(self) -> bool:
        """Driver-excluded or reset-required devices report unavailable utilization."""

        return self.utilization_percent < 100

    def to_value(self) -> dict[str, object]:
        return {
            "compute_pids": list(self.compute_pids),
            "memory_free_mib": self.memory_free_mib,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "name": self.name,
            "physical_index": self.physical_index,
            "utilization_percent": self.utilization_percent,
            "uuid": self.uuid,
        }


@dataclass(frozen=True, slots=True)
class ThreeGPURolePolicy:
    """Protocol 10 inference, coordinator, and gradient-worker mapping.

    Sharing is an explicit per-attempt decision.  The default continues to
    require process-idle training devices.
    """

    inference_physical_index: int
    coordinator_physical_index: int
    gradient_primary_physical_index: int
    forbidden_physical_indices: tuple[int, ...]
    minimum_free_memory_mib: int
    training_may_share: bool = False
    format: str = THREE_GPU_ROLE_FORMAT

    def __post_init__(self) -> None:
        if self.format != THREE_GPU_ROLE_FORMAT:
            raise ValueError("unsupported three-GPU role format")
        assigned = self.assigned_physical_indices
        if any(type(index) is not int or index < 0 for index in assigned):
            raise ValueError("assigned GPU indices must be non-negative integers")
        if len(set(assigned)) != 3:
            raise ValueError("the three GPU roles require distinct physical devices")
        if any(type(index) is not int or index < 0 for index in self.forbidden_physical_indices):
            raise ValueError("forbidden GPU indices must be non-negative integers")
        if len(set(self.forbidden_physical_indices)) != len(self.forbidden_physical_indices):
            raise ValueError("forbidden GPU indices must be unique")
        if set(assigned) & set(self.forbidden_physical_indices):
            raise ValueError("an assigned GPU cannot also be forbidden")
        if type(self.minimum_free_memory_mib) is not int or self.minimum_free_memory_mib <= 0:
            raise ValueError("minimum_free_memory_mib must be a positive integer")
        if type(self.training_may_share) is not bool:
            raise TypeError("training_may_share must be boolean")

    @property
    def assigned_physical_indices(self) -> tuple[int, int, int]:
        return (
            self.inference_physical_index,
            self.coordinator_physical_index,
            self.gradient_primary_physical_index,
        )

    def to_value(self) -> dict[str, object]:
        return {
            "coordinator_physical_index": self.coordinator_physical_index,
            "forbidden_physical_indices": list(self.forbidden_physical_indices),
            "format": self.format,
            "gradient_primary_physical_index": self.gradient_primary_physical_index,
            "inference_physical_index": self.inference_physical_index,
            "minimum_free_memory_mib": self.minimum_free_memory_mib,
            "training_may_share": self.training_may_share,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("hardware role configuration must be an object")
        expected = {
            "coordinator_physical_index",
            "forbidden_physical_indices",
            "format",
            "gradient_primary_physical_index",
            "inference_physical_index",
            "minimum_free_memory_mib",
        }
        compatible_fields = (expected, expected | {"training_may_share"})
        if set(value) not in compatible_fields:
            raise ValueError("hardware role configuration has incompatible fields")
        role_fields = (
            "inference_physical_index",
            "coordinator_physical_index",
            "gradient_primary_physical_index",
        )
        if any(type(value[field]) is not int for field in role_fields):
            raise TypeError("physical role indices must be integers")
        forbidden = value["forbidden_physical_indices"]
        if not isinstance(forbidden, list) or any(type(item) is not int for item in forbidden):
            raise TypeError("forbidden_physical_indices must be an integer list")
        minimum = value["minimum_free_memory_mib"]
        if type(minimum) is not int:
            raise TypeError("minimum_free_memory_mib must be an integer")
        training_may_share = value.get("training_may_share", False)
        if type(training_may_share) is not bool:
            raise TypeError("training_may_share must be boolean")
        format_value = value["format"]
        if type(format_value) is not str:
            raise TypeError("format must be text")
        return cls(
            inference_physical_index=value["inference_physical_index"],
            coordinator_physical_index=value["coordinator_physical_index"],
            gradient_primary_physical_index=value["gradient_primary_physical_index"],
            forbidden_physical_indices=tuple(forbidden),
            minimum_free_memory_mib=minimum,
            training_may_share=training_may_share,
            format=format_value,
        )

    @classmethod
    def read_yaml(cls, path: str | Path) -> Self:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)


def validate_three_gpu_roles(
    policy: ThreeGPURolePolicy,
    observations: Sequence[GPUObservation],
) -> None:
    """Revalidate the fixed Protocol 10 roles immediately before launch.

    The external inference device must already host the serving process.  The
    coordinator and gradient device must retain the frozen memory reserve. By
    default they must also remain process-idle; an attempt can explicitly
    authorize co-location with already-running compute processes. Unassigned
    devices remain eligible for a future explicitly checked role mapping; they
    need not be listed as forbidden merely because this run does not use them.
    """

    if not isinstance(policy, ThreeGPURolePolicy):
        raise TypeError("Protocol 10 GPU validation requires a three-role policy")
    by_index: dict[int, GPUObservation] = {}
    for observation in observations:
        if observation.physical_index in by_index:
            raise ValueError("GPU observations contain duplicate physical indices")
        by_index[observation.physical_index] = observation
    expected_indices = set(range(8))
    if set(by_index) != expected_indices:
        raise RuntimeError("Protocol 10 preflight requires physical GPUs 0 through 7")
    if not (set(policy.assigned_physical_indices) | set(policy.forbidden_physical_indices)) <= (
        expected_indices
    ):
        raise RuntimeError("hardware policy refers to a nonexistent physical GPU")
    inference = by_index[policy.inference_physical_index]
    if not inference.compute_pids:
        raise RuntimeError("Protocol 10 inference GPU has no serving compute process")
    for index in (
        policy.coordinator_physical_index,
        policy.gradient_primary_physical_index,
    ):
        observation = by_index[index]
        driver_available = observation.is_driver_available or bool(observation.compute_pids)
        if not driver_available or (
            not policy.training_may_share and not observation.is_process_idle
        ):
            raise RuntimeError(
                f"Protocol 10 training GPU {index} is unavailable or has a compute process"
            )
        if observation.memory_free_mib < policy.minimum_free_memory_mib:
            raise RuntimeError(f"Protocol 10 training GPU {index} lacks the memory reserve")


@dataclass(frozen=True, slots=True)
class FourGPURolePolicy:
    """Fixed project roles shared by preflight and launchers."""

    inference_physical_index: int
    coordinator_physical_index: int
    gradient_primary_physical_index: int
    gradient_standby_physical_index: int
    forbidden_physical_indices: tuple[int, ...]
    minimum_free_memory_mib: int
    standby_minimum_free_memory_mib: int
    standby_may_share: bool
    standby_must_be_cold: bool
    format: str = FOUR_GPU_ROLE_FORMAT

    def __post_init__(self) -> None:
        if self.format != FOUR_GPU_ROLE_FORMAT:
            raise ValueError("unsupported four-GPU role format")
        assigned = self.assigned_physical_indices
        if any(type(index) is not int or index < 0 for index in assigned):
            raise ValueError("assigned GPU indices must be non-negative integers")
        if len(set(assigned)) != 4:
            raise ValueError("the four GPU roles require distinct physical devices")
        if any(type(index) is not int or index < 0 for index in self.forbidden_physical_indices):
            raise ValueError("forbidden GPU indices must be non-negative integers")
        if len(set(self.forbidden_physical_indices)) != len(self.forbidden_physical_indices):
            raise ValueError("forbidden GPU indices must be unique")
        if set(assigned) & set(self.forbidden_physical_indices):
            raise ValueError("an assigned GPU cannot also be forbidden")
        if type(self.minimum_free_memory_mib) is not int or self.minimum_free_memory_mib <= 0:
            raise ValueError("minimum_free_memory_mib must be a positive integer")
        if (
            type(self.standby_minimum_free_memory_mib) is not int
            or self.standby_minimum_free_memory_mib <= 0
        ):
            raise ValueError("standby_minimum_free_memory_mib must be a positive integer")
        if self.standby_minimum_free_memory_mib > self.minimum_free_memory_mib:
            raise ValueError("standby free-memory requirement cannot exceed steady requirement")
        if self.standby_may_share is not False:
            raise ValueError("the OOM standby must be exclusively reserved")
        if self.standby_must_be_cold is not True:
            raise ValueError("the fourth GPU role must be a cold standby")

    @property
    def assigned_physical_indices(self) -> tuple[int, int, int, int]:
        return (
            self.inference_physical_index,
            self.coordinator_physical_index,
            self.gradient_primary_physical_index,
            self.gradient_standby_physical_index,
        )

    def to_value(self) -> dict[str, object]:
        return {
            "forbidden_physical_indices": list(self.forbidden_physical_indices),
            "format": self.format,
            "gradient_primary_physical_index": self.gradient_primary_physical_index,
            "gradient_standby_physical_index": self.gradient_standby_physical_index,
            "inference_physical_index": self.inference_physical_index,
            "coordinator_physical_index": self.coordinator_physical_index,
            "minimum_free_memory_mib": self.minimum_free_memory_mib,
            "standby_may_share": self.standby_may_share,
            "standby_minimum_free_memory_mib": self.standby_minimum_free_memory_mib,
            "standby_must_be_cold": self.standby_must_be_cold,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("hardware role configuration must be an object")
        expected = {
            "inference_physical_index",
            "coordinator_physical_index",
            "gradient_primary_physical_index",
            "gradient_standby_physical_index",
            "forbidden_physical_indices",
            "format",
            "minimum_free_memory_mib",
            "standby_minimum_free_memory_mib",
            "standby_may_share",
            "standby_must_be_cold",
        }
        if set(value) != expected:
            raise ValueError("hardware role configuration has incompatible fields")
        role_fields = (
            "inference_physical_index",
            "coordinator_physical_index",
            "gradient_primary_physical_index",
            "gradient_standby_physical_index",
        )
        if any(type(value[field]) is not int for field in role_fields):
            raise TypeError("physical role indices must be integers")
        forbidden = value["forbidden_physical_indices"]
        if not isinstance(forbidden, list) or any(type(item) is not int for item in forbidden):
            raise TypeError("forbidden_physical_indices must be an integer list")
        minimum = value["minimum_free_memory_mib"]
        if type(minimum) is not int:
            raise TypeError("minimum_free_memory_mib must be an integer")
        standby_minimum = value["standby_minimum_free_memory_mib"]
        if type(standby_minimum) is not int:
            raise TypeError("standby_minimum_free_memory_mib must be an integer")
        standby_may_share = value["standby_may_share"]
        if type(standby_may_share) is not bool:
            raise TypeError("standby_may_share must be boolean")
        standby = value["standby_must_be_cold"]
        if type(standby) is not bool:
            raise TypeError("standby_must_be_cold must be boolean")
        format_value = value["format"]
        if type(format_value) is not str:
            raise TypeError("format must be text")
        return cls(
            inference_physical_index=value["inference_physical_index"],
            coordinator_physical_index=value["coordinator_physical_index"],
            gradient_primary_physical_index=value["gradient_primary_physical_index"],
            gradient_standby_physical_index=value["gradient_standby_physical_index"],
            forbidden_physical_indices=tuple(forbidden),
            minimum_free_memory_mib=minimum,
            standby_minimum_free_memory_mib=standby_minimum,
            standby_may_share=standby_may_share,
            standby_must_be_cold=standby,
            format=format_value,
        )

    @classmethod
    def read_yaml(cls, path: str | Path) -> Self:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)


@dataclass(frozen=True, slots=True)
class FourGPURoleAssignment:
    """Physical device mapping for one run; process-local devices remain ``cuda:0``."""

    inference: int
    coordinator: int
    gradient_primary: int
    gradient_standby: int | None

    def __post_init__(self) -> None:
        values = self.physical_indices
        if any(type(index) is not int or index < 0 for index in values):
            raise ValueError("role assignments must be non-negative physical indices")
        if len(set(values)) != len(values):
            raise ValueError("enabled GPU roles require distinct physical devices")

    @property
    def physical_indices(self) -> tuple[int, ...]:
        values = (
            self.inference,
            self.coordinator,
            self.gradient_primary,
            self.gradient_standby,
        )
        return tuple(index for index in values if index is not None)

    def device_for_role(self, role: str) -> int:
        roles = {
            "inference": self.inference,
            "coordinator": self.coordinator,
            "gradient_primary": self.gradient_primary,
            "gradient_standby": self.gradient_standby,
        }
        try:
            device = roles[role]
        except KeyError as error:
            raise ValueError(f"unknown GPU role: {role}") from error
        if device is None:
            raise ValueError(f"GPU role is disabled: {role}")
        return device

    def visible_devices_for_role(self, role: str, *, expanded: bool = False) -> str:
        if role == "gradient_primary" and expanded:
            if self.gradient_standby is None:
                raise ValueError("gradient standby is disabled")
            return f"{self.gradient_primary},{self.gradient_standby}"
        if role == "gradient_standby" and not expanded:
            raise ValueError("cold standby must not be launched in the steady profile")
        return str(self.device_for_role(role))

    def to_value(self) -> dict[str, int | None]:
        return {
            "coordinator": self.coordinator,
            "gradient_primary": self.gradient_primary,
            "gradient_standby": self.gradient_standby,
            "inference": self.inference,
        }


def select_four_idle_gpus(
    policy: FourGPURolePolicy,
    observations: Sequence[GPUObservation],
) -> FourGPURoleAssignment:
    """Select four process-idle GPUs, preferring low memory use then low index."""

    validate_four_gpu_roles(policy, observations)
    assignment = FourGPURoleAssignment(*policy.assigned_physical_indices)
    return assignment


def validate_four_gpu_roles(
    policy: FourGPURolePolicy,
    observations: Sequence[GPUObservation],
) -> None:
    """Require one serving GPU and three distinct, idle approved training GPUs."""

    if not isinstance(policy, FourGPURolePolicy):
        raise TypeError("Protocol 10 GPU validation requires the four-role policy")
    by_index: dict[int, GPUObservation] = {}
    for observation in observations:
        if observation.physical_index in by_index:
            raise ValueError("GPU observations contain duplicate physical indices")
        by_index[observation.physical_index] = observation
    expected_indices = set(range(8))
    if set(by_index) != expected_indices:
        raise RuntimeError("Protocol 10 preflight requires physical GPUs 0 through 7")
    if not (set(policy.assigned_physical_indices) | set(policy.forbidden_physical_indices)) <= (
        expected_indices
    ):
        raise RuntimeError("hardware policy refers to a nonexistent physical GPU")
    inference = by_index[policy.inference_physical_index]
    if not inference.compute_pids:
        raise RuntimeError("Protocol 10 inference GPU has no serving compute process")
    for index in (policy.coordinator_physical_index, policy.gradient_primary_physical_index):
        observation = by_index[index]
        if not observation.is_launchable_idle:
            raise RuntimeError(
                f"Protocol 10 training GPU {index} is unavailable or has a compute process"
            )
        if observation.memory_free_mib < policy.minimum_free_memory_mib:
            raise RuntimeError(f"Protocol 10 training GPU {index} lacks the memory reserve")
    standby = by_index[policy.gradient_standby_physical_index]
    if not standby.is_launchable_idle:
        raise RuntimeError("Protocol 10 cold standby GPU is unavailable or has a compute process")
    if standby.memory_free_mib < policy.standby_minimum_free_memory_mib:
        raise RuntimeError("Protocol 10 cold standby GPU lacks the memory reserve")


def validate_assignment_is_idle(
    assignment: FourGPURoleAssignment,
    observations: Sequence[GPUObservation],
    *,
    minimum_free_memory_mib: int,
    standby_minimum_free_memory_mib: int | None = None,
    standby_may_share: bool = False,
) -> None:
    """Revalidate an explicit mapping immediately before launching processes."""

    by_index = {item.physical_index: item for item in observations}
    for index in (assignment.coordinator, assignment.gradient_primary):
        observation = by_index.get(index)
        if observation is None:
            raise ValueError(f"no preflight observation for assigned GPU {index}")
        if not observation.is_launchable_idle:
            raise RuntimeError(f"assigned GPU {index} is unavailable or has a compute process")
        if observation.memory_free_mib < minimum_free_memory_mib:
            raise RuntimeError(f"assigned GPU {index} lacks the required free memory")
    if assignment.gradient_standby is None:
        return
    standby = by_index.get(assignment.gradient_standby)
    if standby is None:
        raise ValueError(f"no preflight observation for assigned GPU {assignment.gradient_standby}")
    required = (
        minimum_free_memory_mib
        if standby_minimum_free_memory_mib is None
        else standby_minimum_free_memory_mib
    )
    if standby.memory_free_mib < required:
        raise RuntimeError("assigned standby GPU lacks the fallback memory reserve")
    if not standby.is_driver_available:
        raise RuntimeError("assigned standby GPU is unavailable to the driver")
    if not standby.is_process_idle and not standby_may_share:
        raise RuntimeError("assigned standby GPU is unavailable or has a compute process")


__all__ = [
    "FOUR_GPU_ROLE_FORMAT",
    "THREE_GPU_ROLE_FORMAT",
    "FourGPURoleAssignment",
    "FourGPURolePolicy",
    "GPUObservation",
    "ThreeGPURolePolicy",
    "select_four_idle_gpus",
    "validate_assignment_is_idle",
    "validate_four_gpu_roles",
    "validate_three_gpu_roles",
]
