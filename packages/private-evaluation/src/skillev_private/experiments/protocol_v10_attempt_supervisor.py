"""Single-topology torchrun supervisor for one Protocol 10 method attempt."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from skillev.experiments import FormalMethodV10
from skillev.runtime.formal_preflight import collect_gpu_observations
from skillev.runtime.formal_storage import FormalStorageBinding, validate_formal_storage
from skillev.runtime.gpu_topology import (
    GPUObservation,
    ThreeGPURolePolicy,
    validate_three_gpu_roles,
)

from .protocol_v10_attempt_input import ProtocolV10AdmissionMode, ProtocolV10AttemptInput
from .protocol_v10_launch_package import ProtocolV10LaunchPackage

PROTOCOL_V10_ATTEMPT_WORKER_MODULE = "skillev_private.experiments.protocol_v10_attempt_worker"


@dataclass(frozen=True, slots=True)
class ProtocolV10TorchrunLaunch:
    """One explicit process topology; the inference GPU is never exposed."""

    method: FormalMethodV10
    exact_input: Path
    hardware: ThreeGPURolePolicy
    storage: FormalStorageBinding
    worker_module: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, FormalMethodV10):
            raise TypeError("Protocol 10 launch requires a formal method")
        if not self.exact_input.is_absolute():
            raise ValueError("Protocol 10 exact input path must be absolute")
        if not isinstance(self.hardware, ThreeGPURolePolicy):
            raise TypeError("Protocol 10 launch requires the three-role hardware policy")
        if not isinstance(self.storage, FormalStorageBinding):
            raise TypeError("Protocol 10 launch requires the formal storage binding")
        if not self.worker_module.strip():
            raise ValueError("Protocol 10 worker module cannot be empty")

    @property
    def training_physical_indices(self) -> tuple[int, ...]:
        return (
            self.hardware.coordinator_physical_index,
            self.hardware.gradient_primary_physical_index,
        )

    @property
    def world_size(self) -> int:
        return len(self.training_physical_indices)

    def environment(
        self,
        base: dict[str, str] | None = None,
        *,
        observations: Sequence[GPUObservation] | None = None,
    ) -> dict[str, str]:
        environment = dict(os.environ if base is None else base)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        observed = collect_gpu_observations() if observations is None else observations
        validate_three_gpu_roles(self.hardware, observed)
        visible_uuids: list[str] = []
        for index in self.training_physical_indices:
            matches = tuple(
                item for item in observed if getattr(item, "physical_index", None) == index
            )
            if len(matches) != 1:
                raise RuntimeError("Protocol 10 physical GPU observation is ambiguous")
            uuid = getattr(matches[0], "uuid", None)
            if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
                raise RuntimeError("Protocol 10 physical GPU UUID is unavailable")
            visible_uuids.append(uuid)
        inference_matches = tuple(
            item
            for item in observed
            if item.physical_index == self.hardware.inference_physical_index
        )
        if len(inference_matches) != 1 or not inference_matches[0].uuid.startswith("GPU-"):
            raise RuntimeError("Protocol 10 inference GPU UUID is unavailable")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible_uuids)
        environment["SKILLEV_DISABLE_GRADIENT_STANDBY"] = "1"
        environment["SKILLEV_EXTERNAL_INFERENCE_GPU_UUID"] = inference_matches[0].uuid
        environment["SKILLEV_TRAINING_GPU_MAP"] = ",".join(
            str(index) for index in self.training_physical_indices
        )
        environment["SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES"] = ",".join(visible_uuids)
        environment["SKILLEV_PROTOCOL_V10_METHOD"] = self.method.value
        return environment

    def command(self) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={self.world_size}",
            "-m",
            self.worker_module,
            str(self.exact_input),
        )

    @classmethod
    def from_exact_input(
        cls,
        exact_input: Path,
        *,
        worker_module: str = PROTOCOL_V10_ATTEMPT_WORKER_MODULE,
    ) -> ProtocolV10TorchrunLaunch:
        """Derive method and physical GPU mapping from the one private input."""

        loaded = ProtocolV10AttemptInput.read(exact_input)
        if loaded.admission_mode is ProtocolV10AdmissionMode.FORMAL:
            raise ValueError("formal Protocol 10 launches require the frozen launch package")
        return cls(
            method=loaded.method,
            exact_input=exact_input,
            hardware=loaded.hardware,
            storage=loaded.storage,
            worker_module=worker_module,
        )

    @classmethod
    def from_launch_package(
        cls,
        package_path: Path,
        *,
        method: FormalMethodV10,
        worker_module: str = PROTOCOL_V10_ATTEMPT_WORKER_MODULE,
    ) -> ProtocolV10TorchrunLaunch:
        package = ProtocolV10LaunchPackage.read(package_path)
        package.require_exact_inputs_unchanged()
        matches = tuple(slot for slot in package.slots if slot.method is method)
        if len(matches) != 1:
            raise ValueError("formal launch package has no unique method slot")
        slot = matches[0]
        loaded = ProtocolV10AttemptInput.read(slot.exact_input_path)
        if (
            loaded.method is not method
            or loaded.admission_mode is not ProtocolV10AdmissionMode.FORMAL
        ):
            raise ValueError("formal launch slot differs from its exact input")
        if (
            loaded.hardware != package.hardware
            or loaded.attempt_root != slot.attempt_root
            or loaded.sglang.adapter_namespace != slot.adapter_namespace
        ):
            raise ValueError("formal launch package differs from its deployment input")
        return cls(
            method=method,
            exact_input=slot.exact_input_path,
            hardware=loaded.hardware,
            storage=loaded.storage,
            worker_module=worker_module,
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10TorchrunSupervisor:
    """Run the fixed coordinator/gradient topology exactly once."""

    launch: ProtocolV10TorchrunLaunch
    process_runner: Callable[[ProtocolV10TorchrunLaunch], int] | None = None

    def run(self) -> None:
        validate_formal_storage(self.launch.storage)
        return_code = self._run(self.launch)
        if return_code != 0:
            raise RuntimeError("Protocol 10 distributed attempt failed")

    def _run(self, launch: ProtocolV10TorchrunLaunch) -> int:
        if self.process_runner is not None:
            return self.process_runner(launch)
        completed = subprocess.run(  # noqa: S603 - closed module and argument vector
            launch.command(),
            env=launch.environment(),
            check=False,
        )
        return completed.returncode


def _require_clean_source_commit(expected: str) -> None:
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    dirty = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    if commit != expected or dirty:
        raise RuntimeError("formal Protocol 10 launch requires its clean frozen source commit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-package", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=tuple(method.value for method in FormalMethodV10),
        required=True,
    )
    arguments = parser.parse_args()
    package_path = arguments.launch_package.resolve()
    package = ProtocolV10LaunchPackage.read(package_path)
    _require_clean_source_commit(package.source_commit)
    launch = ProtocolV10TorchrunLaunch.from_launch_package(
        package_path,
        method=FormalMethodV10(arguments.method),
    )
    ProtocolV10TorchrunSupervisor(launch).run()


if __name__ == "__main__":
    main()


__all__ = [
    "PROTOCOL_V10_ATTEMPT_WORKER_MODULE",
    "ProtocolV10TorchrunLaunch",
    "ProtocolV10TorchrunSupervisor",
]
