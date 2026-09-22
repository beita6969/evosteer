"""Private measurement of the implementation build a formal child executes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue
from skillev.experiments import ExecutionHardwareIdentity, ImplementationBuildIdentity
from skillev.experiments.build_identity import read_source_package_provenance
from skillev.runtime import AttemptDomainError, AttemptFailureCode, AttemptFailureStage

PRIVATE_IMPLEMENTATION_BUILD_DEPLOYMENT_FORMAT = "skillev-private-implementation-build@1"


def _absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return value


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _installed_version(distribution: str) -> str:
    value = importlib.metadata.version(distribution)
    if not value:
        raise ValueError(f"installed distribution {distribution} has no version")
    return value


def _require_executed_package_roots(source_tree_root: Path) -> None:
    """Reject a source measurement that is not the code the child imported."""

    expected_roots = {
        "skillev": source_tree_root / "src" / "skillev",
        "skillev_private": source_tree_root
        / "packages"
        / "private-evaluation"
        / "src"
        / "skillev_private",
    }
    for package, expected in expected_roots.items():
        module = importlib.import_module(package)
        location = getattr(module, "__file__", None)
        if not isinstance(location, str):
            raise ValueError(f"executing package {package} has no file location")
        actual = Path(location).resolve().parent
        if actual != expected.resolve():
            raise ValueError(f"executing package {package} differs from formal source tree")


def _backend_flags() -> tuple[bool, bool, bool, bool]:
    """Read the effective runtime flags rather than the requested settings."""

    import torch

    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    if any(
        type(value) is not bool
        for value in (
            deterministic_algorithms,
            cudnn_deterministic,
            cudnn_benchmark,
            matmul_allow_tf32,
        )
    ):
        raise TypeError("torch backend flags must be booleans")
    return (
        deterministic_algorithms,
        cudnn_deterministic,
        cudnn_benchmark,
        matmul_allow_tf32,
    )


def _nvidia_driver_version() -> str:
    """Measure the driver exposed to the current CUDA job without defaults."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("nvidia-smi")
    completed = subprocess.run(  # noqa: S603 -- executable is resolved by shutil.which
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise ValueError("CUDA allocation does not expose one NVIDIA driver version")
    return versions.pop()


def _nccl_version() -> str:
    """Return the concrete NCCL ABI selected by the executing torch build."""

    import torch

    nccl = getattr(torch.cuda, "nccl", None)
    version = nccl.version() if nccl is not None else None
    if (
        not isinstance(version, tuple)
        or not version
        or any(type(item) is not int or item < 0 for item in version)
    ):
        raise ValueError("CUDA runtime does not expose a concrete NCCL version")
    return ".".join(str(item) for item in version)


def measure_formal_execution_hardware() -> ExecutionHardwareIdentity:
    """Measure the CUDA environment required by a formal benchmark child.

    Formal benchmark execution is GPU-only.  A missing CUDA runtime, driver,
    cuDNN, NCCL, or safetensors installation is terminal rather than a reason
    to silently run a different CPU/backend configuration.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("formal benchmark worker requires an available CUDA device")
        device_count = torch.cuda.device_count()
        if type(device_count) is not int or device_count < 1:
            raise RuntimeError("formal benchmark worker has no visible CUDA devices")
        device_index = torch.cuda.current_device()
        if type(device_index) is not int or not 0 <= device_index < device_count:
            raise RuntimeError("formal benchmark worker has an invalid current CUDA device")
        properties = torch.cuda.get_device_properties(device_index)
        accelerator_name = getattr(properties, "name", None)
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        cuda_runtime_version = torch.version.cuda
        cudnn_version = cast(Callable[[], int | None], torch.backends.cudnn.version)()
        if (
            type(accelerator_name) is not str
            or not accelerator_name.strip()
            or type(major) is not int
            or type(minor) is not int
            or type(cuda_runtime_version) is not str
            or not cuda_runtime_version
            or type(cudnn_version) is not int
            or cudnn_version < 1
        ):
            raise ValueError("CUDA runtime did not expose a complete hardware identity")
        return ExecutionHardwareIdentity(
            accelerator_name=accelerator_name,
            compute_capability=f"{major}.{minor}",
            visible_device_count=device_count,
            nvidia_driver_version=_nvidia_driver_version(),
            cuda_runtime_version=cuda_runtime_version,
            cudnn_version=str(cudnn_version),
            nccl_version=_nccl_version(),
            kernel_release=platform.release(),
            safetensors_version=_installed_version("safetensors"),
        )
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        raise FormalExecutionHardwareVerificationError(str(error)) from error


class FormalImplementationBuildVerificationError(AttemptDomainError):
    """The formal child is not executing the build named by its exact input."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.INVALID_EXACT_INPUT,
            stage=AttemptFailureStage.BUILD,
            private_detail=private_detail,
        )


class FormalExecutionHardwareVerificationError(AttemptDomainError):
    """The formal child lacks one required, reportable CUDA runtime fact."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.INVALID_EXACT_INPUT,
            stage=AttemptFailureStage.BUILD,
            private_detail=private_detail,
        )


@dataclass(frozen=True, slots=True)
class PrivateImplementationBuildDeployment:
    """Private locations used to measure a path-free formal build identity."""

    source_tree_root: Path
    public_wheel_path: Path
    private_evaluation_wheel_path: Path
    lockfile_path: Path
    format: str = PRIVATE_IMPLEMENTATION_BUILD_DEPLOYMENT_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "source_tree_root",
            "public_wheel_path",
            "private_evaluation_wheel_path",
            "lockfile_path",
        ):
            _absolute_path(getattr(self, field), field=field)
        if self.format != PRIVATE_IMPLEMENTATION_BUILD_DEPLOYMENT_FORMAT:
            raise ValueError("unsupported private implementation build deployment format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "lockfile_path": self.lockfile_path.as_posix(),
            "private_evaluation_wheel_path": self.private_evaluation_wheel_path.as_posix(),
            "public_wheel_path": self.public_wheel_path.as_posix(),
            "source_tree_root": self.source_tree_root.as_posix(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateImplementationBuildDeployment:
        if not isinstance(value, dict):
            raise TypeError("private implementation build deployment must be an object")
        fields = {
            "format",
            "lockfile_path",
            "private_evaluation_wheel_path",
            "public_wheel_path",
            "source_tree_root",
        }
        if set(value) != fields or any(type(value[field]) is not str for field in fields):
            raise ValueError("private implementation build deployment has incompatible fields")
        return cls(
            source_tree_root=Path(value["source_tree_root"]),
            public_wheel_path=Path(value["public_wheel_path"]),
            private_evaluation_wheel_path=Path(value["private_evaluation_wheel_path"]),
            lockfile_path=Path(value["lockfile_path"]),
            format=value["format"],
        )

    def measure(self) -> ImplementationBuildIdentity:
        """Measure code, package files, versions, and effective backend flags."""

        source_tree_root = self.source_tree_root.resolve()
        if not source_tree_root.is_dir():
            raise NotADirectoryError(source_tree_root)
        _require_executed_package_roots(source_tree_root)
        flags = _backend_flags()
        provenance = read_source_package_provenance(source_tree_root)
        return ImplementationBuildIdentity(
            source_commit=provenance.source_commit,
            source_tree_hash=provenance.source_tree_hash,
            public_wheel_hash=_sha256_file(self.public_wheel_path),
            private_evaluation_wheel_hash=_sha256_file(self.private_evaluation_wheel_path),
            lockfile_hash=_sha256_file(self.lockfile_path),
            python_version=platform.python_version(),
            torch_version=_installed_version("torch"),
            transformers_version=_installed_version("transformers"),
            peft_version=_installed_version("peft"),
            tokenizers_version=_installed_version("tokenizers"),
            deterministic_algorithms=flags[0],
            cudnn_deterministic=flags[1],
            cudnn_benchmark=flags[2],
            matmul_allow_tf32=flags[3],
        )

    def require_exact(
        self,
        expected: ImplementationBuildIdentity,
    ) -> ImplementationBuildIdentity:
        """Measure once and fail before model construction on any mismatch."""

        if not isinstance(expected, ImplementationBuildIdentity):
            raise TypeError("formal implementation build expectation is invalid")
        try:
            actual = self.measure()
        except (FileNotFoundError, ImportError, NotADirectoryError, ValueError) as error:
            raise FormalImplementationBuildVerificationError(str(error)) from error
        if actual != expected:
            raise FormalImplementationBuildVerificationError(
                "executing implementation build differs from the formal exact input"
            )
        return actual


__all__ = [
    "PRIVATE_IMPLEMENTATION_BUILD_DEPLOYMENT_FORMAT",
    "FormalExecutionHardwareVerificationError",
    "FormalImplementationBuildVerificationError",
    "PrivateImplementationBuildDeployment",
    "measure_formal_execution_hardware",
]
