"""Bounded on-demand Apptainer materialization for official SWE OCI images."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from skillev.contracts import stable_hash

from .swebench_official import (
    MaterializedSWEImage,
    PinnedSWEOCIImage,
    SWEHarnessInfrastructureError,
)

MATERIALIZER_POLICY_VERSION = "skillev-apptainer-oci-materializer@2"
_LEASE_DIRECTORY = "leases"
_CACHE_DIRECTORY = "cache"
_TEMP_DIRECTORY = "tmp"
_WORK_DIRECTORY = "work"
_PARTIAL_IMAGE = "image.partial.sif"
_CACHED_IMAGE = "image.sif"
_CACHE_POLICY = "single-entry-process-local@1"
# A recursive footprint walk is intentionally much less frequent than process
# polling: on host-mounted filesystems, walking a growing OCI extraction tree
# every 10 ms can dominate the materialization itself.  Capacity is still
# checked while the process runs and once more before the SIF is published.
_FOOTPRINT_POLL_SECONDS = 1.0
_PROCESS_STOP_SECONDS = 1.0
_SUPPORTED_PLATFORM = ("linux", "amd64")


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    normalized = float(value)
    if not normalized > 0.0 or not normalized < float("inf"):
        raise ValueError(f"{field_name} must be positive and finite")
    return normalized


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, f"sha256:{digest.hexdigest()}"


def _existing_executable(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("apptainer_executable must be an absolute Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("apptainer_executable must be an executable file")
    return resolved


def _existing_storage_root(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("storage_root must be an absolute Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("storage_root must be an existing directory")
    return resolved


def _tree_file_bytes(root: Path) -> int:
    """Return logical bytes for regular files below the controlled root."""

    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=_PROCESS_STOP_SECONDS)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


@dataclass(frozen=True, slots=True)
class MaterializerProcessResult:
    returncode: int

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("materializer process return code must be an integer")


class MaterializerProcessExecutor(Protocol):
    """Small no-shell subprocess seam for deterministic focused tests."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        footprint_root: Path,
        capacity_bytes: int,
        timeout_seconds: float,
    ) -> MaterializerProcessResult: ...


@dataclass(frozen=True, slots=True)
class SubprocessMaterializerProcessExecutor:
    """Execute an absolute Apptainer argv without a shell."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        footprint_root: Path,
        capacity_bytes: int,
        timeout_seconds: float,
    ) -> MaterializerProcessResult:
        try:
            process = subprocess.Popen(  # noqa: S603 - executable is deployment-pinned
                tuple(argv),
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            raise SWEHarnessInfrastructureError(
                "Apptainer OCI materialization process could not complete"
            ) from error
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                if _tree_file_bytes(footprint_root) > capacity_bytes:
                    _stop_process(process)
                    raise SWEHarnessInfrastructureError(
                        "Apptainer OCI materialization exceeded its capacity while running"
                    )
                returncode = process.poll()
                if returncode is not None:
                    return MaterializerProcessResult(returncode)
                if time.monotonic() >= deadline:
                    _stop_process(process)
                    raise SWEHarnessInfrastructureError("Apptainer OCI materialization timed out")
                time.sleep(_FOOTPRINT_POLL_SECONDS)
        except BaseException:
            _stop_process(process)
            raise


@dataclass(slots=True)
class ApptainerOCIImageMaterializer:
    """Materialize one digest-pinned OCI image under one enforced lease budget.

    The provider intentionally has no tag-only or externally supplied SIF
    branch.  It materializes a digest-qualified OCI source and keeps one SIF for
    sequential requests to the same official environment.  Switching images
    atomically replaces that process-local cache; per-pull cache/temp/work trees
    are always removed when their lease exits.
    """

    apptainer_executable: Path
    apptainer_version: str
    storage_root: Path
    capacity_bytes: int
    executor: MaterializerProcessExecutor = field(
        default_factory=SubprocessMaterializerProcessExecutor,
        repr=False,
    )
    timeout_seconds: float = 1800.0
    _materializer_id: str = field(init=False, repr=False)
    _executable_size_bytes: int = field(init=False, repr=False)
    _executable_sha256: str = field(init=False, repr=False)
    _lease_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _lease_active: bool = field(default=False, init=False, repr=False)
    _cached_identity: tuple[str, str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        executable = _existing_executable(self.apptainer_executable)
        version = _text(self.apptainer_version, field_name="apptainer_version")
        root = _existing_storage_root(self.storage_root)
        capacity = _positive_int(self.capacity_bytes, field_name="capacity_bytes")
        timeout = _positive_float(self.timeout_seconds, field_name="timeout_seconds")
        if not callable(getattr(self.executor, "execute", None)):
            raise TypeError("materializer executor must implement execute")
        executable_size, executable_sha = _sha256_file(executable)
        materializer_id = stable_hash(
            {
                "apptainer": {
                    "sha256": executable_sha,
                    "size_bytes": executable_size,
                    "version": version,
                },
                "capacity_bytes": capacity,
                "cache_policy": _CACHE_POLICY,
                "max_active_leases": 1,
                "footprint_poll_seconds": _FOOTPRINT_POLL_SECONDS,
                "policy_version": MATERIALIZER_POLICY_VERSION,
                "source_transport": "docker",
                "timeout_seconds": timeout,
            }
        )
        object.__setattr__(self, "apptainer_executable", executable)
        object.__setattr__(self, "apptainer_version", version)
        object.__setattr__(self, "storage_root", root)
        object.__setattr__(self, "capacity_bytes", capacity)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "_executable_size_bytes", executable_size)
        object.__setattr__(self, "_executable_sha256", executable_sha)
        object.__setattr__(self, "_materializer_id", materializer_id)

    @property
    def materializer_id(self) -> str:
        return self._materializer_id

    @property
    def max_active_leases(self) -> int:
        return 1

    @contextmanager
    def lease(self, image: PinnedSWEOCIImage) -> Iterator[MaterializedSWEImage]:
        if not isinstance(image, PinnedSWEOCIImage):
            raise TypeError("Apptainer materializer requires a pinned OCI image")
        if (image.platform_os, image.platform_architecture) != _SUPPORTED_PLATFORM:
            raise ValueError("Apptainer SWE materializer only supports linux/amd64 OCI images")
        with self._lease_lock:
            if self._lease_active:
                raise SWEHarnessInfrastructureError(
                    "Apptainer OCI materializer already has an active lease"
                )
            self._lease_active = True

        lease_directory: Path | None = None
        try:
            self._verify_executable()
            cached_path = self.storage_root / _CACHED_IMAGE
            cached_identity = (image.environment_image_id, image.manifest_digest)
            if self._cached_identity == cached_identity and cached_path.is_file():
                yield MaterializedSWEImage(
                    environment_image_id=image.environment_image_id,
                    manifest_digest=image.manifest_digest,
                    image_path=cached_path,
                )
                return
            if self._cached_identity is not None and not cached_path.is_file():
                self._cached_identity = None
            if _tree_file_bytes(self.storage_root) >= self.capacity_bytes:
                raise SWEHarnessInfrastructureError(
                    "Apptainer OCI materializer capacity is already exhausted"
                )
            leases_root = self.storage_root / _LEASE_DIRECTORY
            leases_root.mkdir(mode=0o700, exist_ok=True)
            lease_directory = Path(tempfile.mkdtemp(prefix="lease-", dir=leases_root)).resolve(
                strict=True
            )
            cache_root = lease_directory / _CACHE_DIRECTORY
            temp_root = lease_directory / _TEMP_DIRECTORY
            work_root = lease_directory / _WORK_DIRECTORY
            for directory in (cache_root, temp_root, work_root):
                directory.mkdir(mode=0o700)
            partial_path = lease_directory / _PARTIAL_IMAGE
            environment = dict(os.environ)
            environment.update(
                {
                    "APPTAINER_CACHEDIR": str(cache_root),
                    "APPTAINER_TMPDIR": str(temp_root),
                    "TMPDIR": str(temp_root),
                }
            )
            argv = (
                str(self.apptainer_executable),
                "pull",
                "--disable-cache",
                "--arch",
                image.platform_architecture,
                str(partial_path),
                image.immutable_source_reference,
            )
            result = self.executor.execute(
                argv,
                cwd=work_root,
                environment=environment,
                footprint_root=self.storage_root,
                capacity_bytes=self.capacity_bytes,
                timeout_seconds=self.timeout_seconds,
            )
            if not isinstance(result, MaterializerProcessResult):
                raise SWEHarnessInfrastructureError(
                    "Apptainer materializer executor returned an incompatible result"
                )
            if result.returncode != 0:
                raise SWEHarnessInfrastructureError(
                    "Apptainer OCI materialization exited unsuccessfully"
                )
            if not partial_path.is_file():
                raise SWEHarnessInfrastructureError(
                    "Apptainer OCI materialization did not produce a SIF"
                )
            if _tree_file_bytes(self.storage_root) > self.capacity_bytes:
                raise SWEHarnessInfrastructureError(
                    "Apptainer OCI materialization exceeded its capacity"
                )
            os.replace(partial_path, cached_path)
            self._cached_identity = cached_identity
            yield MaterializedSWEImage(
                environment_image_id=image.environment_image_id,
                manifest_digest=image.manifest_digest,
                image_path=cached_path,
            )
        finally:
            if lease_directory is not None:
                shutil.rmtree(lease_directory, ignore_errors=False)
            with self._lease_lock:
                self._lease_active = False

    def _verify_executable(self) -> None:
        try:
            size, digest = _sha256_file(self.apptainer_executable)
        except OSError as error:
            raise SWEHarnessInfrastructureError(
                "pinned Apptainer executable could not be read"
            ) from error
        if size != self._executable_size_bytes or digest != self._executable_sha256:
            raise SWEHarnessInfrastructureError(
                "Apptainer executable bytes differ from the materializer identity"
            )


__all__ = [
    "MATERIALIZER_POLICY_VERSION",
    "ApptainerOCIImageMaterializer",
    "MaterializerProcessExecutor",
    "MaterializerProcessResult",
    "SubprocessMaterializerProcessExecutor",
]
