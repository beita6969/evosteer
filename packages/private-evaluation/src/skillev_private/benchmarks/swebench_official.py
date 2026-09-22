"""Pinned Apptainer boundary for the official SWE-bench Verified harness.

The model-facing package knows only the lightweight workspace protocol.  This
private module binds that protocol to an immutable OCI manifest, a bounded
temporary-SIF materializer, and a pinned host-side worker mounted read-only.
The worker owns repository checkout, patch application, and official test
execution; no Docker client or hidden verifier truth crosses into the model
process.

The wire protocol is deliberately one request per subprocess.  A non-zero exit,
timeout, malformed response, or identity mismatch is evaluator infrastructure
failure.  None of those conditions is converted into an unresolved benchmark
outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from skillev.benchmarks.swebench import (
    SWEBENCH_VERIFIED_BENCHMARK_ID,
    SWEBenchVerifiedPublicCase,
    SWEWorkspaceBackend,
    SWEWorkspaceCommand,
    SWEWorkspacePublicStep,
)
from skillev.benchmarks.task_family import require_benchmark_task_family
from skillev.contracts import JsonValue, canonical_json_bytes, normalize_json, stable_hash
from skillev.runtime import BudgetVector

from .swebench import (
    OfficialSWEVerifierBackend,
    PrivateSWEBenchVerifiedCase,
    PrivateSWEVerifiedTruth,
    SWEVerifierRequest,
    SWEVerifierResult,
)
from .swebench_grader import (
    PinnedSWEOfficialGrader,
    SWEHarnessInfrastructureError,
    SWEOfficialGrader,
    _git_revision,
    _pinned_git_source,
)

SWE_HARNESS_PROTOCOL_VERSION = "skillev-swebench-official-harness@1"
_CONTAINER_WORK_ROOT = "/skillev/work"
_REVISION_LENGTH = 40
_DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
SWEBENCH_VERIFIED_ROW_FIELDS = frozenset(
    {
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "base_commit",
        "created_at",
        "difficulty",
        "environment_setup_commit",
        "hints_text",
        "instance_id",
        "patch",
        "problem_statement",
        "repo",
        "test_patch",
        "version",
    }
)


class SWEHarnessOperation(StrEnum):
    WORKSPACE_INITIALIZE = "workspace-initialize"
    WORKSPACE_EXECUTE = "workspace-execute"
    VERIFY = "verify"


def _text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} has invalid text")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive and finite")
    return normalized


def _source_revision(value: object, *, field_name: str) -> str:
    revision = _text(value, field_name=field_name)
    if len(revision) != _REVISION_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{field_name} must be a full lowercase Git commit")
    return revision


def _test_identities(value: object, *, field_name: str) -> tuple[str, ...]:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a JSON-encoded string")
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name} is not valid JSON") from error
    if type(decoded) is not list:
        raise TypeError(f"{field_name} must encode a JSON array")
    tests = cast(list[object], decoded)
    if any(type(test) is not str or not test.strip() or "\x00" in test for test in tests):
        raise ValueError(f"{field_name} contains an invalid test identity")
    return tuple(cast(list[str], tests))


def _existing_absolute_path(value: Path, *, field_name: str, directory: bool) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{field_name} must be an absolute Path")
    resolved = value.resolve(strict=True)
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field_name} must identify an existing {kind}")
    return resolved


def _container_executable(value: object) -> str:
    text = _text(value, field_name="worker_container_path")
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError("worker_container_path must be an absolute normalized POSIX path")
    return text


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, f"sha256:{digest.hexdigest()}"


def _sha256_identifier(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a canonical SHA-256 identifier")
    return value


def _result_object(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> dict[str, JsonValue]:
    try:
        normalized = normalize_json(value)
    except (TypeError, ValueError) as error:
        raise SWEHarnessInfrastructureError(f"{label} is not normalized JSON") from error
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise SWEHarnessInfrastructureError(f"{label} has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class PinnedSWEOCIImage:
    """One immutable official OCI source for one public SWE instance.

    ``source_reference`` preserves the official registry/tag provenance while
    ``manifest_digest`` is the execution authority.  The materializer must
    always use :attr:`immutable_source_reference`, never the mutable tag alone.
    """

    environment_image_id: str
    source_reference: str
    manifest_digest: str
    platform_os: str = "linux"
    platform_architecture: str = "amd64"

    def __post_init__(self) -> None:
        environment_image_id = _text(
            self.environment_image_id,
            field_name="environment_image_id",
        )
        source = _text(self.source_reference, field_name="source_reference")
        if (
            not source.startswith("docker://")
            or "@" in source
            or any(character.isspace() for character in source)
        ):
            raise ValueError("source_reference must be a tag-addressed docker:// OCI reference")
        tag_separator = source.rfind(":")
        if tag_separator <= source.rfind("/"):
            raise ValueError("source_reference must include an explicit OCI tag")
        digest = _sha256_identifier(self.manifest_digest, field_name="manifest_digest")
        if environment_image_id != f"{source}@{digest}":
            raise ValueError("environment_image_id must equal the immutable OCI source reference")
        _text(self.platform_os, field_name="platform_os")
        _text(self.platform_architecture, field_name="platform_architecture")

    @property
    def immutable_source_reference(self) -> str:
        # Apptainer rejects Docker references containing both a tag and a
        # digest.  The tag remains persisted as provenance, while execution
        # uses the equivalent registry/repository@digest authority.
        tag_separator = self.source_reference.rfind(":")
        repository = self.source_reference[:tag_separator]
        return f"{repository}@{self.manifest_digest}"


@dataclass(frozen=True, slots=True)
class MaterializedSWEImage:
    """One temporary local SIF leased from an immutable OCI source."""

    environment_image_id: str
    manifest_digest: str
    image_path: Path = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.environment_image_id, field_name="environment_image_id")
        _sha256_identifier(self.manifest_digest, field_name="manifest_digest")
        image_path = _existing_absolute_path(
            self.image_path,
            field_name="image_path",
            directory=False,
        )
        object.__setattr__(self, "image_path", image_path)


class SWEImageMaterializer(Protocol):
    """Bounded lease provider for temporary SIF materializations.

    Implementations must enforce ``capacity_bytes`` across the complete
    temporary/cache footprint.  The production deployment fixes
    ``max_active_leases`` to one because its worker owns a single mounted work
    root.  A lease remains live through process completion and releases its SIF
    on context exit according to the materializer's explicit cache policy.
    """

    @property
    def materializer_id(self) -> str: ...

    @property
    def capacity_bytes(self) -> int: ...

    @property
    def max_active_leases(self) -> int: ...

    def lease(self, image: PinnedSWEOCIImage) -> AbstractContextManager[MaterializedSWEImage]: ...


@dataclass(frozen=True, slots=True)
class PinnedSWEHarnessDeployment:
    """Exact launcher, worker, OCI sources, and bounded materializer."""

    apptainer_executable: Path = field(repr=False)
    worker_host_path: Path = field(repr=False)
    worker_container_path: str
    worker_size_bytes: int
    worker_sha256: str
    harness_revision: str
    harness_source_root: Path = field(repr=False)
    work_root: Path = field(repr=False)
    images: tuple[PinnedSWEOCIImage, ...]
    materializer: SWEImageMaterializer = field(repr=False, compare=False)
    request_timeout_seconds: float = 1800.0
    max_request_bytes: int = _DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES
    _materializer_id: str = field(init=False, repr=False)
    _materializer_capacity_bytes: int = field(init=False, repr=False)
    _materializer_max_active_leases: int = field(init=False, repr=False)
    _apptainer_size_bytes: int = field(init=False, repr=False)
    _apptainer_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        apptainer = _existing_absolute_path(
            self.apptainer_executable,
            field_name="apptainer_executable",
            directory=False,
        )
        if not os.access(apptainer, os.X_OK):
            raise ValueError("apptainer_executable must be executable")
        apptainer_size, apptainer_sha = _sha256_file(apptainer)
        work_root = _existing_absolute_path(
            self.work_root,
            field_name="work_root",
            directory=True,
        )
        if any(marker in str(work_root) for marker in (",", ":")):
            raise ValueError("work_root cannot contain Apptainer bind separators")
        worker_host = _existing_absolute_path(
            self.worker_host_path,
            field_name="worker_host_path",
            directory=False,
        )
        if any(marker in str(worker_host) for marker in (",", ":")):
            raise ValueError("worker_host_path cannot contain Apptainer bind separators")
        if not os.access(worker_host, os.X_OK):
            raise ValueError("worker_host_path must be executable")
        worker_container = _container_executable(self.worker_container_path)
        _positive_int(self.worker_size_bytes, field_name="worker_size_bytes")
        worker_sha = _sha256_identifier(self.worker_sha256, field_name="worker_sha256")
        actual_worker_size, actual_worker_sha = _sha256_file(worker_host)
        if actual_worker_size != self.worker_size_bytes or actual_worker_sha != worker_sha:
            raise ValueError("host-side SWE worker bytes differ from their pin")
        revision = _git_revision(self.harness_revision)
        harness_source = _pinned_git_source(
            self.harness_source_root,
            revision=revision,
        )
        timeout = _positive_float(
            self.request_timeout_seconds,
            field_name="request_timeout_seconds",
        )
        _positive_int(self.max_request_bytes, field_name="max_request_bytes")
        _positive_int(self.max_response_bytes, field_name="max_response_bytes")
        if not isinstance(self.images, tuple) or not self.images:
            raise ValueError("SWE harness deployment requires pinned images")
        if any(not isinstance(image, PinnedSWEOCIImage) for image in self.images):
            raise TypeError("SWE harness deployment images are incompatible")
        image_ids = tuple(image.environment_image_id for image in self.images)
        if image_ids != tuple(sorted(image_ids)):
            raise ValueError("SWE harness images must be ordered by environment_image_id")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("SWE harness image identities must be unique")
        if not callable(getattr(self.materializer, "lease", None)):
            raise TypeError("SWE image materializer must implement lease")
        materializer_id = _text(
            self.materializer.materializer_id,
            field_name="materializer_id",
        )
        materializer_capacity = _positive_int(
            self.materializer.capacity_bytes,
            field_name="materializer capacity_bytes",
        )
        materializer_max_leases = _positive_int(
            self.materializer.max_active_leases,
            field_name="materializer max_active_leases",
        )
        if materializer_max_leases != 1:
            raise ValueError("SWE image materializer max_active_leases must equal one")
        object.__setattr__(self, "apptainer_executable", apptainer)
        object.__setattr__(self, "worker_host_path", worker_host)
        object.__setattr__(self, "worker_container_path", worker_container)
        object.__setattr__(self, "worker_sha256", worker_sha)
        object.__setattr__(self, "harness_revision", revision)
        object.__setattr__(self, "harness_source_root", harness_source)
        object.__setattr__(self, "work_root", work_root)
        object.__setattr__(self, "request_timeout_seconds", timeout)
        object.__setattr__(self, "_apptainer_size_bytes", apptainer_size)
        object.__setattr__(self, "_apptainer_sha256", apptainer_sha)
        object.__setattr__(self, "_materializer_id", materializer_id)
        object.__setattr__(self, "_materializer_capacity_bytes", materializer_capacity)
        object.__setattr__(self, "_materializer_max_active_leases", materializer_max_leases)

    @property
    def deployment_id(self) -> str:
        return stable_hash(
            {
                "apptainer": {
                    "sha256": self._apptainer_sha256,
                    "size_bytes": self._apptainer_size_bytes,
                },
                "harness_revision": self.harness_revision,
                "harness_source": {
                    "revision": self.harness_revision,
                    "root": str(self.harness_source_root),
                },
                "images": [
                    {
                        "environment_image_id": image.environment_image_id,
                        "manifest_digest": image.manifest_digest,
                        "platform_architecture": image.platform_architecture,
                        "platform_os": image.platform_os,
                        "source_reference": image.source_reference,
                    }
                    for image in self.images
                ],
                "materializer": {
                    "capacity_bytes": self._materializer_capacity_bytes,
                    "materializer_id": self._materializer_id,
                    "max_active_leases": self._materializer_max_active_leases,
                },
                "protocol_version": SWE_HARNESS_PROTOCOL_VERSION,
                "transport": {
                    "max_request_bytes": self.max_request_bytes,
                    "max_response_bytes": self.max_response_bytes,
                    "request_timeout_seconds": self.request_timeout_seconds,
                },
                "worker": {
                    "container_path": self.worker_container_path,
                    "sha256": self.worker_sha256,
                    "size_bytes": self.worker_size_bytes,
                },
            }
        )

    def image_source(self, environment_image_id: str) -> PinnedSWEOCIImage:
        matches = tuple(
            image for image in self.images if image.environment_image_id == environment_image_id
        )
        if len(matches) != 1:
            raise ValueError("public SWE image has no unique pinned OCI source")
        return matches[0]

    def verify_materializer_identity(self) -> None:
        if (
            self.materializer.materializer_id != self._materializer_id
            or self.materializer.capacity_bytes != self._materializer_capacity_bytes
            or self.materializer.max_active_leases != self._materializer_max_active_leases
        ):
            raise SWEHarnessInfrastructureError(
                "SWE image materializer identity changed after deployment pinning"
            )


@dataclass(frozen=True, slots=True)
class PinnedSWEInstanceImageBinding:
    """Bind one official row identity to one immutable OCI image identity."""

    instance_id: str
    environment_image_id: str

    def __post_init__(self) -> None:
        _text(self.instance_id, field_name="instance_id")
        _text(self.environment_image_id, field_name="environment_image_id")


@dataclass(frozen=True, slots=True)
class OfficialSWEBenchVerifiedCaseConverter:
    """Convert the exact official Verified row without crossing the truth boundary."""

    deployment: PinnedSWEHarnessDeployment = field(repr=False)
    image_bindings: tuple[PinnedSWEInstanceImageBinding, ...]
    task_family: str
    max_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, PinnedSWEHarnessDeployment):
            raise TypeError("official SWE converter requires a pinned harness deployment")
        if not isinstance(self.image_bindings, tuple) or not self.image_bindings:
            raise ValueError("official SWE converter requires instance image bindings")
        if any(
            not isinstance(binding, PinnedSWEInstanceImageBinding)
            for binding in self.image_bindings
        ):
            raise TypeError("official SWE image binding has an incompatible type")
        instance_ids = tuple(binding.instance_id for binding in self.image_bindings)
        image_ids = tuple(binding.environment_image_id for binding in self.image_bindings)
        if instance_ids != tuple(sorted(instance_ids)):
            raise ValueError("official SWE image bindings must be ordered by instance_id")
        if len(set(instance_ids)) != len(instance_ids) or len(set(image_ids)) != len(image_ids):
            raise ValueError("official SWE image bindings must be one-to-one")
        for image_id in image_ids:
            self.deployment.image_source(image_id)
        _text(self.task_family, field_name="task_family")
        require_benchmark_task_family(
            benchmark_id=SWEBENCH_VERIFIED_BENCHMARK_ID,
            task_family=self.task_family,
        )
        _positive_int(self.max_steps, field_name="max_steps")

    def convert(
        self,
        row: Mapping[str, object],
        *,
        dataset_revision: str,
        split: str,
    ) -> PrivateSWEBenchVerifiedCase:
        if type(row) is not dict or set(row) != SWEBENCH_VERIFIED_ROW_FIELDS:
            raise ValueError("official SWE-bench Verified row has an incompatible field set")
        data = cast(dict[str, object], row)
        revision = _text(dataset_revision, field_name="dataset_revision")
        if split != "test":
            raise ValueError("official SWE-bench Verified source must use its test split")

        repo = _text(data["repo"], field_name="repo")
        if repo.count("/") != 1 or any(not part for part in repo.split("/")):
            raise ValueError("official SWE repository identity is invalid")
        instance_id = _text(data["instance_id"], field_name="instance_id")
        if not instance_id.startswith(f"{repo.replace('/', '__')}-"):
            raise ValueError("official SWE instance identity differs from its repository")
        matches = tuple(
            binding for binding in self.image_bindings if binding.instance_id == instance_id
        )
        if len(matches) != 1:
            raise ValueError("official SWE instance has no unique pinned image binding")
        image_id = matches[0].environment_image_id
        self.deployment.image_source(image_id)

        public = SWEBenchVerifiedPublicCase(
            dataset_revision=revision,
            split=split,
            instance_id=instance_id,
            repo=repo,
            version=_text(data["version"], field_name="version"),
            base_commit=_source_revision(data["base_commit"], field_name="base_commit"),
            problem_statement=_text(
                data["problem_statement"],
                field_name="problem_statement",
            ),
            environment_image_id=image_id,
            task_family=self.task_family,
            max_steps=self.max_steps,
        )
        # The remaining public metadata is validated as exact official text but
        # intentionally not projected into H_0.  In particular, patch/test
        # truth remains solely in the private case below.
        _text(data["hints_text"], field_name="hints_text", allow_empty=True)
        _text(data["created_at"], field_name="created_at")
        _source_revision(
            data["environment_setup_commit"],
            field_name="environment_setup_commit",
        )
        _text(data["difficulty"], field_name="difficulty")
        return PrivateSWEBenchVerifiedCase(
            public=public,
            truth=PrivateSWEVerifiedTruth(
                version=_text(data["version"], field_name="version"),
                gold_patch=_text(data["patch"], field_name="patch"),
                test_patch=_text(data["test_patch"], field_name="test_patch"),
                fail_to_pass=_test_identities(
                    data["FAIL_TO_PASS"],
                    field_name="FAIL_TO_PASS",
                ),
                pass_to_pass=_test_identities(
                    data["PASS_TO_PASS"],
                    field_name="PASS_TO_PASS",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class SWEProcessResult:
    returncode: int
    stdout: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("SWE process return code must be an integer")
        if type(self.stdout) is not bytes:
            raise TypeError("SWE process stdout must be bytes")


class SWEProcessExecutor(Protocol):
    """Tiny injectable process seam used by the concrete Apptainer boundary."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes,
        cwd: Path,
        timeout_seconds: float,
    ) -> SWEProcessResult: ...


@dataclass(frozen=True, slots=True)
class SubprocessSWEProcessExecutor:
    """Run one fixed argv without a shell and without exposing stderr."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes,
        cwd: Path,
        timeout_seconds: float,
    ) -> SWEProcessResult:
        try:
            completed = subprocess.run(  # noqa: S603 - argv is deployment-pinned
                tuple(argv),
                input=stdin,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise SWEHarnessInfrastructureError("official SWE harness timed out") from error
        except (OSError, subprocess.SubprocessError) as error:
            raise SWEHarnessInfrastructureError(
                "official SWE harness process could not complete"
            ) from error
        return SWEProcessResult(completed.returncode, completed.stdout)


class SWEHarnessProcess(Protocol):
    @property
    def deployment_id(self) -> str: ...

    @property
    def harness_revision(self) -> str: ...

    def request(
        self,
        *,
        environment_image_id: str,
        operation: SWEHarnessOperation,
        payload: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]: ...


@dataclass(slots=True)
class ApptainerSWEHarnessProcess:
    """Canonical JSON RPC over one leased ``apptainer exec`` invocation."""

    deployment: PinnedSWEHarnessDeployment
    executor: SWEProcessExecutor = field(default_factory=SubprocessSWEProcessExecutor, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, PinnedSWEHarnessDeployment):
            raise TypeError("SWE harness process requires a pinned deployment")
        if not callable(getattr(self.executor, "execute", None)):
            raise TypeError("SWE harness process executor must implement execute")

    @property
    def deployment_id(self) -> str:
        return self.deployment.deployment_id

    @property
    def harness_revision(self) -> str:
        return self.deployment.harness_revision

    def request(
        self,
        *,
        environment_image_id: str,
        operation: SWEHarnessOperation,
        payload: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        if not isinstance(operation, SWEHarnessOperation):
            raise TypeError("SWE harness operation is invalid")
        image_source = self.deployment.image_source(environment_image_id)
        self.deployment.verify_materializer_identity()
        self._verify_host_assets()
        normalized_payload = normalize_json(dict(payload))
        if not isinstance(normalized_payload, dict):
            raise TypeError("SWE harness payload must be a JSON object")
        identity_fields = {
            "deployment_id": self.deployment_id,
            "environment_image_id": environment_image_id,
            "harness_revision": self.harness_revision,
            "operation": operation.value,
            "payload": normalized_payload,
            "protocol_version": SWE_HARNESS_PROTOCOL_VERSION,
        }
        request_id = stable_hash(identity_fields)
        encoded = canonical_json_bytes({**identity_fields, "request_id": request_id})
        if len(encoded) > self.deployment.max_request_bytes:
            raise ValueError("official SWE harness request exceeds its fixed transport limit")
        with self.deployment.materializer.lease(image_source) as materialized:
            if not isinstance(materialized, MaterializedSWEImage):
                raise SWEHarnessInfrastructureError(
                    "SWE image materializer returned an incompatible lease"
                )
            if (
                materialized.environment_image_id != image_source.environment_image_id
                or materialized.manifest_digest != image_source.manifest_digest
            ):
                raise SWEHarnessInfrastructureError(
                    "materialized SWE image differs from its immutable OCI source"
                )
            if not materialized.image_path.is_file():
                raise SWEHarnessInfrastructureError(
                    "materialized SWE image disappeared during its lease"
                )
            argv = (
                str(self.deployment.apptainer_executable),
                "exec",
                "--cleanenv",
                "--containall",
                "--no-home",
                "--bind",
                f"{self.deployment.work_root}:{_CONTAINER_WORK_ROOT}",
                "--bind",
                (f"{self.deployment.worker_host_path}:{self.deployment.worker_container_path}:ro"),
                "--pwd",
                _CONTAINER_WORK_ROOT,
                str(materialized.image_path),
                self.deployment.worker_container_path,
            )
            completed = self.executor.execute(
                argv,
                stdin=encoded,
                cwd=self.deployment.work_root,
                timeout_seconds=self.deployment.request_timeout_seconds,
            )
        if not isinstance(completed, SWEProcessResult):
            raise TypeError("SWE process executor returned an incompatible result")
        if completed.returncode != 0:
            raise SWEHarnessInfrastructureError("official SWE harness exited unsuccessfully")
        if not completed.stdout or len(completed.stdout) > self.deployment.max_response_bytes:
            raise SWEHarnessInfrastructureError(
                "official SWE harness returned an invalid response size"
            )
        try:
            response_value: object = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SWEHarnessInfrastructureError(
                "official SWE harness returned invalid JSON"
            ) from error
        response = _result_object(
            response_value,
            fields={
                "deployment_id",
                "environment_image_id",
                "harness_revision",
                "protocol_version",
                "request_id",
                "result",
            },
            label="official SWE harness response",
        )
        if (
            response["protocol_version"] != SWE_HARNESS_PROTOCOL_VERSION
            or response["request_id"] != request_id
            or response["deployment_id"] != self.deployment_id
            or response["environment_image_id"] != environment_image_id
            or response["harness_revision"] != self.harness_revision
        ):
            raise SWEHarnessInfrastructureError(
                "official SWE harness response identity differs from its request"
            )
        result = response["result"]
        if not isinstance(result, dict):
            raise SWEHarnessInfrastructureError("official SWE harness result must be a JSON object")
        return result

    def _verify_host_assets(self) -> None:
        try:
            apptainer_size, apptainer_digest = _sha256_file(self.deployment.apptainer_executable)
        except OSError as error:
            raise SWEHarnessInfrastructureError(
                "pinned Apptainer executable could not be read"
            ) from error
        if (
            apptainer_size != self.deployment._apptainer_size_bytes
            or apptainer_digest != self.deployment._apptainer_sha256
        ):
            raise SWEHarnessInfrastructureError(
                "Apptainer executable bytes differ from their deployment pin"
            )
        try:
            size, digest = _sha256_file(self.deployment.worker_host_path)
        except OSError as error:
            raise SWEHarnessInfrastructureError(
                "pinned host-side SWE worker could not be read"
            ) from error
        if size != self.deployment.worker_size_bytes or digest != self.deployment.worker_sha256:
            raise SWEHarnessInfrastructureError(
                "host-side SWE worker bytes differ from their deployment pin"
            )


@dataclass(slots=True)
class _ApptainerSWEWorkspaceBackend:
    public: SWEBenchVerifiedPublicCase
    process: SWEHarnessProcess = field(repr=False)
    workspace_id: str
    test_command: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.public, SWEBenchVerifiedPublicCase):
            raise TypeError("official SWE workspace requires a public case")
        _text(self.workspace_id, field_name="workspace_id")
        _text(self.test_command, field_name="test_command")
        result = self.process.request(
            environment_image_id=self.public.environment_image_id,
            operation=SWEHarnessOperation.WORKSPACE_INITIALIZE,
            payload={
                "base_commit": self.public.base_commit,
                "instance_id": self.public.instance_id,
                "max_steps": self.public.max_steps,
                "repo": self.public.repo,
                "test_command": self.test_command,
                "workspace_id": self.workspace_id,
            },
        )
        initialized = _result_object(
            result,
            fields={"base_commit", "initialized", "instance_id", "workspace_id"},
            label="official SWE workspace initialization",
        )
        if initialized != {
            "base_commit": self.public.base_commit,
            "initialized": True,
            "instance_id": self.public.instance_id,
            "workspace_id": self.workspace_id,
        }:
            raise SWEHarnessInfrastructureError(
                "official SWE workspace initialized another repository state"
            )

    @property
    def instance_id(self) -> str:
        return self.public.instance_id

    @property
    def environment_id(self) -> str:
        return self.public.environment_id

    @property
    def max_steps(self) -> int:
        return self.public.max_steps

    async def execute(
        self,
        command: SWEWorkspaceCommand,
        *,
        step_index: int,
    ) -> SWEWorkspacePublicStep:
        if not isinstance(command, SWEWorkspaceCommand):
            raise TypeError("official SWE workspace requires SWEWorkspaceCommand")
        _positive_int(step_index, field_name="step_index")
        result = self.process.request(
            environment_image_id=self.public.environment_image_id,
            operation=SWEHarnessOperation.WORKSPACE_EXECUTE,
            payload={
                "command": command.to_value(),
                "step_index": step_index,
                "workspace_id": self.workspace_id,
            },
        )
        response = _result_object(
            result,
            fields={
                "budget_usage",
                "public_observation",
                "step_index",
                "terminal",
                "workspace_id",
            },
            label="official SWE workspace execution",
        )
        if response["workspace_id"] != self.workspace_id or response["step_index"] != step_index:
            raise SWEHarnessInfrastructureError(
                "official SWE workspace execution response identity differs"
            )
        terminal = response["terminal"]
        if type(terminal) is not bool:
            raise SWEHarnessInfrastructureError(
                "official SWE workspace terminal flag must be boolean"
            )
        try:
            budget = BudgetVector.from_value(response["budget_usage"])
        except (TypeError, ValueError) as error:
            raise SWEHarnessInfrastructureError(
                "official SWE workspace returned invalid measured usage"
            ) from error
        return SWEWorkspacePublicStep(response["public_observation"], terminal, budget)


@dataclass(slots=True)
class ApptainerSWEWorkspaceBackendFactory:
    """Production implementation of :class:`SWEWorkspaceBackendFactory`."""

    process: SWEHarnessProcess = field(repr=False)
    grader: SWEOfficialGrader = field(repr=False)
    _next_workspace_ordinal: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.process, "request", None)):
            raise TypeError("SWE workspace factory requires a harness process")
        if not callable(getattr(self.grader, "test_command", None)):
            raise TypeError("SWE workspace factory requires an official grader")
        _text(self.grader.grader_id, field_name="grader_id")
        if self.grader.harness_revision != self.process.harness_revision:
            raise ValueError("official SWE workspace and grader revisions differ")

    def create(self, public: SWEBenchVerifiedPublicCase) -> SWEWorkspaceBackend:
        if not isinstance(public, SWEBenchVerifiedPublicCase):
            raise TypeError("SWE workspace factory requires a public case")
        self._next_workspace_ordinal += 1
        workspace_id = stable_hash(
            {
                "deployment_id": self.process.deployment_id,
                "grader_id": self.grader.grader_id,
                "instance_id": public.instance_id,
                "ordinal": self._next_workspace_ordinal,
            }
        ).removeprefix("sha256:")
        test_command = self.grader.test_command(public)
        return _ApptainerSWEWorkspaceBackend(
            public,
            self.process,
            workspace_id,
            test_command,
        )


@dataclass(slots=True)
class _ApptainerOfficialSWEVerifierBackend:
    case: PrivateSWEBenchVerifiedCase
    process: SWEHarnessProcess = field(repr=False)
    grader: SWEOfficialGrader = field(repr=False)

    @property
    def instance_id(self) -> str:
        return self.case.public.instance_id

    @property
    def environment_id(self) -> str:
        return self.case.public.environment_id

    @property
    def verifier_version(self) -> str:
        process_digest = self.process.deployment_id.removeprefix("sha256:")[:16]
        grader_digest = self.grader.grader_id.removeprefix("sha256:")[:16]
        return f"official-swebench:{self.process.harness_revision}:{process_digest}:{grader_digest}"

    async def verify(self, request: SWEVerifierRequest) -> SWEVerifierResult:
        if not isinstance(request, SWEVerifierRequest):
            raise TypeError("official SWE verifier requires SWEVerifierRequest")
        public = self.case.public
        if (
            request.instance_id != public.instance_id
            or request.repo != public.repo
            or request.base_commit != public.base_commit
            or request.environment_image_id != public.environment_image_id
        ):
            raise ValueError("official SWE verifier request identity differs from its case")
        eval_script = self.grader.eval_script(self.case)
        payload: dict[str, JsonValue] = {
            "base_commit": public.base_commit,
            "candidate_patch": request.candidate_patch,
            "eval_script": eval_script,
            "instance_id": public.instance_id,
            "repo": public.repo,
            "test_patch": self.case.truth.test_patch,
            "version": public.version,
        }
        result = self.process.request(
            environment_image_id=public.environment_image_id,
            operation=SWEHarnessOperation.VERIFY,
            payload=payload,
        )
        response = _result_object(
            result,
            fields={
                "candidate_patch_applied",
                "instance_id",
                "test_output",
            },
            label="official SWE verifier result",
        )
        if response["instance_id"] != public.instance_id:
            raise SWEHarnessInfrastructureError("official SWE verifier returned another instance")
        applied = response["candidate_patch_applied"]
        output = response["test_output"]
        if type(applied) is not bool or type(output) is not str:
            raise SWEHarnessInfrastructureError("official SWE worker result is invalid")
        candidate_patch = request.candidate_patch
        if candidate_patch is None or candidate_patch == "" or not applied:
            if applied:
                raise SWEHarnessInfrastructureError(
                    "official SWE worker applied an absent candidate patch"
                )
            return SWEVerifierResult(
                resolved=False,
                fail_to_pass_passed=0,
                fail_to_pass_total=len(self.case.truth.fail_to_pass),
                pass_to_pass_passed=0,
                pass_to_pass_total=len(self.case.truth.pass_to_pass),
            )
        return self.grader.grade(
            self.case,
            candidate_patch=candidate_patch,
            test_output=output,
        )


@dataclass(frozen=True, slots=True)
class ApptainerOfficialSWEVerifierFactory:
    """Production implementation of :class:`OfficialSWEVerifierFactory`."""

    process: SWEHarnessProcess = field(repr=False)
    grader: SWEOfficialGrader = field(repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.process, "request", None)):
            raise TypeError("official SWE verifier factory requires a harness process")
        for method in ("eval_script", "grade"):
            if not callable(getattr(self.grader, method, None)):
                raise TypeError("official SWE verifier factory requires a grader")
        _text(self.grader.grader_id, field_name="grader_id")
        if self.grader.harness_revision != self.process.harness_revision:
            raise ValueError("official SWE verifier and process revisions differ")

    def create(self, case: PrivateSWEBenchVerifiedCase) -> OfficialSWEVerifierBackend:
        if not isinstance(case, PrivateSWEBenchVerifiedCase):
            raise TypeError("official SWE verifier factory requires a private case")
        return _ApptainerOfficialSWEVerifierBackend(case, self.process, self.grader)


__all__ = [
    "SWEBENCH_VERIFIED_ROW_FIELDS",
    "SWE_HARNESS_PROTOCOL_VERSION",
    "ApptainerOfficialSWEVerifierFactory",
    "ApptainerSWEHarnessProcess",
    "ApptainerSWEWorkspaceBackendFactory",
    "MaterializedSWEImage",
    "OfficialSWEBenchVerifiedCaseConverter",
    "PinnedSWEHarnessDeployment",
    "PinnedSWEInstanceImageBinding",
    "PinnedSWEOCIImage",
    "PinnedSWEOfficialGrader",
    "SWEHarnessInfrastructureError",
    "SWEHarnessOperation",
    "SWEHarnessProcess",
    "SWEImageMaterializer",
    "SWEOfficialGrader",
    "SWEProcessExecutor",
    "SWEProcessResult",
    "SubprocessSWEProcessExecutor",
]
