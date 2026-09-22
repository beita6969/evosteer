"""Pinned private runtime wiring for the formal eighteen-benchmark worker.

The public attempt identity deliberately excludes deployment paths, official
environment objects, and SWE verifier material.  The formal worker still
needs one *closed* way to rebuild those objects, however.  This module keeps
that authority in the private exact input: there is no catalog registry,
environment-variable discovery, completion-smoke substitute, or optional
domain route.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask
from skillev.training import RolloutSessionFactory
from skillev_private.benchmarks.acquisition import load_benchmark_acquisition_lock
from skillev_private.benchmarks.bird_sql_official import load_bird_sql_session_factory
from skillev_private.benchmarks.catalog import PrivateBenchmarkCatalog
from skillev_private.benchmarks.catalog_freeze import FrozenProductionCatalogBundle
from skillev_private.benchmarks.evalplus_official import load_evalplus_session_factory
from skillev_private.benchmarks.external_process_runtime import (
    ExternalProcessRuntime,
    ExternalProcessSessionFactory,
)
from skillev_private.benchmarks.official_process_deployment import (
    FileOfficialProcessPreparationFactory,
    load_official_process_deployment_config,
)
from skillev_private.benchmarks.process_catalog import (
    LoadedCompleteProductionCatalog,
    ProductionProcessCatalogDependencies,
    load_complete_production_benchmark_catalog,
    load_training_process_catalog,
)
from skillev_private.benchmarks.production_catalog import (
    ExternalSessionFactoryBuilder,
    ProductionCatalogDependencies,
    PyArrowParquetRowReader,
    load_production_benchmark_catalog,
)
from skillev_private.benchmarks.semantic_selection import (
    IIDSemanticSelection,
    load_iid_semantic_selection,
)
from skillev_private.benchmarks.swebench_grader import PinnedSWEOfficialGrader
from skillev_private.benchmarks.swebench_materializer import ApptainerOCIImageMaterializer
from skillev_private.benchmarks.swebench_official import (
    ApptainerOfficialSWEVerifierFactory,
    ApptainerSWEHarnessProcess,
    ApptainerSWEWorkspaceBackendFactory,
    OfficialSWEBenchVerifiedCaseConverter,
    PinnedSWEHarnessDeployment,
    PinnedSWEInstanceImageBinding,
    PinnedSWEOCIImage,
)
from skillev_private.benchmarks.tablebench_official import load_tablebench_session_factory

from .implementation_build import PrivateImplementationBuildDeployment

PRIVATE_BENCHMARK_RUNTIME_FORMAT = "skillev-private-benchmark-runtime@5"
PRIVATE_SWE_RUNTIME_FORMAT = "skillev-private-swe-runtime@1"
PRIVATE_EXTERNAL_COMPLETION_MANIFEST_FORMAT = "skillev-private-external-completion-manifest@1"
_EXTERNAL_COMPLETION_BENCHMARKS = (
    Benchmark.BIRD_SQL,
    Benchmark.MBPP_PLUS,
    Benchmark.TABLEBENCH,
    Benchmark.HUMANEVAL_PLUS,
)
_TRAINING_PROCESS_BENCHMARKS = (Benchmark.APPWORLD,)


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a SHA-256 identity")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 identity") from error
    return text


def _absolute_path(value: object, *, field: str) -> Path:
    path = Path(_text(value, field=field)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return path


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not result > 0.0 or result == float("inf"):
        raise ValueError(f"{field} must be positive and finite")
    return result


def _external_manifest_values(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError("private external completion manifests must be an array")
    return value


@dataclass(frozen=True, slots=True)
class PrivateExternalCompletionManifest:
    """One private evaluator manifest pinned into the formal exact input."""

    benchmark: Benchmark
    path: Path
    sha256: str
    format: str = PRIVATE_EXTERNAL_COMPLETION_MANIFEST_FORMAT

    def __post_init__(self) -> None:
        if self.format != PRIVATE_EXTERNAL_COMPLETION_MANIFEST_FORMAT:
            raise ValueError("unsupported private external completion manifest format")
        if self.benchmark not in _EXTERNAL_COMPLETION_BENCHMARKS:
            raise ValueError("private external completion benchmark is unsupported")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("private external completion manifest path must be absolute")
        _sha256(self.sha256, field="private external completion manifest sha256")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "format": self.format,
            "path": self.path.as_posix(),
            "sha256": self.sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateExternalCompletionManifest:
        data = _object(
            value,
            fields=frozenset({"benchmark", "format", "path", "sha256"}),
            label="private external completion manifest",
        )
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="benchmark")),
            path=_absolute_path(data["path"], field="path"),
            sha256=_sha256(data["sha256"], field="sha256"),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class PrivateExternalProcessRuntime:
    """Private paths needed to instantiate one formal interactive worker."""

    benchmark: Benchmark
    interpreter_path: Path
    source_root: Path
    state_root: Path
    request_timeout_seconds: float
    container_runtime_path: Path | None = None
    container_storage_root: Path | None = None
    image_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.benchmark not in _TRAINING_PROCESS_BENCHMARKS:
            raise ValueError("private external process benchmark is unsupported")
        for field in ("interpreter_path", "source_root", "state_root"):
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field} must be absolute")
        _positive_float(self.request_timeout_seconds, field="request_timeout_seconds")
        if self.benchmark is Benchmark.APPWORLD:
            if any(
                value is not None
                for value in (
                    self.container_runtime_path,
                    self.container_storage_root,
                    self.image_prefix,
                )
            ):
                raise ValueError("AppWorld cannot carry container fields")
        elif (
            not isinstance(self.container_runtime_path, Path)
            or not self.container_runtime_path.is_absolute()
            or not isinstance(self.container_storage_root, Path)
            or not self.container_storage_root.is_absolute()
            or type(self.image_prefix) is not str
            or not self.image_prefix.strip()
        ):
            raise ValueError("SkillFlow-Bench requires its complete container runtime")

    def deployment(self) -> ExternalProcessRuntime:
        return ExternalProcessRuntime(
            benchmark=self.benchmark,
            interpreter_path=self.interpreter_path,
            source_root=self.source_root,
            state_root=self.state_root,
            request_timeout_seconds=self.request_timeout_seconds,
            container_runtime_path=self.container_runtime_path,
            container_storage_root=self.container_storage_root,
            image_prefix=self.image_prefix,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "container_runtime_path": (
                None
                if self.container_runtime_path is None
                else self.container_runtime_path.as_posix()
            ),
            "container_storage_root": (
                None
                if self.container_storage_root is None
                else self.container_storage_root.as_posix()
            ),
            "image_prefix": self.image_prefix,
            "interpreter_path": self.interpreter_path.as_posix(),
            "request_timeout_seconds": self.request_timeout_seconds,
            "source_root": self.source_root.as_posix(),
            "state_root": self.state_root.as_posix(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateExternalProcessRuntime:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmark",
                    "container_runtime_path",
                    "container_storage_root",
                    "image_prefix",
                    "interpreter_path",
                    "request_timeout_seconds",
                    "source_root",
                    "state_root",
                }
            ),
            label="private external process runtime",
        )
        container_runtime = data["container_runtime_path"]
        container_storage = data["container_storage_root"]
        image_prefix = data["image_prefix"]
        if container_runtime is not None and type(container_runtime) is not str:
            raise TypeError("container_runtime_path must be text or null")
        if container_storage is not None and type(container_storage) is not str:
            raise TypeError("container_storage_root must be text or null")
        if image_prefix is not None and type(image_prefix) is not str:
            raise TypeError("image_prefix must be text or null")
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="benchmark")),
            interpreter_path=_absolute_path(data["interpreter_path"], field="interpreter_path"),
            source_root=_absolute_path(data["source_root"], field="source_root"),
            state_root=_absolute_path(data["state_root"], field="state_root"),
            request_timeout_seconds=_positive_float(
                data["request_timeout_seconds"], field="request_timeout_seconds"
            ),
            container_runtime_path=(
                None
                if container_runtime is None
                else _absolute_path(container_runtime, field="container_runtime_path")
            ),
            container_storage_root=(
                None
                if container_storage is None
                else _absolute_path(container_storage, field="container_storage_root")
            ),
            image_prefix=image_prefix,
        )


@dataclass(frozen=True, slots=True)
class _ExternalCompletionFactoryBuilder:
    manifest: PrivateExternalCompletionManifest
    dataset_root: Path

    def build(self, tasks: tuple[RolloutTask, ...]) -> RolloutSessionFactory:
        benchmark = self.manifest.benchmark
        if benchmark is Benchmark.BIRD_SQL:
            return load_bird_sql_session_factory(
                tasks=tasks,
                private_manifest=self.manifest.path,
                expected_private_manifest_sha256=self.manifest.sha256,
                dataset_root=self.dataset_root,
            )
        if benchmark in (Benchmark.MBPP_PLUS, Benchmark.HUMANEVAL_PLUS):
            return load_evalplus_session_factory(
                benchmark=benchmark,
                tasks=tasks,
                private_manifest_path=self.manifest.path,
                expected_sha256=self.manifest.sha256,
            )
        if benchmark is Benchmark.TABLEBENCH:
            return load_tablebench_session_factory(
                tasks=tasks,
                private_manifest=self.manifest.path,
                expected_sha256=self.manifest.sha256,
            )
        raise AssertionError("unhandled external completion benchmark")


@dataclass(frozen=True, slots=True)
class PrivateSWEImagePin:
    """One immutable OCI source; private only because it names test instances."""

    environment_image_id: str
    source_reference: str
    manifest_digest: str
    platform_os: str = "linux"
    platform_architecture: str = "amd64"

    def __post_init__(self) -> None:
        PinnedSWEOCIImage(
            environment_image_id=self.environment_image_id,
            source_reference=self.source_reference,
            manifest_digest=self.manifest_digest,
            platform_os=self.platform_os,
            platform_architecture=self.platform_architecture,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "environment_image_id": self.environment_image_id,
            "manifest_digest": self.manifest_digest,
            "platform_architecture": self.platform_architecture,
            "platform_os": self.platform_os,
            "source_reference": self.source_reference,
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateSWEImagePin:
        data = _object(
            value,
            fields=frozenset(
                {
                    "environment_image_id",
                    "manifest_digest",
                    "platform_architecture",
                    "platform_os",
                    "source_reference",
                }
            ),
            label="private SWE image pin",
        )
        return cls(
            environment_image_id=_text(data["environment_image_id"], field="environment_image_id"),
            source_reference=_text(data["source_reference"], field="source_reference"),
            manifest_digest=_sha256(data["manifest_digest"], field="manifest_digest"),
            platform_os=_text(data["platform_os"], field="platform_os"),
            platform_architecture=_text(
                data["platform_architecture"], field="platform_architecture"
            ),
        )

    def deployment_image(self) -> PinnedSWEOCIImage:
        return PinnedSWEOCIImage(
            environment_image_id=self.environment_image_id,
            source_reference=self.source_reference,
            manifest_digest=self.manifest_digest,
            platform_os=self.platform_os,
            platform_architecture=self.platform_architecture,
        )


@dataclass(frozen=True, slots=True)
class PrivateSWEImageBinding:
    """Private binding from an official SWE task identity to an OCI image."""

    instance_id: str
    environment_image_id: str

    def __post_init__(self) -> None:
        PinnedSWEInstanceImageBinding(
            instance_id=self.instance_id,
            environment_image_id=self.environment_image_id,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "environment_image_id": self.environment_image_id,
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateSWEImageBinding:
        data = _object(
            value,
            fields=frozenset({"environment_image_id", "instance_id"}),
            label="private SWE image binding",
        )
        return cls(
            instance_id=_text(data["instance_id"], field="instance_id"),
            environment_image_id=_text(data["environment_image_id"], field="environment_image_id"),
        )

    def deployment_binding(self) -> PinnedSWEInstanceImageBinding:
        return PinnedSWEInstanceImageBinding(
            instance_id=self.instance_id,
            environment_image_id=self.environment_image_id,
        )


@dataclass(frozen=True, slots=True)
class PrivateSWEHarnessRuntime:
    """All private pins needed to create the one official SWE route."""

    apptainer_executable: Path
    apptainer_version: str
    materializer_storage_root: Path
    materializer_capacity_bytes: int
    materializer_timeout_seconds: float
    worker_host_path: Path
    worker_container_path: str
    worker_size_bytes: int
    worker_sha256: str
    harness_revision: str
    harness_source_root: Path
    work_root: Path
    images: tuple[PrivateSWEImagePin, ...]
    bindings: tuple[PrivateSWEImageBinding, ...]
    task_family: str
    max_steps: int
    request_timeout_seconds: float = 1800.0
    max_request_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    format: str = PRIVATE_SWE_RUNTIME_FORMAT

    def __post_init__(self) -> None:
        if self.format != PRIVATE_SWE_RUNTIME_FORMAT:
            raise ValueError("unsupported private SWE runtime format")
        for field in (
            "apptainer_executable",
            "materializer_storage_root",
            "worker_host_path",
            "harness_source_root",
            "work_root",
        ):
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field} must be an absolute path")
        _text(self.apptainer_version, field="apptainer_version")
        _text(self.worker_container_path, field="worker_container_path")
        _sha256(self.worker_sha256, field="worker_sha256")
        _text(self.harness_revision, field="harness_revision")
        _text(self.task_family, field="task_family")
        _positive_int(self.materializer_capacity_bytes, field="materializer_capacity_bytes")
        _positive_int(self.worker_size_bytes, field="worker_size_bytes")
        _positive_int(self.max_steps, field="max_steps")
        _positive_int(self.max_request_bytes, field="max_request_bytes")
        _positive_int(self.max_response_bytes, field="max_response_bytes")
        _positive_float(self.materializer_timeout_seconds, field="materializer_timeout_seconds")
        _positive_float(self.request_timeout_seconds, field="request_timeout_seconds")
        if not isinstance(self.images, tuple) or not self.images:
            raise ValueError("private SWE runtime requires image pins")
        if any(not isinstance(item, PrivateSWEImagePin) for item in self.images):
            raise TypeError("private SWE runtime image pins are invalid")
        image_ids = tuple(item.environment_image_id for item in self.images)
        if image_ids != tuple(sorted(image_ids)) or len(set(image_ids)) != len(image_ids):
            raise ValueError("private SWE images must be sorted and unique")
        if not isinstance(self.bindings, tuple) or not self.bindings:
            raise ValueError("private SWE runtime requires image bindings")
        if any(not isinstance(item, PrivateSWEImageBinding) for item in self.bindings):
            raise TypeError("private SWE runtime image bindings are invalid")
        instance_ids = tuple(item.instance_id for item in self.bindings)
        binding_images = tuple(item.environment_image_id for item in self.bindings)
        if instance_ids != tuple(sorted(instance_ids)) or len(set(instance_ids)) != len(
            instance_ids
        ):
            raise ValueError("private SWE bindings must be sorted and unique")
        if any(item not in image_ids for item in binding_images):
            raise ValueError("private SWE binding refers to an unpinned image")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "apptainer_executable": self.apptainer_executable.as_posix(),
            "apptainer_version": self.apptainer_version,
            "bindings": [item.to_value() for item in self.bindings],
            "format": self.format,
            "harness_revision": self.harness_revision,
            "harness_source_root": self.harness_source_root.as_posix(),
            "images": [item.to_value() for item in self.images],
            "materializer_capacity_bytes": self.materializer_capacity_bytes,
            "materializer_storage_root": self.materializer_storage_root.as_posix(),
            "materializer_timeout_seconds": self.materializer_timeout_seconds,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_steps": self.max_steps,
            "request_timeout_seconds": self.request_timeout_seconds,
            "task_family": self.task_family,
            "work_root": self.work_root.as_posix(),
            "worker_container_path": self.worker_container_path,
            "worker_host_path": self.worker_host_path.as_posix(),
            "worker_sha256": self.worker_sha256,
            "worker_size_bytes": self.worker_size_bytes,
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateSWEHarnessRuntime:
        data = _object(
            value,
            fields=frozenset(
                {
                    "apptainer_executable",
                    "apptainer_version",
                    "bindings",
                    "format",
                    "harness_revision",
                    "harness_source_root",
                    "images",
                    "materializer_capacity_bytes",
                    "materializer_storage_root",
                    "materializer_timeout_seconds",
                    "max_request_bytes",
                    "max_response_bytes",
                    "max_steps",
                    "request_timeout_seconds",
                    "task_family",
                    "work_root",
                    "worker_container_path",
                    "worker_host_path",
                    "worker_sha256",
                    "worker_size_bytes",
                }
            ),
            label="private SWE runtime",
        )
        images = data["images"]
        bindings = data["bindings"]
        if not isinstance(images, list) or not isinstance(bindings, list):
            raise ValueError("private SWE runtime images and bindings must be arrays")
        return cls(
            apptainer_executable=_absolute_path(
                data["apptainer_executable"], field="apptainer_executable"
            ),
            apptainer_version=_text(data["apptainer_version"], field="apptainer_version"),
            materializer_storage_root=_absolute_path(
                data["materializer_storage_root"], field="materializer_storage_root"
            ),
            materializer_capacity_bytes=_positive_int(
                data["materializer_capacity_bytes"], field="materializer_capacity_bytes"
            ),
            materializer_timeout_seconds=_positive_float(
                data["materializer_timeout_seconds"], field="materializer_timeout_seconds"
            ),
            worker_host_path=_absolute_path(data["worker_host_path"], field="worker_host_path"),
            worker_container_path=_text(
                data["worker_container_path"], field="worker_container_path"
            ),
            worker_size_bytes=_positive_int(data["worker_size_bytes"], field="worker_size_bytes"),
            worker_sha256=_sha256(data["worker_sha256"], field="worker_sha256"),
            harness_revision=_text(data["harness_revision"], field="harness_revision"),
            harness_source_root=_absolute_path(
                data["harness_source_root"], field="harness_source_root"
            ),
            work_root=_absolute_path(data["work_root"], field="work_root"),
            images=tuple(PrivateSWEImagePin.from_value(item) for item in images),
            bindings=tuple(PrivateSWEImageBinding.from_value(item) for item in bindings),
            task_family=_text(data["task_family"], field="task_family"),
            max_steps=_positive_int(data["max_steps"], field="max_steps"),
            request_timeout_seconds=_positive_float(
                data["request_timeout_seconds"], field="request_timeout_seconds"
            ),
            max_request_bytes=_positive_int(data["max_request_bytes"], field="max_request_bytes"),
            max_response_bytes=_positive_int(
                data["max_response_bytes"], field="max_response_bytes"
            ),
            format=_text(data["format"], field="format"),
        )

    def legacy_swe_dependencies(
        self,
    ) -> tuple[
        OfficialSWEBenchVerifiedCaseConverter,
        ApptainerSWEWorkspaceBackendFactory,
        ApptainerOfficialSWEVerifierFactory,
    ]:
        """Construct legacy SWE objects only for explicit historical tooling."""

        materializer = ApptainerOCIImageMaterializer(
            apptainer_executable=self.apptainer_executable,
            apptainer_version=self.apptainer_version,
            storage_root=self.materializer_storage_root,
            capacity_bytes=self.materializer_capacity_bytes,
            timeout_seconds=self.materializer_timeout_seconds,
        )
        deployment = PinnedSWEHarnessDeployment(
            apptainer_executable=self.apptainer_executable,
            worker_host_path=self.worker_host_path,
            worker_container_path=self.worker_container_path,
            worker_size_bytes=self.worker_size_bytes,
            worker_sha256=self.worker_sha256,
            harness_revision=self.harness_revision,
            harness_source_root=self.harness_source_root,
            work_root=self.work_root,
            images=tuple(item.deployment_image() for item in self.images),
            materializer=materializer,
            request_timeout_seconds=self.request_timeout_seconds,
            max_request_bytes=self.max_request_bytes,
            max_response_bytes=self.max_response_bytes,
        )
        grader = PinnedSWEOfficialGrader(
            harness_source_root=self.harness_source_root,
            harness_revision=self.harness_revision,
            work_root=self.work_root,
        )
        harness = ApptainerSWEHarnessProcess(deployment=deployment)
        converter = OfficialSWEBenchVerifiedCaseConverter(
            deployment=deployment,
            image_bindings=tuple(item.deployment_binding() for item in self.bindings),
            task_family=self.task_family,
            max_steps=self.max_steps,
        )
        return (
            converter,
            ApptainerSWEWorkspaceBackendFactory(
                process=harness,
                grader=grader,
            ),
            ApptainerOfficialSWEVerifierFactory(
                process=harness,
                grader=grader,
            ),
        )


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkRuntime:
    """Closed private catalog loader authority for the fixed benchmark worker."""

    acquisition_lock_path: Path
    acquisition_lock_content_hash: str
    official_process_deployment_path: Path
    official_process_deployment_content_hash: str
    iid_semantic_selection_path: Path
    iid_semantic_selection_content_hash: str
    implementation_build_deployment: PrivateImplementationBuildDeployment
    swe: PrivateSWEHarnessRuntime
    external_completion_manifests: tuple[PrivateExternalCompletionManifest, ...]
    external_process_runtimes: tuple[PrivateExternalProcessRuntime, ...]
    format: str = PRIVATE_BENCHMARK_RUNTIME_FORMAT

    def __post_init__(self) -> None:
        if self.format != PRIVATE_BENCHMARK_RUNTIME_FORMAT:
            raise ValueError("unsupported private benchmark runtime format")
        for field in (
            "acquisition_lock_path",
            "official_process_deployment_path",
            "iid_semantic_selection_path",
        ):
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field} must be an absolute path")
        _sha256(self.acquisition_lock_content_hash, field="acquisition_lock_content_hash")
        _sha256(
            self.official_process_deployment_content_hash,
            field="official_process_deployment_content_hash",
        )
        _sha256(
            self.iid_semantic_selection_content_hash,
            field="iid_semantic_selection_content_hash",
        )
        if not isinstance(
            self.implementation_build_deployment, PrivateImplementationBuildDeployment
        ):
            raise TypeError("private benchmark runtime requires an implementation build deployment")
        if not isinstance(self.swe, PrivateSWEHarnessRuntime):
            raise TypeError("private benchmark runtime requires a pinned SWE runtime")
        if any(
            not isinstance(item, PrivateExternalCompletionManifest)
            for item in self.external_completion_manifests
        ):
            raise TypeError("private benchmark runtime external manifests are invalid")
        if tuple(item.benchmark for item in self.external_completion_manifests) != (
            _EXTERNAL_COMPLETION_BENCHMARKS
        ):
            raise ValueError("private external completion manifests must follow protocol order")
        if tuple(item.benchmark for item in self.external_process_runtimes) != (
            _TRAINING_PROCESS_BENCHMARKS
        ):
            raise ValueError("private external process runtimes must follow training order")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "acquisition_lock_content_hash": self.acquisition_lock_content_hash,
            "acquisition_lock_path": self.acquisition_lock_path.as_posix(),
            "format": self.format,
            "iid_semantic_selection_content_hash": (self.iid_semantic_selection_content_hash),
            "iid_semantic_selection_path": self.iid_semantic_selection_path.as_posix(),
            "implementation_build_deployment": self.implementation_build_deployment.to_value(),
            "external_completion_manifests": [
                item.to_value() for item in self.external_completion_manifests
            ],
            "external_process_runtimes": [
                item.to_value() for item in self.external_process_runtimes
            ],
            "official_process_deployment_content_hash": (
                self.official_process_deployment_content_hash
            ),
            "official_process_deployment_path": self.official_process_deployment_path.as_posix(),
            "swe": self.swe.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateBenchmarkRuntime:
        data = _object(
            value,
            fields=frozenset(
                {
                    "acquisition_lock_content_hash",
                    "acquisition_lock_path",
                    "format",
                    "iid_semantic_selection_content_hash",
                    "iid_semantic_selection_path",
                    "implementation_build_deployment",
                    "external_completion_manifests",
                    "external_process_runtimes",
                    "official_process_deployment_content_hash",
                    "official_process_deployment_path",
                    "swe",
                }
            ),
            label="private benchmark runtime",
        )
        return cls(
            acquisition_lock_path=_absolute_path(
                data["acquisition_lock_path"], field="acquisition_lock_path"
            ),
            acquisition_lock_content_hash=_sha256(
                data["acquisition_lock_content_hash"], field="acquisition_lock_content_hash"
            ),
            official_process_deployment_path=_absolute_path(
                data["official_process_deployment_path"],
                field="official_process_deployment_path",
            ),
            official_process_deployment_content_hash=_sha256(
                data["official_process_deployment_content_hash"],
                field="official_process_deployment_content_hash",
            ),
            iid_semantic_selection_path=_absolute_path(
                data["iid_semantic_selection_path"],
                field="iid_semantic_selection_path",
            ),
            iid_semantic_selection_content_hash=_sha256(
                data["iid_semantic_selection_content_hash"],
                field="iid_semantic_selection_content_hash",
            ),
            implementation_build_deployment=PrivateImplementationBuildDeployment.from_value(
                data["implementation_build_deployment"]
            ),
            swe=PrivateSWEHarnessRuntime.from_value(data["swe"]),
            external_completion_manifests=tuple(
                PrivateExternalCompletionManifest.from_value(item)
                for item in _external_manifest_values(data["external_completion_manifests"])
            ),
            external_process_runtimes=tuple(
                PrivateExternalProcessRuntime.from_value(item)
                for item in _external_manifest_values(data["external_process_runtimes"])
            ),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def load_iid_semantic_selection(self) -> IIDSemanticSelection:
        """Load the exact reviewed training population named by this runtime."""

        selection = load_iid_semantic_selection(self.iid_semantic_selection_path)
        if selection.content_hash != self.iid_semantic_selection_content_hash:
            raise ValueError("private IID semantic selection differs from exact input")
        return selection

    def production_dependencies(self, *, dataset_root: Path) -> ProductionCatalogDependencies:
        """Build every frozen non-process evaluator before catalog hydration."""

        if not isinstance(dataset_root, Path) or not dataset_root.is_absolute():
            raise ValueError("formal dataset root must be an absolute Path")
        builders: tuple[tuple[Benchmark, ExternalSessionFactoryBuilder], ...] = tuple(
            (
                item.benchmark,
                _ExternalCompletionFactoryBuilder(
                    manifest=item,
                    dataset_root=dataset_root,
                ),
            )
            for item in self.external_completion_manifests
        )
        return ProductionCatalogDependencies(
            parquet_reader=PyArrowParquetRowReader(),
            external_session_factory_builders=builders,
        )

    def load_complete_catalog(
        self,
        bundle: FrozenProductionCatalogBundle,
    ) -> LoadedCompleteProductionCatalog:
        """Hydrate all eighteen pinned workloads, or fail before any rollout.

        This is deliberately a single closed loader rather than a registry:
        the private exact input names every source/deployment authority and no
        missing benchmark can fall back to a smoke environment.
        """

        if not isinstance(bundle, FrozenProductionCatalogBundle):
            raise TypeError("formal catalog loader requires FrozenProductionCatalogBundle")
        lock = load_benchmark_acquisition_lock(self.acquisition_lock_path)
        if lock.content_hash != self.acquisition_lock_content_hash:
            raise ValueError("private acquisition lock differs from exact input")
        process_deployment = load_official_process_deployment_config(
            self.official_process_deployment_path
        )
        if process_deployment.content_hash != self.official_process_deployment_content_hash:
            raise ValueError("private process deployment differs from exact input")
        process_deployment_runtime = FileOfficialProcessPreparationFactory(
            self.official_process_deployment_path
        ).build(lock=lock, target_root=bundle.process.dataset_root)
        process_dependencies = process_deployment_runtime.dependencies
        if not isinstance(process_dependencies, ProductionProcessCatalogDependencies):
            raise TypeError("private process deployment did not build catalog dependencies")
        return load_complete_production_benchmark_catalog(
            non_process_config=bundle.non_process,
            non_process_dependencies=self.production_dependencies(
                dataset_root=bundle.non_process.dataset_root
            ),
            process_config=bundle.process,
            process_dependencies=process_dependencies,
        )

    def load_training_catalog(
        self,
        bundle: FrozenProductionCatalogBundle,
    ) -> LoadedCompleteProductionCatalog:
        """Hydrate every static and interactive workload used by frozen B2."""

        if not isinstance(bundle, FrozenProductionCatalogBundle):
            raise TypeError("formal training catalog requires FrozenProductionCatalogBundle")
        lock = load_benchmark_acquisition_lock(self.acquisition_lock_path)
        if lock.content_hash != self.acquisition_lock_content_hash:
            raise ValueError("private acquisition lock differs from exact input")
        deployment_config = load_official_process_deployment_config(
            self.official_process_deployment_path
        )
        if deployment_config.content_hash != self.official_process_deployment_content_hash:
            raise ValueError("private process deployment differs from exact input")
        deployment = FileOfficialProcessPreparationFactory(
            self.official_process_deployment_path
        ).build(lock=lock, target_root=bundle.process.dataset_root)
        base = deployment.dependencies
        external = tuple(
            (
                item.benchmark,
                ExternalProcessSessionFactory(item.deployment()),
            )
            for item in self.external_process_runtimes
        )
        process_dependencies = ProductionProcessCatalogDependencies(
            webshop=base.webshop,
            alfworld=base.alfworld,
            scienceworld=base.scienceworld,
            external_session_factories=external,
        )
        process = load_training_process_catalog(bundle.process, process_dependencies)
        non_process = load_production_benchmark_catalog(
            bundle.non_process,
            self.production_dependencies(dataset_root=bundle.non_process.dataset_root),
        )
        try:
            catalog = PrivateBenchmarkCatalog((*non_process.catalog.workloads, *process.workloads))
        except Exception:
            non_process.close()
            raise
        return LoadedCompleteProductionCatalog(catalog, non_process)


__all__ = [
    "PRIVATE_BENCHMARK_RUNTIME_FORMAT",
    "PRIVATE_EXTERNAL_COMPLETION_MANIFEST_FORMAT",
    "PRIVATE_SWE_RUNTIME_FORMAT",
    "PrivateBenchmarkRuntime",
    "PrivateExternalCompletionManifest",
    "PrivateExternalProcessRuntime",
    "PrivateImplementationBuildDeployment",
    "PrivateSWEHarnessRuntime",
    "PrivateSWEImageBinding",
    "PrivateSWEImagePin",
]
