from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from skillev_private.benchmarks.swebench_materializer import (
    ApptainerOCIImageMaterializer,
    MaterializerProcessResult,
    SubprocessMaterializerProcessExecutor,
)
from skillev_private.benchmarks.swebench_official import (
    PinnedSWEOCIImage,
    SWEHarnessInfrastructureError,
)

IMAGE_DIGEST = f"sha256:{'a' * 64}"
IMAGE_SOURCE = "docker://docker.io/swebench/sweb.eval.x86_64.example_1776_project-1:latest"
IMAGE_ID = f"{IMAGE_SOURCE}@{IMAGE_DIGEST}"


def _executable(path: Path, content: bytes = b"#!/bin/sh\nexit 0\n") -> Path:
    path.write_bytes(content)
    path.chmod(0o700)
    return path.resolve()


def _image() -> PinnedSWEOCIImage:
    return PinnedSWEOCIImage(
        environment_image_id=IMAGE_ID,
        source_reference=IMAGE_SOURCE,
        manifest_digest=IMAGE_DIGEST,
    )


def _other_image() -> PinnedSWEOCIImage:
    digest = f"sha256:{'b' * 64}"
    source = "docker://docker.io/swebench/sweb.eval.x86_64.other_1776_project-2:latest"
    return PinnedSWEOCIImage(
        environment_image_id=f"{source}@{digest}",
        source_reference=source,
        manifest_digest=digest,
    )


@dataclass(slots=True)
class _FakeMaterializerExecutor:
    output: bytes = b"synthetic leased SIF"
    returncode: int = 0
    error: OSError | None = None
    calls: list[tuple[tuple[str, ...], Path, dict[str, str], float]] = field(default_factory=list)

    def execute(
        self,
        argv: object,
        *,
        cwd: Path,
        environment: object,
        footprint_root: Path,
        capacity_bytes: int,
        timeout_seconds: float,
    ) -> MaterializerProcessResult:
        del footprint_root, capacity_bytes
        arguments = tuple(argv)  # type: ignore[arg-type]
        variables = dict(environment)  # type: ignore[arg-type]
        self.calls.append((arguments, cwd, variables, timeout_seconds))
        if self.error is not None:
            raise self.error
        Path(arguments[-2]).write_bytes(self.output)
        return MaterializerProcessResult(self.returncode)


def _materializer(
    tmp_path: Path,
    *,
    executor: _FakeMaterializerExecutor | None = None,
    capacity_bytes: int = 1024 * 1024,
    executable_content: bytes = b"#!/bin/sh\nexit 0\n",
    version: str = "apptainer version 1.3.6",
    timeout_seconds: float = 17.0,
) -> tuple[ApptainerOCIImageMaterializer, _FakeMaterializerExecutor]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = _executable(tmp_path / "apptainer", executable_content)
    storage_root = tmp_path / "materializer"
    storage_root.mkdir()
    process = executor or _FakeMaterializerExecutor()
    return (
        ApptainerOCIImageMaterializer(
            apptainer_executable=executable,
            apptainer_version=version,
            storage_root=storage_root.resolve(),
            capacity_bytes=capacity_bytes,
            executor=process,
            timeout_seconds=timeout_seconds,
        ),
        process,
    )


def _lease_directories(root: Path) -> tuple[Path, ...]:
    leases = root / "leases"
    return tuple(leases.iterdir()) if leases.is_dir() else ()


def test_materializer_uses_exact_digest_argv_and_reuses_atomic_sif(
    tmp_path: Path,
) -> None:
    materializer, executor = _materializer(tmp_path)
    image = _image()

    with materializer.lease(image) as materialized:
        assert materialized.environment_image_id == image.environment_image_id
        assert materialized.manifest_digest == image.manifest_digest
        assert materialized.image_path.name == "image.sif"
        assert materialized.image_path.read_bytes() == b"synthetic leased SIF"
        assert not (materialized.image_path.parent / "image.partial.sif").exists()
        assert materialized.image_path.is_relative_to(materializer.storage_root)

        assert len(executor.calls) == 1
        argv, cwd, environment, timeout = executor.calls[0]
        assert argv == (
            str(materializer.apptainer_executable),
            "pull",
            "--disable-cache",
            "--arch",
            "amd64",
            str(executor.calls[0][1].parent / "image.partial.sif"),
            image.immutable_source_reference,
        )
        assert argv[-1] != image.source_reference
        lease_root = cwd.parent
        assert cwd == lease_root / "work"
        assert environment["APPTAINER_CACHEDIR"] == str(lease_root / "cache")
        assert environment["APPTAINER_TMPDIR"] == str(lease_root / "tmp")
        assert environment["TMPDIR"] == str(lease_root / "tmp")
        assert timeout == 17.0

    assert materialized.image_path.exists()
    assert _lease_directories(materializer.storage_root) == ()

    with materializer.lease(image) as reused:
        assert reused.image_path == materialized.image_path
        assert reused.image_path.read_bytes() == b"synthetic leased SIF"
    assert len(executor.calls) == 1


def test_materializer_switches_its_single_cached_image_and_repulls_old_identity(
    tmp_path: Path,
) -> None:
    materializer, executor = _materializer(tmp_path)
    first_image = _image()
    second_image = _other_image()

    with materializer.lease(first_image) as first:
        cached_path = first.image_path
        assert cached_path.read_bytes() == b"synthetic leased SIF"

    executor.output = b"second synthetic SIF"
    with materializer.lease(second_image) as second:
        assert second.image_path == cached_path
        assert second.environment_image_id == second_image.environment_image_id
        assert second.manifest_digest == second_image.manifest_digest
        assert second.image_path.read_bytes() == b"second synthetic SIF"

    assert len(executor.calls) == 2
    assert executor.calls[1][0][-1] == second_image.immutable_source_reference
    assert _lease_directories(materializer.storage_root) == ()

    executor.output = b"first identity repulled"
    with materializer.lease(first_image) as first_again:
        assert first_again.image_path == cached_path
        assert first_again.image_path.read_bytes() == b"first identity repulled"

    assert len(executor.calls) == 3
    assert executor.calls[2][0][-1] == first_image.immutable_source_reference
    assert _lease_directories(materializer.storage_root) == ()


def test_materializer_rejects_over_capacity_and_removes_partial_tree(
    tmp_path: Path,
) -> None:
    executor = _FakeMaterializerExecutor(output=b"x" * 33)
    materializer, _ = _materializer(
        tmp_path,
        executor=executor,
        capacity_bytes=32,
    )

    with pytest.raises(SWEHarnessInfrastructureError):
        with materializer.lease(_image()):
            raise AssertionError("an over-capacity SIF must never be published")

    assert len(executor.calls) == 1
    assert _lease_directories(materializer.storage_root) == ()


def test_materializer_cleans_partial_tree_after_subprocess_failure(tmp_path: Path) -> None:
    executor = _FakeMaterializerExecutor(output=b"partial", returncode=9)
    materializer, _ = _materializer(tmp_path, executor=executor)

    with pytest.raises(SWEHarnessInfrastructureError):
        with materializer.lease(_image()):
            raise AssertionError("a failed process must not publish a lease")

    assert len(executor.calls) == 1
    assert _lease_directories(materializer.storage_root) == ()


def test_materializer_retains_sif_when_lease_body_fails(tmp_path: Path) -> None:
    materializer, executor = _materializer(tmp_path)
    leased_path: Path | None = None

    def fail_inside_lease() -> None:
        nonlocal leased_path
        with materializer.lease(_image()) as materialized:
            leased_path = materialized.image_path
            raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError):
        fail_inside_lease()

    assert leased_path is not None
    assert leased_path.exists()
    assert _lease_directories(materializer.storage_root) == ()
    with materializer.lease(_image()) as reused:
        assert reused.image_path == leased_path
    assert len(executor.calls) == 1


def test_materializer_rejects_a_concurrent_second_lease(tmp_path: Path) -> None:
    materializer, executor = _materializer(tmp_path)

    with materializer.lease(_image()) as first:
        assert first.image_path.is_file()
        with pytest.raises(SWEHarnessInfrastructureError):
            with materializer.lease(_image()):
                raise AssertionError("concurrent lease unexpectedly opened")

    assert len(executor.calls) == 1
    assert materializer.max_active_leases == 1
    assert _lease_directories(materializer.storage_root) == ()


def test_materializer_id_binds_executable_version_policy_and_capacity(
    tmp_path: Path,
) -> None:
    first, _ = _materializer(tmp_path / "first")
    same, _ = _materializer(tmp_path / "same")
    different_version, _ = _materializer(
        tmp_path / "version",
        version="apptainer version 1.4.0",
    )
    different_capacity, _ = _materializer(
        tmp_path / "capacity",
        capacity_bytes=2 * 1024 * 1024,
    )
    different_executable, _ = _materializer(
        tmp_path / "executable",
        executable_content=b"#!/bin/sh\nexit 7\n",
    )
    different_policy, _ = _materializer(
        tmp_path / "policy",
        timeout_seconds=18.0,
    )

    assert same.materializer_id == first.materializer_id
    assert different_version.materializer_id != first.materializer_id
    assert different_capacity.materializer_id != first.materializer_id
    assert different_executable.materializer_id != first.materializer_id
    assert different_policy.materializer_id != first.materializer_id


def test_materializer_rejects_changed_apptainer_before_subprocess(tmp_path: Path) -> None:
    materializer, executor = _materializer(tmp_path)
    materializer.apptainer_executable.write_bytes(b"changed executable")

    with pytest.raises(SWEHarnessInfrastructureError):
        with materializer.lease(_image()):
            raise AssertionError("changed executable unexpectedly materialized")

    assert executor.calls == []
    assert _lease_directories(materializer.storage_root) == ()


@pytest.mark.parametrize(
    ("platform_os", "platform_architecture"),
    [("windows", "amd64"), ("linux", "arm64")],
)
def test_materializer_rejects_unsupported_platform_before_subprocess(
    tmp_path: Path,
    platform_os: str,
    platform_architecture: str,
) -> None:
    materializer, executor = _materializer(tmp_path)
    image = PinnedSWEOCIImage(
        environment_image_id=IMAGE_ID,
        source_reference=IMAGE_SOURCE,
        manifest_digest=IMAGE_DIGEST,
        platform_os=platform_os,
        platform_architecture=platform_architecture,
    )

    with pytest.raises(ValueError):
        with materializer.lease(image):
            raise AssertionError("unsupported platform unexpectedly materialized")

    assert executor.calls == []


_OVER_CAPACITY_WRITER = r"""#!/usr/bin/env python3
import os
import pathlib
import sys
import time

output = pathlib.Path(sys.argv[-2])
with output.open("wb") as stream:
    while output.stat().st_size <= 128 * 1024:
        stream.write(b"x" * 4096)
        stream.flush()
        os.fsync(stream.fileno())
        time.sleep(0.002)
time.sleep(2.0)
pathlib.Path(os.environ["FAKE_COMPLETION_MARKER"]).write_text("not-stopped")
"""


def test_subprocess_executor_stops_writer_while_capacity_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path / "fake-apptainer",
        _OVER_CAPACITY_WRITER.encode(),
    )
    storage_root = tmp_path / "materializer"
    storage_root.mkdir()
    completion_marker = tmp_path / "writer-completed"
    monkeypatch.setenv("FAKE_COMPLETION_MARKER", str(completion_marker))
    materializer = ApptainerOCIImageMaterializer(
        apptainer_executable=executable,
        apptainer_version="fake-apptainer@1",
        storage_root=storage_root.resolve(),
        capacity_bytes=16 * 1024,
        executor=SubprocessMaterializerProcessExecutor(),
        timeout_seconds=5.0,
    )

    with pytest.raises(SWEHarnessInfrastructureError):
        with materializer.lease(_image()):
            raise AssertionError("over-capacity writer unexpectedly published a lease")

    assert not completion_marker.exists()
    assert _lease_directories(materializer.storage_root) == ()
