"""Formal-child build measurement and source-attestation regressions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import skillev_private.experiments.benchmark_attempt_builders as attempt_builders
import skillev_private.experiments.implementation_build as implementation_build
from skillev_private.experiments.implementation_build import (
    FormalImplementationBuildVerificationError,
    PrivateImplementationBuildDeployment,
)

from skillev.audit.formal_build_attestation import (
    require_formal_implementation_build_attestation,
)
from skillev.audit.formal_hardware_attestation import (
    require_formal_execution_hardware_attestation,
)
from skillev.contracts import stable_hash
from skillev.experiments import (
    ExecutionHardwareIdentity,
    FormalExecutionHardwareAttestation,
    FormalImplementationBuildAttestation,
    ImplementationBuildIdentity,
    PublishedAttemptIdentity,
)
from skillev.experiments.build_identity import SourcePackageProvenance
from skillev.runtime import AttemptBuilderKind, AttemptRequest, EventEnvelope, EventType


def _build_identity() -> ImplementationBuildIdentity:
    return ImplementationBuildIdentity(
        source_commit="1" * 40,
        source_tree_hash=stable_hash("source-tree"),
        public_wheel_hash=stable_hash("public-wheel"),
        private_evaluation_wheel_hash=stable_hash("private-wheel"),
        lockfile_hash=stable_hash("lockfile"),
        python_version="3.13.0",
        torch_version="2.13.0",
        transformers_version="5.14.1",
        peft_version="0.19.1",
        tokenizers_version="0.22.2",
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        matmul_allow_tf32=False,
    )


def _hardware_identity() -> ExecutionHardwareIdentity:
    return ExecutionHardwareIdentity(
        accelerator_name="NVIDIA H200",
        compute_capability="9.0",
        visible_device_count=1,
        nvidia_driver_version="570.12",
        cuda_runtime_version="12.8",
        cudnn_version="91000",
        nccl_version="2.27.5",
        kernel_release="6.12.0-test",
        safetensors_version="0.5.3",
    )


def _deployment(
    source_tree_root: Path = Path("/private/source"),
) -> PrivateImplementationBuildDeployment:
    return PrivateImplementationBuildDeployment(
        source_tree_root=source_tree_root,
        public_wheel_path=Path("/private/wheels/skillev.whl"),
        private_evaluation_wheel_path=Path("/private/wheels/skillev-private.whl"),
        lockfile_path=Path("/private/source/uv.lock"),
    )


def test_private_deployment_measures_one_complete_build_without_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = _build_identity()
    source_root = (tmp_path / "source").resolve()
    source_root.mkdir()
    deployment = _deployment(source_root)
    hashes = {
        deployment.public_wheel_path: expected.public_wheel_hash,
        deployment.private_evaluation_wheel_path: expected.private_evaluation_wheel_hash,
        deployment.lockfile_path: expected.lockfile_hash,
    }
    versions = {
        "torch": expected.torch_version,
        "transformers": expected.transformers_version,
        "peft": expected.peft_version,
        "tokenizers": expected.tokenizers_version,
    }
    monkeypatch.setattr(
        implementation_build,
        "read_source_package_provenance",
        lambda root: SourcePackageProvenance(
            source_commit=expected.source_commit,
            source_tree_hash=expected.source_tree_hash,
        ),
    )
    monkeypatch.setattr(implementation_build, "_sha256_file", lambda path: hashes[path])
    monkeypatch.setattr(implementation_build, "_installed_version", lambda name: versions[name])
    monkeypatch.setattr(
        implementation_build,
        "_require_executed_package_roots",
        lambda root: None,
    )
    monkeypatch.setattr(
        implementation_build,
        "_backend_flags",
        lambda: (
            expected.deterministic_algorithms,
            expected.cudnn_deterministic,
            expected.cudnn_benchmark,
            expected.matmul_allow_tf32,
        ),
    )
    monkeypatch.setattr(
        implementation_build.platform,
        "python_version",
        lambda: expected.python_version,
    )

    assert PrivateImplementationBuildDeployment.from_value(deployment.to_value()) == deployment
    assert deployment.require_exact(expected) == expected
    with pytest.raises(FormalImplementationBuildVerificationError):
        deployment.require_exact(
            ImplementationBuildIdentity(
                source_commit=expected.source_commit,
                source_tree_hash=stable_hash("another-source-tree"),
                public_wheel_hash=expected.public_wheel_hash,
                private_evaluation_wheel_hash=expected.private_evaluation_wheel_hash,
                lockfile_hash=expected.lockfile_hash,
                python_version=expected.python_version,
                torch_version=expected.torch_version,
                transformers_version=expected.transformers_version,
                peft_version=expected.peft_version,
                tokenizers_version=expected.tokenizers_version,
                deterministic_algorithms=expected.deterministic_algorithms,
                cudnn_deterministic=expected.cudnn_deterministic,
                cudnn_benchmark=expected.cudnn_benchmark,
                matmul_allow_tf32=expected.matmul_allow_tf32,
            )
        )


@dataclass(slots=True)
class _BuildDeployment:
    expected: ImplementationBuildIdentity
    failure: Exception | None = None
    calls: int = 0

    def require_exact(self, expected: ImplementationBuildIdentity) -> ImplementationBuildIdentity:
        self.calls += 1
        assert expected == self.expected
        if self.failure is not None:
            raise self.failure
        return self.expected


@dataclass(slots=True)
class _Runtime:
    implementation_build_deployment: _BuildDeployment
    load_calls: int = 0

    def load_training_catalog(self, bundle: object) -> object:
        del bundle
        self.load_calls += 1
        raise _StopAfterAttestationError()


class _StopAfterAttestationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _PublicIdentity:
    content_hash: str


@dataclass(frozen=True, slots=True)
class _Exact:
    builder_kind: AttemptBuilderKind
    implementation_build: ImplementationBuildIdentity
    runtime: _Runtime
    protocol: object

    @property
    def catalog_bundle(self) -> object:
        """Supply the immutable catalog handle consumed after both attestations."""

        return object()

    def public_identity(self, *, exact_input_sha256: str) -> _PublicIdentity:
        assert exact_input_sha256 == stable_hash("exact-input")
        return _PublicIdentity(content_hash=stable_hash("public-identity"))


def _request(tmp_path: Path) -> AttemptRequest:
    return AttemptRequest(
        run_id="formal-build-test",
        attempt_id="formal-build-attempt",
        builder_kind=AttemptBuilderKind.FULL,
        exact_input_path=tmp_path / "exact.json",
        exact_input_sha256=stable_hash("exact-input"),
        private_bundle_directory=tmp_path / "private",
    )


def _install_fake_exact(
    monkeypatch: pytest.MonkeyPatch,
    exact: _Exact,
) -> None:
    monkeypatch.setattr(
        attempt_builders.PrivateBenchmarkAttemptInput,
        "read_verified",
        classmethod(lambda cls, path, expected_sha256: exact),
    )
    monkeypatch.setattr(
        attempt_builders,
        "measure_formal_execution_hardware",
        _hardware_identity,
    )


def test_formal_builder_rejects_an_unmeasured_build_before_catalog_hydration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _build_identity()
    deployment = _BuildDeployment(
        expected=expected,
        failure=FormalImplementationBuildVerificationError("fixture mismatch"),
    )
    runtime = _Runtime(implementation_build_deployment=deployment)
    exact = _Exact(
        builder_kind=AttemptBuilderKind.FULL,
        implementation_build=expected,
        runtime=runtime,
        protocol=SimpleNamespace(
            formal_execution=SimpleNamespace(content_hash=stable_hash("freeze"))
        ),
    )
    _install_fake_exact(monkeypatch, exact)

    with pytest.raises(FormalImplementationBuildVerificationError):
        attempt_builders.build_private_benchmark_attempt(_request(tmp_path))

    assert deployment.calls == 1
    assert runtime.load_calls == 0
    assert not (tmp_path / "private" / "events.jsonl").exists()


def test_formal_builder_writes_the_build_event_before_catalog_hydration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _build_identity()
    deployment = _BuildDeployment(expected=expected)
    runtime = _Runtime(implementation_build_deployment=deployment)
    exact = _Exact(
        builder_kind=AttemptBuilderKind.FULL,
        implementation_build=expected,
        runtime=runtime,
        protocol=SimpleNamespace(
            formal_execution=SimpleNamespace(content_hash=stable_hash("freeze"))
        ),
    )
    _install_fake_exact(monkeypatch, exact)

    with pytest.raises(_StopAfterAttestationError):
        attempt_builders.build_private_benchmark_attempt(_request(tmp_path))

    envelopes = tuple(
        EventEnvelope.from_value(json.loads(line))
        for line in (tmp_path / "private" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    )
    build = FormalImplementationBuildAttestation.from_value(envelopes[0].payload)
    hardware = FormalExecutionHardwareAttestation.from_value(envelopes[1].payload)
    assert tuple(event.event_type for event in envelopes) == (
        EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED,
        EventType.FORMAL_EXECUTION_HARDWARE_ATTESTED,
    )
    assert tuple(event.producer_seq for event in envelopes) == (1, 2)
    assert build.measured_implementation_build == expected
    assert hardware.measured_hardware == _hardware_identity()
    assert deployment.calls == 1
    assert runtime.load_calls == 1


def test_audit_requires_the_attested_build_to_match_the_formal_freeze() -> None:
    expected = _build_identity()
    attestation = FormalImplementationBuildAttestation(
        attempt_id="formal-build-attempt",
        builder_kind=AttemptBuilderKind.FULL,
        exact_input_sha256=stable_hash("exact-input"),
        public_identity_content_hash=stable_hash("public-identity"),
        formal_execution_content_hash=stable_hash("freeze"),
        expected_implementation_build=expected,
        measured_implementation_build=expected,
    )
    envelope = EventEnvelope.create(
        event_type=EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED,
        run_id="formal-build-test",
        attempt_id=attestation.attempt_id,
        producer_id="skillev-formal-build",
        producer_seq=1,
        occurred_at="1970-01-01T00:00:00Z",
        payload=attestation.to_value(),
    )
    identity = cast(
        PublishedAttemptIdentity,
        SimpleNamespace(
            builder_kind=AttemptBuilderKind.FULL,
            content_hash=attestation.public_identity_content_hash,
            exact_input_sha256=attestation.exact_input_sha256,
            formal_execution=SimpleNamespace(
                content_hash=attestation.formal_execution_content_hash,
                implementation_build=expected,
            ),
        ),
    )

    assert (
        require_formal_implementation_build_attestation(
            (envelope,),
            identity=identity,
            attempt_id=attestation.attempt_id,
        )
        == attestation
    )

    tampered = attestation.to_value()
    tampered["formal_execution_content_hash"] = stable_hash("another-freeze")
    with pytest.raises(ValueError):
        require_formal_implementation_build_attestation(
            (
                EventEnvelope.create(
                    event_type=EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED,
                    run_id="formal-build-test",
                    attempt_id=attestation.attempt_id,
                    producer_id="skillev-formal-build",
                    producer_seq=1,
                    occurred_at="1970-01-01T00:00:00Z",
                    payload=tampered,
                ),
            ),
            identity=identity,
            attempt_id=attestation.attempt_id,
        )


def test_audit_requires_the_attested_hardware_to_follow_build_attestation() -> None:
    expected = _build_identity()
    hardware = _hardware_identity()
    build = FormalImplementationBuildAttestation(
        attempt_id="formal-build-attempt",
        builder_kind=AttemptBuilderKind.FULL,
        exact_input_sha256=stable_hash("exact-input"),
        public_identity_content_hash=stable_hash("public-identity"),
        formal_execution_content_hash=stable_hash("freeze"),
        expected_implementation_build=expected,
        measured_implementation_build=expected,
    )
    attestation = FormalExecutionHardwareAttestation(
        attempt_id=build.attempt_id,
        builder_kind=build.builder_kind,
        exact_input_sha256=build.exact_input_sha256,
        public_identity_content_hash=build.public_identity_content_hash,
        formal_execution_content_hash=build.formal_execution_content_hash,
        measured_hardware=hardware,
    )
    envelopes = (
        EventEnvelope.create(
            event_type=EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED,
            run_id="formal-build-test",
            attempt_id=build.attempt_id,
            producer_id="skillev-formal-build",
            producer_seq=1,
            occurred_at="1970-01-01T00:00:00Z",
            payload=build.to_value(),
        ),
        EventEnvelope.create(
            event_type=EventType.FORMAL_EXECUTION_HARDWARE_ATTESTED,
            run_id="formal-build-test",
            attempt_id=build.attempt_id,
            producer_id="skillev-formal-build",
            producer_seq=2,
            occurred_at="1970-01-01T00:00:00Z",
            payload=attestation.to_value(),
        ),
    )
    identity = cast(
        PublishedAttemptIdentity,
        SimpleNamespace(
            builder_kind=build.builder_kind,
            content_hash=build.public_identity_content_hash,
            exact_input_sha256=build.exact_input_sha256,
            formal_execution=SimpleNamespace(content_hash=build.formal_execution_content_hash),
        ),
    )

    assert (
        require_formal_execution_hardware_attestation(
            envelopes,
            identity=identity,
            attempt_id=build.attempt_id,
        )
        == attestation
    )
    with pytest.raises(ValueError):
        require_formal_execution_hardware_attestation(
            (envelopes[1], envelopes[0]),
            identity=identity,
            attempt_id=build.attempt_id,
        )
