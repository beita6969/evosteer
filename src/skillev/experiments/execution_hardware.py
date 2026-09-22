"""Path-free CUDA runtime provenance for formal training attempts.

Formal inputs deliberately do not name a particular allocated device: the
scheduler chooses that resource.  A child therefore measures its CUDA runtime
before it loads a model and records the result as a source event.  Seven-arm
aggregation accepts results only when those measured identities agree.

This module is intentionally dependency-light.  CUDA interrogation lives in
the private worker package; this public side only owns the canonical wire and
the source-event admission rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256
from skillev.runtime import AttemptBuilderKind, EventEnvelope, EventType

if TYPE_CHECKING:
    from .attempt_identity import PublishedAttemptIdentity


EXECUTION_HARDWARE_IDENTITY_FORMAT = "skillev-execution-hardware@1"
FORMAL_EXECUTION_HARDWARE_ATTESTATION_FORMAT = "skillev-formal-execution-hardware-attestation@1"


def _required_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionHardwareIdentity:
    """The measured CUDA hardware/runtime set for one formal child.

    The identity intentionally omits hostnames, device UUIDs, paths, and job
    identifiers.  Those values would make equivalent allocations incomparable
    without constraining a scientific control.  It retains every runtime value
    that can affect CUDA kernels, numerical reductions, or serialization.
    """

    accelerator_name: str
    compute_capability: str
    visible_device_count: int
    nvidia_driver_version: str
    cuda_runtime_version: str
    cudnn_version: str
    nccl_version: str
    kernel_release: str
    safetensors_version: str
    format: str = EXECUTION_HARDWARE_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "accelerator_name",
            "compute_capability",
            "nvidia_driver_version",
            "cuda_runtime_version",
            "cudnn_version",
            "nccl_version",
            "kernel_release",
            "safetensors_version",
        ):
            _required_text(getattr(self, field), field=field)
        if type(self.visible_device_count) is not int or self.visible_device_count < 1:
            raise ValueError("visible_device_count must be positive")
        if self.format != EXECUTION_HARDWARE_IDENTITY_FORMAT:
            raise ValueError("unsupported execution hardware identity format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "accelerator_name": self.accelerator_name,
            "compute_capability": self.compute_capability,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cudnn_version": self.cudnn_version,
            "format": self.format,
            "kernel_release": self.kernel_release,
            "nccl_version": self.nccl_version,
            "nvidia_driver_version": self.nvidia_driver_version,
            "safetensors_version": self.safetensors_version,
            "visible_device_count": self.visible_device_count,
        }

    @classmethod
    def from_value(cls, value: object) -> ExecutionHardwareIdentity:
        normalized = normalize_json(value)
        fields = {
            "accelerator_name",
            "compute_capability",
            "cuda_runtime_version",
            "cudnn_version",
            "format",
            "kernel_release",
            "nccl_version",
            "nvidia_driver_version",
            "safetensors_version",
            "visible_device_count",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("execution hardware identity has incompatible fields")
        text_fields = fields - {"visible_device_count"}
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("execution hardware identity text fields must be strings")
        if type(normalized["visible_device_count"]) is not int:
            raise TypeError("execution hardware visible device count must be an integer")
        return cls(
            accelerator_name=normalized["accelerator_name"],
            compute_capability=normalized["compute_capability"],
            visible_device_count=normalized["visible_device_count"],
            nvidia_driver_version=normalized["nvidia_driver_version"],
            cuda_runtime_version=normalized["cuda_runtime_version"],
            cudnn_version=normalized["cudnn_version"],
            nccl_version=normalized["nccl_version"],
            kernel_release=normalized["kernel_release"],
            safetensors_version=normalized["safetensors_version"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalExecutionHardwareAttestation:
    """One source event proving which CUDA environment a formal child used."""

    attempt_id: str
    builder_kind: AttemptBuilderKind
    exact_input_sha256: str
    public_identity_content_hash: str
    formal_execution_content_hash: str
    measured_hardware: ExecutionHardwareIdentity
    format: str = FORMAL_EXECUTION_HARDWARE_ATTESTATION_FORMAT

    def __post_init__(self) -> None:
        _required_text(self.attempt_id, field="formal hardware attestation attempt_id")
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("formal hardware attestation requires a closed builder kind")
        for field in (
            "exact_input_sha256",
            "public_identity_content_hash",
            "formal_execution_content_hash",
        ):
            validate_sha256(getattr(self, field))
        if not isinstance(self.measured_hardware, ExecutionHardwareIdentity):
            raise TypeError("formal hardware attestation requires a hardware identity")
        if self.format != FORMAL_EXECUTION_HARDWARE_ATTESTATION_FORMAT:
            raise ValueError("unsupported formal execution hardware attestation format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "exact_input_sha256": self.exact_input_sha256,
            "formal_execution_content_hash": self.formal_execution_content_hash,
            "format": self.format,
            "measured_hardware": self.measured_hardware.to_value(),
            "public_identity_content_hash": self.public_identity_content_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalExecutionHardwareAttestation:
        normalized = normalize_json(value)
        fields = {
            "attempt_id",
            "builder_kind",
            "exact_input_sha256",
            "formal_execution_content_hash",
            "format",
            "measured_hardware",
            "public_identity_content_hash",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal execution hardware attestation has incompatible fields")
        for field in (
            "attempt_id",
            "builder_kind",
            "exact_input_sha256",
            "formal_execution_content_hash",
            "format",
            "public_identity_content_hash",
        ):
            if type(normalized[field]) is not str:
                raise TypeError("formal execution hardware attestation text fields must be text")
        return cls(
            attempt_id=normalized["attempt_id"],
            builder_kind=AttemptBuilderKind(normalized["builder_kind"]),
            exact_input_sha256=normalized["exact_input_sha256"],
            public_identity_content_hash=normalized["public_identity_content_hash"],
            formal_execution_content_hash=normalized["formal_execution_content_hash"],
            measured_hardware=ExecutionHardwareIdentity.from_value(normalized["measured_hardware"]),
            format=normalized["format"],
        )


def require_formal_execution_hardware_attestation(
    envelopes: Sequence[EventEnvelope],
    *,
    identity: PublishedAttemptIdentity,
    attempt_id: str,
) -> FormalExecutionHardwareAttestation | None:
    """Require the measured CUDA event immediately after build attestation.

    Correctness fixtures have no formal execution freeze and deliberately do
    not need this event.  Formal children must emit it before catalog/model
    hydration so an aggregate cannot mix runtimes invisibly.
    """

    if identity.formal_execution is None:
        return None
    if len(envelopes) < 2:
        raise ValueError("formal source lacks the mandatory pre-model provenance events")
    build_event = envelopes[0]
    if (
        build_event.event_type is not EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED
        or build_event.producer_id != "skillev-formal-build"
        or build_event.producer_seq != 1
        or build_event.attempt_id != attempt_id
    ):
        raise ValueError("formal execution hardware attestation lacks a leading build attestation")
    matches = tuple(
        (index, event)
        for index, event in enumerate(envelopes)
        if event.event_type is EventType.FORMAL_EXECUTION_HARDWARE_ATTESTED
    )
    if len(matches) != 1:
        raise ValueError("formal source requires exactly one execution hardware attestation")
    index, event = matches[0]
    if (
        index != 1
        or event.producer_id != "skillev-formal-build"
        or event.producer_seq != 2
        or event.attempt_id != attempt_id
    ):
        raise ValueError("formal execution hardware attestation must follow build attestation")
    attestation = FormalExecutionHardwareAttestation.from_value(event.payload)
    if (
        attestation.attempt_id != attempt_id
        or attestation.builder_kind is not identity.builder_kind
        or attestation.exact_input_sha256 != identity.exact_input_sha256
        or attestation.public_identity_content_hash != identity.content_hash
        or attestation.formal_execution_content_hash != identity.formal_execution.content_hash
    ):
        raise ValueError("formal execution hardware attestation differs from public identity")
    return attestation


__all__ = [
    "EXECUTION_HARDWARE_IDENTITY_FORMAT",
    "FORMAL_EXECUTION_HARDWARE_ATTESTATION_FORMAT",
    "ExecutionHardwareIdentity",
    "FormalExecutionHardwareAttestation",
    "require_formal_execution_hardware_attestation",
]
