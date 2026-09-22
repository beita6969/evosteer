"""Private deployment input for one canonical Protocol 10 torchrun attempt.

The file routes already-frozen scientific inputs to the formal composition
root.  It deliberately contains paths and SGLang deployment controls, so it
must remain in the private run directory and must never be committed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from skillev.contracts import JsonValue, normalize_json
from skillev.experiments import FormalMethodV10
from skillev.runtime.formal_sglang_runtime import FormalSGLangRuntimeBinding
from skillev.runtime.formal_storage import FormalStorageBinding
from skillev.runtime.gpu_topology import ThreeGPURolePolicy

PROTOCOL_V10_ATTEMPT_INPUT_FORMAT: Final = "skillev-private-protocol-v10-attempt-input@7"


class ProtocolV10AdmissionMode(StrEnum):
    FORMAL = "formal"
    INTEGRATION_SMOKE = "integration-smoke"


def _absolute_path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be non-empty text")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


@dataclass(frozen=True, slots=True)
class ProtocolV10AttemptInput:
    """One method-specific private input shared by every torchrun rank."""

    method: FormalMethodV10
    protocol_config: Path
    experiment_config: Path
    catalog_root: Path
    session_deployments: Path
    application_input: Path
    hardware: ThreeGPURolePolicy
    sglang: FormalSGLangRuntimeBinding
    storage: FormalStorageBinding
    attempt_root: Path
    resume_snapshot: Path | None = None
    admission_mode: ProtocolV10AdmissionMode = ProtocolV10AdmissionMode.FORMAL
    integration_smoke_steps: int | None = None
    format: str = PROTOCOL_V10_ATTEMPT_INPUT_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V10_ATTEMPT_INPUT_FORMAT:
            raise ValueError("unsupported Protocol 10 attempt input format")
        if not isinstance(self.method, FormalMethodV10):
            raise TypeError("Protocol 10 attempt input requires a formal method")
        if not isinstance(self.admission_mode, ProtocolV10AdmissionMode):
            raise TypeError("Protocol 10 attempt admission mode is not closed")
        required_files = (
            self.protocol_config,
            self.experiment_config,
            self.session_deployments,
            self.application_input,
        )
        if any(not isinstance(path, Path) or not path.is_absolute() for path in required_files):
            raise ValueError("Protocol 10 input file routes must be absolute Paths")
        required_directories = (self.catalog_root, self.attempt_root)
        if any(
            not isinstance(path, Path) or not path.is_absolute() for path in required_directories
        ):
            raise ValueError("Protocol 10 private roots must be absolute Paths")
        if self.resume_snapshot is not None and (
            not isinstance(self.resume_snapshot, Path) or not self.resume_snapshot.is_absolute()
        ):
            raise ValueError("Protocol 10 resume snapshot must be an absolute Path")
        if not isinstance(self.hardware, ThreeGPURolePolicy):
            raise TypeError("Protocol 10 input requires the three-role hardware policy")
        if not isinstance(self.sglang, FormalSGLangRuntimeBinding):
            raise TypeError("Protocol 10 input requires the formal SGLang binding")
        if not isinstance(self.storage, FormalStorageBinding):
            raise TypeError("Protocol 10 input requires the formal storage binding")
        if self.storage.output_root.resolve() != self.attempt_root.parent.resolve():
            raise ValueError("Protocol 10 attempt must be placed directly under its output root")
        if (
            self.sglang.workflow.max_inflight_model_requests
            > self.sglang.workflow.max_resident_trajectories
        ):
            raise ValueError("Protocol 10 model concurrency exceeds resident trajectories")
        if self.sglang.adapter_export_root != (self.attempt_root / "adapters").resolve():
            raise ValueError("Protocol 10 adapters must remain in the attempt-private root")
        if (
            self.admission_mode is ProtocolV10AdmissionMode.INTEGRATION_SMOKE
            and "non-formal-smoke" not in self.attempt_root.parts
        ):
            raise ValueError("integration smoke output must be explicitly non-formal")
        if self.admission_mode is ProtocolV10AdmissionMode.FORMAL:
            if self.integration_smoke_steps is not None:
                raise ValueError("formal attempts cannot have a smoke step limit")
        elif (
            type(self.integration_smoke_steps) is not int
            or not 1 <= self.integration_smoke_steps <= 22
        ):
            raise ValueError("integration smoke must execute between 1 and 22 committed steps")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "admission_mode": self.admission_mode.value,
            "application_input": str(self.application_input),
            "attempt_root": str(self.attempt_root),
            "catalog_root": str(self.catalog_root),
            "experiment_config": str(self.experiment_config),
            "format": self.format,
            "hardware": normalize_json(self.hardware.to_value()),
            "integration_smoke_steps": self.integration_smoke_steps,
            "method": self.method.value,
            "protocol_config": str(self.protocol_config),
            "resume_snapshot": str(self.resume_snapshot) if self.resume_snapshot else None,
            "session_deployments": str(self.session_deployments),
            "sglang": self.sglang.to_value(),
            "storage": self.storage.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        fields = {
            "admission_mode",
            "application_input",
            "attempt_root",
            "catalog_root",
            "experiment_config",
            "format",
            "hardware",
            "integration_smoke_steps",
            "method",
            "protocol_config",
            "resume_snapshot",
            "session_deployments",
            "sglang",
            "storage",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Protocol 10 attempt input has incompatible fields")
        format_value = normalized["format"]
        method_value = normalized["method"]
        admission_value = normalized["admission_mode"]
        if any(type(value) is not str for value in (format_value, method_value, admission_value)):
            raise TypeError("Protocol 10 attempt format, method, and admission must be text")
        resume_value = normalized["resume_snapshot"]
        if resume_value is not None and type(resume_value) is not str:
            raise TypeError("Protocol 10 resume snapshot must be text or null")
        smoke_steps = normalized["integration_smoke_steps"]
        if smoke_steps is not None and type(smoke_steps) is not int:
            raise TypeError("Protocol 10 integration smoke steps must be an integer or null")
        return cls(
            method=FormalMethodV10(method_value),
            admission_mode=ProtocolV10AdmissionMode(admission_value),
            protocol_config=_absolute_path(normalized["protocol_config"], label="protocol_config"),
            experiment_config=_absolute_path(
                normalized["experiment_config"], label="experiment_config"
            ),
            catalog_root=_absolute_path(normalized["catalog_root"], label="catalog_root"),
            session_deployments=_absolute_path(
                normalized["session_deployments"], label="session_deployments"
            ),
            application_input=_absolute_path(
                normalized["application_input"], label="application_input"
            ),
            hardware=ThreeGPURolePolicy.from_mapping(normalized["hardware"]),
            integration_smoke_steps=smoke_steps,
            sglang=FormalSGLangRuntimeBinding.from_value(normalized["sglang"]),
            storage=FormalStorageBinding.from_value(normalized["storage"]),
            attempt_root=_absolute_path(normalized["attempt_root"], label="attempt_root"),
            resume_snapshot=(
                _absolute_path(resume_value, label="resume_snapshot")
                if resume_value is not None
                else None
            ),
            format=format_value,
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        if not path.is_absolute():
            raise ValueError("Protocol 10 exact input path must be absolute")
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "PROTOCOL_V10_ATTEMPT_INPUT_FORMAT",
    "ProtocolV10AdmissionMode",
    "ProtocolV10AttemptInput",
]
