"""Dependency-light execution evidence for one-shot Gate 4c processes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256

GATE_4C_CHILD_TERMINAL_FORMAT: Final = "skillev-gate-4c-child-terminal@3"
GATE_4C_OPERATOR_TERMINAL_FORMAT: Final = "skillev-gate-4c-operator-terminal@1"
GATE_4C_STAGE_EVENT_FORMAT: Final = "skillev-gate-4c-stage-event@1"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _optional_hash(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{field} must be text or null")
    validate_sha256(value)
    return value


class Gate4cStage(StrEnum):
    PROCESS_START = "process-start"
    SPEC_READ = "spec-read"
    DETERMINISM_CONFIGURED = "determinism-configured"
    HARDWARE_ATTESTED = "hardware-attested"
    BACKEND_ATTESTED = "backend-attested"
    BUNDLE_VERIFIED = "bundle-verified"
    CHECKPOINT_METADATA_VERIFIED = "checkpoint-metadata-verified"
    MODEL_CONFIG_CONSTRUCTED = "model-config-constructed"
    BACKBONE_CONSTRUCT = "backbone-construct"
    INITIAL_CHECKPOINT_LOAD = "initial-checkpoint-load"
    INITIAL_STATE_BIND = "initial-state-bind"
    WORST_SHAPE_FIXTURE = "worst-shape-fixture"
    WORST_SHAPE_SCORE = "worst-shape-score"
    STATELESS_ROLLOUT = "stateless-rollout"
    CACHED_ROLLOUT = "cached-rollout"
    CACHE_EQUIVALENCE = "cache-equivalence"
    PRODUCTION_SCORE = "production-score"
    OPTIMIZER_STEP = "optimizer-step"
    SOURCE_COMMIT = "source-commit"
    CHECKPOINT_SAVE = "checkpoint-save"
    CHECKPOINT_RESTORE = "checkpoint-restore"
    RESULT_WRITE = "result-write"
    PROCESS_SUCCESS = "process-success"


class Gate4cStageState(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class Gate4cFailureCode(StrEnum):
    PYTHON_EXCEPTION = "python-exception"
    PROCESS_EXITED_WITHOUT_CHILD_TERMINAL = "process-exited-without-child-terminal"
    PROCESS_SIGNALLED = "process-signalled"
    LOG_CAPTURE_FAILED = "log-capture-failed"
    GPU_SAMPLER_FAILED = "gpu-sampler-failed"


class Gate4cChildStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class Gate4cOperatorStatus(StrEnum):
    PASSED = "passed"
    FAILED_CONSUMED_NO_RETRY = "failed-consumed-no-retry"


@dataclass(frozen=True, slots=True)
class Gate4cCudaMemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    free_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.allocated_bytes,
            self.reserved_bytes,
            self.max_allocated_bytes,
            self.max_reserved_bytes,
            self.free_bytes,
            self.total_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Gate 4c CUDA memory values must be non-negative integers")
        if self.total_bytes < self.free_bytes:
            raise ValueError("Gate 4c CUDA free memory exceeds total memory")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "free_bytes": self.free_bytes,
            "max_allocated_bytes": self.max_allocated_bytes,
            "max_reserved_bytes": self.max_reserved_bytes,
            "reserved_bytes": self.reserved_bytes,
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_value(cls, value: object) -> Gate4cCudaMemorySnapshot:
        normalized = normalize_json(value)
        fields = {
            "allocated_bytes",
            "free_bytes",
            "max_allocated_bytes",
            "max_reserved_bytes",
            "reserved_bytes",
            "total_bytes",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Gate 4c CUDA memory snapshot has incompatible fields")
        if any(type(normalized[field]) is not int for field in fields):
            raise TypeError("Gate 4c CUDA memory fields must be integers")
        return cls(**{field: normalized[field] for field in fields})


@dataclass(frozen=True, slots=True)
class Gate4cStageEvent:
    ordinal: int
    stage: Gate4cStage
    state: Gate4cStageState
    monotonic_ns: int
    previous_event_hash: str | None
    cuda_memory: Gate4cCudaMemorySnapshot | None
    format: str = GATE_4C_STAGE_EVENT_FORMAT

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise ValueError("Gate 4c stage ordinal must be positive")
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 0:
            raise ValueError("Gate 4c stage monotonic time must be non-negative")
        if self.previous_event_hash is not None:
            validate_sha256(self.previous_event_hash)
        if self.format != GATE_4C_STAGE_EVENT_FORMAT:
            raise ValueError("unsupported Gate 4c stage event format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "cuda_memory": self.cuda_memory.to_value() if self.cuda_memory is not None else None,
            "format": self.format,
            "monotonic_ns": self.monotonic_ns,
            "ordinal": self.ordinal,
            "previous_event_hash": self.previous_event_hash,
            "stage": self.stage.value,
            "state": self.state.value,
        }

    @classmethod
    def from_value(cls, value: object) -> Gate4cStageEvent:
        normalized = normalize_json(value)
        fields = {
            "cuda_memory",
            "format",
            "monotonic_ns",
            "ordinal",
            "previous_event_hash",
            "stage",
            "state",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Gate 4c stage event has incompatible fields")
        if type(normalized["ordinal"]) is not int or type(normalized["monotonic_ns"]) is not int:
            raise TypeError("Gate 4c stage event counters must be integers")
        for field in ("format", "stage", "state"):
            if type(normalized[field]) is not str:
                raise TypeError("Gate 4c stage event text fields must be text")
        previous = normalized["previous_event_hash"]
        if previous is not None and type(previous) is not str:
            raise TypeError("Gate 4c previous event hash must be text or null")
        return cls(
            ordinal=normalized["ordinal"],
            stage=Gate4cStage(normalized["stage"]),
            state=Gate4cStageState(normalized["state"]),
            monotonic_ns=normalized["monotonic_ns"],
            previous_event_hash=previous,
            cuda_memory=(
                Gate4cCudaMemorySnapshot.from_value(normalized["cuda_memory"])
                if normalized["cuda_memory"] is not None
                else None
            ),
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class Gate4cChildTerminal:
    status: Gate4cChildStatus
    spec_content_hash: str | None
    final_stage: Gate4cStage
    exception_type: str | None
    traceback_sha256: str | None
    result_sha256: str | None
    stage_journal_sha256: str
    elapsed_ns: int
    format: str = GATE_4C_CHILD_TERMINAL_FORMAT

    def __post_init__(self) -> None:
        _optional_hash(self.spec_content_hash, field="spec_content_hash")
        _optional_hash(self.traceback_sha256, field="traceback_sha256")
        _optional_hash(self.result_sha256, field="result_sha256")
        validate_sha256(self.stage_journal_sha256)
        if type(self.elapsed_ns) is not int or self.elapsed_ns < 0:
            raise ValueError("Gate 4c child elapsed time must be non-negative")
        if self.status is Gate4cChildStatus.PASSED:
            if self.exception_type is not None or self.traceback_sha256 is not None:
                raise ValueError("successful Gate 4c child cannot contain failure evidence")
            if self.result_sha256 is None or self.spec_content_hash is None:
                raise ValueError("successful Gate 4c child requires result and spec identities")
        else:
            _text(self.exception_type, field="exception_type")
            if self.traceback_sha256 is None or self.result_sha256 is not None:
                raise ValueError("failed Gate 4c child requires traceback and no result")
        if self.format != GATE_4C_CHILD_TERMINAL_FORMAT:
            raise ValueError("unsupported Gate 4c child terminal format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "elapsed_ns": self.elapsed_ns,
            "exception_type": self.exception_type,
            "final_stage": self.final_stage.value,
            "format": self.format,
            "result_sha256": self.result_sha256,
            "spec_content_hash": self.spec_content_hash,
            "stage_journal_sha256": self.stage_journal_sha256,
            "status": self.status.value,
            "traceback_sha256": self.traceback_sha256,
        }

    @classmethod
    def passed(
        cls,
        *,
        spec_content_hash: str,
        final_stage: Gate4cStage,
        result_sha256: str,
        stage_journal_sha256: str,
        elapsed_ns: int,
    ) -> Gate4cChildTerminal:
        return cls(
            status=Gate4cChildStatus.PASSED,
            spec_content_hash=spec_content_hash,
            final_stage=final_stage,
            exception_type=None,
            traceback_sha256=None,
            result_sha256=result_sha256,
            stage_journal_sha256=stage_journal_sha256,
            elapsed_ns=elapsed_ns,
        )

    @classmethod
    def failed(
        cls,
        *,
        spec_content_hash: str | None,
        final_stage: Gate4cStage,
        exception_type: str,
        traceback_sha256: str,
        stage_journal_sha256: str,
        elapsed_ns: int,
    ) -> Gate4cChildTerminal:
        return cls(
            status=Gate4cChildStatus.FAILED,
            spec_content_hash=spec_content_hash,
            final_stage=final_stage,
            exception_type=exception_type,
            traceback_sha256=traceback_sha256,
            result_sha256=None,
            stage_journal_sha256=stage_journal_sha256,
            elapsed_ns=elapsed_ns,
        )

    @classmethod
    def from_value(cls, value: object) -> Gate4cChildTerminal:
        normalized = normalize_json(value)
        fields = {
            "elapsed_ns",
            "exception_type",
            "final_stage",
            "format",
            "result_sha256",
            "spec_content_hash",
            "stage_journal_sha256",
            "status",
            "traceback_sha256",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Gate 4c child terminal has incompatible fields")
        if type(normalized["elapsed_ns"]) is not int:
            raise TypeError("Gate 4c child terminal elapsed time must be an integer")
        for field in ("final_stage", "format", "stage_journal_sha256", "status"):
            if type(normalized[field]) is not str:
                raise TypeError("Gate 4c child terminal text fields must be text")
        for field in (
            "exception_type",
            "result_sha256",
            "spec_content_hash",
            "traceback_sha256",
        ):
            if normalized[field] is not None and type(normalized[field]) is not str:
                raise TypeError("Gate 4c child terminal optional fields must be text or null")
        return cls(
            status=Gate4cChildStatus(normalized["status"]),
            spec_content_hash=normalized["spec_content_hash"],
            final_stage=Gate4cStage(normalized["final_stage"]),
            exception_type=normalized["exception_type"],
            traceback_sha256=normalized["traceback_sha256"],
            result_sha256=normalized["result_sha256"],
            stage_journal_sha256=normalized["stage_journal_sha256"],
            elapsed_ns=normalized["elapsed_ns"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class Gate4cOperatorTerminal:
    status: Gate4cOperatorStatus
    failure_code: Gate4cFailureCode | None
    spec_content_hash: str
    physical_gpu: int
    expected_gpu_uuid: str
    child_pid: int | None
    child_exit_code: int | None
    child_signal: int | None
    stdout_sha256: str
    stderr_sha256: str
    gpu_samples_sha256: str
    child_terminal_present: bool
    child_terminal_sha256: str | None
    elapsed_ns: int
    format: str = GATE_4C_OPERATOR_TERMINAL_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "spec_content_hash",
            "stdout_sha256",
            "stderr_sha256",
            "gpu_samples_sha256",
        ):
            validate_sha256(getattr(self, field))
        _text(self.expected_gpu_uuid, field="expected_gpu_uuid")
        if type(self.physical_gpu) is not int or self.physical_gpu < 0:
            raise ValueError("Gate 4c physical GPU must be non-negative")
        if self.child_pid is not None and (type(self.child_pid) is not int or self.child_pid < 1):
            raise ValueError("Gate 4c child PID must be positive or null")
        if self.child_pid is None:
            if self.child_exit_code is not None or self.child_signal is not None:
                raise ValueError("Gate 4c absent child cannot have an exit status")
        elif (self.child_exit_code is None) == (self.child_signal is None):
            raise ValueError("Gate 4c operator requires exactly one exit code or signal")
        if type(self.child_terminal_present) is not bool:
            raise TypeError("Gate 4c child terminal presence must be boolean")
        if self.child_terminal_present:
            if self.child_terminal_sha256 is None:
                raise ValueError("Gate 4c present child terminal requires its identity")
            validate_sha256(self.child_terminal_sha256)
        elif self.child_terminal_sha256 is not None:
            raise ValueError("Gate 4c absent child terminal cannot have an identity")
        if type(self.elapsed_ns) is not int or self.elapsed_ns < 0:
            raise ValueError("Gate 4c operator elapsed time must be non-negative")
        if self.status is Gate4cOperatorStatus.PASSED:
            if self.failure_code is not None or self.child_exit_code != 0:
                raise ValueError("passed Gate 4c operator cannot contain failure evidence")
            if self.child_pid is None or not self.child_terminal_present:
                raise ValueError("passed Gate 4c operator requires a child terminal")
        elif self.failure_code is None:
            raise ValueError("failed Gate 4c operator requires a closed failure code")
        if self.format != GATE_4C_OPERATOR_TERMINAL_FORMAT:
            raise ValueError("unsupported Gate 4c operator terminal format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "child_exit_code": self.child_exit_code,
            "child_pid": self.child_pid,
            "child_signal": self.child_signal,
            "child_terminal_present": self.child_terminal_present,
            "child_terminal_sha256": self.child_terminal_sha256,
            "elapsed_ns": self.elapsed_ns,
            "expected_gpu_uuid": self.expected_gpu_uuid,
            "failure_code": self.failure_code.value if self.failure_code is not None else None,
            "format": self.format,
            "gpu_samples_sha256": self.gpu_samples_sha256,
            "physical_gpu": self.physical_gpu,
            "spec_content_hash": self.spec_content_hash,
            "status": self.status.value,
            "stderr_sha256": self.stderr_sha256,
            "stdout_sha256": self.stdout_sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> Gate4cOperatorTerminal:
        normalized = normalize_json(value)
        fields = {
            "child_exit_code",
            "child_pid",
            "child_signal",
            "child_terminal_present",
            "child_terminal_sha256",
            "elapsed_ns",
            "expected_gpu_uuid",
            "failure_code",
            "format",
            "gpu_samples_sha256",
            "physical_gpu",
            "spec_content_hash",
            "status",
            "stderr_sha256",
            "stdout_sha256",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Gate 4c operator terminal has incompatible fields")
        int_fields = {
            "child_exit_code",
            "child_pid",
            "child_signal",
            "elapsed_ns",
            "physical_gpu",
        }
        if any(
            normalized[field] is not None and type(normalized[field]) is not int
            for field in int_fields
        ):
            raise TypeError("Gate 4c operator terminal counters must be integers or null")
        if type(normalized["child_terminal_present"]) is not bool:
            raise TypeError("Gate 4c child terminal presence must be boolean")
        text_fields = {
            "expected_gpu_uuid",
            "format",
            "gpu_samples_sha256",
            "spec_content_hash",
            "status",
            "stderr_sha256",
            "stdout_sha256",
        }
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("Gate 4c operator terminal text fields must be text")
        for field in ("child_terminal_sha256", "failure_code"):
            if normalized[field] is not None and type(normalized[field]) is not str:
                raise TypeError("Gate 4c operator optional fields must be text or null")
        return cls(
            status=Gate4cOperatorStatus(normalized["status"]),
            failure_code=(
                Gate4cFailureCode(normalized["failure_code"])
                if normalized["failure_code"] is not None
                else None
            ),
            spec_content_hash=normalized["spec_content_hash"],
            physical_gpu=normalized["physical_gpu"],
            expected_gpu_uuid=normalized["expected_gpu_uuid"],
            child_pid=normalized["child_pid"],
            child_exit_code=normalized["child_exit_code"],
            child_signal=normalized["child_signal"],
            stdout_sha256=normalized["stdout_sha256"],
            stderr_sha256=normalized["stderr_sha256"],
            gpu_samples_sha256=normalized["gpu_samples_sha256"],
            child_terminal_present=normalized["child_terminal_present"],
            child_terminal_sha256=normalized["child_terminal_sha256"],
            elapsed_ns=normalized["elapsed_ns"],
            format=normalized["format"],
        )


__all__ = [
    "GATE_4C_CHILD_TERMINAL_FORMAT",
    "GATE_4C_OPERATOR_TERMINAL_FORMAT",
    "GATE_4C_STAGE_EVENT_FORMAT",
    "Gate4cChildStatus",
    "Gate4cChildTerminal",
    "Gate4cCudaMemorySnapshot",
    "Gate4cFailureCode",
    "Gate4cOperatorStatus",
    "Gate4cOperatorTerminal",
    "Gate4cStage",
    "Gate4cStageEvent",
    "Gate4cStageState",
]
