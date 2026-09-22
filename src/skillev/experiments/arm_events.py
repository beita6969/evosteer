"""Tagged source-event stream owned by experiment-arm application graphs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.runtime.attempt_failures import EventAppendFailedError

from .protocol import AblationArm

ARM_EVENT_ENVELOPE_FORMAT = "skillev-arm-event-envelope@2"


class ArmEventType(StrEnum):
    FLOW_ONLY_LIBRARY_INITIALIZED = "flow_only_library_initialized"
    FLOW_ONLY_TRAINING_STEP_COMMITTED = "flow_only_training_step_committed"
    FLOW_ONLY_PHASE_OPENED = "flow_only_phase_opened"
    FLOW_ONLY_CYCLE_COMMITTED = "flow_only_cycle_committed"
    FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED = "flow_only_phase_checkpoint_published"


@dataclass(frozen=True, slots=True)
class ArmEventEnvelope:
    arm: AblationArm
    event_type: ArmEventType
    run_id: str
    attempt_id: str
    sequence: int
    payload: JsonValue
    format: str = ARM_EVENT_ENVELOPE_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.arm, AblationArm):
            raise TypeError("arm event requires AblationArm")
        if not isinstance(self.event_type, ArmEventType):
            raise TypeError("arm event requires ArmEventType")
        if not self.run_id.strip() or not self.attempt_id.strip():
            raise ValueError("arm event identity cannot be empty")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("arm event sequence must be positive")
        if self.format != ARM_EVENT_ENVELOPE_FORMAT:
            raise ValueError("unsupported arm event envelope format")
        object.__setattr__(self, "payload", normalize_json(self.payload))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arm": self.arm.value,
            "attempt_id": self.attempt_id,
            "event_type": self.event_type.value,
            "format": self.format,
            "payload": self.payload,
            "run_id": self.run_id,
            "sequence": self.sequence,
        }

    @classmethod
    def from_value(cls, value: object) -> ArmEventEnvelope:
        normalized = normalize_json(value)
        fields = {
            "arm",
            "attempt_id",
            "event_type",
            "format",
            "payload",
            "run_id",
            "sequence",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("ArmEventEnvelope has incompatible fields")
        for field in ("arm", "attempt_id", "event_type", "format", "run_id"):
            if type(normalized[field]) is not str:
                raise TypeError(f"arm event {field} must be text")
        if type(normalized["sequence"]) is not int:
            raise TypeError("arm event sequence must be an integer")
        return cls(
            arm=AblationArm(normalized["arm"]),
            event_type=ArmEventType(normalized["event_type"]),
            run_id=normalized["run_id"],
            attempt_id=normalized["attempt_id"],
            sequence=normalized["sequence"],
            payload=normalized["payload"],
            format=normalized["format"],
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


class LiveArmEventLog:
    """Exclusive, write-only source stream for one explicit arm attempt."""

    def __init__(self, path: Path, *, arm: AblationArm, run_id: str, attempt_id: str) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path.resolve()
        self._stream = self._path.open("x", encoding="utf-8", newline="\n")
        self._arm = arm
        self._run_id = run_id
        self._attempt_id = attempt_id
        self._sequence = 0

    @classmethod
    def resume(
        cls,
        path: Path,
        *,
        arm: AblationArm,
        run_id: str,
        attempt_id: str,
    ) -> LiveArmEventLog:
        """Continue one exact arm stream after validating its complete prefix."""

        events = read_arm_event_history(path)
        if events and (
            events[0].arm is not arm
            or events[0].run_id != run_id
            or events[0].attempt_id != attempt_id
        ):
            raise ValueError("arm source event prefix belongs to another attempt")
        instance = cls.__new__(cls)
        instance._path = path.resolve()
        instance._stream = instance._path.open("a", encoding="utf-8", newline="\n")
        instance._arm = arm
        instance._run_id = run_id
        instance._attempt_id = attempt_id
        instance._sequence = len(events)
        return instance

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event_type: ArmEventType, payload: object) -> ArmEventEnvelope:
        if self._stream.closed:
            raise RuntimeError("cannot append after arm event log is closed")
        self._sequence += 1
        envelope = ArmEventEnvelope(
            arm=self._arm,
            event_type=event_type,
            run_id=self._run_id,
            attempt_id=self._attempt_id,
            sequence=self._sequence,
            payload=normalize_json(payload),
        )
        try:
            self._stream.write(canonical_json(envelope.to_value()) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except OSError as error:
            raise EventAppendFailedError("flow-only source event append failed") from error
        return envelope

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


def read_arm_event_history(path: Path) -> tuple[ArmEventEnvelope, ...]:
    """Read one sealed arm source stream with an exact identity/sequence chain."""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    events: list[ArmEventEnvelope] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"arm source event log has a blank line at {line_number}")
        try:
            event = ArmEventEnvelope.from_value(json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"arm source event log is invalid at line {line_number}") from error
        if event.sequence != line_number:
            raise ValueError("arm source event sequence is not contiguous")
        if events and (
            event.arm is not events[0].arm
            or event.run_id != events[0].run_id
            or event.attempt_id != events[0].attempt_id
        ):
            raise ValueError("arm source event identity changes within one log")
        events.append(event)
    return tuple(events)


__all__ = [
    "ARM_EVENT_ENVELOPE_FORMAT",
    "ArmEventEnvelope",
    "ArmEventType",
    "LiveArmEventLog",
    "read_arm_event_history",
]
