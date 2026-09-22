"""Write-once admission and terminal proof for formal frozen evaluations."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256
from skillev.experiments import (
    ExecutionHardwareIdentity,
    FrozenTaskSequenceIdentity,
    ImplementationBuildIdentity,
)
from skillev.runtime import AttemptBuilderKind
from skillev.runtime.attempt_publication import fsync_directory

FORMAL_EVALUATION_SLOT_FORMAT = "skillev-formal-evaluation-slot@1"
FORMAL_EVALUATION_MANIFEST_FORMAT = "skillev-formal-evaluation-manifest@1"
FORMAL_EVALUATION_CLAIM_FORMAT = "skillev-formal-evaluation-claim@1"
FORMAL_EVALUATION_LAUNCH_FORMAT = "skillev-formal-evaluation-launch@1"
FORMAL_EVALUATION_TERMINAL_FORMAT = "skillev-formal-evaluation-terminal@1"
FORMAL_EVALUATION_HARDWARE_POLICY_HASH = stable_hash(
    {"policy": "single-GPU-exact-runtime-identity@1"}
)

_MANIFEST_FILE = "formal-evaluation-manifest.json"
_CLAIMS = "claims"
_LAUNCHES = "launches"
_TERMINALS = "terminals"


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _write_once(path: Path, value: dict[str, JsonValue]) -> None:
    encoded = canonical_json(value).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def _read(path: Path) -> object:
    return json.loads(path.read_text("utf-8"))


def _token_hash(token: str) -> str:
    return f"sha256:{hashlib.sha256(token.encode('utf-8')).hexdigest()}"


class FormalEvaluationKind(StrEnum):
    INITIAL_IID = "initial-iid"
    INITIAL_OOD = "initial-ood"
    PROGRESS_IID = "progress-iid"
    FINAL_IID = "final-iid"
    FINAL_OOD = "final-ood"


@dataclass(frozen=True, slots=True)
class FormalEvaluationSlot:
    """One result-blind state/schedule evaluation that may execute once."""

    source_formal_run_group_id: str
    source_builder_kind: AttemptBuilderKind
    source_training_attempt_id: str
    source_training_identity_hash: str
    kind: FormalEvaluationKind
    anchor_ordinal: int
    frozen_state_hash: str
    policy_snapshot_id: str
    library_version: str
    task_sequence_identity: FrozenTaskSequenceIdentity
    sampling_schedule_hash: str
    implementation_build_hash: str
    execution_hardware_policy_hash: str
    evaluation_id: str
    format: str = FORMAL_EVALUATION_SLOT_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "source_formal_run_group_id",
            "source_training_identity_hash",
            "frozen_state_hash",
            "library_version",
            "sampling_schedule_hash",
            "implementation_build_hash",
            "execution_hardware_policy_hash",
            "evaluation_id",
        ):
            validate_sha256(getattr(self, field))
        if not isinstance(self.source_builder_kind, AttemptBuilderKind):
            raise TypeError("formal evaluation slot requires a closed source arm")
        _text(self.source_training_attempt_id, field="source_training_attempt_id")
        _text(self.policy_snapshot_id, field="policy_snapshot_id")
        if not isinstance(self.kind, FormalEvaluationKind):
            raise TypeError("formal evaluation slot kind is invalid")
        if type(self.anchor_ordinal) is not int or self.anchor_ordinal < 0:
            raise ValueError("formal evaluation anchor ordinal must be non-negative")
        if self.kind is FormalEvaluationKind.PROGRESS_IID:
            if self.anchor_ordinal < 1:
                raise ValueError("progress evaluation requires a positive phase ordinal")
        elif self.anchor_ordinal != 0:
            raise ValueError("non-progress evaluation anchor ordinal must be zero")
        if not isinstance(self.task_sequence_identity, FrozenTaskSequenceIdentity):
            raise TypeError("formal evaluation slot requires a task sequence identity")
        if self.format != FORMAL_EVALUATION_SLOT_FORMAT:
            raise ValueError("unsupported formal evaluation slot format")
        if self.execution_hardware_policy_hash != FORMAL_EVALUATION_HARDWARE_POLICY_HASH:
            raise ValueError("formal evaluation hardware policy differs from protocol")
        if self.evaluation_id != stable_hash(self._identity_value()):
            raise ValueError("formal evaluation ID differs from its scientific slot")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "anchor_ordinal": self.anchor_ordinal,
            "execution_hardware_policy_hash": self.execution_hardware_policy_hash,
            "format": self.format,
            "frozen_state_hash": self.frozen_state_hash,
            "implementation_build_hash": self.implementation_build_hash,
            "kind": self.kind.value,
            "library_version": self.library_version,
            "policy_snapshot_id": self.policy_snapshot_id,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "source_builder_kind": self.source_builder_kind.value,
            "source_formal_run_group_id": self.source_formal_run_group_id,
            "source_training_attempt_id": self.source_training_attempt_id,
            "source_training_identity_hash": self.source_training_identity_hash,
            "task_sequence_identity": self.task_sequence_identity.to_value(),
        }

    @classmethod
    def create(cls, **values: object) -> FormalEvaluationSlot:
        identity = {
            **values,
            "format": FORMAL_EVALUATION_SLOT_FORMAT,
        }
        wire = {
            key: (
                value.value
                if isinstance(value, StrEnum)
                else value.to_value()
                if isinstance(value, FrozenTaskSequenceIdentity)
                else value
            )
            for key, value in identity.items()
        }
        return cls(**values, evaluation_id=stable_hash(wire))  # type: ignore[arg-type]

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._identity_value(), "evaluation_id": self.evaluation_id}

    @classmethod
    def from_value(cls, value: object) -> FormalEvaluationSlot:
        fields = {
            "anchor_ordinal",
            "evaluation_id",
            "execution_hardware_policy_hash",
            "format",
            "frozen_state_hash",
            "implementation_build_hash",
            "kind",
            "library_version",
            "policy_snapshot_id",
            "sampling_schedule_hash",
            "source_builder_kind",
            "source_formal_run_group_id",
            "source_training_attempt_id",
            "source_training_identity_hash",
            "task_sequence_identity",
        }
        data = _object(value, fields=fields, label="formal evaluation slot")
        if type(data["anchor_ordinal"]) is not int:
            raise TypeError("formal evaluation anchor ordinal must be an integer")
        text_fields = fields - {"anchor_ordinal", "task_sequence_identity"}
        for field in text_fields:
            _text(data[field], field=field)
        return cls(
            source_formal_run_group_id=data["source_formal_run_group_id"],
            source_builder_kind=AttemptBuilderKind(data["source_builder_kind"]),
            source_training_attempt_id=data["source_training_attempt_id"],
            source_training_identity_hash=data["source_training_identity_hash"],
            kind=FormalEvaluationKind(data["kind"]),
            anchor_ordinal=data["anchor_ordinal"],
            frozen_state_hash=data["frozen_state_hash"],
            policy_snapshot_id=data["policy_snapshot_id"],
            library_version=data["library_version"],
            task_sequence_identity=FrozenTaskSequenceIdentity.from_value(
                data["task_sequence_identity"]
            ),
            sampling_schedule_hash=data["sampling_schedule_hash"],
            implementation_build_hash=data["implementation_build_hash"],
            execution_hardware_policy_hash=data["execution_hardware_policy_hash"],
            evaluation_id=data["evaluation_id"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalEvaluationManifest:
    protocol_hash: str
    protocol_freeze_id: str
    source_formal_run_group_id: str
    slots: tuple[FormalEvaluationSlot, ...]
    manifest_id: str
    format: str = FORMAL_EVALUATION_MANIFEST_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "protocol_hash",
            "protocol_freeze_id",
            "source_formal_run_group_id",
            "manifest_id",
        ):
            validate_sha256(getattr(self, field))
        if not self.slots or any(not isinstance(item, FormalEvaluationSlot) for item in self.slots):
            raise ValueError("formal evaluation manifest requires slots")
        if tuple(sorted(self.slots, key=lambda item: item.evaluation_id)) != self.slots:
            raise ValueError("formal evaluation slots must use evaluation-ID order")
        if len({item.evaluation_id for item in self.slots}) != len(self.slots):
            raise ValueError("formal evaluation manifest repeats a slot")
        if any(
            item.source_formal_run_group_id != self.source_formal_run_group_id
            for item in self.slots
        ):
            raise ValueError("formal evaluation slots belong to another training group")
        if self.format != FORMAL_EVALUATION_MANIFEST_FORMAT:
            raise ValueError("unsupported formal evaluation manifest format")
        if self.manifest_id != stable_hash(self._identity_value()):
            raise ValueError("formal evaluation manifest ID differs from its slots")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "slots": [item.to_value() for item in self.slots],
            "source_formal_run_group_id": self.source_formal_run_group_id,
        }

    @classmethod
    def create(
        cls,
        *,
        protocol_hash: str,
        protocol_freeze_id: str,
        source_formal_run_group_id: str,
        slots: tuple[FormalEvaluationSlot, ...],
    ) -> FormalEvaluationManifest:
        ordered = tuple(sorted(slots, key=lambda item: item.evaluation_id))
        identity = {
            "format": FORMAL_EVALUATION_MANIFEST_FORMAT,
            "protocol_freeze_id": protocol_freeze_id,
            "protocol_hash": protocol_hash,
            "slots": [item.to_value() for item in ordered],
            "source_formal_run_group_id": source_formal_run_group_id,
        }
        return cls(
            protocol_hash=protocol_hash,
            protocol_freeze_id=protocol_freeze_id,
            source_formal_run_group_id=source_formal_run_group_id,
            slots=ordered,
            manifest_id=stable_hash(identity),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._identity_value(), "manifest_id": self.manifest_id}

    @classmethod
    def from_value(cls, value: object) -> FormalEvaluationManifest:
        data = _object(
            value,
            fields={
                "format",
                "manifest_id",
                "protocol_freeze_id",
                "protocol_hash",
                "slots",
                "source_formal_run_group_id",
            },
            label="formal evaluation manifest",
        )
        rows = data["slots"]
        if not isinstance(rows, list):
            raise TypeError("formal evaluation manifest slots must be an array")
        return cls(
            protocol_hash=_text(data["protocol_hash"], field="protocol_hash"),
            protocol_freeze_id=_text(data["protocol_freeze_id"], field="protocol_freeze_id"),
            source_formal_run_group_id=_text(
                data["source_formal_run_group_id"], field="source_formal_run_group_id"
            ),
            slots=tuple(FormalEvaluationSlot.from_value(item) for item in rows),
            manifest_id=_text(data["manifest_id"], field="manifest_id"),
            format=_text(data["format"], field="format"),
        )

    def slot(self, evaluation_id: str) -> FormalEvaluationSlot:
        matches = tuple(item for item in self.slots if item.evaluation_id == evaluation_id)
        if len(matches) != 1:
            raise ValueError("evaluation ID is absent from the formal manifest")
        return matches[0]


@dataclass(frozen=True, slots=True)
class FormalEvaluationClaim:
    manifest_id: str
    slot: FormalEvaluationSlot
    token_hash: str
    format: str = FORMAL_EVALUATION_CLAIM_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.manifest_id)
        validate_sha256(self.token_hash)
        if self.format != FORMAL_EVALUATION_CLAIM_FORMAT:
            raise ValueError("unsupported formal evaluation claim format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "manifest_id": self.manifest_id,
            "slot": self.slot.to_value(),
            "token_hash": self.token_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalEvaluationClaim:
        data = _object(
            value,
            fields={"format", "manifest_id", "slot", "token_hash"},
            label="formal evaluation claim",
        )
        return cls(
            manifest_id=_text(data["manifest_id"], field="manifest_id"),
            slot=FormalEvaluationSlot.from_value(data["slot"]),
            token_hash=_text(data["token_hash"], field="token_hash"),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FormalEvaluationClaimSecret:
    ledger_directory: Path
    claim: FormalEvaluationClaim
    token: str


@dataclass(frozen=True, slots=True)
class FormalEvaluationLaunch:
    ledger_directory: Path
    manifest_id: str
    evaluation_id: str
    token: str
    format: str = FORMAL_EVALUATION_LAUNCH_FORMAT

    def __post_init__(self) -> None:
        if not self.ledger_directory.is_absolute():
            raise ValueError("formal evaluation launch requires an absolute ledger path")
        validate_sha256(self.manifest_id)
        validate_sha256(self.evaluation_id)
        _text(self.token, field="formal evaluation token")
        if self.format != FORMAL_EVALUATION_LAUNCH_FORMAT:
            raise ValueError("unsupported formal evaluation launch format")


class FormalEvaluationTerminalStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FormalEvaluationTerminal:
    manifest_id: str
    slot: FormalEvaluationSlot
    token_hash: str
    status: FormalEvaluationTerminalStatus
    outcome_hash: str | None
    implementation_build: ImplementationBuildIdentity | None
    execution_hardware: ExecutionHardwareIdentity | None
    format: str = FORMAL_EVALUATION_TERMINAL_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.manifest_id)
        validate_sha256(self.token_hash)
        if self.outcome_hash is not None:
            validate_sha256(self.outcome_hash)
        if self.status is FormalEvaluationTerminalStatus.SUCCEEDED:
            if (
                self.outcome_hash is None
                or not isinstance(self.implementation_build, ImplementationBuildIdentity)
                or not isinstance(self.execution_hardware, ExecutionHardwareIdentity)
            ):
                raise ValueError("successful formal evaluation terminal lacks provenance")
            if self.implementation_build.content_hash != self.slot.implementation_build_hash:
                raise ValueError("formal evaluation used another implementation build")
        elif self.implementation_build is not None or self.execution_hardware is not None:
            raise ValueError("failed formal evaluation cannot publish success provenance")
        if self.format != FORMAL_EVALUATION_TERMINAL_FORMAT:
            raise ValueError("unsupported formal evaluation terminal format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "execution_hardware": (
                self.execution_hardware.to_value() if self.execution_hardware else None
            ),
            "format": self.format,
            "implementation_build": (
                self.implementation_build.to_value() if self.implementation_build else None
            ),
            "manifest_id": self.manifest_id,
            "outcome_hash": self.outcome_hash,
            "slot": self.slot.to_value(),
            "status": self.status.value,
            "token_hash": self.token_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalEvaluationTerminal:
        data = _object(
            value,
            fields={
                "execution_hardware",
                "format",
                "implementation_build",
                "manifest_id",
                "outcome_hash",
                "slot",
                "status",
                "token_hash",
            },
            label="formal evaluation terminal",
        )
        build = data["implementation_build"]
        hardware = data["execution_hardware"]
        return cls(
            manifest_id=_text(data["manifest_id"], field="manifest_id"),
            slot=FormalEvaluationSlot.from_value(data["slot"]),
            token_hash=_text(data["token_hash"], field="token_hash"),
            status=FormalEvaluationTerminalStatus(_text(data["status"], field="status")),
            outcome_hash=(
                _text(data["outcome_hash"], field="outcome_hash")
                if data["outcome_hash"] is not None
                else None
            ),
            implementation_build=(
                ImplementationBuildIdentity.from_value(build) if build is not None else None
            ),
            execution_hardware=(
                ExecutionHardwareIdentity.from_value(hardware) if hardware is not None else None
            ),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class FormalEvaluationLedger:
    directory: Path
    manifest: FormalEvaluationManifest

    @classmethod
    def create(
        cls, *, directory: Path, manifest: FormalEvaluationManifest
    ) -> FormalEvaluationLedger:
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        for name in (_CLAIMS, _LAUNCHES, _TERMINALS):
            (directory / name).mkdir(mode=0o700)
        _write_once(directory / _MANIFEST_FILE, manifest.to_value())
        return cls(directory.resolve(), manifest)

    @classmethod
    def open(cls, directory: Path) -> FormalEvaluationLedger:
        root = directory.resolve()
        manifest = FormalEvaluationManifest.from_value(_read(root / _MANIFEST_FILE))
        for name in (_CLAIMS, _LAUNCHES, _TERMINALS):
            if not (root / name).is_dir():
                raise FileNotFoundError(root / name)
        return cls(root, manifest)

    def _path(self, kind: str, slot: FormalEvaluationSlot) -> Path:
        return self.directory / kind / f"{slot.evaluation_id.removeprefix('sha256:')}.json"

    def claim(self, evaluation_id: str) -> FormalEvaluationClaimSecret:
        slot = self.manifest.slot(evaluation_id)
        claim_path = self._path(_CLAIMS, slot)
        terminal_path = self._path(_TERMINALS, slot)
        if claim_path.exists() or terminal_path.exists():
            raise ValueError("formal evaluation slot was already claimed")
        token = secrets.token_urlsafe(32)
        claim = FormalEvaluationClaim(self.manifest.manifest_id, slot, _token_hash(token))
        _write_once(claim_path, claim.to_value())
        return FormalEvaluationClaimSecret(self.directory, claim, token)

    def launch(self, secret: FormalEvaluationClaimSecret) -> FormalEvaluationLaunch:
        if secret.ledger_directory != self.directory:
            raise ValueError("formal evaluation claim belongs to another ledger")
        durable = FormalEvaluationClaim.from_value(_read(self._path(_CLAIMS, secret.claim.slot)))
        if durable != secret.claim or not hmac.compare_digest(
            durable.token_hash, _token_hash(secret.token)
        ):
            raise ValueError("formal evaluation claim secret differs from durable claim")
        return FormalEvaluationLaunch(
            ledger_directory=self.directory,
            manifest_id=self.manifest.manifest_id,
            evaluation_id=durable.slot.evaluation_id,
            token=secret.token,
        )

    def consume(self, launch: FormalEvaluationLaunch) -> FormalEvaluationSlot:
        if launch.ledger_directory.resolve() != self.directory:
            raise ValueError("formal evaluation launch belongs to another ledger")
        if launch.manifest_id != self.manifest.manifest_id:
            raise ValueError("formal evaluation launch belongs to another manifest")
        slot = self.manifest.slot(launch.evaluation_id)
        claim = FormalEvaluationClaim.from_value(_read(self._path(_CLAIMS, slot)))
        if not hmac.compare_digest(claim.token_hash, _token_hash(launch.token)):
            raise ValueError("formal evaluation launch token is invalid")
        _write_once(self._path(_LAUNCHES, slot), claim.to_value())
        return slot

    def _launched_claim(self, slot: FormalEvaluationSlot) -> FormalEvaluationClaim:
        claim = FormalEvaluationClaim.from_value(_read(self._path(_CLAIMS, slot)))
        launched = FormalEvaluationClaim.from_value(_read(self._path(_LAUNCHES, slot)))
        if launched != claim:
            raise ValueError("formal evaluation slot was not consumed by the child")
        return claim

    def record_success(
        self,
        *,
        slot: FormalEvaluationSlot,
        outcome_hash: str,
        implementation_build: ImplementationBuildIdentity,
        execution_hardware: ExecutionHardwareIdentity,
    ) -> FormalEvaluationTerminal:
        claim = self._launched_claim(slot)
        terminal = FormalEvaluationTerminal(
            manifest_id=self.manifest.manifest_id,
            slot=slot,
            token_hash=claim.token_hash,
            status=FormalEvaluationTerminalStatus.SUCCEEDED,
            outcome_hash=outcome_hash,
            implementation_build=implementation_build,
            execution_hardware=execution_hardware,
        )
        _write_once(self._path(_TERMINALS, slot), terminal.to_value())
        return terminal

    def record_failure(self, *, slot: FormalEvaluationSlot) -> FormalEvaluationTerminal:
        claim = FormalEvaluationClaim.from_value(_read(self._path(_CLAIMS, slot)))
        terminal = FormalEvaluationTerminal(
            manifest_id=self.manifest.manifest_id,
            slot=slot,
            token_hash=claim.token_hash,
            status=FormalEvaluationTerminalStatus.FAILED,
            outcome_hash=None,
            implementation_build=None,
            execution_hardware=None,
        )
        _write_once(self._path(_TERMINALS, slot), terminal.to_value())
        return terminal

    def require_success(self, evaluation_id: str) -> FormalEvaluationTerminal:
        slot = self.manifest.slot(evaluation_id)
        terminal = FormalEvaluationTerminal.from_value(_read(self._path(_TERMINALS, slot)))
        if terminal.status is not FormalEvaluationTerminalStatus.SUCCEEDED:
            raise ValueError("formal evaluation did not terminate successfully")
        if terminal.manifest_id != self.manifest.manifest_id or terminal.slot != slot:
            raise ValueError("formal evaluation terminal belongs to another slot")
        return terminal


__all__ = [
    "FORMAL_EVALUATION_CLAIM_FORMAT",
    "FORMAL_EVALUATION_HARDWARE_POLICY_HASH",
    "FORMAL_EVALUATION_LAUNCH_FORMAT",
    "FORMAL_EVALUATION_MANIFEST_FORMAT",
    "FORMAL_EVALUATION_SLOT_FORMAT",
    "FORMAL_EVALUATION_TERMINAL_FORMAT",
    "FormalEvaluationClaim",
    "FormalEvaluationClaimSecret",
    "FormalEvaluationKind",
    "FormalEvaluationLaunch",
    "FormalEvaluationLedger",
    "FormalEvaluationManifest",
    "FormalEvaluationSlot",
    "FormalEvaluationTerminal",
    "FormalEvaluationTerminalStatus",
]
