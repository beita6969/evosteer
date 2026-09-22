"""Result-blind seven-arm manifest, launch, terminal, and publication proof.

Formal benchmark execution is deliberately more constrained than an ordinary
attempt. A manifest fixes every slot before results, a unique slot is claimed
once, the private child consumes that claim once, the child outcome is
terminalized durably, and only that terminal can authorize publication. The
manifest itself is the execution input; deprecated external approval and Git
receipt machinery is deliberately absent from this path.
"""

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
from skillev.runtime import AttemptBuilderKind, AttemptRequest
from skillev.runtime.attempt_protocol import FormalPublicationAdmission
from skillev.runtime.attempt_publication import (
    PreparedSuccessfulAttempt,
    PublishedSuccessfulAttemptBundle,
    fsync_directory,
    sha256_file,
)

from .protocol import ProtocolFreeze

FORMAL_RUN_MANIFEST_FORMAT = "skillev-formal-run-manifest@2"
FORMAL_RUN_SLOT_FORMAT = "skillev-formal-run-slot@1"
FORMAL_RUN_CLAIM_FORMAT = "skillev-formal-run-claim@1"
FORMAL_ATTEMPT_LAUNCH_FORMAT = "skillev-formal-attempt-launch@1"
FORMAL_RUN_TERMINAL_FORMAT = "skillev-formal-run-terminal@2"

_MANIFEST_FILE = "formal-run-manifest.json"
_CLAIMS_DIRECTORY = "claims"
_LAUNCHES_DIRECTORY = "launches"
_TERMINALS_DIRECTORY = "terminals"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _write_once(path: Path, value: dict[str, JsonValue]) -> None:
    encoded = canonical_json(value).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def _read_value(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_sha256(token: str) -> str:
    return f"sha256:{hashlib.sha256(token.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class FormalRunSlot:
    """The only predeclared input and attempt ID for one formal arm."""

    builder_kind: AttemptBuilderKind
    attempt_id: str
    exact_input_sha256: str
    public_identity_content_hash: str
    format: str = FORMAL_RUN_SLOT_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("formal run slot requires a closed builder kind")
        _text(self.attempt_id, field="attempt_id")
        validate_sha256(self.exact_input_sha256)
        validate_sha256(self.public_identity_content_hash)
        if self.format != FORMAL_RUN_SLOT_FORMAT:
            raise ValueError("unsupported formal run slot format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "exact_input_sha256": self.exact_input_sha256,
            "format": self.format,
            "public_identity_content_hash": self.public_identity_content_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalRunSlot:
        data = _object(
            value,
            fields=frozenset(
                {
                    "attempt_id",
                    "builder_kind",
                    "exact_input_sha256",
                    "format",
                    "public_identity_content_hash",
                }
            ),
            label="formal run slot",
        )
        if any(type(data[field]) is not str for field in data):
            raise TypeError("formal run slot fields must be text")
        return cls(
            builder_kind=AttemptBuilderKind(data["builder_kind"]),
            attempt_id=data["attempt_id"],
            exact_input_sha256=data["exact_input_sha256"],
            public_identity_content_hash=data["public_identity_content_hash"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalRunManifest:
    """Result-blind seven-arm execution plan derived from one protocol freeze."""

    protocol_hash: str
    protocol_freeze_id: str
    slots: tuple[FormalRunSlot, ...]
    run_group_id: str
    format: str = FORMAL_RUN_MANIFEST_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.protocol_hash)
        validate_sha256(self.protocol_freeze_id)
        if not isinstance(self.slots, tuple) or any(
            not isinstance(slot, FormalRunSlot) for slot in self.slots
        ):
            raise TypeError("formal run manifest slots must be FormalRunSlot values")
        if tuple(slot.builder_kind for slot in self.slots) != tuple(AttemptBuilderKind):
            raise ValueError("formal run manifest must predeclare each arm exactly once")
        attempt_ids = tuple(slot.attempt_id for slot in self.slots)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("formal run manifest repeats an attempt ID")
        input_hashes = tuple(slot.exact_input_sha256 for slot in self.slots)
        if len(set(input_hashes)) != len(input_hashes):
            raise ValueError("formal run manifest repeats an exact input")
        if self.format != FORMAL_RUN_MANIFEST_FORMAT:
            raise ValueError("unsupported formal run manifest format")
        validate_sha256(self.run_group_id)
        if self.run_group_id != stable_hash(self._identity_value()):
            raise ValueError("formal run group ID differs from its frozen slots")

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "slots": [slot.to_value() for slot in self.slots],
        }

    @classmethod
    def create(
        cls,
        *,
        protocol_freeze: ProtocolFreeze,
        slots: tuple[FormalRunSlot, ...],
    ) -> FormalRunManifest:
        if not isinstance(protocol_freeze, ProtocolFreeze):
            raise TypeError("formal run manifest requires a protocol freeze")
        identity = {
            "format": FORMAL_RUN_MANIFEST_FORMAT,
            "protocol_freeze_id": protocol_freeze.freeze_id,
            "protocol_hash": protocol_freeze.protocol_hash,
            "slots": [slot.to_value() for slot in slots],
        }
        return cls(
            protocol_hash=protocol_freeze.protocol_hash,
            protocol_freeze_id=protocol_freeze.freeze_id,
            slots=slots,
            run_group_id=stable_hash(identity),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def slot_for_request(self, request: AttemptRequest) -> FormalRunSlot:
        if not isinstance(request, AttemptRequest):
            raise TypeError("formal run request must be AttemptRequest")
        matches = tuple(slot for slot in self.slots if slot.builder_kind is request.builder_kind)
        if len(matches) != 1:  # pragma: no cover - closed-slot invariant
            raise RuntimeError("formal run manifest lost a builder slot")
        slot = matches[0]
        if (
            request.attempt_id != slot.attempt_id
            or request.exact_input_sha256 != slot.exact_input_sha256
        ):
            raise ValueError("attempt request differs from its formal run slot")
        return slot

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "run_group_id": self.run_group_id,
            "slots": [slot.to_value() for slot in self.slots],
        }

    @classmethod
    def from_value(cls, value: object) -> FormalRunManifest:
        data = _object(
            value,
            fields=frozenset(
                {"format", "protocol_freeze_id", "protocol_hash", "run_group_id", "slots"}
            ),
            label="formal run manifest",
        )
        raw_slots = data["slots"]
        if not isinstance(raw_slots, list):
            raise TypeError("formal run manifest slots must be an array")
        for field in ("format", "protocol_freeze_id", "protocol_hash", "run_group_id"):
            if type(data[field]) is not str:
                raise TypeError("formal run manifest hash fields must be text")
        return cls(
            protocol_hash=data["protocol_hash"],
            protocol_freeze_id=data["protocol_freeze_id"],
            slots=tuple(FormalRunSlot.from_value(item) for item in raw_slots),
            run_group_id=data["run_group_id"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalRunClaim:
    """Durable claim identity; the random token itself never enters this wire."""

    run_group_id: str
    slot: FormalRunSlot
    token_sha256: str
    format: str = FORMAL_RUN_CLAIM_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.run_group_id)
        if not isinstance(self.slot, FormalRunSlot):
            raise TypeError("formal claim requires a formal run slot")
        validate_sha256(self.token_sha256)
        if self.format != FORMAL_RUN_CLAIM_FORMAT:
            raise ValueError("unsupported formal run claim format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "run_group_id": self.run_group_id,
            "slot": self.slot.to_value(),
            "token_sha256": self.token_sha256,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalRunClaim:
        data = _object(
            value,
            fields=frozenset({"format", "run_group_id", "slot", "token_sha256"}),
            label="formal run claim",
        )
        text_fields = ("format", "run_group_id", "token_sha256")
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("formal run claim text fields are invalid")
        return cls(
            run_group_id=data["run_group_id"],
            slot=FormalRunSlot.from_value(data["slot"]),
            token_sha256=data["token_sha256"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalRunClaimSecret:
    """Private in-memory launch capability created once by :meth:`claim`."""

    ledger_directory: Path
    claim: FormalRunClaim
    token: str

    def __post_init__(self) -> None:
        if not isinstance(self.ledger_directory, Path) or not self.ledger_directory.is_absolute():
            raise ValueError("formal claim secret requires an absolute ledger directory")
        if not isinstance(self.claim, FormalRunClaim):
            raise TypeError("formal claim secret requires FormalRunClaim")
        _text(self.token, field="formal claim token")
        if not hmac.compare_digest(_token_sha256(self.token), self.claim.token_sha256):
            raise ValueError("formal claim secret differs from its durable claim identity")


@dataclass(frozen=True, slots=True)
class FormalAttemptLaunch:
    """Private IPC envelope accepted by the sole formal benchmark worker."""

    request: AttemptRequest
    ledger_directory: Path
    run_group_id: str
    claim_token: str
    format: str = FORMAL_ATTEMPT_LAUNCH_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.request, AttemptRequest):
            raise TypeError("formal attempt launch requires AttemptRequest")
        if not isinstance(self.ledger_directory, Path) or not self.ledger_directory.is_absolute():
            raise ValueError("formal attempt launch requires an absolute ledger directory")
        validate_sha256(self.run_group_id)
        _text(self.claim_token, field="formal attempt launch token")
        if self.format != FORMAL_ATTEMPT_LAUNCH_FORMAT:
            raise ValueError("unsupported formal attempt launch format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "claim_token": self.claim_token,
            "format": self.format,
            "ledger_directory": self.ledger_directory.as_posix(),
            "request": self.request.to_value(),
            "run_group_id": self.run_group_id,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalAttemptLaunch:
        data = _object(
            value,
            fields=frozenset(
                {"claim_token", "format", "ledger_directory", "request", "run_group_id"}
            ),
            label="formal attempt launch",
        )
        text_fields = ("claim_token", "format", "ledger_directory", "run_group_id")
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("formal attempt launch text fields are invalid")
        return cls(
            request=AttemptRequest.from_value(data["request"]),
            ledger_directory=Path(data["ledger_directory"]),
            run_group_id=data["run_group_id"],
            claim_token=data["claim_token"],
            format=data["format"],
        )


class FormalRunTerminalStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FormalRunTerminal:
    """The one immutable terminal outcome for a predeclared arm slot."""

    run_group_id: str
    slot: FormalRunSlot
    claim_token_sha256: str
    status: FormalRunTerminalStatus
    outcome_sha256: str | None
    format: str = FORMAL_RUN_TERMINAL_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.run_group_id)
        if not isinstance(self.slot, FormalRunSlot):
            raise TypeError("formal terminal requires one formal run slot")
        validate_sha256(self.claim_token_sha256)
        if not isinstance(self.status, FormalRunTerminalStatus):
            raise TypeError("formal terminal status is invalid")
        if self.outcome_sha256 is not None:
            validate_sha256(self.outcome_sha256)
        if self.status is FormalRunTerminalStatus.SUCCEEDED and self.outcome_sha256 is None:
            raise ValueError("successful formal terminal requires an outcome hash")
        if self.format != FORMAL_RUN_TERMINAL_FORMAT:
            raise ValueError("unsupported formal run terminal format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "claim_token_sha256": self.claim_token_sha256,
            "format": self.format,
            "outcome_sha256": self.outcome_sha256,
            "run_group_id": self.run_group_id,
            "slot": self.slot.to_value(),
            "status": self.status.value,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalRunTerminal:
        data = _object(
            value,
            fields=frozenset(
                {"claim_token_sha256", "format", "outcome_sha256", "run_group_id", "slot", "status"}
            ),
            label="formal run terminal",
        )
        text_fields = ("claim_token_sha256", "format", "run_group_id", "status")
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("formal terminal text fields are invalid")
        if data["outcome_sha256"] is not None and type(data["outcome_sha256"]) is not str:
            raise TypeError("formal terminal outcome hash must be text or null")
        return cls(
            run_group_id=data["run_group_id"],
            slot=FormalRunSlot.from_value(data["slot"]),
            claim_token_sha256=data["claim_token_sha256"],
            status=FormalRunTerminalStatus(data["status"]),
            outcome_sha256=data["outcome_sha256"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalRunLedger:
    """Filesystem-backed one-shot state for one immutable manifest."""

    directory: Path
    manifest: FormalRunManifest

    @classmethod
    def create(
        cls,
        *,
        directory: Path,
        manifest: FormalRunManifest,
    ) -> FormalRunLedger:
        if not isinstance(directory, Path):
            raise TypeError("formal run ledger directory must be Path")
        if not isinstance(manifest, FormalRunManifest):
            raise TypeError("formal run ledger requires a manifest")
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        (directory / _CLAIMS_DIRECTORY).mkdir(mode=0o700)
        (directory / _LAUNCHES_DIRECTORY).mkdir(mode=0o700)
        (directory / _TERMINALS_DIRECTORY).mkdir(mode=0o700)
        _write_once(directory / _MANIFEST_FILE, manifest.to_value())
        fsync_directory(directory)
        return cls(directory=directory.resolve(), manifest=manifest)

    @classmethod
    def open(cls, *, directory: Path) -> FormalRunLedger:
        if not isinstance(directory, Path):
            raise TypeError("formal run ledger directory must be Path")
        root = directory.resolve()
        manifest = FormalRunManifest.from_value(_read_value(root / _MANIFEST_FILE))
        for child in (_CLAIMS_DIRECTORY, _LAUNCHES_DIRECTORY, _TERMINALS_DIRECTORY):
            if not (root / child).is_dir():
                raise FileNotFoundError(root / child)
        return cls(directory=root, manifest=manifest)

    @classmethod
    def open_exact(cls, *, directory: Path, manifest: FormalRunManifest) -> FormalRunLedger:
        if not isinstance(manifest, FormalRunManifest):
            raise TypeError("formal run ledger requires a manifest")
        ledger = cls.open(directory=directory)
        if ledger.manifest != manifest:
            raise ValueError("formal run ledger belongs to another manifest")
        return ledger

    @property
    def _claims_directory(self) -> Path:
        return self.directory / _CLAIMS_DIRECTORY

    @property
    def _launches_directory(self) -> Path:
        return self.directory / _LAUNCHES_DIRECTORY

    @property
    def _terminals_directory(self) -> Path:
        return self.directory / _TERMINALS_DIRECTORY

    @staticmethod
    def _slot_filename(slot: FormalRunSlot, *, suffix: str) -> str:
        return f"{slot.builder_kind.value}.{suffix}.json"

    def _claim_path(self, slot: FormalRunSlot) -> Path:
        return self._claims_directory / self._slot_filename(slot, suffix="claim")

    def _launch_path(self, slot: FormalRunSlot) -> Path:
        return self._launches_directory / self._slot_filename(slot, suffix="launch")

    def _terminal_path(self, slot: FormalRunSlot) -> Path:
        return self._terminals_directory / self._slot_filename(slot, suffix="terminal")

    def _claimed_claim(self, request: AttemptRequest) -> FormalRunClaim:
        slot = self.manifest.slot_for_request(request)
        path = self._claim_path(slot)
        if not path.is_file():
            raise ValueError("formal operation requires a prior slot claim")
        claim = FormalRunClaim.from_value(_read_value(path))
        if claim.run_group_id != self.manifest.run_group_id or claim.slot != slot:
            raise ValueError("formal claim differs from its manifest slot")
        return claim

    def _launched_claim(self, request: AttemptRequest) -> FormalRunClaim:
        claim = self._claimed_claim(request)
        path = self._launch_path(claim.slot)
        if not path.is_file():
            raise ValueError("formal success requires a consumed child launch claim")
        launched = FormalRunClaim.from_value(_read_value(path))
        if launched != claim:
            raise ValueError("formal child launch differs from its durable claim")
        return claim

    def claim(self, request: AttemptRequest) -> FormalRunClaimSecret:
        """Permanently reserve the one predeclared slot before child launch."""

        slot = self.manifest.slot_for_request(request)
        if self._terminal_path(slot).exists() or self._claim_path(slot).exists():
            raise ValueError("formal arm has already been claimed or terminalized")
        token = secrets.token_urlsafe(32)
        claim = FormalRunClaim(
            run_group_id=self.manifest.run_group_id,
            slot=slot,
            token_sha256=_token_sha256(token),
        )
        _write_once(self._claim_path(slot), claim.to_value())
        return FormalRunClaimSecret(
            ledger_directory=self.directory,
            claim=claim,
            token=token,
        )

    def launch_for(
        self,
        *,
        request: AttemptRequest,
        claim_secret: FormalRunClaimSecret,
    ) -> FormalAttemptLaunch:
        """Construct the only private envelope accepted by the formal child."""

        if not isinstance(claim_secret, FormalRunClaimSecret):
            raise TypeError("formal child launch requires a claim secret")
        if claim_secret.ledger_directory != self.directory:
            raise ValueError("formal claim secret belongs to another ledger")
        claim = self._claimed_claim(request)
        if claim != claim_secret.claim:
            raise ValueError("formal claim secret differs from durable claim")
        return FormalAttemptLaunch(
            request=request,
            ledger_directory=self.directory,
            run_group_id=self.manifest.run_group_id,
            claim_token=claim_secret.token,
        )

    def consume_private_launch(self, launch: FormalAttemptLaunch) -> FormalRunClaim:
        """Validate and consume a private child launch exactly once."""

        if not isinstance(launch, FormalAttemptLaunch):
            raise TypeError("formal worker requires FormalAttemptLaunch")
        if launch.ledger_directory.resolve() != self.directory:
            raise ValueError("formal attempt launch belongs to another ledger")
        if launch.run_group_id != self.manifest.run_group_id:
            raise ValueError("formal attempt launch belongs to another run group")
        claim = self._claimed_claim(launch.request)
        if not hmac.compare_digest(_token_sha256(launch.claim_token), claim.token_sha256):
            raise ValueError("formal attempt launch has an invalid claim token")
        if self._terminal_path(claim.slot).exists():
            raise ValueError("formal attempt launch follows a terminal outcome")
        _write_once(self._launch_path(claim.slot), claim.to_value())
        return claim

    def _record(self, terminal: FormalRunTerminal) -> None:
        if terminal.run_group_id != self.manifest.run_group_id:
            raise ValueError("formal terminal belongs to another run group")
        if terminal.slot not in self.manifest.slots:
            raise ValueError("formal terminal slot is absent from the manifest")
        _write_once(self._terminal_path(terminal.slot), terminal.to_value())

    def record_success(
        self,
        *,
        request: AttemptRequest,
        prepared: PreparedSuccessfulAttempt,
    ) -> FormalPublicationAdmission:
        """Durably terminalize a verified private child before publication."""

        if not isinstance(prepared, PreparedSuccessfulAttempt):
            raise TypeError("formal success requires a prepared private success")
        if prepared.request != request:
            raise ValueError("prepared attempt differs from formal request")
        claim = self._launched_claim(request)
        slot = claim.slot
        if (
            prepared.formal_run_group_id != self.manifest.run_group_id
            or prepared.request.attempt_id != slot.attempt_id
            or prepared.request.builder_kind is not slot.builder_kind
            or prepared.request.exact_input_sha256 != slot.exact_input_sha256
            or prepared.public_identity_content_hash != slot.public_identity_content_hash
        ):
            raise ValueError("prepared attempt differs from its formal run slot")
        terminal = FormalRunTerminal(
            run_group_id=self.manifest.run_group_id,
            slot=slot,
            claim_token_sha256=claim.token_sha256,
            status=FormalRunTerminalStatus.SUCCEEDED,
            outcome_sha256=prepared.outcome_sha256,
        )
        self._record(terminal)
        return FormalPublicationAdmission(
            run_group_id=self.manifest.run_group_id,
            manifest_content_hash=self.manifest.content_hash,
            slot_builder_kind=slot.builder_kind,
            slot_attempt_id=slot.attempt_id,
            slot_exact_input_sha256=slot.exact_input_sha256,
            slot_public_identity_content_hash=slot.public_identity_content_hash,
            claim_token_sha256=claim.token_sha256,
            terminal_outcome_sha256=prepared.outcome_sha256,
        )

    def record_failure(
        self,
        *,
        request: AttemptRequest,
        outcome_path: Path | None,
    ) -> FormalRunTerminal:
        """Record an unrecoverable failure; the slot cannot be replaced."""

        if outcome_path is not None and not isinstance(outcome_path, Path):
            raise TypeError("formal failed outcome path must be Path or None")
        claim = self._claimed_claim(request)
        terminal = FormalRunTerminal(
            run_group_id=self.manifest.run_group_id,
            slot=claim.slot,
            claim_token_sha256=claim.token_sha256,
            status=FormalRunTerminalStatus.FAILED,
            outcome_sha256=(
                sha256_file(outcome_path) if outcome_path and outcome_path.is_file() else None
            ),
        )
        self._record(terminal)
        return terminal

    def terminals(self) -> tuple[FormalRunTerminal, ...]:
        """Return terminal states in frozen arm order, rejecting unknown files."""

        expected_names = {
            self._slot_filename(slot, suffix="terminal") for slot in self.manifest.slots
        }
        actual_names = {path.name for path in self._terminals_directory.iterdir() if path.is_file()}
        if not actual_names <= expected_names:
            raise ValueError("formal run ledger contains an undeclared terminal file")
        result: list[FormalRunTerminal] = []
        for slot in self.manifest.slots:
            path = self._terminal_path(slot)
            if not path.is_file():
                continue
            terminal = FormalRunTerminal.from_value(_read_value(path))
            if terminal.run_group_id != self.manifest.run_group_id or terminal.slot != slot:
                raise ValueError("formal terminal differs from its manifest slot")
            claim = self._claimed_claim(
                AttemptRequest(
                    run_id="ledger-validation",
                    attempt_id=slot.attempt_id,
                    builder_kind=slot.builder_kind,
                    exact_input_path=Path("/formal-ledger-validation"),
                    exact_input_sha256=slot.exact_input_sha256,
                    private_bundle_directory=Path("/formal-ledger-validation"),
                )
            )
            if terminal.claim_token_sha256 != claim.token_sha256:
                raise ValueError("formal terminal differs from its durable claim")
            result.append(terminal)
        return tuple(result)

    def require_bundle_terminal_success(
        self,
        bundle: PublishedSuccessfulAttemptBundle,
    ) -> FormalRunTerminal:
        """Require a published formal bundle to match this terminal exactly."""

        if not isinstance(bundle, PublishedSuccessfulAttemptBundle):
            raise TypeError("formal terminal proof requires a published success bundle")
        formal = bundle.formal_publication
        if formal is None:
            raise ValueError("formal published bundle lacks terminal publication admission")
        if (
            formal.run_group_id != self.manifest.run_group_id
            or formal.manifest_content_hash != self.manifest.content_hash
        ):
            raise ValueError("formal publication admission belongs to another ledger")
        slot = self.manifest.slot_for_request(
            AttemptRequest(
                run_id=bundle.run_id,
                attempt_id=bundle.attempt_id,
                builder_kind=bundle.builder_kind,
                exact_input_path=Path("/formal-ledger-validation"),
                exact_input_sha256=bundle.exact_input_sha256,
                private_bundle_directory=Path("/formal-ledger-validation"),
            )
        )
        terminal_path = self._terminal_path(slot)
        if not terminal_path.is_file():
            raise ValueError("formal published bundle has no terminal outcome")
        terminal = FormalRunTerminal.from_value(_read_value(terminal_path))
        if (
            terminal.status is not FormalRunTerminalStatus.SUCCEEDED
            or terminal.run_group_id != self.manifest.run_group_id
            or terminal.slot != slot
            or terminal.outcome_sha256 != bundle.outcome_sha256
            or formal.slot_builder_kind is not slot.builder_kind
            or formal.slot_attempt_id != slot.attempt_id
            or formal.slot_exact_input_sha256 != slot.exact_input_sha256
            or formal.slot_public_identity_content_hash != slot.public_identity_content_hash
            or formal.claim_token_sha256 != terminal.claim_token_sha256
            or formal.terminal_outcome_sha256 != terminal.outcome_sha256
        ):
            raise ValueError("formal published bundle differs from its terminal proof")
        return terminal

    def require_complete_successes(
        self,
        bundles: tuple[PublishedSuccessfulAttemptBundle, ...],
    ) -> tuple[PublishedSuccessfulAttemptBundle, ...]:
        """Accept only the seven manifest-selected terminal successes in arm order."""

        if not isinstance(bundles, tuple) or any(
            not isinstance(bundle, PublishedSuccessfulAttemptBundle) for bundle in bundles
        ):
            raise TypeError("formal aggregate requires published attempt bundles")
        if len(bundles) != len(self.manifest.slots):
            raise ValueError("formal aggregate requires every manifest-selected arm")
        by_kind = {bundle.builder_kind: bundle for bundle in bundles}
        if len(by_kind) != len(bundles):
            raise ValueError("formal aggregate repeats an arm")
        terminals = {terminal.slot.builder_kind: terminal for terminal in self.terminals()}
        if len(terminals) != len(self.manifest.slots):
            raise ValueError("formal run has an arm without a terminal outcome")
        ordered: list[PublishedSuccessfulAttemptBundle] = []
        for slot in self.manifest.slots:
            terminal = terminals[slot.builder_kind]
            if terminal.status is not FormalRunTerminalStatus.SUCCEEDED:
                raise ValueError("formal run contains a failed arm and cannot aggregate")
            bundle = by_kind.get(slot.builder_kind)
            if bundle is None:
                raise ValueError("formal aggregate omitted a manifest arm")
            self.require_bundle_terminal_success(bundle)
            ordered.append(bundle)
        return tuple(ordered)


__all__ = [
    "FORMAL_ATTEMPT_LAUNCH_FORMAT",
    "FORMAL_RUN_CLAIM_FORMAT",
    "FORMAL_RUN_MANIFEST_FORMAT",
    "FORMAL_RUN_SLOT_FORMAT",
    "FORMAL_RUN_TERMINAL_FORMAT",
    "FormalAttemptLaunch",
    "FormalRunClaim",
    "FormalRunClaimSecret",
    "FormalRunLedger",
    "FormalRunManifest",
    "FormalRunSlot",
    "FormalRunTerminal",
    "FormalRunTerminalStatus",
]
