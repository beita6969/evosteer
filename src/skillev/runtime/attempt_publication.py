"""Atomic publication boundary for attempt-private scientific source logs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from skillev.contracts import (
    JsonValue,
    PhaseCheckpointArtifact,
    RunCursorValue,
    canonical_json,
    normalize_json,
    stable_hash,
)

from .attempt_protocol import (
    AttemptBuilderKind,
    AttemptFailed,
    AttemptRequest,
    AttemptSourceLogDigest,
    AttemptSourceLogKind,
    AttemptSucceeded,
    FinalTrainingArtifact,
    FormalPublicationAdmission,
    attempt_outcome_from_value,
    expected_source_log_layout,
    validate_source_log_layout,
)

EVENTS_FILE = "events.jsonl"
ARM_EVENTS_FILE = "arm-events.jsonl"
PUBLIC_IDENTITY_FILE = "public-identity.json"
OUTCOME_FILE = "outcome.json"
TRACEBACK_FILE = "private-traceback.txt"
MANIFEST_FILE = "published-manifest.json"
PUBLISHED_ATTEMPT_FORMAT = "skillev-published-attempt@4"


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha256_tree(directory: Path) -> str:
    """Hash a completed private checkpoint by relative file identity and bytes."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if not files:
        raise ValueError("final training artifact has no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(path).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def describe_final_training_artifact(
    directory: Path,
    *,
    policy_snapshot_id: str,
    library_version: str,
    optimizer_step: int,
) -> FinalTrainingArtifact:
    """Bind a finished private checkpoint to the public success outcome."""

    root = directory.resolve()
    runtime_state = root / "runtime_state.json"
    if not runtime_state.is_file():
        raise FileNotFoundError(runtime_state)
    return FinalTrainingArtifact(
        artifact_sha256=sha256_tree(root),
        runtime_state_sha256=sha256_file(runtime_state),
        policy_snapshot_id=policy_snapshot_id,
        library_version=library_version,
        optimizer_step=optimizer_step,
    )


def describe_phase_checkpoint_artifact(
    directory: Path,
    *,
    phase_event_id: str,
    policy_snapshot_id: str,
    library_version: str,
    optimizer_step: int,
    run_cursor_after: RunCursorValue,
) -> PhaseCheckpointArtifact:
    """Bind one named post-cycle checkpoint without exposing its path.

    Phase anchors select this artifact only by the source-event phase ID; the
    private resolver later checks the resulting tree and runtime-state bytes.
    """

    root = directory.resolve()
    runtime_state = root / "runtime_state.json"
    if not runtime_state.is_file():
        raise FileNotFoundError(runtime_state)
    return PhaseCheckpointArtifact(
        artifact_sha256=sha256_tree(root),
        runtime_state_sha256=sha256_file(runtime_state),
        policy_snapshot_id=policy_snapshot_id,
        library_version=library_version,
        optimizer_step=optimizer_step,
        phase_event_id=phase_event_id,
        run_cursor_after=run_cursor_after,
    )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"published attempt {field} must be text")
    return value


@dataclass(frozen=True, slots=True)
class UnpublishedAttemptBundle:
    """Worker-writable bundle that becomes audit input only after publication."""

    directory: Path

    @classmethod
    def create(cls, directory: Path) -> UnpublishedAttemptBundle:
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        return cls(directory.resolve())

    @classmethod
    def open_exact(cls, directory: Path) -> UnpublishedAttemptBundle:
        resolved = directory.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        return cls(resolved)

    @property
    def event_log_path(self) -> Path:
        return self.directory / EVENTS_FILE

    @property
    def arm_event_log_path(self) -> Path:
        return self.directory / ARM_EVENTS_FILE

    @property
    def public_identity_path(self) -> Path:
        return self.directory / PUBLIC_IDENTITY_FILE

    @property
    def outcome_path(self) -> Path:
        return self.directory / OUTCOME_FILE

    @property
    def private_traceback_path(self) -> Path:
        return self.directory / TRACEBACK_FILE

    def source_log_path(self, name: str) -> Path:
        if type(name) is not str or not name or Path(name).name != name:
            raise ValueError("source log name must be one path component")
        return self.directory / name

    def write_public_identity_once(self, value: object) -> tuple[str, str]:
        normalized = normalize_json(value)
        encoded = canonical_json(normalized).encode("utf-8") + b"\n"
        with self.public_identity_path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return sha256_file(self.public_identity_path), stable_hash(normalized)

    def source_log_digests(
        self,
        builder_kind: AttemptBuilderKind,
    ) -> tuple[AttemptSourceLogDigest, ...]:
        result: list[AttemptSourceLogDigest] = []
        for name, kind in expected_source_log_layout(builder_kind):
            path = self.source_log_path(name)
            if not path.is_file():
                raise RuntimeError(f"attempt produced no required source log {name!r}")
            result.append(AttemptSourceLogDigest(name=name, kind=kind, sha256=sha256_file(path)))
        return tuple(result)

    def require_exact_success_file_set(
        self,
        builder_kind: AttemptBuilderKind,
        *,
        include_manifest: bool,
    ) -> None:
        """Reject undeclared regular files before a success bundle is published."""

        expected = {
            PUBLIC_IDENTITY_FILE,
            OUTCOME_FILE,
            *(name for name, _kind in expected_source_log_layout(builder_kind)),
        }
        if include_manifest:
            expected.add(MANIFEST_FILE)
        actual = {path.name for path in self.directory.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError("successful attempt contains an undeclared regular file")

    def write_outcome_once(self, value: object) -> None:
        normalized = normalize_json(value)
        encoded = canonical_json(normalized).encode("utf-8") + b"\n"
        with self.outcome_path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True, slots=True)
class PreparedSuccessfulAttempt:
    """Validated private success that is not yet eligible for publication.

    Formal supervisors deliberately stop at this boundary, write the durable
    formal terminal, and only then pass the returned object to
    :meth:`AttemptPublisher.publish_prepared`.
    """

    private: UnpublishedAttemptBundle
    request: AttemptRequest
    public_identity_sha256: str
    public_identity_content_hash: str
    source_logs: tuple[AttemptSourceLogDigest, ...]
    final_training_artifact: FinalTrainingArtifact
    outcome_sha256: str
    formal_run_group_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.private, UnpublishedAttemptBundle):
            raise TypeError("prepared attempt requires an unpublished bundle")
        if not isinstance(self.request, AttemptRequest):
            raise TypeError("prepared attempt requires its exact request")
        for field in (
            "public_identity_sha256",
            "public_identity_content_hash",
            "outcome_sha256",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise ValueError(f"prepared attempt {field} must be text")
        if not isinstance(self.final_training_artifact, FinalTrainingArtifact):
            raise TypeError("prepared attempt requires a final training artifact")
        validate_source_log_layout(self.request.builder_kind, self.source_logs)
        if self.formal_run_group_id is not None and (
            type(self.formal_run_group_id) is not str or not self.formal_run_group_id
        ):
            raise TypeError("prepared attempt formal_run_group_id must be text or None")


@dataclass(frozen=True, slots=True)
class PublishedSuccessfulAttemptBundle:
    """The only bundle type accepted by scientific audit entrypoints."""

    directory: Path
    run_id: str
    attempt_id: str
    builder_kind: AttemptBuilderKind
    exact_input_sha256: str
    public_identity_sha256: str
    public_identity_content_hash: str
    source_logs: tuple[AttemptSourceLogDigest, ...]
    final_training_artifact: FinalTrainingArtifact
    outcome_sha256: str
    formal_publication: FormalPublicationAdmission | None = None

    @property
    def event_log_path(self) -> Path:
        """Operational source log (all builder kinds publish it)."""

        return self.directory / EVENTS_FILE

    @property
    def public_identity_path(self) -> Path:
        return self.directory / PUBLIC_IDENTITY_FILE

    @property
    def outcome_path(self) -> Path:
        return self.directory / OUTCOME_FILE

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_FILE

    def source_log_path(self, item: AttemptSourceLogDigest) -> Path:
        return self.directory / item.name

    def source_log(self, kind: AttemptSourceLogKind) -> AttemptSourceLogDigest:
        selected = tuple(item for item in self.source_logs if item.kind is kind)
        if len(selected) != 1:
            raise ValueError(f"published attempt requires exactly one {kind.value} log")
        return selected[0]

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "exact_input_sha256": self.exact_input_sha256,
            "final_training_artifact": self.final_training_artifact.to_value(),
            "format": PUBLISHED_ATTEMPT_FORMAT,
            "formal_publication": (
                self.formal_publication.to_value() if self.formal_publication is not None else None
            ),
            "outcome_sha256": self.outcome_sha256,
            "public_identity_content_hash": self.public_identity_content_hash,
            "public_identity_sha256": self.public_identity_sha256,
            "run_id": self.run_id,
            "source_logs": [item.to_value() for item in self.source_logs],
        }

    @classmethod
    def open_exact(cls, directory: Path) -> PublishedSuccessfulAttemptBundle:
        resolved = directory.resolve()
        value = json.loads((resolved / MANIFEST_FILE).read_text(encoding="utf-8"))
        fields = {
            "attempt_id",
            "builder_kind",
            "exact_input_sha256",
            "final_training_artifact",
            "format",
            "formal_publication",
            "outcome_sha256",
            "public_identity_content_hash",
            "public_identity_sha256",
            "run_id",
            "source_logs",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("published attempt manifest has incompatible fields")
        if value["format"] != PUBLISHED_ATTEMPT_FORMAT:
            raise ValueError("unsupported published attempt format")
        raw_logs = value["source_logs"]
        if type(raw_logs) is not list:
            raise TypeError("published source_logs must be an array")
        raw_formal_publication = value["formal_publication"]
        if raw_formal_publication is not None and not isinstance(raw_formal_publication, dict):
            raise TypeError("published formal_publication must be an object or null")
        published = cls(
            directory=resolved,
            run_id=_text(value["run_id"], field="run_id"),
            attempt_id=_text(value["attempt_id"], field="attempt_id"),
            builder_kind=AttemptBuilderKind(_text(value["builder_kind"], field="builder_kind")),
            exact_input_sha256=_text(value["exact_input_sha256"], field="exact_input_sha256"),
            public_identity_sha256=_text(
                value["public_identity_sha256"], field="public_identity_sha256"
            ),
            public_identity_content_hash=_text(
                value["public_identity_content_hash"],
                field="public_identity_content_hash",
            ),
            source_logs=tuple(AttemptSourceLogDigest.from_value(item) for item in raw_logs),
            final_training_artifact=FinalTrainingArtifact.from_value(
                value["final_training_artifact"]
            ),
            outcome_sha256=_text(value["outcome_sha256"], field="outcome_sha256"),
            formal_publication=(
                FormalPublicationAdmission.from_value(raw_formal_publication)
                if raw_formal_publication is not None
                else None
            ),
        )
        validate_source_log_layout(published.builder_kind, published.source_logs)
        expected_regular_files = {
            MANIFEST_FILE,
            PUBLIC_IDENTITY_FILE,
            OUTCOME_FILE,
            *(item.name for item in published.source_logs),
        }
        actual_regular_files = {path.name for path in resolved.iterdir() if path.is_file()}
        if actual_regular_files != expected_regular_files:
            raise ValueError("published success directory has an incompatible file set")
        if sha256_file(published.public_identity_path) != published.public_identity_sha256:
            raise ValueError("published public identity hash differs from manifest")
        if sha256_file(published.outcome_path) != published.outcome_sha256:
            raise ValueError("published outcome hash differs from manifest")
        for item in published.source_logs:
            if sha256_file(published.source_log_path(item)) != item.sha256:
                raise ValueError(f"published source log {item.name!r} hash differs")
        outcome = attempt_outcome_from_value(
            json.loads(published.outcome_path.read_text(encoding="utf-8"))
        )
        if not isinstance(outcome, AttemptSucceeded):
            raise ValueError("published scientific attempt did not succeed")
        if (
            outcome.attempt_id != published.attempt_id
            or outcome.builder_kind is not published.builder_kind
            or outcome.exact_input_sha256 != published.exact_input_sha256
            or outcome.public_identity_sha256 != published.public_identity_sha256
            or outcome.public_identity_content_hash != published.public_identity_content_hash
            or outcome.source_logs != published.source_logs
            or outcome.final_training_artifact != published.final_training_artifact
        ):
            raise ValueError("published manifest and success outcome identities differ")
        formal = published.formal_publication
        if formal is None:
            if outcome.formal_run_group_id is not None:
                raise ValueError("formal outcome lacks publication admission")
        else:
            if (
                outcome.formal_run_group_id != formal.run_group_id
                or published.builder_kind is not formal.slot_builder_kind
                or published.attempt_id != formal.slot_attempt_id
                or published.exact_input_sha256 != formal.slot_exact_input_sha256
                or published.public_identity_content_hash
                != formal.slot_public_identity_content_hash
                or published.outcome_sha256 != formal.terminal_outcome_sha256
            ):
                raise ValueError("formal publication admission differs from published success")
        return published


@dataclass(frozen=True, slots=True)
class QuarantinedFailedAttemptBundle:
    directory: Path


class AttemptPublisher:
    def __init__(self, *, published_root: Path, quarantine_root: Path) -> None:
        self._published_root = published_root.resolve()
        self._quarantine_root = quarantine_root.resolve()
        self._published_root.mkdir(parents=True, exist_ok=True)
        self._quarantine_root.mkdir(parents=True, exist_ok=True)

    def finalize(
        self,
        private: UnpublishedAttemptBundle,
        *,
        request: AttemptRequest,
    ) -> PublishedSuccessfulAttemptBundle | QuarantinedFailedAttemptBundle:
        outcome = attempt_outcome_from_value(
            json.loads(private.outcome_path.read_text(encoding="utf-8"))
        )
        if outcome.attempt_id != request.attempt_id:
            raise ValueError("attempt outcome identity differs from request")
        if outcome.builder_kind is not request.builder_kind:
            raise ValueError("attempt outcome builder differs from request")
        if outcome.exact_input_sha256 != request.exact_input_sha256:
            raise ValueError("attempt outcome exact-input identity differs from request")
        if isinstance(outcome, AttemptFailed):
            return self._move_to_quarantine(private, request.attempt_id)

        return self.publish_prepared(self.prepare_success(private, request=request))

    def prepare_success(
        self,
        private: UnpublishedAttemptBundle,
        *,
        request: AttemptRequest,
    ) -> PreparedSuccessfulAttempt:
        """Validate a success before a formal terminal grants publication.

        Unlike :meth:`finalize`, this method never writes a published manifest
        and never moves the private directory.  That is the actual boundary
        needed by formal runs: success bytes first, durable terminal second,
        publication third.
        """

        outcome = attempt_outcome_from_value(
            json.loads(private.outcome_path.read_text(encoding="utf-8"))
        )
        if not isinstance(outcome, AttemptSucceeded):
            raise ValueError("prepared publication requires a successful outcome")
        if (
            outcome.attempt_id != request.attempt_id
            or outcome.builder_kind is not request.builder_kind
            or outcome.exact_input_sha256 != request.exact_input_sha256
        ):
            raise ValueError("successful outcome identity differs from request")
        actual_identity_hash = sha256_file(private.public_identity_path)
        raw_identity = json.loads(private.public_identity_path.read_text(encoding="utf-8"))
        actual_identity_content_hash = stable_hash(normalize_json(raw_identity))
        actual_logs = private.source_log_digests(request.builder_kind)
        if outcome.public_identity_sha256 != actual_identity_hash:
            raise ValueError("successful outcome has the wrong public identity hash")
        if outcome.public_identity_content_hash != actual_identity_content_hash:
            raise ValueError("successful outcome has the wrong identity content hash")
        if outcome.source_logs != actual_logs:
            raise ValueError("successful outcome has the wrong source-log identities")
        private.require_exact_success_file_set(request.builder_kind, include_manifest=False)
        return PreparedSuccessfulAttempt(
            private=private,
            request=request,
            public_identity_sha256=actual_identity_hash,
            public_identity_content_hash=actual_identity_content_hash,
            source_logs=actual_logs,
            final_training_artifact=outcome.final_training_artifact,
            outcome_sha256=sha256_file(private.outcome_path),
            formal_run_group_id=outcome.formal_run_group_id,
        )

    def publish_prepared(
        self,
        prepared: PreparedSuccessfulAttempt,
        *,
        formal_publication: FormalPublicationAdmission | None = None,
    ) -> PublishedSuccessfulAttemptBundle:
        """Publish one previously validated private success.

        A formal publication admission is not optional for a formal child
        outcome.  Conversely a regular attempt cannot manufacture formal
        provenance after execution because it has no formal run-group outcome.
        """

        if not isinstance(prepared, PreparedSuccessfulAttempt):
            raise TypeError("publication requires a prepared successful attempt")
        if formal_publication is None:
            if prepared.formal_run_group_id is not None:
                raise ValueError("formal child outcome lacks terminal publication admission")
        else:
            if (
                prepared.formal_run_group_id != formal_publication.run_group_id
                or prepared.request.builder_kind is not formal_publication.slot_builder_kind
                or prepared.request.attempt_id != formal_publication.slot_attempt_id
                or prepared.request.exact_input_sha256 != formal_publication.slot_exact_input_sha256
                or prepared.public_identity_content_hash
                != formal_publication.slot_public_identity_content_hash
                or prepared.outcome_sha256 != formal_publication.terminal_outcome_sha256
            ):
                raise ValueError("terminal publication admission differs from prepared attempt")
        published = PublishedSuccessfulAttemptBundle(
            directory=prepared.private.directory,
            run_id=prepared.request.run_id,
            attempt_id=prepared.request.attempt_id,
            builder_kind=prepared.request.builder_kind,
            exact_input_sha256=prepared.request.exact_input_sha256,
            public_identity_sha256=prepared.public_identity_sha256,
            public_identity_content_hash=prepared.public_identity_content_hash,
            source_logs=prepared.source_logs,
            final_training_artifact=prepared.final_training_artifact,
            outcome_sha256=prepared.outcome_sha256,
            formal_publication=formal_publication,
        )
        with published.manifest_path.open("xb") as stream:
            stream.write(canonical_json(published.to_value()).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(prepared.private.directory)
        target = self._published_root / prepared.request.attempt_id
        os.replace(prepared.private.directory, target)
        fsync_directory(self._published_root)
        return PublishedSuccessfulAttemptBundle.open_exact(target)

    def _move_to_quarantine(
        self,
        private: UnpublishedAttemptBundle,
        attempt_id: str,
    ) -> QuarantinedFailedAttemptBundle:
        target = self._quarantine_root / attempt_id
        os.replace(private.directory, target)
        fsync_directory(self._quarantine_root)
        return QuarantinedFailedAttemptBundle(target)

    def quarantine_incomplete(
        self,
        private: UnpublishedAttemptBundle,
        *,
        attempt_id: str,
    ) -> QuarantinedFailedAttemptBundle:
        return self._move_to_quarantine(private, attempt_id)


__all__ = [
    "ARM_EVENTS_FILE",
    "EVENTS_FILE",
    "MANIFEST_FILE",
    "OUTCOME_FILE",
    "PUBLIC_IDENTITY_FILE",
    "PUBLISHED_ATTEMPT_FORMAT",
    "AttemptPublisher",
    "PreparedSuccessfulAttempt",
    "PublishedSuccessfulAttemptBundle",
    "QuarantinedFailedAttemptBundle",
    "UnpublishedAttemptBundle",
    "describe_final_training_artifact",
    "fsync_directory",
    "sha256_bytes",
    "sha256_file",
    "sha256_tree",
]
