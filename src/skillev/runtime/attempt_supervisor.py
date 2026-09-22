"""Fixed production supervisor for exactly one non-retryable child attempt."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .attempt_protocol import AttemptFailed, AttemptRequest, AttemptSucceeded, read_attempt_outcome
from .attempt_publication import (
    AttemptPublisher,
    PublishedSuccessfulAttemptBundle,
    QuarantinedFailedAttemptBundle,
    UnpublishedAttemptBundle,
    fsync_directory,
    sha256_bytes,
)


class AttemptProcessFailedError(RuntimeError):
    """The one-shot child failed and its private bundle was quarantined."""

    def __init__(self, quarantine: QuarantinedFailedAttemptBundle) -> None:
        super().__init__("attempt process failed")
        self.quarantine = quarantine


@dataclass(frozen=True, slots=True)
class ExactInputReference:
    """The one byte sequence whose hash is sent across the worker boundary."""

    path: Path
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> ExactInputReference:
        resolved = path.resolve()
        return cls(path=resolved, sha256=sha256_bytes(resolved.read_bytes()))


@dataclass(frozen=True, slots=True)
class AttemptSupervisor:
    publisher: AttemptPublisher

    def run(self, request: AttemptRequest) -> PublishedSuccessfulAttemptBundle:
        captured = ExactInputReference.capture(request.exact_input_path)
        if captured.path != request.exact_input_path.resolve():
            raise ValueError("attempt request has a non-canonical exact-input path")
        if captured.sha256 != request.exact_input_sha256:
            raise ValueError("exact input content changed after request capture")
        private_root = request.private_bundle_directory.parent.resolve()
        private_root.mkdir(parents=True, exist_ok=True)
        private = UnpublishedAttemptBundle.create(request.private_bundle_directory)
        request_path = private_root / f".{request.attempt_id}.request.json"
        encoded = request.to_json().encode("utf-8") + b"\n"
        with request_path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(private_root)

        completed = subprocess.run(  # noqa: S603
            (sys.executable, "-m", "skillev.runtime.attempt_worker", str(request_path)),
            check=False,
        )
        if not private.outcome_path.is_file():
            quarantined = self.publisher.quarantine_incomplete(
                private,
                attempt_id=request.attempt_id,
            )
            raise AttemptProcessFailedError(quarantined)

        outcome = read_attempt_outcome(private.outcome_path)
        child_status_matches = (
            completed.returncode == 0 and isinstance(outcome, AttemptSucceeded)
        ) or (completed.returncode != 0 and isinstance(outcome, AttemptFailed))
        if not child_status_matches:
            quarantined = self.publisher.quarantine_incomplete(
                private,
                attempt_id=request.attempt_id,
            )
            raise AttemptProcessFailedError(quarantined)

        finalized = self.publisher.finalize(private, request=request)
        if isinstance(finalized, QuarantinedFailedAttemptBundle):
            raise AttemptProcessFailedError(finalized)
        return finalized


__all__ = [
    "AttemptProcessFailedError",
    "AttemptSupervisor",
    "ExactInputReference",
]
