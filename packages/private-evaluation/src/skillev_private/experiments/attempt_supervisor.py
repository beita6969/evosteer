"""One-shot supervisor for the fixed, claim-gated private formal worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

from skillev.experiments import FormalRunClaimSecret, FormalRunLedger
from skillev.runtime.attempt_protocol import (
    AttemptFailed,
    AttemptRequest,
    AttemptSucceeded,
    read_attempt_outcome,
)
from skillev.runtime.attempt_publication import (
    OUTCOME_FILE,
    AttemptPublisher,
    PreparedSuccessfulAttempt,
    PublishedSuccessfulAttemptBundle,
    QuarantinedFailedAttemptBundle,
    UnpublishedAttemptBundle,
    fsync_directory,
)
from skillev.runtime.attempt_supervisor import (
    AttemptProcessFailedError,
    ExactInputReference,
)


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkAttemptSupervisor:
    """Launch exactly one ledger-claimed private formal worker.

    A formal worker has no unledgered mode.  The sequence is deliberately
    durable and one-way: claim → child launch → child outcome → terminal →
    publication.  A terminalized success that cannot be published is not
    retryable and therefore cannot be turned into a selected rerun.
    """

    publisher: AttemptPublisher
    formal_run_ledger: FormalRunLedger

    def __post_init__(self) -> None:
        if not isinstance(self.publisher, AttemptPublisher):
            raise TypeError("formal private supervisor requires AttemptPublisher")
        if not isinstance(self.formal_run_ledger, FormalRunLedger):
            raise TypeError("formal private supervisor requires FormalRunLedger")

    def run(self, request: AttemptRequest) -> PublishedSuccessfulAttemptBundle:
        claim = self.formal_run_ledger.claim(request)
        try:
            prepared = self._run_claimed(request, claim)
        except AttemptProcessFailedError as error:
            self.formal_run_ledger.record_failure(
                request=request,
                outcome_path=error.quarantine.directory / OUTCOME_FILE,
            )
            raise
        except BaseException:
            self.formal_run_ledger.record_failure(request=request, outcome_path=None)
            raise

        try:
            admission = self.formal_run_ledger.record_success(
                request=request,
                prepared=prepared,
            )
        except BaseException as error:
            quarantined = self.publisher.quarantine_incomplete(
                prepared.private,
                attempt_id=request.attempt_id,
            )
            # A failed terminal admission is itself the one terminal outcome.
            # If the prior write reached disk, the slot is already immutable;
            # otherwise this write supplies the required failure terminal.
            try:
                self.formal_run_ledger.record_failure(
                    request=request,
                    outcome_path=quarantined.directory / OUTCOME_FILE,
                )
            except FileExistsError:
                pass
            raise AttemptProcessFailedError(quarantined) from error

        try:
            return self.publisher.publish_prepared(
                prepared,
                formal_publication=admission,
            )
        except BaseException as error:
            # The terminal was already durable.  Preserve the private bytes in
            # quarantine and fail closed rather than creating a second child.
            quarantined = self.publisher.quarantine_incomplete(
                prepared.private,
                attempt_id=request.attempt_id,
            )
            raise AttemptProcessFailedError(quarantined) from error

    def _run_claimed(
        self,
        request: AttemptRequest,
        claim: FormalRunClaimSecret,
    ) -> PreparedSuccessfulAttempt:
        captured = ExactInputReference.capture(request.exact_input_path)
        if captured.path != request.exact_input_path.resolve():
            raise ValueError("attempt request has a non-canonical exact-input path")
        if captured.sha256 != request.exact_input_sha256:
            raise ValueError("exact input content changed after request capture")
        private_root = request.private_bundle_directory.parent.resolve()
        private_root.mkdir(parents=True, exist_ok=True)
        private = UnpublishedAttemptBundle.create(request.private_bundle_directory)
        launch = self.formal_run_ledger.launch_for(request=request, claim_secret=claim)
        launch_path = private_root / f".{request.attempt_id}.formal-launch.json"
        with launch_path.open("xb") as stream:
            stream.write(json.dumps(launch.to_value(), sort_keys=True).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(private_root)

        completed = subprocess.run(  # noqa: S603 - fixed private module and launch path
            (
                sys.executable,
                "-m",
                "skillev_private.experiments.attempt_worker",
                str(launch_path),
            ),
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
        if isinstance(outcome, AttemptFailed):
            finalized = self.publisher.finalize(private, request=request)
            if not isinstance(finalized, QuarantinedFailedAttemptBundle):
                raise RuntimeError("failed formal child was unexpectedly published")
            raise AttemptProcessFailedError(finalized)
        return self.publisher.prepare_success(private, request=request)


__all__ = ["PrivateBenchmarkAttemptSupervisor"]
