"""Child-process boundary for one non-retryable method attempt."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from skillev.scoring import TrajectoryScoringMemoryError

from .attempt_builders import BuiltAttempt, build_public_exact_attempt
from .attempt_failures import AttemptDomainError
from .attempt_protocol import (
    AttemptFailed,
    AttemptFailureCode,
    AttemptFailureStage,
    AttemptRequest,
    AttemptSucceeded,
)
from .attempt_publication import UnpublishedAttemptBundle, describe_final_training_artifact


class TypedAttemptError(AttemptDomainError):
    """A fixed public classification plus private-only diagnostic detail."""

    def __init__(
        self,
        code: AttemptFailureCode,
        stage: AttemptFailureStage,
        private_detail: str,
    ) -> None:
        super().__init__(code=code, stage=stage, private_detail=private_detail)


AttemptBuilder = Callable[[AttemptRequest], BuiltAttempt]


async def execute_attempt_request(
    request: AttemptRequest,
    *,
    build: AttemptBuilder,
    formal_run_group_id: str | None = None,
) -> None:
    """Execute one closed builder under the shared child publication contract."""

    if not isinstance(request, AttemptRequest):
        raise TypeError("attempt worker requires AttemptRequest")
    if not callable(build):
        raise TypeError("attempt worker build must be callable")
    if formal_run_group_id is not None:
        # A formal child receives this only after its private launch proof was
        # consumed by FormalRunLedger.  The public outcome repeats the group
        # identity so publication cannot be detached from that terminal.
        if type(formal_run_group_id) is not str:
            raise TypeError("formal_run_group_id must be text or None")
    bundle = UnpublishedAttemptBundle.open_exact(request.private_bundle_directory)
    built = None
    try:
        built = build(request)
        identity_sha256, identity_content_hash = bundle.write_public_identity_once(
            built.public_identity.to_value()
        )
        summary = await built.application.evolution_loop.run(built.run_plan)
        built.application.training_loop.ledger.assert_fully_settled()
        built.validate_summary(summary)
        final_training_artifact = describe_final_training_artifact(
            built.application.final_training_snapshot_directory,
            policy_snapshot_id=summary.final_policy_snapshot_id,
            library_version=summary.final_library_version,
            optimizer_step=summary.final_optimizer_step,
        )
        built.close_source_logs()
        source_logs = bundle.source_log_digests(request.builder_kind)
        bundle.write_outcome_once(
            AttemptSucceeded(
                attempt_id=request.attempt_id,
                builder_kind=request.builder_kind,
                exact_input_sha256=request.exact_input_sha256,
                public_identity_sha256=identity_sha256,
                public_identity_content_hash=identity_content_hash,
                source_logs=source_logs,
                summary=summary,
                final_training_artifact=final_training_artifact,
                formal_run_group_id=formal_run_group_id,
            ).to_value()
        )
    except AttemptDomainError as error:
        _write_private_traceback(bundle.private_traceback_path)
        bundle.write_outcome_once(
            AttemptFailed(
                attempt_id=request.attempt_id,
                builder_kind=request.builder_kind,
                exact_input_sha256=request.exact_input_sha256,
                code=error.code,
                stage=error.stage,
                exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            ).to_value()
        )
        raise
    except TrajectoryScoringMemoryError as error:
        _write_private_traceback(bundle.private_traceback_path)
        bundle.write_outcome_once(
            AttemptFailed(
                attempt_id=request.attempt_id,
                builder_kind=request.builder_kind,
                exact_input_sha256=request.exact_input_sha256,
                code=AttemptFailureCode.TRAINING_MEMORY_EXHAUSTED,
                stage=AttemptFailureStage.TTB_SCORING,
                exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            ).to_value()
        )
        raise
    except BaseException as error:
        _write_private_traceback(bundle.private_traceback_path)
        bundle.write_outcome_once(
            AttemptFailed(
                attempt_id=request.attempt_id,
                builder_kind=request.builder_kind,
                exact_input_sha256=request.exact_input_sha256,
                code=AttemptFailureCode.INTERNAL_ATTEMPT_FAILURE,
                stage=AttemptFailureStage.INTERNAL,
                exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            ).to_value()
        )
        raise
    finally:
        if built is not None:
            built.close_source_logs()


async def run_attempt_worker(request: AttemptRequest) -> None:
    """Run the sole public completion-smoke builder (never formal benchmark input)."""

    await execute_attempt_request(request, build=build_public_exact_attempt)


def _write_private_traceback(path: Path) -> None:
    with path.open("xb") as stream:
        stream.write(traceback.format_exc().encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def main(request_path: str) -> None:
    request = AttemptRequest.from_value(json.loads(Path(request_path).read_text(encoding="utf-8")))
    asyncio.run(run_attempt_worker(request))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m skillev.runtime.attempt_worker REQUEST.json")
    main(sys.argv[1])


__all__ = [
    "AttemptBuilder",
    "TypedAttemptError",
    "execute_attempt_request",
    "main",
    "run_attempt_worker",
]
