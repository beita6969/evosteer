"""Closed production dispatch from preregistered arm to an exact builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, assert_never, cast

from .attempt_protocol import (
    AttemptBuilderKind,
    AttemptRequest,
    AttemptRunSummary,
    AttemptSourceLogKind,
    expected_source_log_layout,
)
from .attempt_run_plan import ExactAttemptRunPlan

if TYPE_CHECKING:
    from skillev.experiments.attempt_identity import PublishedAttemptIdentity


class AttemptLedger(Protocol):
    def assert_fully_settled(self) -> None: ...


class AttemptTrainingLoop(Protocol):
    @property
    def ledger(self) -> AttemptLedger: ...


class AttemptEvolutionLoop(Protocol):
    async def run(self, plan: ExactAttemptRunPlan) -> AttemptRunSummary: ...


class ExactAttemptApplication(Protocol):
    @property
    def training_loop(self) -> AttemptTrainingLoop: ...

    @property
    def evolution_loop(self) -> AttemptEvolutionLoop: ...

    @property
    def final_training_snapshot_directory(self) -> Path: ...


def _accept_summary(summary: AttemptRunSummary) -> None:
    """Ordinary correctness fixtures have no additional terminal admission."""

    del summary


@dataclass(frozen=True, slots=True)
class BuiltAttempt:
    application: ExactAttemptApplication
    run_plan: ExactAttemptRunPlan
    public_identity: "PublishedAttemptIdentity"  # noqa: UP037
    source_logs: tuple[tuple[str, AttemptSourceLogKind], ...]
    close_source_logs_callback: Callable[[], None]
    validate_summary_callback: Callable[[AttemptRunSummary], None] = _accept_summary

    def __post_init__(self) -> None:
        expected = expected_source_log_layout(self.public_identity.builder_kind)
        if self.source_logs != expected:
            raise ValueError("built attempt source layout differs from its builder")
        if self.public_identity.run_plan != self.run_plan:
            raise ValueError("built attempt identity has another run plan")

    def close_source_logs(self) -> None:
        """Flush every builder-owned persistent source stream before hashing."""

        self.close_source_logs_callback()

    def validate_summary(self, summary: AttemptRunSummary) -> None:
        """Apply a builder-owned terminal admission before success publication."""

        self.validate_summary_callback(summary)


def build_exact_attempt(
    kind: AttemptBuilderKind,
    exact_input_path: Path,
    *,
    expected_exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    private_bundle_directory: Path,
) -> BuiltAttempt:
    """Build one fixed graph without reflective entrypoints or mutable registries."""

    event_log_path = private_bundle_directory / "events.jsonl"
    arm_event_log_path = private_bundle_directory / "arm-events.jsonl"
    match kind:
        case AttemptBuilderKind.FULL:
            from skillev.experiments.exact_attempts import build_full_application

            built = build_full_application(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
            )
        case AttemptBuilderKind.NO_BAYESIAN:
            from skillev.experiments.exact_attempts import build_no_bayesian_application

            built = build_no_bayesian_application(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            )
        case AttemptBuilderKind.UNIT_FLOW:
            from skillev.experiments.exact_attempts import build_unit_flow_application

            built = build_unit_flow_application(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            )
        case AttemptBuilderKind.CAPPED_FLOW:
            from skillev.experiments.exact_attempts import build_capped_flow_application

            built = build_capped_flow_application(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            )
        case AttemptBuilderKind.CLIPPED_IMPORTANCE:
            from skillev.experiments.exact_attempts import (
                build_clipped_importance_application_exact,
            )

            built = build_clipped_importance_application_exact(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            )
        case AttemptBuilderKind.POSTERIOR_MEAN:
            from skillev.experiments.exact_attempts import build_posterior_mean_application

            built = build_posterior_mean_application(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            )
        case AttemptBuilderKind.RESIDUAL_ONLY_PHASE:
            from skillev.experiments.exact_attempts import build_residual_only_application

            built = build_residual_only_application(
                exact_input_path,
                exact_input_sha256=expected_exact_input_sha256,
                run_id=run_id,
                attempt_id=attempt_id,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            )
        case _ as unreachable:
            assert_never(unreachable)
    return BuiltAttempt(
        application=cast(ExactAttemptApplication, built.application),
        run_plan=built.run_plan,
        public_identity=built.public_identity,
        source_logs=expected_source_log_layout(kind),
        close_source_logs_callback=built.close_source_logs,
    )


def build_public_exact_attempt(request: AttemptRequest) -> BuiltAttempt:
    """Fixed public-smoke builder used by the public child worker only."""

    if not isinstance(request, AttemptRequest):
        raise TypeError("public exact builder requires AttemptRequest")
    return build_exact_attempt(
        request.builder_kind,
        request.exact_input_path,
        expected_exact_input_sha256=request.exact_input_sha256,
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        private_bundle_directory=request.private_bundle_directory,
    )


__all__ = [
    "BuiltAttempt",
    "ExactAttemptApplication",
    "build_exact_attempt",
    "build_public_exact_attempt",
]
