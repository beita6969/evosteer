"""Formal B1 terminal admission runs before a worker publishes success."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from skillev.contracts import JsonValue, TrainingStepReportValue, stable_hash
from skillev.experiments.b1_run_admission import B1TerminalRequirements
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptFailed,
    AttemptRequest,
    AttemptRunSummary,
    AttemptSourceLogKind,
    AttemptSucceeded,
    ExactAttemptRunPlan,
    FullAttemptSummary,
    UnpublishedAttemptBundle,
    read_attempt_outcome,
)
from skillev.runtime.attempt_builders import BuiltAttempt
from skillev.runtime.attempt_worker import execute_attempt_request

_INITIAL_LIBRARY = "sha256:" + "1" * 64
_FINAL_LIBRARY = "sha256:" + "2" * 64


@dataclass(frozen=True)
class _Identity:
    run_plan: ExactAttemptRunPlan
    builder_kind: AttemptBuilderKind = AttemptBuilderKind.FULL

    def to_value(self) -> dict[str, JsonValue]:
        return {"format": "b1-terminal-worker-test@1"}


class _Ledger:
    def assert_fully_settled(self) -> None:
        return None


@dataclass(frozen=True)
class _TrainingLoop:
    ledger: _Ledger = field(default_factory=_Ledger)


@dataclass(frozen=True)
class _EvolutionLoop:
    summary: AttemptRunSummary

    async def run(self, plan: ExactAttemptRunPlan) -> AttemptRunSummary:
        del plan
        return self.summary


@dataclass(frozen=True)
class _Application:
    evolution_loop: _EvolutionLoop
    final_training_snapshot_directory: Path
    training_loop: _TrainingLoop = _TrainingLoop()


def _request(tmp_path: Path, *, attempt_id: str) -> AttemptRequest:
    bundle = UnpublishedAttemptBundle.create(tmp_path / attempt_id)
    return AttemptRequest(
        run_id="b1-terminal-worker-test",
        attempt_id=attempt_id,
        builder_kind=AttemptBuilderKind.FULL,
        exact_input_path=bundle.directory / "unused-input.json",
        exact_input_sha256=stable_hash({"attempt": attempt_id}),
        private_bundle_directory=bundle.directory,
    )


def _report(step: int) -> TrainingStepReportValue:
    return TrainingStepReportValue(
        optimizer_step=step,
        batch_id=f"batch-{step}",
        torch_batch_loss=0.25,
        audited_batch_loss=0.25,
        mean_reward=0.5,
        grad_norm_forward=1.0,
        grad_norm_backward=1.0,
        grad_norm_z=1.0,
        forward_adapter_version=f"forward@{step}",
        backward_adapter_version=f"backward@{step}",
        z_version=f"z@{step}",
        started_at="2026-08-02T00:00:00Z",
        completed_at="2026-08-02T00:00:01Z",
    )


def _summary(
    *, cycles: int = 1, actions: int = 1, final: str = _FINAL_LIBRARY
) -> FullAttemptSummary:
    return FullAttemptSummary(
        reports=(_report(1), _report(2)),
        planned_training_steps_this_attempt=2,
        completed_training_steps_this_attempt=2,
        actions_committed_this_attempt=actions,
        cycles_committed_this_attempt=cycles,
        cycles_committed_in_run=cycles,
        initial_optimizer_step=0,
        final_optimizer_step=2,
        final_library_version=final,
        final_policy_snapshot_id="b1-final-snapshot",
    )


def _builder(
    request: AttemptRequest,
    *,
    summary: FullAttemptSummary,
    closure_library: str,
) -> BuiltAttempt:
    plan = ExactAttemptRunPlan(phase_search_steps=1, closure_steps=1, maximum_cycles=1)
    snapshot = request.private_bundle_directory / "snapshot"
    snapshot.mkdir()
    (snapshot / "runtime_state.json").write_text("{}\n", encoding="utf-8")
    event_log = request.private_bundle_directory / "events.jsonl"
    event_log.write_text("", encoding="utf-8")
    terminal = B1TerminalRequirements(
        planned_training_steps=2,
        closure_steps=1,
        minimum_committed_cycles=1,
        minimum_committed_actions=1,
        initial_library_version=_INITIAL_LIBRARY,
    )

    def validate(actual: AttemptRunSummary) -> None:
        terminal.require_success(
            actual,
            ordered_training_library_versions=(_INITIAL_LIBRARY, closure_library),
        )

    return BuiltAttempt(
        application=_Application(_EvolutionLoop(summary), snapshot),  # type: ignore[arg-type]
        run_plan=plan,
        public_identity=_Identity(plan),  # type: ignore[arg-type]
        source_logs=(("events.jsonl", AttemptSourceLogKind.FULL_METHOD),),
        close_source_logs_callback=lambda: None,
        validate_summary_callback=validate,
    )


@pytest.mark.parametrize(
    ("summary", "closure_library"),
    [
        (_summary(cycles=0, actions=0), _FINAL_LIBRARY),
        (_summary(final=_INITIAL_LIBRARY), _INITIAL_LIBRARY),
        (_summary(), _INITIAL_LIBRARY),
    ],
)
def test_worker_emits_failure_when_b1_terminal_admission_fails(
    tmp_path: Path,
    summary: FullAttemptSummary,
    closure_library: str,
) -> None:
    request = _request(tmp_path, attempt_id=f"failed-{len(tuple(tmp_path.iterdir()))}")

    with pytest.raises(ValueError):
        asyncio.run(
            execute_attempt_request(
                request,
                build=lambda source: _builder(
                    source,
                    summary=summary,
                    closure_library=closure_library,
                ),
            )
        )

    assert isinstance(
        read_attempt_outcome(request.private_bundle_directory / "outcome.json"),
        AttemptFailed,
    )


def test_worker_publishes_success_after_valid_b1_terminal_admission(tmp_path: Path) -> None:
    request = _request(tmp_path, attempt_id="valid")
    summary = _summary()
    asyncio.run(
        execute_attempt_request(
            request,
            build=lambda source: _builder(
                source,
                summary=summary,
                closure_library=_FINAL_LIBRARY,
            ),
        )
    )

    assert isinstance(
        read_attempt_outcome(request.private_bundle_directory / "outcome.json"),
        AttemptSucceeded,
    )
