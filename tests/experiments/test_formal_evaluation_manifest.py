from __future__ import annotations

from pathlib import Path

import pytest
from skillev_private.experiments.formal_evaluation_manifest import (
    FORMAL_EVALUATION_HARDWARE_POLICY_HASH,
    FormalEvaluationKind,
    FormalEvaluationLedger,
    FormalEvaluationManifest,
    FormalEvaluationSlot,
    FormalEvaluationTerminalStatus,
)

from skillev.contracts import scientific_sampling_schedule_hash, stable_hash
from skillev.experiments import (
    Benchmark,
    BenchmarkTaskCount,
    ExecutionHardwareIdentity,
    FrozenTaskSequenceIdentity,
    SchedulePurpose,
)
from skillev.runtime import AttemptBuilderKind
from tests.experiments.formal_execution_helpers import make_formal_artifacts


def _sequence() -> FrozenTaskSequenceIdentity:
    return FrozenTaskSequenceIdentity(
        purpose=SchedulePurpose.IID_EVALUATION,
        ordered_task_ids_hash=stable_hash(["task-1"]),
        task_count=1,
        benchmark_counts=(BenchmarkTaskCount(Benchmark.WEBSHOP, 1),),
        schedule_algorithm="formal-evaluation-test@1",
    )


def _slot(*, kind: AttemptBuilderKind = AttemptBuilderKind.FULL) -> FormalEvaluationSlot:
    artifacts = make_formal_artifacts()
    return FormalEvaluationSlot.create(
        source_formal_run_group_id=stable_hash({"run-group": "one"}),
        source_builder_kind=kind,
        source_training_attempt_id=f"formal-{kind.value}",
        source_training_identity_hash=stable_hash({"identity": kind.value}),
        kind=FormalEvaluationKind.FINAL_IID,
        anchor_ordinal=0,
        frozen_state_hash=stable_hash({"state": kind.value}),
        policy_snapshot_id=f"policy-{kind.value}@10",
        library_version=stable_hash({"library": kind.value}),
        task_sequence_identity=_sequence(),
        sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=20260721),
        implementation_build_hash=artifacts.implementation.content_hash,
        execution_hardware_policy_hash=FORMAL_EVALUATION_HARDWARE_POLICY_HASH,
    )


def _manifest(slot: FormalEvaluationSlot) -> FormalEvaluationManifest:
    return FormalEvaluationManifest.create(
        protocol_hash=stable_hash({"protocol": "evaluation-ledger"}),
        protocol_freeze_id=stable_hash({"freeze": "evaluation-ledger"}),
        source_formal_run_group_id=slot.source_formal_run_group_id,
        slots=(slot,),
    )


def _hardware() -> ExecutionHardwareIdentity:
    return ExecutionHardwareIdentity(
        accelerator_name="NVIDIA H200",
        compute_capability="9.0",
        visible_device_count=1,
        nvidia_driver_version="570.12",
        cuda_runtime_version="12.8",
        cudnn_version="91000",
        nccl_version="2.27.5",
        kernel_release="6.12.0-test",
        safetensors_version="0.5.3",
    )


def test_formal_evaluation_slot_and_manifest_round_trip() -> None:
    slot = _slot()
    manifest = _manifest(slot)

    assert FormalEvaluationSlot.from_value(slot.to_value()) == slot
    assert FormalEvaluationManifest.from_value(manifest.to_value()) == manifest
    assert (
        slot.evaluation_id
        == FormalEvaluationSlot.create(
            source_formal_run_group_id=slot.source_formal_run_group_id,
            source_builder_kind=slot.source_builder_kind,
            source_training_attempt_id=slot.source_training_attempt_id,
            source_training_identity_hash=slot.source_training_identity_hash,
            kind=slot.kind,
            anchor_ordinal=slot.anchor_ordinal,
            frozen_state_hash=slot.frozen_state_hash,
            policy_snapshot_id=slot.policy_snapshot_id,
            library_version=slot.library_version,
            task_sequence_identity=slot.task_sequence_identity,
            sampling_schedule_hash=slot.sampling_schedule_hash,
            implementation_build_hash=slot.implementation_build_hash,
            execution_hardware_policy_hash=slot.execution_hardware_policy_hash,
        ).evaluation_id
    )


def test_formal_evaluation_claim_launch_and_terminal_are_write_once(tmp_path: Path) -> None:
    slot = _slot()
    ledger = FormalEvaluationLedger.create(directory=tmp_path / "ledger", manifest=_manifest(slot))
    secret = ledger.claim(slot.evaluation_id)
    with pytest.raises(ValueError):
        ledger.claim(slot.evaluation_id)

    launch = ledger.launch(secret)
    assert ledger.consume(launch) == slot
    with pytest.raises(FileExistsError):
        ledger.consume(launch)

    build = make_formal_artifacts().implementation
    terminal = ledger.record_success(
        slot=slot,
        outcome_hash=stable_hash({"private-episode-results": "exact"}),
        implementation_build=build,
        execution_hardware=_hardware(),
    )
    assert terminal.status is FormalEvaluationTerminalStatus.SUCCEEDED
    assert ledger.require_success(slot.evaluation_id) == terminal
    with pytest.raises(FileExistsError):
        ledger.record_success(
            slot=slot,
            outcome_hash=terminal.outcome_hash or "",
            implementation_build=build,
            execution_hardware=_hardware(),
        )


def test_failed_formal_evaluation_slot_cannot_be_replaced(tmp_path: Path) -> None:
    slot = _slot()
    ledger = FormalEvaluationLedger.create(directory=tmp_path / "ledger", manifest=_manifest(slot))
    secret = ledger.claim(slot.evaluation_id)
    ledger.consume(ledger.launch(secret))
    terminal = ledger.record_failure(slot=slot)

    assert terminal.status is FormalEvaluationTerminalStatus.FAILED
    with pytest.raises(ValueError):
        ledger.require_success(slot.evaluation_id)
    with pytest.raises(ValueError):
        ledger.claim(slot.evaluation_id)
