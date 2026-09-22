from __future__ import annotations

from pathlib import Path
from typing import Any

from skillev_private.experiments.formal_evaluation_manifest import (
    FORMAL_EVALUATION_HARDWARE_POLICY_HASH,
    FormalEvaluationKind,
    FormalEvaluationLaunch,
    FormalEvaluationSlot,
    FormalEvaluationTerminal,
    FormalEvaluationTerminalStatus,
)

from skillev.contracts import scientific_sampling_schedule_hash, stable_hash
from skillev.experiments import ExecutionHardwareIdentity, FrozenTaskSequenceIdentity
from skillev.runtime import AttemptBuilderKind
from tests.experiments.formal_execution_helpers import make_formal_artifacts


def make_evaluation_hardware() -> ExecutionHardwareIdentity:
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


def make_formal_evaluation_launch() -> FormalEvaluationLaunch:
    return FormalEvaluationLaunch(
        ledger_directory=Path("/private/formal-evaluation-ledger"),
        manifest_id=stable_hash({"manifest": "test"}),
        evaluation_id=stable_hash({"evaluation": "test"}),
        token="test-formal-evaluation-token",
    )


def make_successful_evaluation_terminal(
    *,
    state: Any,
    task_sequence_identity: FrozenTaskSequenceIdentity,
    source_builder_kind: AttemptBuilderKind = AttemptBuilderKind.FULL,
) -> FormalEvaluationTerminal:
    source_attempt_id = getattr(
        state,
        "source_training_attempt_id",
        getattr(state, "source_attempt_id", None),
    )
    source_identity_hash = getattr(
        state,
        "source_training_identity_hash",
        getattr(state, "source_identity_hash", None),
    )
    build = make_formal_artifacts().implementation
    slot = FormalEvaluationSlot.create(
        source_formal_run_group_id=state.source_formal_run_group_id,
        source_builder_kind=source_builder_kind,
        source_training_attempt_id=source_attempt_id,
        source_training_identity_hash=source_identity_hash,
        kind=FormalEvaluationKind.FINAL_IID,
        anchor_ordinal=0,
        frozen_state_hash=state.content_hash,
        policy_snapshot_id=state.policy_snapshot_id,
        library_version=state.library.current_version,
        task_sequence_identity=task_sequence_identity,
        sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=20260721),
        implementation_build_hash=build.content_hash,
        execution_hardware_policy_hash=FORMAL_EVALUATION_HARDWARE_POLICY_HASH,
    )
    return FormalEvaluationTerminal(
        manifest_id=stable_hash({"manifest": slot.evaluation_id}),
        slot=slot,
        token_hash=stable_hash({"token": slot.evaluation_id}),
        status=FormalEvaluationTerminalStatus.SUCCEEDED,
        outcome_hash=stable_hash({"outcome": slot.evaluation_id}),
        implementation_build=build,
        execution_hardware=make_evaluation_hardware(),
    )
