"""Cross-component recovery checks at the existing step transaction boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skillev.contracts import EvolutionCycleCommitted, TrainingStepCommit
from skillev.runtime import (
    EventType,
    StepTransactionJournal,
    StepTransactionReconciler,
    StepTransactionRecord,
    StepTransactionState,
)
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


def require_pending_step_matches(
    application: SKILLEVApplication, record: StepTransactionRecord
) -> None:
    if record.policy_snapshot_after != application.training_loop.policy_snapshot_id:
        raise ValueError("restored model policy differs from the checkpointed transaction")
    steps = tuple(
        TrainingStepCommit.from_value(event.payload)
        for event in record.source_events
        if event.event_type is EventType.TRAINING_STEP_COMMITTED
    )
    if len(steps) != 1:
        raise ValueError("checkpointed transaction requires exactly one training step")
    step = steps[0]
    provenance = application.projections.posterior_provenance
    if not provenance.batches:
        raise ValueError("checkpointed transaction has no posterior batch cursor")
    latest = provenance.batches[-1]
    if (
        step.batch_id != record.batch_id
        or step.optimizer_step != record.optimizer_step
        or latest.posterior != step.posterior_batch
        or latest.policy_snapshot_id != step.policy_snapshot_before
        or latest.library_version != step.library_version
        or latest.trajectory_ids != tuple(item.trajectory_id for item in step.records)
    ):
        raise ValueError("restored posterior evidence differs from the checkpointed training step")
    library_version = step.library_version
    for event in record.source_events:
        if event.event_type is EventType.EVOLUTION_CYCLE_COMMITTED:
            cycle = EvolutionCycleCommitted.from_value(event.payload)
            library_version = cycle.library_version_after
            if cycle.z_version_after_reset != application.training_loop.backbone.z_version:
                raise ValueError("restored partition differs from the committed evolution reset")
    if library_version != application.library.current_version:
        raise ValueError("restored skill library differs from the committed evolution")


def reconcile_application_step(
    application: SKILLEVApplication, journal: StepTransactionJournal, snapshot_directory: Path
) -> None:
    pending = journal.pending()
    if len(pending) > 1:
        raise RuntimeError("formal resume found multiple unfinished optimizer steps")
    if not pending:
        step = application.training_loop.optimizer_step
        if step:
            # A trained snapshot without its original completed transaction is
            # not a fresh run: adapter/event publication cannot be reconstructed.
            try:
                committed = journal.load(step)
            except FileNotFoundError as error:
                raise RuntimeError("restored checkpoint is missing its step journal") from error
            if committed.state is not StepTransactionState.COMMITTED:
                raise RuntimeError("restored checkpoint has no completed transaction")
            require_pending_step_matches(application, committed)
        return
    record = pending[0]
    if record.state in {
        StepTransactionState.PREPARED,
        StepTransactionState.OPTIMIZER_APPLIED,
        StepTransactionState.PROJECTION_INSTALLED,
        StepTransactionState.EVOLUTION_RESOLVED,
    }:
        if record.optimizer_step != application.training_loop.optimizer_step + 1:
            raise RuntimeError("formal rollback differs from the next optimizer step")
        if record.policy_snapshot_before != application.training_loop.policy_snapshot_id:
            raise RuntimeError("formal rollback policy differs from the last committed step")
        FilesystemTrainingCheckpointStore(root=snapshot_directory.parent).archive_uncommitted_step(
            record.optimizer_step
        )
        journal.rollback_before_checkpoint(record)
        return
    if record.checkpoint_name != snapshot_directory.resolve().name:
        raise RuntimeError("formal resume checkpoint differs from pending transaction")
    if record.optimizer_step != application.training_loop.optimizer_step:
        raise RuntimeError("formal resume optimizer differs from pending transaction")

    def ensure_adapter(record: StepTransactionRecord) -> str:
        generation = application.evolution_loop.publish_restored_adapter()
        return "no-external-adapter" if generation is None else generation.adapter_revision

    StepTransactionReconciler(
        journal=journal,
        emitter=application.emitter,
        require_checkpoint=lambda item: require_pending_step_matches(application, item),
        ensure_adapter=ensure_adapter,
    ).reconcile(record)
