"""Explicit domain removal at a saved batch boundary, without resetting learning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from skillev.runtime import RuntimeSnapshotIdentity

if TYPE_CHECKING:
    from skillev.application import ApplicationPublicIdentity


@dataclass(frozen=True, slots=True)
class DomainSubsetContinuation:
    source: ApplicationPublicIdentity
    optimizer_step: int
    completed_tasks: int
    source_task_ids: tuple[str, ...]
    target_task_ids: tuple[str, ...]

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        old, new = self.source.application_config, target.application_config
        before, after = old.trainer.execution.batch_size, new.trainer.execution.batch_size
        if self.optimizer_step < 0 or not 0 < after < before:
            raise ValueError("domain removal requires a saved boundary and a smaller batch")
        modes = new.trainer.rollout.reasoning_by_domain
        if (
            tuple(v for v in old.trainer.rollout.reasoning_by_domain if v[0] in dict(modes))
            != modes
        ):
            raise ValueError("domain removal must preserve thinking for retained domains")
        expected = replace(
            old,
            trainer=replace(
                old.trainer,
                execution=replace(old.trainer.execution, batch_size=after),
                rollout=replace(old.trainer.rollout, reasoning_by_domain=modes),
            ),
        )
        if (
            expected != new
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("domain removal cannot change the learning or rollout controls")
        cursor = self.completed_tasks
        if not 0 <= cursor <= len(self.source_task_ids):
            raise ValueError("invalid completed source cursor")
        if self.source_task_ids[:cursor] != self.target_task_ids[:cursor]:
            raise ValueError("domain removal must retain the entire consumed task prefix")
        remaining = self.source.run_plan.total_training_steps - self.optimizer_step
        if (
            len(self.source_task_ids) - cursor != remaining * before
            or len(self.target_task_ids) - cursor != remaining * after
        ):
            raise ValueError("domain removal must preserve every remaining complete batch")
        retained = set(self.target_task_ids[cursor:])
        if (
            tuple(t for t in self.source_task_ids[cursor:] if t in retained)
            != self.target_task_ids[cursor:]
        ):
            raise ValueError("remaining tasks must be an ordered subset, not replacements")
        source, destination = (
            self.source.runtime_snapshot_identity(),
            target.runtime_snapshot_identity(),
        )
        if (
            replace(
                source,
                application_config_hash=destination.application_config_hash,
                public_identity_content_hash=destination.public_identity_content_hash,
                protocol_hash=destination.protocol_hash,
                protocol_freeze_id=destination.protocol_freeze_id,
                ordered_task_sequence_hash=destination.ordered_task_sequence_hash,
                sampling_schedule_hash=destination.sampling_schedule_hash,
            )
            != destination
        ):
            raise ValueError("domain removal cannot change model, scorer or sampling settings")
        return source
