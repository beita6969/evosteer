"""Narrow, explicitly selected condition transitions at a complete checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

from skillev.contracts.action_wire import NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE
from skillev.contracts.skill_exposure import can_continue_skill_exposure
from skillev.runtime import RuntimeSnapshotIdentity
from skillev.runtime.attempt_run_plan import AttemptRunCursorState, ExactAttemptRunPlan

if TYPE_CHECKING:
    from skillev.application import ApplicationPublicIdentity


@dataclass(frozen=True, slots=True)
class HorizonContinuation:
    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        """Keep all scientific state/identity except the declared horizon budget.

        This is a changed training condition, not exact same-condition recovery.
        Historical records/cells/windows remain unmodified and keep their source
        horizon features. The caller must persist the condition boundary.
        """
        old = self.source.application_config
        new = target.application_config
        before, after = old.trainer.rollout, new.trainer.rollout
        if self.optimizer_step < 0:
            raise ValueError("continuation needs a complete nonnegative step")
        if any(
            getattr(before.per_rollout_maximum, field.name) * after.max_turns
            != getattr(after.per_rollout_maximum, field.name) * before.max_turns
            for field in fields(after.per_rollout_maximum)
        ):
            raise ValueError("only the horizon-proportional call budget may change")
        replaced = replace(
            old,
            trainer=replace(
                old.trainer,
                rollout=replace(
                    before, max_turns=after.max_turns, per_rollout_maximum=after.per_rollout_maximum
                ),
            ),
        )
        if replaced != new or self.source.run_plan != target.run_plan:
            raise ValueError("horizon continuation cannot change optimizer, method or run plan")
        source_identity = self.source.runtime_snapshot_identity()
        target_identity = target.runtime_snapshot_identity()
        if (
            replace(
                source_identity,
                application_config_hash=target_identity.application_config_hash,
                public_identity_content_hash=target_identity.public_identity_content_hash,
            )
            != target_identity
        ):
            raise ValueError(
                "horizon continuation cannot change model, scorer or sampling identities"
            )
        return source_identity


@dataclass(frozen=True, slots=True)
class ActionWireContinuation:
    """Explicit native carrier v2 to v3 handoff repair, not a general prompt override."""

    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        old, new = self.source.application_config, target.application_config
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("action wire continuation needs a complete nonnegative step")
        if (old.trainer.rollout.action_wire, new.trainer.rollout.action_wire) != (
            NATIVE_TOOL_CARRIER_WIRE,
            NATIVE_TOOL_HANDOFF_WIRE,
        ):
            raise ValueError("only the declared native carrier v2 to v3 repair is supported")
        expected = replace(
            old,
            trainer=replace(
                old.trainer,
                rollout=replace(old.trainer.rollout, action_wire=NATIVE_TOOL_HANDOFF_WIRE),
            ),
        )
        if (
            expected != new
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("action wire continuation changes an undeclared method/run field")
        source_identity, target_identity = (
            self.source.runtime_snapshot_identity(),
            target.runtime_snapshot_identity(),
        )
        if (
            replace(
                source_identity,
                application_config_hash=target_identity.application_config_hash,
                public_identity_content_hash=target_identity.public_identity_content_hash,
            )
            != target_identity
        ):
            raise ValueError(
                "action wire continuation cannot change model, scorer, data or sampling"
            )
        return source_identity


@dataclass(frozen=True, slots=True)
class TokenBudgetNoticeContinuation:
    """Opt into runtime token-cap notices without changing the caps or other prompts."""

    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        old, new = self.source.application_config, target.application_config
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("token notice continuation needs a complete nonnegative step")
        if old.trainer.rollout.token_budget_notice or not new.trainer.rollout.token_budget_notice:
            raise ValueError("token notice continuation only enables the explicit notice")
        expected = replace(
            old,
            trainer=replace(
                old.trainer, rollout=replace(old.trainer.rollout, token_budget_notice=True)
            ),
        )
        if (
            expected != new
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("token notice continuation changes an undeclared method/run field")
        source_identity, target_identity = (
            self.source.runtime_snapshot_identity(),
            target.runtime_snapshot_identity(),
        )
        if (
            replace(
                source_identity,
                application_config_hash=target_identity.application_config_hash,
                public_identity_content_hash=target_identity.public_identity_content_hash,
            )
            != target_identity
        ):
            raise ValueError(
                "token notice continuation cannot change model, scorer, data or sampling"
            )
        return source_identity


@dataclass(frozen=True, slots=True)
class SkillExposureContinuation:
    """Move full-inline skills behind an explicit read at a complete checkpoint."""

    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        old, new = self.source.application_config, target.application_config
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("skill exposure continuation needs a complete nonnegative step")
        if not can_continue_skill_exposure(
            old.trainer.rollout.skill_exposure, new.trainer.rollout.skill_exposure
        ):
            raise ValueError("unsupported skill catalog condition continuation")
        expected = replace(
            old,
            trainer=replace(
                old.trainer,
                rollout=replace(
                    old.trainer.rollout, skill_exposure=new.trainer.rollout.skill_exposure
                ),
            ),
        )
        if (
            expected != new
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("skill exposure continuation changes an undeclared method/run field")
        source_identity, target_identity = (
            self.source.runtime_snapshot_identity(),
            target.runtime_snapshot_identity(),
        )
        if (
            replace(
                source_identity,
                application_config_hash=target_identity.application_config_hash,
                public_identity_content_hash=target_identity.public_identity_content_hash,
            )
            != target_identity
        ):
            raise ValueError("skill exposure continuation cannot change model, scorer or sampling")
        return source_identity


@dataclass(frozen=True, slots=True)
class SkillColdStartContinuation(SkillExposureContinuation):
    """Declare Generate cold start + optional catalog reads at a complete boundary.

    Old evidence keeps its original labels and provenance. Model, optimizer,
    task/seed order, all thresholds and the global cycle budget stay unchanged.
    This is a method-extension condition, NOT same-condition recovery.
    """

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        from skillev.contracts.skill_exposure import CATALOG_EXPOSURE_VERSION
        from skillev.evolution.cold_start_config import ColdStartConfig

        old, new = self.source.application_config, target.application_config
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("cold start needs a complete checkpoint")
        if old.evolution.cold_start is not None or not isinstance(
            new.evolution.cold_start, ColdStartConfig
        ):
            raise ValueError("cold-start continuation only enables the declared extension")
        if new.trainer.rollout.skill_exposure != CATALOG_EXPOSURE_VERSION:
            raise ValueError("cold start requires optional catalog reads")
        # Reuse the strict state/identity comparison without treating the method
        # change as a prompt-only transition. Nothing in the source is mutated.
        expected = replace(
            old,
            evolution=replace(old.evolution, cold_start=new.evolution.cold_start),
            trainer=replace(
                old.trainer,
                rollout=replace(old.trainer.rollout, skill_exposure=CATALOG_EXPOSURE_VERSION),
            ),
        )
        if (
            expected != new
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("cold-start continuation changes an undeclared method/run field")
        before, after = self.source.runtime_snapshot_identity(), target.runtime_snapshot_identity()
        if (
            replace(
                before,
                application_config_hash=after.application_config_hash,
                public_identity_content_hash=after.public_identity_content_hash,
            )
            != after
        ):
            raise ValueError("cold start cannot change model, scorer, data or sampling")
        return before


@dataclass(frozen=True, slots=True)
class ReasoningContinuation:
    """Change only the public reasoning tool catalog and reasoning token envelope.

    Domain-specific execution profiles live outside ApplicationConfig: their
    explicit declaration and comparison remain the caller's responsibility.
    This does not permit changing thinking mode, horizon, sampling, or action
    semantics. Branch naming/storage is independent of scientific identity.
    """

    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        old, new = self.source.application_config, target.application_config
        before, after = old.trainer.rollout, new.trainer.rollout
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("reasoning continuation needs a complete nonnegative step")
        # More reasoning tokens change only this derived output allowance, not
        # call counts, input capacity, tool time, or action/environment horizons.
        maximum = replace(
            before.per_rollout_maximum,
            output_tokens=before.max_turns
            * (after.max_reasoning_tokens + before.max_action_tokens),
        )
        expected = replace(
            old,
            trainer=replace(
                old.trainer,
                rollout=replace(
                    before,
                    reasoning_tool_catalog=after.reasoning_tool_catalog,
                    max_reasoning_tokens=after.max_reasoning_tokens,
                    per_rollout_maximum=maximum,
                ),
            ),
        )
        if (
            expected != new
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("reasoning continuation changes an undeclared method/run field")
        source_identity = self.source.runtime_snapshot_identity()
        target_identity = target.runtime_snapshot_identity()
        if (
            replace(
                source_identity,
                application_config_hash=target_identity.application_config_hash,
                public_identity_content_hash=target_identity.public_identity_content_hash,
            )
            != target_identity
        ):
            raise ValueError("reasoning continuation cannot change model, scorer, data or sampling")
        return source_identity


@dataclass(frozen=True, slots=True)
class PlanContinuation:
    """Append work after a complete plan; no reset or retrospective phase check."""

    source: ApplicationPublicIdentity
    optimizer_step: int
    source_task_ids: tuple[str, ...]
    target_task_ids: tuple[str, ...]

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        old, new = self.source.run_plan, target.run_plan
        segments = old.segments or ((old.phase_search_steps, old.closure_steps),)
        if (
            self.optimizer_step != old.total_training_steps
            or not new.segments
            or new.segments[:-1] != segments
            or new.maximum_cycles != old.maximum_cycles
        ):
            raise ValueError("plan continuation must append to the complete unchanged prefix")
        if self.source.application_config != target.application_config:
            raise ValueError("append-plan continuation cannot change model/rollout/method config")
        batch = target.application_config.trainer.execution.batch_size
        if (
            len(self.source_task_ids) != old.total_training_steps * batch
            or len(self.target_task_ids) != new.total_training_steps * batch
            or self.target_task_ids[: len(self.source_task_ids)] != self.source_task_ids
        ):
            raise ValueError("append-plan task order must preserve the entire original prefix")
        if self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals:
            raise ValueError("append-plan continuation cannot reset or expand evolution cycles")
        before, after = self.source.runtime_snapshot_identity(), target.runtime_snapshot_identity()
        if (
            replace(
                before,
                run_plan_hash=after.run_plan_hash,
                public_identity_content_hash=after.public_identity_content_hash,
            )
            != after
        ):
            raise ValueError("append-plan continuation changed model/scorer/sampling identity")
        return before

    def migrate_cursor(
        self, cursor: AttemptRunCursorState, target_plan: ExactAttemptRunPlan
    ) -> AttemptRunCursorState:
        """Keep actual completed steps and all cumulative evolution counters."""
        cursor.require_plan(self.source.run_plan)
        if cursor.completed_training_steps != self.optimizer_step:
            raise ValueError("append-plan boundary differs from the complete saved cursor")
        return replace(cursor, run_plan_hash=target_plan.content_hash)
