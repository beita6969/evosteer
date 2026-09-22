"""A complete-state boundary for a training-only terminal reward change."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from skillev.evaluation.format_content_review import format_review_condition
from skillev.runtime import RuntimeSnapshotIdentity

if TYPE_CHECKING:
    from skillev.application import ApplicationPublicIdentity


@dataclass(frozen=True, slots=True)
class FormatReviewContinuation:
    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("format review requires a complete source checkpoint")
        if (
            self.source.application_config != target.application_config
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("format review cannot change learning or rollout controls")
        before, after = self.source.runtime_snapshot_identity(), target.runtime_snapshot_identity()
        old = json.loads(before.terminal_evaluation_conditions_json or "{}")
        new = json.loads(after.terminal_evaluation_conditions_json or "{}")
        if "format_content_review" in old or new != {
            **old,
            "format_content_review": format_review_condition(self.optimizer_step + 1),
        }:
            raise ValueError("only adding format review at the next step is supported")
        if (
            replace(
                before,
                terminal_evaluation_conditions_json=after.terminal_evaluation_conditions_json,
                public_identity_content_hash=after.public_identity_content_hash,
            )
            != after
        ):
            raise ValueError("format review cannot change models, sources or other scorers")
        return before
