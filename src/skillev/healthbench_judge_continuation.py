"""Owner-declared terminal judge change, only at a complete training boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from skillev.evaluation.external_judge_policy import (
    DIRECT_HEALTHBENCH_PROFILE,
    GATEWAY_HEALTHBENCH_PROFILE,
)
from skillev.evaluation.healthbench_luna_profile import PROFILE_ID, healthbench_condition
from skillev.runtime import RuntimeSnapshotIdentity

if TYPE_CHECKING:
    from skillev.application import ApplicationPublicIdentity


@dataclass(frozen=True, slots=True)
class HealthBenchJudgeContinuation:
    source: ApplicationPublicIdentity
    optimizer_step: int

    def require_target(self, target: ApplicationPublicIdentity) -> RuntimeSnapshotIdentity:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("judge continuation requires a complete checkpoint")
        if (
            self.source.application_config != target.application_config
            or self.source.run_plan != target.run_plan
            or self.source.initial_run_cursor != target.initial_run_cursor
            or self.source.phase_checkpoint_cycle_ordinals != target.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("judge continuation cannot change method or rollout conditions")
        before, after = self.source.runtime_snapshot_identity(), target.runtime_snapshot_identity()
        if (
            before.terminal_evaluation_conditions_json is None
            or after.terminal_evaluation_conditions_json is None
        ):
            raise ValueError("judge continuation requires an explicit original scorer condition")
        old = json.loads(before.terminal_evaluation_conditions_json)
        new = json.loads(after.terminal_evaluation_conditions_json)
        previous = old.get("healthbench")
        if previous is not None and previous not in (
            healthbench_condition(DIRECT_HEALTHBENCH_PROFILE),
            healthbench_condition(GATEWAY_HEALTHBENCH_PROFILE),
        ):
            raise ValueError("HealthBench continuation source is not a declared predecessor")
        selected = new.get("healthbench")
        if selected not in (
            healthbench_condition(GATEWAY_HEALTHBENCH_PROFILE),
            healthbench_condition(PROFILE_ID),
        ):
            raise ValueError("unsupported target HealthBench judge condition")
        expected = {**old, "healthbench": selected}
        if new != expected:
            raise ValueError("only the declared HealthBench Judge transition is supported")
        if (
            replace(
                before,
                terminal_evaluation_conditions_json=after.terminal_evaluation_conditions_json,
                public_identity_content_hash=after.public_identity_content_hash,
            )
            != after
        ):
            raise ValueError("judge continuation cannot change model, data, seeds or other scorers")
        # Restore the original model, optimizer and evidence. Historical rewards
        # are never re-judged; the caller persists a new condition checkpoint.
        return before
