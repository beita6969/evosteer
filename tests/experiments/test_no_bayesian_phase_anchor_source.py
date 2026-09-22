"""No-Bayesian phase-anchor source reduction without posterior placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillev_private.phase_anchor import _source_flow_only_phase_artifact

from skillev.contracts import (
    AuthoringUsageValue,
    PhaseCheckpointArtifact,
    RunCursorValue,
    stable_hash,
)
from skillev.experiments.arm_events import ArmEventType, LiveArmEventLog
from skillev.experiments.arms.no_bayesian_contracts import (
    FlowOnlyActionResult,
    FlowOnlyCycleResult,
    FlowOnlyGenerateEvidence,
    FlowOnlyPhaseCheckpointPublished,
    FlowOnlyPhaseOpened,
)
from skillev.experiments.protocol import AblationArm
from skillev.runtime import AttemptBuilderKind, SkillLibraryState
from tests.v3_helpers import make_phase_event, make_skill_document


@dataclass(frozen=True, slots=True)
class _FlowOnlyPublishedSource:
    """Minimal source-log surface consumed by the native flow-only reducer."""

    path: Path
    run_id: str
    attempt_id: str
    builder_kind: AttemptBuilderKind = AttemptBuilderKind.NO_BAYESIAN

    def source_log(self, _: object) -> Path:
        return self.path

    def source_log_path(self, item: Path) -> Path:
        return item


def test_flow_only_phase_checkpoint_reduces_native_arm_events(tmp_path: Path) -> None:
    source = make_skill_document("source")
    generated = make_skill_document("generated")
    before = SkillLibraryState.from_seed_documents((source,))
    after = SkillLibraryState.from_seed_documents((source, generated))
    phase = make_phase_event(library_version=before.current_version)
    run_plan_hash = stable_hash("flow-only-phase-anchor-plan")
    phase_cursor = RunCursorValue(
        run_plan_hash=run_plan_hash,
        completed_training_steps=phase.triggered_at_step,
        committed_cycles=0,
        committed_actions=0,
    )
    cursor_after = RunCursorValue(
        run_plan_hash=run_plan_hash,
        completed_training_steps=phase.triggered_at_step,
        committed_cycles=1,
        committed_actions=1,
    )
    proposal_hash = stable_hash("flow-only-generate-proposal")
    action = FlowOnlyActionResult(
        action_id="flow-only-action-1",
        proposal_content_hash=proposal_hash,
        action_kind="generate",
        target_skill_ids=(),
        produced_skill_ids=(generated.manifest.skill_id,),
        evidence=FlowOnlyGenerateEvidence(("gap:1",), 0.1, 0.9, "absolute-log-density-ratio@1"),
        rationale_text="cover one native flow-only gap",
    )
    cycle = FlowOnlyCycleResult(
        phase_event_id=phase.event_id,
        optimizer_step=phase.triggered_at_step,
        library_version_before=before.current_version,
        library_version_after=after.current_version,
        new_documents=(generated.to_value(),),
        active_skill_ids_after=after.active_skill_ids,
        actions=(action,),
        decision_content_hash=stable_hash("flow-only-decision"),
        proposal_content_hashes=(proposal_hash,),
        authoring_reservation_ids=("flow-only-authoring-1",),
        authoring_usage=AuthoringUsageValue(input_tokens=1, output_tokens=1, model_calls=1),
        z_reset_seed=7,
        z_version_after_reset="z@reset-7",
        run_cursor_after=cursor_after,
    )
    artifact = PhaseCheckpointArtifact(
        artifact_sha256=stable_hash("flow-only-phase-artifact"),
        runtime_state_sha256=stable_hash("flow-only-phase-runtime"),
        policy_snapshot_id="flow-only-policy@2",
        library_version=after.current_version,
        optimizer_step=phase.triggered_at_step,
        phase_event_id=phase.event_id,
        run_cursor_after=cursor_after,
    )
    path = tmp_path / "arm-events.jsonl"
    log = LiveArmEventLog(
        path,
        arm=AblationArm.SKILLFLOW_DISABLED,
        run_id="flow-only-run",
        attempt_id="flow-only-attempt",
    )
    log.append(
        ArmEventType.FLOW_ONLY_PHASE_OPENED,
        FlowOnlyPhaseOpened(phase, phase_cursor).to_value(),
    )
    log.append(ArmEventType.FLOW_ONLY_CYCLE_COMMITTED, cycle.to_value())
    log.append(
        ArmEventType.FLOW_ONLY_PHASE_CHECKPOINT_PUBLISHED,
        FlowOnlyPhaseCheckpointPublished(phase.event_id, artifact).to_value(),
    )
    log.close()

    reduced, phase_count = _source_flow_only_phase_artifact(
        _FlowOnlyPublishedSource(path, "flow-only-run", "flow-only-attempt"),
        phase_event_id=phase.event_id,
    )

    assert reduced == artifact
    assert phase_count == 1
    assert "posterior" not in FlowOnlyPhaseCheckpointPublished(phase.event_id, artifact).to_value()
