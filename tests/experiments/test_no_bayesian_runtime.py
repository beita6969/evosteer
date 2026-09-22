from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from skillev.contracts import RunCursorValue, TrainingStepReportValue, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    AuthoredSkillDraft,
    AuthoringActionKind,
    AuthoringResult,
    EvolutionConfig,
    PosteriorEvidenceView,
    SkillAuthoringAuthority,
    TrajectoryEvidenceView,
    WindowFlowView,
    build_evidence_pack,
)
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.experiments import AblationArm
from skillev.experiments.arm_events import (
    ArmEventType,
    LiveArmEventLog,
    read_arm_event_history,
)
from skillev.experiments.arms import (
    FlowOnlyDecision,
    FlowOnlyGenerateEvidence,
    FlowOnlyGenerateProposal,
    FlowOnlyProjectionPipeline,
    FlowOnlyTrainingSource,
    FlowOnlyTrainingStepCommit,
    build_flow_only_evidence_pack,
)
from skillev.experiments.arms.no_bayesian_execution import (
    build_flow_only_mutation,
    required_flow_only_phi_budget,
)
from skillev.runtime import (
    BudgetVector,
    SkillApplicability,
    SkillLibrary,
    SkillLibraryState,
    SkillRequirement,
)
from tests.fakes.skill_author import ScriptedSkillAuthor
from tests.v3_helpers import (
    TEST_CONTEXT_ID,
    TEST_TASK_FAMILY,
    CharacterTokenizer,
    make_artifact,
    make_authoring_edge,
    make_phase_event,
    make_skill_document,
    make_source,
)


def _report(step: int = 1, *, batch_id: str | None = None) -> TrainingStepReportValue:
    return TrainingStepReportValue(
        optimizer_step=step,
        batch_id=batch_id or f"batch-{step}",
        torch_batch_loss=1.0,
        audited_batch_loss=1.0,
        mean_reward=0.5,
        grad_norm_forward=1.0,
        grad_norm_backward=1.0,
        grad_norm_z=1.0,
        forward_adapter_version=f"forward@{step}",
        backward_adapter_version=f"backward@{step}",
        z_version=f"z@{step}",
        started_at="2026-07-25T00:00:00Z",
        completed_at="2026-07-25T00:00:01Z",
    )


def test_flow_only_training_source_has_no_full_posterior_payload(tmp_path: Path) -> None:
    source = make_source()
    commit = FlowOnlyTrainingStepCommit(
        batch_id=source.batch_id,
        optimizer_step=source.optimizer_step,
        policy_snapshot_before="policy-before",
        policy_snapshot_after="policy-after",
        library_version=source.stats.library_version,
        records=tuple(artifact.record for artifact in source.artifacts),
        edge_records=source.edge_records,
        stats=source.stats,
        report=_report(source.optimizer_step, batch_id=source.batch_id),
        run_cursor_after=RunCursorValue(
            run_plan_hash=stable_hash({"run_plan": "flow-only-unit-test"}),
            completed_training_steps=source.optimizer_step,
            committed_cycles=0,
            committed_actions=0,
        ),
    )
    path = tmp_path / "flow-only-events.jsonl"
    log = LiveArmEventLog(
        path,
        arm=AblationArm.SKILLFLOW_DISABLED,
        run_id="run-flow-only",
        attempt_id="attempt-flow-only",
    )
    log.append(ArmEventType.FLOW_ONLY_TRAINING_STEP_COMMITTED, commit.to_value())
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert "posterior_batch" not in commit.to_value()
    assert "posterior_batch" not in persisted["payload"]


def test_flow_only_event_log_resumes_the_same_attempt_prefix(tmp_path: Path) -> None:
    path = tmp_path / "flow-only-events.jsonl"
    first = LiveArmEventLog(
        path,
        arm=AblationArm.SKILLFLOW_DISABLED,
        run_id="run-flow-only",
        attempt_id="attempt-flow-only",
    )
    first.append(ArmEventType.FLOW_ONLY_LIBRARY_INITIALIZED, {"value": 1})
    first.close()

    resumed = LiveArmEventLog.resume(
        path,
        arm=AblationArm.SKILLFLOW_DISABLED,
        run_id="run-flow-only",
        attempt_id="attempt-flow-only",
    )
    resumed.append(ArmEventType.FLOW_ONLY_TRAINING_STEP_COMMITTED, {"value": 2})
    resumed.close()

    history = read_arm_event_history(path)
    assert tuple(event.sequence for event in history) == (1, 2)
    with pytest.raises(ValueError):
        LiveArmEventLog.resume(
            path,
            arm=AblationArm.SKILLFLOW_DISABLED,
            run_id="another-run",
            attempt_id="attempt-flow-only",
        )


def test_flow_only_projection_has_a_closed_state_without_posterior_placeholders() -> None:
    source = make_source()
    config = DiagnosticsConfig(window_size=1)
    projection = FlowOnlyProjectionPipeline.fresh(
        config,
        library_version=source.stats.library_version,
    )
    flow_source = FlowOnlyTrainingSource(
        batch_id=source.batch_id,
        optimizer_step=source.optimizer_step,
        records=tuple(artifact.record for artifact in source.artifacts),
        stats=source.stats,
        edge_records=source.edge_records,
    )
    transition = projection.preview(flow_source)
    projection.commit(transition)
    state = projection.runtime_state()
    restored = FlowOnlyProjectionPipeline.from_runtime_state(config, state)

    assert state.kind == "flow-only"
    assert state.format == "skillev-flow-only-projection-runtime-state@2"
    assert "calibration_cells" not in state.to_value()
    assert "event_ids_by_skill" not in state.to_value()
    assert restored.runtime_state().to_value() == state.to_value()
    with pytest.raises(ValueError):
        type(state).from_value(
            {
                **state.to_value(),
                "format": "skillev-flow-only-projection-runtime-state@1",
            }
        )


def test_flow_only_generate_uses_the_same_answer_free_evidence_and_exact_budget() -> None:
    seed = make_skill_document("alpha")
    library = SkillLibrary(SkillLibraryState.from_seed_documents((seed,)))
    phase = make_phase_event(library_version=library.current_version)
    exemplar = make_authoring_edge("gap:1")
    decision = FlowOnlyDecision(
        phase_event_id=phase.event_id,
        proposals=(
            FlowOnlyGenerateProposal(
                evidence=FlowOnlyGenerateEvidence(
                    (exemplar.edge_id,), 0.1, 0.9, "absolute-log-density-ratio@1"
                ),
                rationale_text="cover the uncovered edge",
                edge_exemplars=(exemplar,),
            ),
        ),
    )
    authored = AuthoringResult(
        drafts=(
            AuthoredSkillDraft(
                title="Flow-only gap skill",
                summary="Covers one public gap.",
                instructions="Apply the public flow-only gap procedure.",
                applicability=SkillApplicability(
                    task_families=(TEST_TASK_FAMILY,),
                    contexts=(TEST_CONTEXT_ID,),
                    required_tools=(),
                    excluded_contexts=(),
                ),
                requirements=(
                    SkillRequirement(
                        requirement_id=uncovered_edge_requirement_id(exemplar.edge_id),
                        text="Cover the public gap.",
                        kind="evolvable-strategy",
                    ),
                ),
            ),
        )
    )
    author = ScriptedSkillAuthor(CharacterTokenizer(), (authored,))

    assert required_flow_only_phi_budget(
        decision,
        max_authoring_completion_tokens=4096,
        max_authoring_prompt_tokens=2048,
    ) == BudgetVector(model_calls=1, input_tokens=2048, output_tokens=4096)
    mutation = build_flow_only_mutation(
        decision,
        author=author,
        authority=SkillAuthoringAuthority(
            input_schema_id="input@3",
            output_schema_id="output@3",
            license_id="unit-test",
            allowed_task_families=(TEST_TASK_FAMILY,),
            allowed_tools=(),
        ),
        library=library,
        phase_event=phase,
        optimizer_step=2,
        max_skill_instruction_tokens_per_draft=1024,
        base_seed=17,
        cycle_ordinal=1,
    )

    assert len(mutation.actions) == 1
    assert mutation.library_version_before == library.current_version
    assert mutation.library_version_after != library.current_version
    assert author.requests[0].edge_exemplars == (exemplar,)


def test_full_and_flow_only_generate_population_share_terminal_eligibility() -> None:
    library_version = "library-generate-eligibility"
    previous = make_source(
        batch_id="batch-eligibility-1",
        optimizer_step=1,
        library_version=library_version,
        artifacts=(
            make_artifact(
                "trajectory-tool",
                skills_by_step=((),),
                library_version=library_version,
            ),
            make_artifact(
                "trajectory-invalid",
                skills_by_step=((),),
                library_version=library_version,
            ),
        ),
        importances=((100.0,), (50.0,)),
    )
    current = make_source(
        batch_id="batch-eligibility-2",
        optimizer_step=2,
        library_version=library_version,
        artifacts=(
            make_artifact(
                "trajectory-complete",
                skills_by_step=((),),
                library_version=library_version,
            ),
        ),
        importances=((1000.0,),),
    )
    diagnostics_config = DiagnosticsConfig(window_size=1)
    projection = FlowOnlyProjectionPipeline.fresh(
        diagnostics_config,
        library_version=library_version,
    )
    for source in (previous, current):
        transition = projection.preview(
            FlowOnlyTrainingSource(
                batch_id=source.batch_id,
                optimizer_step=source.optimizer_step,
                records=tuple(item.record for item in source.artifacts),
                stats=source.stats,
                edge_records=source.edge_records,
            )
        )
        projection.commit(transition)
    phase = make_phase_event(
        library_version=library_version,
        previous_batch_id=previous.batch_id,
        current_batch_id=current.batch_id,
    )
    diagnostics = projection.diagnostics_for_batch_ids((previous.batch_id, current.batch_id))
    window = WindowFlowView(
        phase_event=phase,
        diagnostics=diagnostics,
        active_skill_ids=(),
        applicability_by_skill={},
        task_family_universe=(TEST_TASK_FAMILY,),
    )
    tool = replace(
        make_authoring_edge("trajectory-tool:1"),
        action_kind=AuthoringActionKind.TOOL,
        tool_or_skill_name="debug.tool",
        absolute_log_importance=100.0,
    )
    invalid = replace(
        make_authoring_edge("trajectory-invalid:1"),
        action_kind=AuthoringActionKind.INVALID,
        tool_or_skill_name=None,
        absolute_log_importance=50.0,
    )
    complete = replace(
        make_authoring_edge("trajectory-complete:1"),
        action_kind=AuthoringActionKind.COMPLETE,
        tool_or_skill_name="complete",
        absolute_log_importance=1000.0,
    )
    trajectories = TrajectoryEvidenceView(
        authoring_by_edge_id={item.edge_id: item for item in (tool, invalid, complete)}
    )
    config = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        importance_quantile=0.5,
    )

    full = build_evidence_pack(
        window_flow=window,
        posterior=PosteriorEvidenceView(by_skill={}),
        trajectories=trajectories,
        config=config,
    )
    flow_only = build_flow_only_evidence_pack(
        window_flow=window,
        trajectories=trajectories,
        config=config,
    )

    expected = ("trajectory-complete:1",)
    assert full.uncovered_importance_edges == expected
    assert flow_only.uncovered_edge_ids == expected
    assert "trajectory-invalid:1" not in full.uncovered_importance_edges
    assert "trajectory-invalid:1" not in flow_only.uncovered_edge_ids
