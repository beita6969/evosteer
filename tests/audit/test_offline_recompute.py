from __future__ import annotations

from skillev.audit.calibration_recompute import (
    FullCalibrationAuditKernel,
    recompute_posterior_batches,
)
from skillev.audit.diagnostics_recompute import (
    FullDiagnosticsAuditKernel,
    recompute_diagnostics,
)
from skillev.audit.source_reducer import CommittedLibrarySegment
from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import (
    RunCursorValue,
    TrainingStepCommit,
    TrainingStepReportValue,
    stable_hash,
)
from skillev.diagnostics import ComparableResidualWindows, DiagnosticsConfig
from skillev.runtime import SkillLibraryState
from skillev.training import MethodProjectionPipeline
from tests.v3_helpers import make_artifact, make_source


def _report(step: int, batch_id: str) -> TrainingStepReportValue:
    return TrainingStepReportValue(
        optimizer_step=step,
        batch_id=batch_id,
        torch_batch_loss=1.0,
        audited_batch_loss=1.0,
        mean_reward=1.0,
        grad_norm_forward=1.0,
        grad_norm_backward=1.0,
        grad_norm_z=1.0,
        forward_adapter_version=f"forward@{step}",
        backward_adapter_version=f"backward@{step}",
        z_version=f"z@{step}",
        started_at="2026-07-25T00:00:00Z",
        completed_at="2026-07-25T00:00:01Z",
    )


def test_offline_audit_recomputes_online_diagnostics_and_posterior_exactly() -> None:
    diagnostics_config = DiagnosticsConfig(window_size=1)
    calibration_config = CalibrationConfig()
    library = SkillLibraryState.from_seed_documents(())
    projections = MethodProjectionPipeline.fresh(
        diagnostics_config=diagnostics_config,
        calibration=CalibrationEngine(calibration_config),
        library_version=library.current_version,
    )
    commits: list[TrainingStepCommit] = []
    run_plan_hash = stable_hash({"offline": "recompute"})
    for step in (1, 2):
        batch_id = f"batch-{step}"
        source = make_source(
            batch_id=batch_id,
            optimizer_step=step,
            library_version=library.current_version,
            artifacts=(
                make_artifact(
                    f"trajectory-{step}",
                    library_version=library.current_version,
                    skills_by_step=(("skill-alpha",),),
                ),
            ),
            importances=((0.25 * step,),),
            deltas=(1.0,),
        )
        transition = projections.preview(source)
        projections.commit(transition)
        report = _report(step, batch_id)
        commits.append(
            TrainingStepCommit(
                batch_id=batch_id,
                optimizer_step=step,
                policy_snapshot_before=f"policy@{step - 1}",
                policy_snapshot_after=f"policy@{step}",
                library_version=source.stats.library_version,
                records=tuple(item.record for item in source.artifacts),
                edge_records=source.edge_records,
                stats=source.stats,
                posterior_batch=transition.posterior_batch,
                report=report,
                run_cursor_after=RunCursorValue(
                    run_plan_hash=run_plan_hash,
                    completed_training_steps=step,
                    committed_cycles=0,
                    committed_actions=0,
                ),
            )
        )
    segments = (
        CommittedLibrarySegment(
            library_state=library,
            cursor_before=RunCursorValue(
                run_plan_hash=run_plan_hash,
                completed_training_steps=0,
                committed_cycles=0,
                committed_actions=0,
            ),
            cursor_after=RunCursorValue(
                run_plan_hash=run_plan_hash,
                completed_training_steps=2,
                committed_cycles=0,
                committed_actions=0,
            ),
            training_steps=tuple(commits),
            phase=None,
            cycle=None,
        ),
    )

    offline_diagnostics = recompute_diagnostics(
        segments,
        config=diagnostics_config,
        kernel=FullDiagnosticsAuditKernel(),
    )
    offline_posterior, _ = recompute_posterior_batches(
        segments,
        offline_diagnostics,
        config=calibration_config,
        kernel=FullCalibrationAuditKernel(),
    )

    assert tuple(item.to_value() for item in offline_diagnostics) == tuple(
        item.to_value() for item in projections.diagnostics_history
    )
    assert tuple(item.to_value() for item in offline_posterior) == tuple(
        item.to_value() for item in projections.calibration_cells
    )


def test_offline_equivalence_spans_a_complete_two_window_history() -> None:
    diagnostics_config = DiagnosticsConfig(window_size=2)
    calibration_config = CalibrationConfig()
    library = SkillLibraryState.from_seed_documents(())
    projections = MethodProjectionPipeline.fresh(
        diagnostics_config=diagnostics_config,
        calibration=CalibrationEngine(calibration_config),
        library_version=library.current_version,
    )
    commits: list[TrainingStepCommit] = []
    run_plan_hash = stable_hash({"offline": "full-two-window"})
    for step in range(1, 5):
        batch_id = f"full-window-batch-{step}"
        artifacts = tuple(
            make_artifact(
                f"full-window-{step}-{index}",
                library_version=library.current_version,
                skills_by_step=(("skill-alpha",), ("skill-beta",)),
                reward_success=(step + index) % 2 == 0,
            )
            for index in range(3)
        )
        source = make_source(
            batch_id=batch_id,
            optimizer_step=step,
            library_version=library.current_version,
            artifacts=artifacts,
            importances=tuple((0.1 * step, -0.05 * index) for index in range(3)),
            deltas=tuple(1.0 + 0.05 * step + 0.01 * index for index in range(3)),
        )
        transition = projections.preview(source)
        projections.commit(transition)
        commits.append(
            TrainingStepCommit(
                batch_id=batch_id,
                optimizer_step=step,
                policy_snapshot_before=f"policy@{step - 1}",
                policy_snapshot_after=f"policy@{step}",
                library_version=source.stats.library_version,
                records=tuple(item.record for item in source.artifacts),
                edge_records=source.edge_records,
                stats=source.stats,
                posterior_batch=transition.posterior_batch,
                report=_report(step, batch_id),
                run_cursor_after=RunCursorValue(
                    run_plan_hash=run_plan_hash,
                    completed_training_steps=step,
                    committed_cycles=0,
                    committed_actions=0,
                ),
            )
        )
    segments = (
        CommittedLibrarySegment(
            library_state=library,
            cursor_before=RunCursorValue(
                run_plan_hash=run_plan_hash,
                completed_training_steps=0,
                committed_cycles=0,
                committed_actions=0,
            ),
            cursor_after=RunCursorValue(
                run_plan_hash=run_plan_hash,
                completed_training_steps=4,
                committed_cycles=0,
                committed_actions=0,
            ),
            training_steps=tuple(commits),
            phase=None,
            cycle=None,
        ),
    )

    offline_diagnostics = recompute_diagnostics(
        segments,
        config=diagnostics_config,
        kernel=FullDiagnosticsAuditKernel(),
    )
    offline_posterior, _ = recompute_posterior_batches(
        segments,
        offline_diagnostics,
        config=calibration_config,
        kernel=FullCalibrationAuditKernel(),
    )

    assert len(offline_diagnostics) == 4
    assert isinstance(offline_diagnostics[-1].residual_window, ComparableResidualWindows)
    assert tuple(item.to_value() for item in offline_diagnostics) == tuple(
        item.to_value() for item in projections.diagnostics_history
    )
    assert tuple(item.to_value() for item in offline_posterior) == tuple(
        item.to_value() for item in projections.calibration_cells
    )
