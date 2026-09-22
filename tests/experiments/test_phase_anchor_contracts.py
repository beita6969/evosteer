"""CPU-only contracts for phase-bound IID progress anchor admission."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence
from skillev_private.evaluation.result_contracts import (
    EvaluationEpisodeTelemetry,
    NativeMetricValue,
    ObservationStatusCount,
    PrivateEvaluationEpisodeResult,
    aggregate_private_episodes,
)
from skillev_private.experiments.frozen_evaluation_worker import (
    PhaseAnchorEvaluationDeployment,
)
from skillev_private.phase_anchor import PhaseAnchorEvaluationInput, PhaseAnchorInferenceState

from skillev.contracts import (
    PhaseCheckpointArtifact,
    RunCursorValue,
    SuccessRule,
    TerminalReward,
    stable_hash,
)
from skillev.experiments import (
    Benchmark,
    BenchmarkTaskCount,
    EvaluationReportingProtocol,
    FrozenTaskSequenceIdentity,
    SchedulePurpose,
    TrainingEvolutionCounts,
)
from skillev.rollout import RolloutTermination
from skillev.runtime import BudgetVector, SkillLibraryState
from tests.experiments.formal_evaluation_helpers import (
    make_formal_evaluation_launch,
    make_successful_evaluation_terminal,
)
from tests.experiments.formal_execution_helpers import make_formal_artifact_resolver


def _state_value(*, snapshot_directory: str) -> dict[str, object]:
    library = SkillLibraryState.from_seed_documents(())
    artifact = PhaseCheckpointArtifact(
        artifact_sha256=stable_hash("phase-checkpoint-artifact"),
        runtime_state_sha256=stable_hash("phase-runtime-state"),
        policy_snapshot_id="policy-phase@4",
        library_version=library.current_version,
        optimizer_step=4,
        phase_event_id="phase-event-1",
        run_cursor_after=RunCursorValue(
            run_plan_hash=stable_hash("phase-anchor-plan"),
            completed_training_steps=4,
            committed_cycles=1,
            committed_actions=1,
        ),
    )
    return {
        "calibration_cells": [],
        "format": "skillev-phase-anchor-inference-state@3",
        "evolution_counts": TrainingEvolutionCounts(
            training_step_count=4,
            phase_count=1,
            cycle_count=1,
            action_count=1,
        ).to_value(),
        "library": library.to_value(),
        "method_identity_hash": stable_hash("phase-anchor-method"),
        "phase_checkpoint_artifact": artifact.to_value(),
        "policy_snapshot_directory": snapshot_directory,
        "policy_snapshot_hash": artifact.artifact_sha256,
        "source_exact_input_sha256": stable_hash("phase-anchor-exact-input"),
        "source_formal_run_group_id": stable_hash("phase-anchor-run-group"),
        "source_phase_event_id": artifact.phase_event_id,
        "source_training_attempt_id": "formal-training-attempt",
        "source_training_identity_hash": stable_hash("phase-anchor-identity"),
        "source_training_outcome_sha256": stable_hash("phase-anchor-outcome"),
    }


def _progress_sequence() -> PrivateFrozenTaskSequence:
    task_ids = ("phase-progress-task",)
    return PrivateFrozenTaskSequence(
        identity=FrozenTaskSequenceIdentity(
            purpose=SchedulePurpose.IID_PROGRESS,
            ordered_task_ids_hash=stable_hash(list(task_ids)),
            task_count=1,
            benchmark_counts=(BenchmarkTaskCount(benchmark=Benchmark.AIME_2026, count=1),),
            schedule_algorithm="phase-anchor-test@1",
        ),
        task_ids=task_ids,
    )


def test_phase_anchor_is_path_independent_and_progress_only() -> None:
    first = PhaseAnchorEvaluationInput.from_value(
        {
            "format": "skillev-private-phase-anchor-input@2",
            "state": _state_value(snapshot_directory="/private/phases/one"),
            "task_sequence": _progress_sequence().to_value(),
        }
    )
    second = PhaseAnchorEvaluationInput.from_value(
        {
            "format": "skillev-private-phase-anchor-input@2",
            "state": _state_value(snapshot_directory="/other/private/phases/two"),
            "task_sequence": _progress_sequence().to_value(),
        }
    )

    assert first.state.content_hash == second.state.content_hash
    assert first.state.optimizer_step == 4
    assert first.state.policy_snapshot_id == "policy-phase@4"
    assert first.to_value()["state"] != second.to_value()["state"]

    iid_evaluation = PrivateFrozenTaskSequence(
        identity=FrozenTaskSequenceIdentity(
            purpose=SchedulePurpose.IID_EVALUATION,
            ordered_task_ids_hash=stable_hash(["wrong-purpose-task"]),
            task_count=1,
            benchmark_counts=(BenchmarkTaskCount(benchmark=Benchmark.AIME_2026, count=1),),
            schedule_algorithm="phase-anchor-test@1",
        ),
        task_ids=("wrong-purpose-task",),
    )
    with pytest.raises(ValueError):
        PhaseAnchorEvaluationInput(state=first.state, task_sequence=iid_evaluation)

    with pytest.raises(TypeError):
        PhaseAnchorInferenceState(
            policy_snapshot_directory=first.state.policy_snapshot_directory,
            policy_snapshot_hash=first.state.policy_snapshot_hash,
            phase_checkpoint_artifact=first.state.phase_checkpoint_artifact,
            library=first.state.library,
            calibration_cells=first.state.calibration_cells,
            method_identity_hash=first.state.method_identity_hash,
            source_training_attempt_id=first.state.source_training_attempt_id,
            source_formal_run_group_id=first.state.source_formal_run_group_id,
            source_training_identity_hash=first.state.source_training_identity_hash,
            source_exact_input_sha256=first.state.source_exact_input_sha256,
            source_training_outcome_sha256=first.state.source_training_outcome_sha256,
            source_phase_event_id=first.state.source_phase_event_id,
            evolution_counts=first.state.evolution_counts,
            _admission=object(),
        )


def test_phase_anchor_aggregate_uses_admitted_state() -> None:
    request = PhaseAnchorEvaluationInput.from_value(
        {
            "format": "skillev-private-phase-anchor-input@2",
            "state": _state_value(snapshot_directory="/private/phases/anchor"),
            "task_sequence": _progress_sequence().to_value(),
        }
    )
    episode = PrivateEvaluationEpisodeResult(
        task_id="phase-progress-task",
        benchmark=Benchmark.AIME_2026,
        reward=TerminalReward(
            value=1.0,
            success=True,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="accuracy",
            native_payload={"private": "not exported"},
            environment_id="private-evaluation-test",
            verifier_version="private-evaluation-test@1",
        ),
        native_metrics=(NativeMetricValue(metric_name="accuracy", value=1.0),),
        telemetry=EvaluationEpisodeTelemetry(
            resource_usage=BudgetVector(model_calls=2, agent_turns=1),
            action_count=1,
            action_token_count=3,
            reasoning_token_count=2,
            elapsed_wall_time_milliseconds=7,
            observation_status_counts=(
                ObservationStatusCount(observation_status="success", count=1),
            ),
            termination=RolloutTermination.COMPLETED,
        ),
        frozen_state_hash=request.state.content_hash,
        task_sequence_hash=request.task_sequence.identity.ordered_task_ids_hash,
        sequence_position=0,
        policy_snapshot_id=request.state.policy_snapshot_id,
        library_version=request.state.library.current_version,
    )

    aggregate = aggregate_private_episodes(
        (episode,),
        benchmark=Benchmark.AIME_2026,
        task_sequence_identity=request.task_sequence.identity,
        expected_task_ids=request.task_sequence.task_ids,
        expected_sequence_positions=(0,),
        state=request.state,
        reporting=EvaluationReportingProtocol(),
        formal_evaluation_terminal=make_successful_evaluation_terminal(
            state=request.state,
            task_sequence_identity=request.task_sequence.identity,
        ),
    )

    assert aggregate.source_training_attempt_id == request.state.source_training_attempt_id
    assert aggregate.source_formal_run_group_id == request.state.source_formal_run_group_id
    assert aggregate.source_training_identity_hash == request.state.source_training_identity_hash
    assert aggregate.source_evolution_counts == request.state.evolution_counts
    assert aggregate.frozen_state_hash == request.state.content_hash


def test_phase_anchor_worker_requires_one_explicit_checkpoint_directory() -> None:
    deployment = PhaseAnchorEvaluationDeployment(
        source_training_input_path=Path("/private/input/training.json"),
        published_training_bundle_directory=Path("/private/published/training-attempt"),
        formal_run_ledger_directory=Path("/private/formal-ledger"),
        phase_checkpoint_directory=Path("/private/checkpoints/phase-00000001"),
        output_directory=Path("/private/evaluation-output"),
        run_id="phase-anchor-evaluation-run",
        formal_artifact_resolver=make_formal_artifact_resolver(),
        formal_evaluation_launch=make_formal_evaluation_launch(),
    )

    assert deployment.phase_checkpoint_directory.is_absolute()
    assert "phase_checkpoint_directory" in {
        field.name for field in fields(PhaseAnchorEvaluationDeployment)
    }
    with pytest.raises(ValueError):
        PhaseAnchorEvaluationDeployment(
            source_training_input_path=Path("private/input/training.json"),
            published_training_bundle_directory=Path("/private/published/training-attempt"),
            formal_run_ledger_directory=Path("/private/formal-ledger"),
            phase_checkpoint_directory=Path("/private/checkpoints/phase-00000001"),
            output_directory=Path("/private/evaluation-output"),
            run_id="phase-anchor-evaluation-run",
            formal_artifact_resolver=make_formal_artifact_resolver(),
            formal_evaluation_launch=make_formal_evaluation_launch(),
        )
    with pytest.raises(TypeError):
        PhaseAnchorEvaluationDeployment(
            source_training_input_path=Path("/private/input/training.json"),
            published_training_bundle_directory=Path("/private/published/training-attempt"),
            formal_run_ledger_directory=Path("/private/formal-ledger"),
            phase_checkpoint_directory=Path("/private/checkpoints/phase-00000001"),
            output_directory=Path("/private/evaluation-output"),
            run_id="phase-anchor-evaluation-run",
            formal_artifact_resolver=object(),  # type: ignore[arg-type]
            formal_evaluation_launch=make_formal_evaluation_launch(),
        )
