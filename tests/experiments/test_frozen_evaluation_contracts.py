"""CPU-only contracts for the closed frozen-evaluation boundary."""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence
from skillev_private.evaluation.frozen_runner import (
    FrozenEvaluationRunner,
    _capture_frozen_skill_binding,
    _evaluation_sampling_coordinate,
    _require_frozen_skill_binding_unchanged,
)
from skillev_private.evaluation.result_contracts import (
    EvaluationEpisodeTelemetry,
    NativeMetricValue,
    ObservationStatusCount,
    PrivateEvaluationEpisodeResult,
    PublicBenchmarkAggregateResult,
    aggregate_private_episodes,
)
from skillev_private.experiments.frozen_evaluation_worker import (
    FrozenEvaluationDeployment,
    execute_frozen_evaluation,
)
from skillev_private.frozen_state import FrozenEvaluationInput

from skillev.contracts import SuccessRule, TerminalReward, stable_hash
from skillev.evolution import TaskConditionedSkillRetriever
from skillev.experiments import (
    Benchmark,
    BenchmarkTaskCount,
    EvaluationReportingProtocol,
    FrozenTaskSequenceIdentity,
    SchedulePurpose,
    TrainingEvolutionCounts,
)
from skillev.rollout import RolloutTask, RolloutTermination
from skillev.runtime import BudgetVector, FinalTrainingArtifact, SkillLibrary, SkillLibraryState
from tests.experiments.formal_evaluation_helpers import (
    make_formal_evaluation_launch,
    make_successful_evaluation_terminal,
)
from tests.experiments.formal_execution_helpers import make_formal_artifact_resolver
from tests.v3_helpers import TEST_CONTEXT_ID, TEST_TASK_FAMILY, make_skill_document


def _state_value(*, snapshot_directory: str) -> dict[str, object]:
    artifact = FinalTrainingArtifact(
        artifact_sha256=stable_hash("final-artifact"),
        runtime_state_sha256=stable_hash("runtime-state"),
        policy_snapshot_id="policy-final@2",
        library_version=SkillLibraryState.from_seed_documents(()).current_version,
        optimizer_step=2,
    )
    return {
        "calibration_cells": [],
        "final_training_artifact": artifact.to_value(),
        "format": "skillev-frozen-inference-state@5",
        "evolution_counts": TrainingEvolutionCounts(
            training_step_count=2,
            phase_count=1,
            cycle_count=1,
            action_count=1,
        ).to_value(),
        "library": SkillLibraryState.from_seed_documents(()).to_value(),
        "method_identity_hash": stable_hash("method"),
        "policy_snapshot_directory": snapshot_directory,
        "policy_snapshot_hash": artifact.artifact_sha256,
        "source_exact_input_sha256": stable_hash("exact-input"),
        "source_formal_run_group_id": stable_hash("formal-run-group"),
        "source_training_attempt_id": "formal-training-attempt",
        "source_training_identity_hash": stable_hash("published-training-identity"),
        "source_training_outcome_sha256": stable_hash("published-training-outcome"),
    }


def _sequence() -> PrivateFrozenTaskSequence:
    task_ids = ("progress-task",)
    return PrivateFrozenTaskSequence(
        identity=FrozenTaskSequenceIdentity(
            purpose=SchedulePurpose.IID_PROGRESS,
            ordered_task_ids_hash=stable_hash(list(task_ids)),
            task_count=1,
            benchmark_counts=(BenchmarkTaskCount(benchmark=Benchmark.AIME_2026, count=1),),
            schedule_algorithm="unit-test-frozen-order@1",
        ),
        task_ids=task_ids,
    )


def test_frozen_skill_binding_uses_one_active_library_snapshot() -> None:
    document = make_skill_document("frozen-binding")
    initial = SkillLibraryState.from_seed_documents((document,))
    library = SkillLibrary(initial)
    retriever = TaskConditionedSkillRetriever(library=library)
    task = RolloutTask(
        task_id="frozen-binding-task",
        environment_id="debug-environment",
        task_family=TEST_TASK_FAMILY,
        context_id=TEST_CONTEXT_ID,
        query="Use the active skill.",
        available_tools=(),
        public_context={},
    )

    captured, retrieved = _capture_frozen_skill_binding(
        library=library,
        retriever=retriever,
        task=task,
    )
    assert captured == initial
    assert tuple(item.metadata.skill_id for item in retrieved) == initial.active_skill_ids
    _require_frozen_skill_binding_unchanged(library=library, expected_state=captured)

    library.apply(SkillLibraryState.from_seed_documents(()))
    with pytest.raises(RuntimeError):
        _require_frozen_skill_binding_unchanged(library=library, expected_state=captured)


def test_evaluation_sampling_coordinate_excludes_state_and_output_namespaces() -> None:
    sequence = _sequence()
    coordinate = _evaluation_sampling_coordinate(
        sampling_schedule_hash=stable_hash({"sampling": "common"}),
        sequence=sequence,
        sequence_position=0,
        task_id=sequence.task_ids[0],
    )

    assert coordinate == _evaluation_sampling_coordinate(
        sampling_schedule_hash=stable_hash({"sampling": "common"}),
        sequence=sequence,
        sequence_position=0,
        task_id=sequence.task_ids[0],
    )
    assert set(coordinate.to_value()).isdisjoint(
        {"run_id", "attempt_id", "output_directory", "source_arm_identity", "state_hash"}
    )


def _input(snapshot_directory: str) -> FrozenEvaluationInput:
    return FrozenEvaluationInput.from_value(
        {
            "format": "skillev-private-frozen-evaluation-input@4",
            "state": _state_value(snapshot_directory=snapshot_directory),
            "task_sequence": _sequence().to_value(),
        }
    )


def test_frozen_state_binds_source_outcome_but_not_private_checkpoint_location() -> None:
    first = _input("/private/checkpoints/one")
    second = _input("/another-private-host/checkpoints/two")

    assert first.state.source_training_outcome_sha256 == stable_hash("published-training-outcome")
    assert first.state.content_hash == second.state.content_hash
    assert (
        first.state.to_value()["policy_snapshot_directory"]
        != second.state.to_value()["policy_snapshot_directory"]
    )

    missing_outcome = first.to_value()
    state = missing_outcome["state"]
    assert isinstance(state, dict)
    del state["source_training_outcome_sha256"]
    with pytest.raises(ValueError):
        FrozenEvaluationInput.from_value(missing_outcome)


def test_frozen_evaluation_worker_has_one_published_training_graph() -> None:
    deployment = FrozenEvaluationDeployment(
        source_training_input_path=Path("/private/input/training.json"),
        published_training_bundle_directory=Path("/private/published/training-attempt"),
        formal_run_ledger_directory=Path("/private/formal-ledger"),
        output_directory=Path("/private/evaluation-output"),
        run_id="frozen-evaluation-run",
        formal_artifact_resolver=make_formal_artifact_resolver(),
        formal_evaluation_launch=make_formal_evaluation_launch(),
    )

    assert deployment.published_training_bundle_directory.is_absolute()
    assert tuple(inspect.signature(execute_frozen_evaluation).parameters) == (
        "request",
        "deployment",
    )
    assert "execute_episode" not in {field.name for field in fields(FrozenEvaluationRunner)}
    with pytest.raises(ValueError):
        FrozenEvaluationDeployment(
            source_training_input_path=Path("relative-input.json"),
            published_training_bundle_directory=Path("/private/published/training-attempt"),
            formal_run_ledger_directory=Path("/private/formal-ledger"),
            output_directory=Path("/private/evaluation-output"),
            run_id="frozen-evaluation-run",
            formal_artifact_resolver=make_formal_artifact_resolver(),
            formal_evaluation_launch=make_formal_evaluation_launch(),
        )
    with pytest.raises(TypeError):
        FrozenEvaluationDeployment(
            source_training_input_path=Path("/private/input/training.json"),
            published_training_bundle_directory=Path("/private/published/training-attempt"),
            formal_run_ledger_directory=Path("/private/formal-ledger"),
            output_directory=Path("/private/evaluation-output"),
            run_id="frozen-evaluation-run",
            formal_artifact_resolver=object(),  # type: ignore[arg-type]
            formal_evaluation_launch=make_formal_evaluation_launch(),
        )


def test_public_aggregate_derives_provenance_from_the_admitted_frozen_state() -> None:
    frozen = _input("/private/checkpoints/final")
    sequence = frozen.task_sequence
    episode = PrivateEvaluationEpisodeResult(
        task_id="progress-task",
        benchmark=Benchmark.AIME_2026,
        reward=TerminalReward(
            value=1.0,
            success=True,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="accuracy",
            native_payload={
                "answer": "not exported",
            },
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
        frozen_state_hash=frozen.state.content_hash,
        task_sequence_hash=sequence.identity.ordered_task_ids_hash,
        sequence_position=0,
        policy_snapshot_id=frozen.state.final_training_artifact.policy_snapshot_id,
        library_version=frozen.state.library.current_version,
    )

    aggregate = aggregate_private_episodes(
        (episode,),
        benchmark=Benchmark.AIME_2026,
        task_sequence_identity=sequence.identity,
        expected_task_ids=sequence.task_ids,
        expected_sequence_positions=(0,),
        state=frozen.state,
        reporting=EvaluationReportingProtocol(),
        formal_evaluation_terminal=make_successful_evaluation_terminal(
            state=frozen.state,
            task_sequence_identity=sequence.identity,
        ),
    )

    assert aggregate.source_training_attempt_id == frozen.state.source_training_attempt_id
    assert aggregate.source_formal_run_group_id == frozen.state.source_formal_run_group_id
    assert aggregate.source_training_identity_hash == frozen.state.source_training_identity_hash
    assert aggregate.source_evolution_counts == frozen.state.evolution_counts
    assert aggregate.method_identity_hash == frozen.state.method_identity_hash
    assert aggregate.frozen_state_hash == frozen.state.content_hash
    assert tuple(item.metric_name for item in aggregate.native_metrics) == ("accuracy",)
    assert aggregate.cost.resource_usage == BudgetVector(model_calls=2, agent_turns=1)
    assert aggregate.observation_status_counts == (
        ObservationStatusCount(observation_status="success", count=1),
    )
    assert aggregate.termination_counts[0].termination is RolloutTermination.COMPLETED
    assert "not exported" not in str(aggregate.to_value())
    assert PublicBenchmarkAggregateResult.from_value(aggregate.to_value()) == aggregate
    aggregate.require_comparable_wall_time(aggregate)
    with pytest.raises(ValueError):
        aggregate.require_comparable_wall_time(
            replace(
                aggregate,
                evaluation_execution_hardware=replace(
                    aggregate.evaluation_execution_hardware,
                    accelerator_name="NVIDIA A100",
                ),
            )
        )
    with pytest.raises(ValueError):
        replace(
            aggregate,
            native_metrics=(replace(aggregate.native_metrics[0], sample_count=2),),
        )
    with pytest.raises(TypeError):
        aggregate_private_episodes(
            (episode,),
            benchmark=Benchmark.AIME_2026,
            task_sequence_identity=sequence.identity,
            expected_task_ids=sequence.task_ids,
            expected_sequence_positions=(0,),
            state=object(),
            reporting=EvaluationReportingProtocol(),
            formal_evaluation_terminal=make_successful_evaluation_terminal(
                state=frozen.state,
                task_sequence_identity=sequence.identity,
            ),
        )
