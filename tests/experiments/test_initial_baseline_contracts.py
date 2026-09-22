"""Contract tests for the exact formal pre-training baseline route."""

from __future__ import annotations

import json
from dataclasses import fields, replace
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
    InitialBaselineEvaluationDeployment,
)
from skillev_private.initial_baseline import (
    InitialBaselineEvaluationInput,
    InitialBaselineInferenceState,
)

from skillev.contracts import SuccessRule, TerminalReward, stable_hash
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments import (
    AttemptPurpose,
    Benchmark,
    BenchmarkTaskCount,
    EvaluationReportingProtocol,
    FormalRunLedger,
    FormalTrainingBinding,
    FrozenTaskSequenceIdentity,
    PublishedAttemptIdentity,
    SchedulePurpose,
    arm_protocol_for_builder_kind,
)
from skillev.experiments.protocol import FREEZE_FORMAT
from skillev.policy import PrivateInitialCheckpointBinding, TrainableStateIdentity
from skillev.rollout import RolloutTermination
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    BudgetVector,
    ExactAttemptRunPlan,
    PublishedSuccessfulAttemptBundle,
    SkillLibraryState,
)
from tests.experiments.formal_evaluation_helpers import (
    make_formal_evaluation_launch,
    make_successful_evaluation_terminal,
)
from tests.experiments.formal_execution_helpers import (
    make_formal_application,
    make_formal_artifact_resolver,
    make_formal_artifacts,
    make_formal_execution,
    make_formal_run_ledger_for_identity,
    publish_formal_success_for_identity,
)
from tests.v3_helpers import make_skill_document


def _formal_source(
    tmp_path: Path,
) -> tuple[
    PrivateInitialCheckpointBinding,
    tuple,
    PublishedAttemptIdentity,
    FormalRunLedger,
    PublishedSuccessfulAttemptBundle,
]:
    seed_documents = (make_skill_document("baseline-seed"),)
    application = make_formal_application()
    plan = ExactAttemptRunPlan(phase_search_steps=1, closure_steps=1, maximum_cycles=1)
    training_sequence = FrozenTaskSequenceIdentity(
        purpose=SchedulePurpose.IID_TRAINING,
        ordered_task_ids_hash=stable_hash("baseline-training-order"),
        task_count=plan.total_training_steps,
        benchmark_counts=(
            BenchmarkTaskCount(benchmark=Benchmark.HOTPOT_QA, count=plan.total_training_steps),
        ),
        schedule_algorithm="initial-baseline-test@1",
    )
    deployment_hash = stable_hash("baseline-deployment")
    initial_state = TrainableStateIdentity.create(
        backbone_deployment_hash=deployment_hash,
        forward_adapter_hash=stable_hash("baseline-forward"),
        backward_adapter_hash=stable_hash("baseline-backward"),
        z_head_hash=stable_hash("baseline-z"),
    )
    authority = SkillAuthoringAuthority(
        input_schema_id="baseline-input@1",
        output_schema_id="baseline-output@1",
        license_id="unit-test",
        allowed_task_families=("debug/task-family",),
        allowed_tools=(),
    )
    attempt_budget = BudgetVector(input_tokens=100, output_tokens=100, model_calls=100)
    phi_budget = BudgetVector(input_tokens=4, output_tokens=4, model_calls=1)
    catalog_hash = stable_hash("baseline-catalog")
    formal_execution = make_formal_execution(
        application=application,
        run_plan=plan,
        training_sequence=training_sequence,
        backbone_deployment_hash=deployment_hash,
        initial_trainable_state=initial_state,
        initial_library_version=SkillLibraryState.from_seed_documents(
            seed_documents
        ).current_version,
        initial_skill_library_state_hash=SkillLibraryState.from_seed_documents(
            seed_documents
        ).state_hash,
        authoring_authority=authority,
        attempt_budget=attempt_budget,
        phi_per_cycle_maximum=phi_budget,
        catalog_freeze_hash=catalog_hash,
    )
    artifacts = make_formal_artifacts()
    protocol_hash = stable_hash("baseline-protocol")
    freeze_id = stable_hash(
        {
            "format": FREEZE_FORMAT,
            "protocol_hash": protocol_hash,
            "result_ingestion_count": 0,
        }
    )
    identity = PublishedAttemptIdentity(
        builder_kind=AttemptBuilderKind.FULL,
        purpose=AttemptPurpose.FORMAL_BENCHMARK_TRAINING,
        arm_protocol=arm_protocol_for_builder_kind(AttemptBuilderKind.FULL),
        application_config=application,
        run_plan=plan,
        initial_optimizer_step=0,
        initial_run_cursor=AttemptRunCursorState.fresh(plan),
        authoring_authority=authority,
        attempt_budget=attempt_budget,
        phi_per_cycle_maximum=phi_budget,
        protocol_hash=protocol_hash,
        protocol_freeze_id=freeze_id,
        exact_input_sha256=stable_hash("baseline-exact-input"),
        backbone_deployment_hash=deployment_hash,
        initial_trainable_state=initial_state,
        tokenizer_identity=artifacts.public_tokenizer,
        initial_library_version=SkillLibraryState.from_seed_documents(
            seed_documents
        ).current_version,
        initial_skill_library_state_hash=SkillLibraryState.from_seed_documents(
            seed_documents
        ).state_hash,
        ordered_task_sequence_hash=training_sequence.ordered_task_ids_hash,
        formal_training_binding=FormalTrainingBinding(
            training_sequence=training_sequence,
            catalog_freeze_hash=catalog_hash,
        ),
        formal_execution=formal_execution,
        base_model_artifact=artifacts.base_model,
        tokenizer_artifact=artifacts.tokenizer,
        implementation_build=artifacts.implementation,
    )
    checkpoint_directory = tmp_path / "initial-checkpoint"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "policy_state.json").write_text(
        json.dumps(
            {
                "backbone_id": deployment_hash,
                "backward_version": "backward@0",
                "format": "skillev-policy-state@1",
                "forward_version": "forward@0",
                "optimizer_step": 0,
                "trainable_state": initial_state.to_value(),
                "z_version": "z@0",
            }
        ),
        encoding="utf-8",
    )
    checkpoint = PrivateInitialCheckpointBinding(
        directory=str(checkpoint_directory),
        trainable_state=initial_state,
    )
    ledger = make_formal_run_ledger_for_identity(tmp_path, identity)
    source_bundle = publish_formal_success_for_identity(tmp_path, identity, ledger)
    return (
        checkpoint,
        seed_documents,
        identity,
        ledger,
        source_bundle,
    )


def _evaluation_sequence() -> PrivateFrozenTaskSequence:
    task_ids = ("baseline-evaluation-task",)
    return PrivateFrozenTaskSequence(
        identity=FrozenTaskSequenceIdentity(
            purpose=SchedulePurpose.IID_EVALUATION,
            ordered_task_ids_hash=stable_hash(list(task_ids)),
            task_count=1,
            benchmark_counts=(BenchmarkTaskCount(benchmark=Benchmark.HOTPOT_QA, count=1),),
            schedule_algorithm="initial-baseline-test@1",
        ),
        task_ids=task_ids,
    )


def test_initial_baseline_is_the_exact_preregistered_formal_initial_state(
    tmp_path: Path,
) -> None:
    checkpoint, seed_documents, identity, ledger, source_bundle = _formal_source(tmp_path)

    state = InitialBaselineInferenceState.from_formal_input(
        initial_checkpoint=checkpoint,
        seed_documents=seed_documents,
        source_identity=identity,
        source_bundle=source_bundle,
        formal_run_ledger=ledger,
    )

    assert state.calibration_cells == ()
    assert state.optimizer_step == 0
    assert state.source_training_attempt_id == "formal-full"
    assert InitialBaselineInferenceState.from_value(state.to_value()) == state
    with pytest.raises(TypeError):
        InitialBaselineInferenceState(
            policy_snapshot_directory=state.policy_snapshot_directory,
            policy_snapshot_hash=state.policy_snapshot_hash,
            policy_snapshot=state.policy_snapshot,
            initial_trainable_state=state.initial_trainable_state,
            library=state.library,
            calibration_cells=(),
            method_identity_hash=state.method_identity_hash,
            source_attempt_id=state.source_attempt_id,
            source_formal_run_group_id=state.source_formal_run_group_id,
            source_identity_hash=state.source_identity_hash,
            source_exact_input_sha256=state.source_exact_input_sha256,
            _admission=object(),
        )
    state.require_formal_initial(
        initial_checkpoint=checkpoint,
        seed_documents=seed_documents,
        source_identity=identity,
        source_bundle=source_bundle,
        formal_run_ledger=ledger,
    )

    wrong_ledger = make_formal_run_ledger_for_identity(tmp_path / "wrong-ledger", identity)
    with pytest.raises(ValueError):
        InitialBaselineInferenceState.from_formal_input(
            initial_checkpoint=checkpoint,
            seed_documents=seed_documents,
            source_identity=identity,
            source_bundle=source_bundle,
            formal_run_ledger=wrong_ledger,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "2"),
        ("input_schema_id", "alternate-input@1"),
        ("output_schema_id", "alternate-output@1"),
        ("license_id", "alternate-license"),
        ("provenance_hash", stable_hash({"provenance": "alternate"})),
    ],
)
def test_formal_initial_baseline_rejects_each_seed_manifest_wire_mutation(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    checkpoint, seed_documents, identity, ledger, source_bundle = _formal_source(tmp_path)
    seed = seed_documents[0]
    mutated = replace(seed, manifest=replace(seed.manifest, **{field: value}))

    with pytest.raises(ValueError):
        InitialBaselineInferenceState.from_formal_input(
            initial_checkpoint=checkpoint,
            seed_documents=(mutated,),
            source_identity=identity,
            source_bundle=source_bundle,
            formal_run_ledger=ledger,
        )


def test_initial_baseline_disallows_posterior_or_progress_schedule(tmp_path: Path) -> None:
    checkpoint, seed_documents, identity, ledger, source_bundle = _formal_source(tmp_path)
    state = InitialBaselineInferenceState.from_formal_input(
        initial_checkpoint=checkpoint,
        seed_documents=seed_documents,
        source_identity=identity,
        source_bundle=source_bundle,
        formal_run_ledger=ledger,
    )
    value = state.to_value()
    value["calibration_cells"] = [{}]
    with pytest.raises(ValueError):
        InitialBaselineInferenceState.from_value(value)

    progress = PrivateFrozenTaskSequence(
        identity=FrozenTaskSequenceIdentity(
            purpose=SchedulePurpose.IID_PROGRESS,
            ordered_task_ids_hash=stable_hash(["baseline-progress-task"]),
            task_count=1,
            benchmark_counts=(BenchmarkTaskCount(benchmark=Benchmark.AIME_2026, count=1),),
            schedule_algorithm="initial-baseline-test@1",
        ),
        task_ids=("baseline-progress-task",),
    )
    with pytest.raises(ValueError):
        InitialBaselineEvaluationInput(state=state, task_sequence=progress)


def test_initial_baseline_aggregates_without_a_completed_training_artifact(tmp_path: Path) -> None:
    checkpoint, seed_documents, identity, ledger, source_bundle = _formal_source(tmp_path)
    state = InitialBaselineInferenceState.from_formal_input(
        initial_checkpoint=checkpoint,
        seed_documents=seed_documents,
        source_identity=identity,
        source_bundle=source_bundle,
        formal_run_ledger=ledger,
    )
    sequence = _evaluation_sequence()
    episode = PrivateEvaluationEpisodeResult(
        task_id=sequence.task_ids[0],
        benchmark=Benchmark.HOTPOT_QA,
        reward=TerminalReward(
            value=1.0,
            success=True,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="token-f1",
            native_payload={
                "private": "not exported",
                "public_metrics": {"exact-match": 1.0},
            },
            environment_id="private-evaluation-test",
            verifier_version="private-evaluation-test@1",
        ),
        native_metrics=(
            NativeMetricValue(metric_name="exact-match", value=1.0),
            NativeMetricValue(metric_name="token-f1", value=1.0),
        ),
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
        frozen_state_hash=state.content_hash,
        task_sequence_hash=sequence.identity.ordered_task_ids_hash,
        sequence_position=0,
        policy_snapshot_id=state.policy_snapshot_id,
        library_version=state.library.current_version,
    )

    aggregate = aggregate_private_episodes(
        (episode,),
        benchmark=Benchmark.HOTPOT_QA,
        task_sequence_identity=sequence.identity,
        expected_task_ids=sequence.task_ids,
        expected_sequence_positions=(0,),
        state=state,
        reporting=EvaluationReportingProtocol(),
        formal_evaluation_terminal=make_successful_evaluation_terminal(
            state=state,
            task_sequence_identity=sequence.identity,
        ),
    )

    assert aggregate.source_training_attempt_id == "formal-full"
    assert aggregate.source_formal_run_group_id == state.source_formal_run_group_id
    assert aggregate.source_training_identity_hash == state.source_identity_hash
    assert aggregate.source_evolution_counts == state.evolution_counts
    assert aggregate.frozen_state_hash == state.content_hash

    missing_exact_match = replace(
        episode.reward,
        native_payload={"private": "not exported"},
    )
    with pytest.raises(ValueError):
        replace(
            episode,
            reward=missing_exact_match,
            native_metrics=(NativeMetricValue(metric_name="token-f1", value=1.0),),
        )

    extra_metric = replace(
        episode.reward,
        native_payload={
            "private": "not exported",
            "public_metrics": {"exact-match": 1.0, "unregistered": 0.5},
        },
    )
    with pytest.raises(ValueError):
        replace(
            episode,
            reward=extra_metric,
            native_metrics=(
                NativeMetricValue(metric_name="exact-match", value=1.0),
                NativeMetricValue(metric_name="token-f1", value=1.0),
                NativeMetricValue(metric_name="unregistered", value=0.5),
            ),
        )


def test_initial_baseline_worker_requires_terminalized_source_bundle() -> None:
    deployment = InitialBaselineEvaluationDeployment(
        source_training_input_path=Path("/private/input/training.json"),
        published_training_bundle_directory=Path("/private/published/training-attempt"),
        formal_run_ledger_directory=Path("/private/formal-ledger"),
        output_directory=Path("/private/evaluation-output"),
        run_id="initial-baseline-evaluation-run",
        formal_artifact_resolver=make_formal_artifact_resolver(),
        formal_evaluation_launch=make_formal_evaluation_launch(),
    )

    assert deployment.published_training_bundle_directory.is_absolute()
    assert "formal_run_ledger_directory" in {
        field.name for field in fields(InitialBaselineEvaluationDeployment)
    }
    with pytest.raises(ValueError):
        InitialBaselineEvaluationDeployment(
            source_training_input_path=Path("relative-input.json"),
            published_training_bundle_directory=Path("/private/published/training-attempt"),
            formal_run_ledger_directory=Path("/private/formal-ledger"),
            output_directory=Path("/private/evaluation-output"),
            run_id="initial-baseline-evaluation-run",
            formal_artifact_resolver=make_formal_artifact_resolver(),
            formal_evaluation_launch=make_formal_evaluation_launch(),
        )
    with pytest.raises(TypeError):
        InitialBaselineEvaluationDeployment(
            source_training_input_path=Path("/private/input/training.json"),
            published_training_bundle_directory=Path("/private/published/training-attempt"),
            formal_run_ledger_directory=Path("/private/formal-ledger"),
            output_directory=Path("/private/evaluation-output"),
            run_id="initial-baseline-evaluation-run",
            formal_artifact_resolver=object(),  # type: ignore[arg-type]
            formal_evaluation_launch=make_formal_evaluation_launch(),
        )
