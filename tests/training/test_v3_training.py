from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from skillev.contracts import RunCursorValue, stable_hash
from skillev.evolution import AwaitingDetectorSegment
from skillev.policy import AdapterRole, QwenPolicyBackbone
from skillev.rollout import RolloutInfrastructureError
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    EventEnvelope,
    EventType,
    ExactAttemptRunPlan,
    FullRuntimeExecutionState,
    RuntimeEventEmitter,
    RuntimeSnapshotIdentity,
)
from skillev.runtime.event_log_reader import read_event_history
from skillev.scoring import (
    ScoringConfig,
    materialize_edge_records,
    materialize_residual,
    score_trajectory,
)
from skillev.training import (
    AppliedTrainingState,
    CollectedTrainingBatch,
    IdleTrainingState,
    OptimizerConfig,
    TrainerConfig,
    TrainingCheckpointSnapshot,
    TrainingStepExecutionContext,
)
from skillev.training.distributed_ttb import (
    DistributedTTBTopology,
    partition_training_batch,
    should_inject_distributed_ttb_oom,
    training_artifact_token_cost,
)
from skillev.training.step_math import (
    compute_ttb_gradient_shard,
    create_ttb_optimizer,
    merge_ttb_gradient_shards,
    prepare_ttb_step,
)
from tests.training.fakes import TrainingHarness


def _trainable_values(backbone: QwenPolicyBackbone) -> tuple[torch.Tensor, ...]:
    groups = backbone.parameter_groups()
    return tuple(
        parameter.detach().clone()
        for parameter in (*groups.forward, *groups.backward, *groups.z_head)
    )


def _copy_trainables(source: QwenPolicyBackbone, target: QwenPolicyBackbone) -> None:
    with torch.no_grad():
        for source_parameter, target_parameter in zip(
            _all_trainables(source),
            _all_trainables(target),
            strict=True,
        ):
            target_parameter.copy_(source_parameter)


def _all_trainables(backbone: QwenPolicyBackbone) -> tuple[torch.nn.Parameter, ...]:
    groups = backbone.parameter_groups()
    return (*groups.forward, *groups.backward, *groups.z_head)


def _step_context(step: int) -> TrainingStepExecutionContext:
    return TrainingStepExecutionContext(
        RunCursorValue(
            run_plan_hash=stable_hash({"fixture": "v3-training-plan"}),
            completed_training_steps=step,
            committed_cycles=0,
            committed_actions=0,
        )
    )


async def _run_steps(harness: TrainingHarness, count: int):
    return tuple([await harness.loop.run_one(_step_context(step)) for step in range(1, count + 1)])


def _checkpoint_identity_and_cursor(
    harness: TrainingHarness,
) -> tuple[RuntimeSnapshotIdentity, AttemptRunCursorState]:
    """Construct the immutable execution identity required by snapshot v6."""

    plan = ExactAttemptRunPlan(
        phase_search_steps=2,
        closure_steps=1,
        maximum_cycles=1,
    )
    cursor = AttemptRunCursorState.fresh(plan)
    for _ in range(harness.loop.optimizer_step):
        cursor = cursor.after_training_step(plan)
    return (
        RuntimeSnapshotIdentity(
            builder_kind=AttemptBuilderKind.FULL,
            public_identity_content_hash=stable_hash({"public": "v3-training"}),
            application_config_hash=stable_hash({"config": "v3-training"}),
            method_identity_hash=stable_hash({"method": "v3-training"}),
            protocol_hash=stable_hash({"protocol": "v3-training"}),
            protocol_freeze_id=stable_hash({"freeze": "v3-training"}),
            run_plan_hash=plan.content_hash,
            initial_library_version=harness.library.current_version,
            initial_skill_library_state_hash=harness.library.state.state_hash,
            initial_trainable_state_hash=harness.backbone.initial_trainable_state_hash,
            ordered_task_sequence_hash=stable_hash({"tasks": "v3-training"}),
            sampling_schedule_algorithm="skillev-scientific-sampling@1",
            sampling_schedule_hash=stable_hash({"sampling": "v3-training"}),
        ),
        cursor,
    )


def test_full_config_has_zero_decay_and_no_gradient_clip(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness()
    value = harness.config.to_value()

    assert TrainerConfig.from_value(value) == harness.config
    assert harness.config.optimizer.weight_decay == 0.0
    assert "gradient_clip_norm" not in harness.config.optimizer.to_value()
    with pytest.raises(ValueError):
        OptimizerConfig(
            adapter_learning_rate=1e-3,
            z_learning_rate=1e-3,
            weight_decay=0.01,
        )


def test_run_is_the_real_collect_then_train_entrypoint_for_two_iterations(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness()
    before = _trainable_values(harness.backbone)

    reports = asyncio.run(_run_steps(harness, 2))

    assert tuple(report.optimizer_step for report in reports) == (1, 2)
    assert isinstance(harness.loop.state, IdleTrainingState)
    assert harness.task_provider.cursor == 4
    assert len(harness.generator.calls) == 8
    assert all(
        not torch.equal(left, right)
        for left, right in zip(
            before,
            _trainable_values(harness.backbone),
            strict=True,
        )
    )
    source_events = tuple(
        event.event_type
        for event in read_event_history(harness.event_log.path)
        if event.event_type is EventType.TRAINING_STEP_COMMITTED
    )
    assert source_events == (
        EventType.TRAINING_STEP_COMMITTED,
        EventType.TRAINING_STEP_COMMITTED,
    )
    assert (
        harness.backbone.adapter_version(AdapterRole.FORWARD_POLICY)
        == reports[-1].forward_adapter_version
    )


def test_streaming_batch_mean_gradients_and_adam_step_match_monolithic_reference(
    make_training_harness,
    make_training_backbone,
) -> None:
    harness: TrainingHarness = make_training_harness(raw_rewards=(0.0, 0.8))
    batch = asyncio.run(harness.loop.collect_batch())
    performance = harness.loop.last_rollout_performance
    assert performance is not None
    assert performance.batch_size == len(batch.artifacts)
    assert performance.model_calls == 2 * len(batch.artifacts)
    assert performance.prompt_tokens > 0
    assert performance.completion_tokens > 0
    assert performance.model.high_water_mark == 1
    assert performance.session_setup.calls == len(batch.artifacts)
    assert performance.session_cleanup.calls == len(batch.artifacts)
    assert "query" not in performance.to_value()
    reference = make_training_backbone()
    _copy_trainables(harness.backbone, reference)
    actual_optimizer, actual_groups = create_ttb_optimizer(
        harness.backbone,
        harness.config.optimizer,
    )
    reference_optimizer, _ = create_ttb_optimizer(reference, harness.config.optimizer)

    reference_optimizer.zero_grad(set_to_none=True)
    expected_residuals = []
    expected_edges = []
    expected_losses = []
    for artifact in batch.artifacts:
        score = score_trajectory(
            reference,
            artifact.record,
            artifact.initial_context.text,
            ScoringConfig(temperature_beta=harness.config.method.temperature_beta),
        )
        (score.loss / len(batch.artifacts)).backward()
        expected_losses.append(float(score.loss.detach()))
        expected_residuals.append(
            materialize_residual(score, raw_reward=artifact.record.reward.value)
        )
        expected_edges.extend(
            materialize_edge_records(
                score,
                forward_adapter_version=reference.adapter_version(AdapterRole.FORWARD_POLICY),
                backward_adapter_version=reference.adapter_version(AdapterRole.BACKWARD_POLICY),
                batch_id=batch.batch_id,
                policy_snapshot_id=batch.policy_snapshot_id,
                library_version=batch.library_version,
                action_token_ids=tuple(step.action_token_ids for step in artifact.record.steps),
                action_token_counts=tuple(
                    step.action_token_count for step in artifact.record.steps
                ),
            )
        )

    prepared = prepare_ttb_step(
        backbone=harness.backbone,
        optimizer=actual_optimizer,
        parameters=actual_groups,
        batch=batch,
        snapshot_before=harness.generator.snapshot(),
        temperature_beta=harness.config.method.temperature_beta,
        clock=lambda: "2026-08-10T00:00:00Z",
    )

    assert prepared.residuals == tuple(expected_residuals)
    assert prepared.edges == tuple(expected_edges)
    assert prepared.detached_losses == pytest.approx(expected_losses, rel=2e-5, abs=2e-5)
    for actual_parameter, expected_parameter in zip(
        _all_trainables(harness.backbone),
        _all_trainables(reference),
        strict=True,
    ):
        assert actual_parameter.grad is not None
        assert expected_parameter.grad is not None
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=2e-5,
            atol=2e-6,
        )

    actual_optimizer.step()
    reference_optimizer.step()
    for actual_parameter, expected_parameter in zip(
        _all_trainables(harness.backbone),
        _all_trainables(reference),
        strict=True,
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=2e-5, atol=2e-6)


def test_disjoint_gradient_shards_match_one_worker_reference(
    make_training_harness,
    make_training_backbone,
) -> None:
    harness: TrainingHarness = make_training_harness(raw_rewards=(0.0, 0.8))
    batch = asyncio.run(harness.loop.collect_batch())
    reference = make_training_backbone()
    coordinator = make_training_backbone()
    workers = (make_training_backbone(), make_training_backbone())
    for replica in (reference, coordinator, *workers):
        _copy_trainables(harness.backbone, replica)
    _, reference_groups = create_ttb_optimizer(reference, harness.config.optimizer)
    _, coordinator_groups = create_ttb_optimizer(coordinator, harness.config.optimizer)
    worker_groups = tuple(
        create_ttb_optimizer(worker, harness.config.optimizer)[1] for worker in workers
    )
    kwargs = {
        "batch": batch,
        "global_batch_size": len(batch.artifacts),
        "temperature_beta": harness.config.method.temperature_beta,
    }

    reference_shard = compute_ttb_gradient_shard(
        backbone=reference,
        parameters=reference_groups,
        positions=(0, 1),
        **kwargs,
    )
    worker_shards = tuple(
        compute_ttb_gradient_shard(
            backbone=worker,
            parameters=groups,
            positions=(position,),
            **kwargs,
        )
        for position, (worker, groups) in enumerate(zip(workers, worker_groups, strict=True))
    )
    expected = merge_ttb_gradient_shards(
        parameters=reference_groups,
        batch=batch,
        snapshot_before=harness.generator.snapshot(),
        shards=(reference_shard,),
        clock=lambda: "2026-08-10T00:00:01Z",
        started_at="2026-08-10T00:00:00Z",
    )
    actual = merge_ttb_gradient_shards(
        parameters=coordinator_groups,
        batch=batch,
        snapshot_before=harness.generator.snapshot(),
        shards=worker_shards,
        clock=lambda: "2026-08-10T00:00:01Z",
        started_at="2026-08-10T00:00:00Z",
    )

    from skillev.training.projections import TrainingStepSource

    transitions = tuple(
        harness.projections.preview(
            TrainingStepSource(
                batch.batch_id,
                batch.optimizer_step,
                batch.artifacts,
                result.stats,
                result.edges,
            )
        )
        for result in (expected, actual)
    )
    assert transitions[0].diagnostic == transitions[1].diagnostic
    assert transitions[0].posterior_batch == transitions[1].posterior_batch
    assert actual.residuals == expected.residuals
    assert actual.edges == expected.edges
    assert actual.detached_losses == pytest.approx(expected.detached_losses)
    for actual_parameter, expected_parameter in zip(
        _all_trainables(coordinator),
        _all_trainables(reference),
        strict=True,
    ):
        assert actual_parameter.grad is not None
        assert expected_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_collected_batch_round_trips_without_retokenizing_actions(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness(raw_rewards=(0.0, 0.8))
    batch = asyncio.run(harness.loop.collect_batch())

    restored = CollectedTrainingBatch.from_value(
        batch.to_value(),
        tokenizer=harness.backbone.tokenizer,
    )

    assert restored == batch


def test_distributed_partition_covers_batch_once_by_private_token_cost(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness(raw_rewards=(0.0, 0.8))
    batch = asyncio.run(harness.loop.collect_batch())

    partitions = partition_training_batch(batch, worker_count=2)

    assert sorted(position for partition in partitions for position in partition) == [0, 1]
    assert all(training_artifact_token_cost(batch, position) > 0 for position in (0, 1))
    assert DistributedTTBTopology(rank=0, world_size=3, local_rank=0, backend="nccl")
    with pytest.raises(ValueError):
        DistributedTTBTopology(rank=0, world_size=1, local_rank=0, backend="nccl")


def test_controlled_ttb_oom_stops_after_expanding_process_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLEV_INJECT_TTB_PRIMARY_OOM_UNTIL_EXPANSION", "1")
    steady = DistributedTTBTopology(rank=1, world_size=2, local_rank=1, backend="nccl")
    expanded = DistributedTTBTopology(rank=1, world_size=3, local_rank=1, backend="nccl")

    assert should_inject_distributed_ttb_oom(steady, "batch-1")
    assert not should_inject_distributed_ttb_oom(expanded, "batch-1")


def test_gradient_shard_merge_rejects_duplicate_or_missing_trajectory_coverage(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness(raw_rewards=(0.0, 0.8))
    batch = asyncio.run(harness.loop.collect_batch())
    _, groups = create_ttb_optimizer(harness.backbone, harness.config.optimizer)
    shard = compute_ttb_gradient_shard(
        backbone=harness.backbone,
        parameters=groups,
        batch=batch,
        positions=(0,),
        global_batch_size=len(batch.artifacts),
        temperature_beta=harness.config.method.temperature_beta,
    )

    with pytest.raises(ValueError):
        merge_ttb_gradient_shards(
            parameters=groups,
            batch=batch,
            snapshot_before=harness.generator.snapshot(),
            shards=(shard, shard),
            clock=lambda: "2026-08-10T00:00:01Z",
            started_at="2026-08-10T00:00:00Z",
        )


def test_horizon_without_submission_is_zero_reward_and_commits_one_optimizer_step(
    make_training_harness,
) -> None:
    baseline: TrainingHarness = make_training_harness()
    config = replace(
        baseline.config,
        execution=replace(baseline.config.execution, batch_size=1),
    )
    harness: TrainingHarness = make_training_harness(
        config=config,
        raw_rewards=(1.0,),
        action_text="not valid action json",
    )

    report = asyncio.run(harness.loop.run_one(_step_context(1)))

    assert report.optimizer_step == 1
    assert report.mean_reward == 0.0
    assert len(harness.generator.calls) == 2
    assert harness.generator.calls[0].phase.value == "reasoning"
    committed = tuple(
        event
        for event in read_event_history(harness.event_log.path)
        if event.event_type is EventType.TRAINING_STEP_COMMITTED
    )
    assert len(committed) == 1


def test_experiment_namespace_does_not_select_rollout_randomness(
    make_training_harness,
) -> None:
    first: TrainingHarness = make_training_harness()
    second: TrainingHarness = make_training_harness(
        config=replace(
            first.config,
            execution=replace(first.config.execution, experiment_id="another-artifact-namespace"),
        )
    )

    first_batch = asyncio.run(first.loop.collect_batch())
    second_batch = asyncio.run(second.loop.collect_batch())

    assert tuple(call.seed for call in first.generator.calls) == tuple(
        call.seed for call in second.generator.calls
    )
    assert tuple(call.input_ids for call in first.generator.calls) == tuple(
        call.input_ids for call in second.generator.calls
    )
    assert tuple(item.record.trajectory_id for item in first_batch.artifacts) != tuple(
        item.record.trajectory_id for item in second_batch.artifacts
    )


def test_fixed_population_infrastructure_failure_ends_attempt_without_replacement(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness(fail_on_generation_call=3)

    with pytest.raises(RolloutInfrastructureError):
        asyncio.run(harness.loop.collect_batch())

    assert isinstance(harness.loop.state, IdleTrainingState)
    assert harness.task_provider.cursor == 2
    assert len(harness.generator.calls) == 3
    assert harness.session_factory.cleanup_count == 2


def test_training_source_append_failure_requires_checkpoint_restart(
    make_training_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness: TrainingHarness = make_training_harness()
    asyncio.run(harness.loop.collect_batch())
    emit = RuntimeEventEmitter.emit

    def fail_training_source(
        emitter: RuntimeEventEmitter,
        event_type: EventType,
        payload: object,
    ) -> EventEnvelope:
        if event_type is EventType.TRAINING_STEP_COMMITTED:
            raise OSError("injected source append failure")
        return emit(emitter, event_type, payload)

    monkeypatch.setattr(RuntimeEventEmitter, "emit", fail_training_source)

    with pytest.raises(OSError, match="injected source append failure"):
        harness.loop.train_step(_step_context(1))

    assert isinstance(harness.loop.state, AppliedTrainingState)
    assert harness.loop.state.projection_installed is True
    with pytest.raises(RuntimeError, match="ready batch"):
        harness.loop.train_step(_step_context(1))


def test_optimizer_apply_projection_install_and_commit_are_distinct(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness()
    asyncio.run(harness.loop.collect_batch())
    diagnostics_before = harness.projections.diagnostics_state

    report = harness.loop.apply_step(_step_context(1))

    assert report.optimizer_step == 1
    assert isinstance(harness.loop.state, AppliedTrainingState)
    assert harness.loop.state.projection_installed is False
    assert harness.projections.diagnostics_state == diagnostics_before
    assert EventType.TRAINING_STEP_COMMITTED not in {
        event.event_type for event in read_event_history(harness.event_log.path)
    }

    harness.loop.install_applied_projection()

    assert isinstance(harness.loop.state, AppliedTrainingState)
    assert harness.loop.state.projection_installed is True
    assert harness.projections.diagnostics_state != diagnostics_before
    assert EventType.TRAINING_STEP_COMMITTED not in {
        event.event_type for event in read_event_history(harness.event_log.path)
    }

    finalized = harness.loop.finalize_applied_step()

    assert finalized == report
    assert isinstance(harness.loop.state, IdleTrainingState)
    assert (
        sum(
            event.event_type is EventType.TRAINING_STEP_COMMITTED
            for event in read_event_history(harness.event_log.path)
        )
        == 1
    )


def test_batch_is_single_use_and_versions_advance_after_one_update(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    before = harness.generator.snapshot()

    report = harness.loop.train_step(_step_context(1))

    after = harness.generator.snapshot()
    assert before.snapshot_id == batch.policy_snapshot_id
    assert after.snapshot_id != before.snapshot_id
    assert report.optimizer_step == 1
    with pytest.raises(RuntimeError):
        harness.loop.train_step(_step_context(2))


def test_reset_partition_is_seeded_and_only_deletes_z_optimizer_state(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness()
    asyncio.run(_run_steps(harness, 1))
    groups = harness.backbone.parameter_groups()
    optimizer_state = harness.loop.optimizer.state
    adapter_state = {
        parameter: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in optimizer_state[parameter].items()
        }
        for parameter in (*groups.forward, *groups.backward)
    }

    version = harness.loop.reset_partition(20260725)

    assert version == harness.backbone.z_version
    assert all(parameter not in optimizer_state for parameter in groups.z_head)
    assert all(parameter.grad is None for parameter in groups.z_head)
    harness.loop.reset_partition(20260725)
    assert all(parameter not in optimizer_state for parameter in groups.z_head)
    for parameter, expected in adapter_state.items():
        actual = optimizer_state[parameter]
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_checkpoint_store_uses_exact_named_path_without_locator_fallback(
    make_training_harness,
    tmp_path: Path,
) -> None:
    harness: TrainingHarness = make_training_harness()
    asyncio.run(_run_steps(harness, 1))
    identity, run_cursor = _checkpoint_identity_and_cursor(harness)
    snapshot = TrainingCheckpointSnapshot(
        optimizer_step=harness.loop.optimizer_step,
        experiment_id=harness.config.execution.experiment_id,
        identity=identity,
        backbone=harness.backbone,
        optimizer=harness.loop.optimizer,
        execution_state=FullRuntimeExecutionState(
            task_cursor=harness.task_provider.runtime_state,
            run_cursor=run_cursor,
            library=harness.library.state,
            projections=harness.projections.runtime_state(),
            detector=AwaitingDetectorSegment(harness.library.current_version),
        ),
    )
    explicit = harness.checkpoint_store.save_as(
        snapshot,
        name="phase-step-00000001",
    )

    assert harness.checkpoint_store.load_metadata(explicit).optimizer_step == 1
    assert (explicit / "COMPLETE").read_text(encoding="ascii") == "complete\n"
    with pytest.raises((FileNotFoundError, ValueError)):
        harness.checkpoint_store.load_metadata(tmp_path / "latest")

    (explicit / "COMPLETE").unlink()
    with pytest.raises(ValueError):
        harness.checkpoint_store.load_metadata(explicit)


def test_checkpoint_retention_only_removes_old_ordinary_steps(make_training_harness) -> None:
    harness: TrainingHarness = make_training_harness()
    identity, run_cursor = _checkpoint_identity_and_cursor(harness)
    snapshot = TrainingCheckpointSnapshot(
        optimizer_step=0,
        experiment_id=harness.config.execution.experiment_id,
        identity=identity,
        backbone=harness.backbone,
        optimizer=harness.loop.optimizer,
        execution_state=FullRuntimeExecutionState(
            task_cursor=harness.task_provider.runtime_state,
            run_cursor=run_cursor,
            library=harness.library.state,
            projections=harness.projections.runtime_state(),
            detector=AwaitingDetectorSegment(harness.library.current_version),
        ),
    )
    root = harness.checkpoint_store.root
    for name in ("step-00000001", "step-00000002", "step-00000003", "final-step-00000003"):
        harness.checkpoint_store.save_as(snapshot, name=name)

    removed = harness.checkpoint_store.retain_recent(keep_recent=2)

    assert tuple(path.name for path in removed) == ("step-00000001",)
    assert {path.name for path in root.iterdir()} == {
        "step-00000002",
        "step-00000003",
        "final-step-00000003",
    }


def test_checkpoint_failure_preserves_staging_for_offline_maintenance(
    make_training_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness: TrainingHarness = make_training_harness()
    identity, run_cursor = _checkpoint_identity_and_cursor(harness)
    snapshot = TrainingCheckpointSnapshot(
        optimizer_step=0,
        experiment_id=harness.config.execution.experiment_id,
        identity=identity,
        backbone=harness.backbone,
        optimizer=harness.loop.optimizer,
        execution_state=FullRuntimeExecutionState(
            task_cursor=harness.task_provider.runtime_state,
            run_cursor=run_cursor,
            library=harness.library.state,
            projections=harness.projections.runtime_state(),
            detector=AwaitingDetectorSegment(harness.library.current_version),
        ),
    )

    def fail_after_writing(
        backbone: QwenPolicyBackbone,
        directory: str,
    ) -> None:
        del backbone
        path = Path(directory)
        path.mkdir(parents=True)
        (path / "partial-marker").write_text("injected failure\n", encoding="utf-8")
        raise OSError("injected checkpoint failure")

    monkeypatch.setattr(QwenPolicyBackbone, "save_checkpoint", fail_after_writing)
    with pytest.raises(OSError):
        harness.checkpoint_store.save_as(snapshot, name="phase-step-00000000")

    root = harness.checkpoint_store.root
    staging = tuple(root.glob(".phase-step-00000000.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "policy" / "partial-marker").is_file()
    assert not (root / "phase-step-00000000").exists()
    assert not hasattr(harness.checkpoint_store, "prune")
    assert not hasattr(harness.checkpoint_store, "cleanup")


def test_optimizer_config_rejects_removed_clip_field_on_wire(
    make_training_harness,
) -> None:
    harness: TrainingHarness = make_training_harness()
    raw = dict(harness.config.optimizer.to_value())
    raw["gradient_clip_norm"] = None
    with pytest.raises(ValueError):
        OptimizerConfig.from_value(raw)
