from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
import torch

from skillev.training.planning import PlannedRollout, TrainingBatchPlan
from skillev.training.step_math import (
    compute_ttb_gradient_shard,
    create_ttb_optimizer,
    named_ttb_parameters,
)
from skillev.training.streaming_step import LocalGradientStepStream


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("buffered_contributions", [1, 3])
def test_local_ready_first_fixed_reduction_and_failed_batch_discard(
    make_training_harness, monkeypatch, abort, buffered_contributions
):
    import skillev.training.streaming_step as streaming

    torch.set_num_threads(1)
    harness = make_training_harness()
    config = replace(harness.config, execution=replace(harness.config.execution, batch_size=4))
    harness = make_training_harness(config=config)
    batch = asyncio.run(harness.loop.collect_batch())
    backbone = harness.backbone
    optimizer, parameters = create_ttb_optimizer(backbone, config.optimizer)
    named = named_ttb_parameters(parameters)
    before = {n: p.detach().clone() for n, p in named.items()}
    reference = compute_ttb_gradient_shard(
        backbone=backbone,
        parameters=parameters,
        batch=batch,
        positions=(0, 1, 2, 3),
        global_batch_size=4,
        temperature_beta=config.method.temperature_beta,
    )
    plan = TrainingBatchPlan(
        batch.batch_id,
        batch.optimizer_step,
        batch.policy_snapshot_id,
        batch.library_version,
        tuple(
            PlannedRollout(
                i + 1,
                harness.task_provider.tasks[i],
                a.record.trajectory_id,
                harness.loop.decoding_snapshot,
            )
            for i, a in enumerate(batch.artifacts)
        ),
    )
    order = []
    completed = threading.Event()
    all_out_of_order_completed = threading.Event()
    compute = streaming.compute_ttb_artifact_contribution

    def observed(**kwargs):
        result = compute(**kwargs)
        order.append(kwargs["position"])
        return result

    monkeypatch.setattr(streaming, "compute_ttb_artifact_contribution", observed)
    one_contribution = sum(p.numel() * p.element_size() for p in named.values())
    stream = LocalGradientStepStream(
        backbone=backbone,
        parameters=parameters,
        optimizer=optimizer,
        temperature_beta=config.method.temperature_beta,
        clock=lambda: "2026-09-08T00:00:00Z",
        max_buffer_bytes=one_contribution * buffered_contributions,
    )

    record_complete = stream._record_complete

    def recorded(position):
        record_complete(position)
        completed.set()
        if stream._completed == 3:
            all_out_of_order_completed.set()

    monkeypatch.setattr(stream, "_record_complete", recorded)

    async def run():
        await stream.begin(plan, harness.generator.snapshot())
        try:
            await stream.accept(3, batch.artifacts[3])
            # A real contribution finishes even though canonical position zero is absent.
            assert await asyncio.to_thread(completed.wait, 10)
            assert order == [3]
            assert all(p.grad is None for p in named.values())
            if abort:
                await stream.discard_uncommitted()
                return
            # Buffer is full: zero must bypass later ready positions, not deadlock.
            for position in (2, 1):
                await stream.accept(position, batch.artifacts[position])
            if buffered_contributions == 3:
                assert await asyncio.to_thread(all_out_of_order_completed.wait, 10)
                assert order[0] == 3
                assert set(order) == {1, 2, 3}
            await stream.accept(0, batch.artifacts[0])
            prepared = await stream.seal(batch)
            assert [r.trajectory_id for r in prepared.residuals] == [
                a.record.trajectory_id for a in batch.artifacts
            ]
            assert stream.completed_before_last_rollout >= 1
            metrics = stream.rank_metrics[0]
            assert (
                metrics["gradient_buffer_peak_bytes"] == one_contribution * buffered_contributions
            )
            assert sorted(order) == [0, 1, 2, 3]
            assert order[1] == 0 if buffered_contributions == 1 else order[-1] == 0
            assert metrics["queue_wait_seconds"] == pytest.approx(
                metrics["artifact_wait_seconds"]
                + metrics["buffer_backpressure_seconds"]
                + metrics["remote_compute_wait_seconds"]
            )
        finally:
            if stream.prepared is None:
                await stream.discard_uncommitted()

    asyncio.run(run())
    assert not optimizer.state
    for name, parameter in named.items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        if abort:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.gradients[name], rtol=0, atol=0)


def test_kernel_enables_local_overlap_only_for_an_external_model_owner(make_training_harness):
    from skillev.rollout import ExternalSGLangRolloutConfig, ExternalSGLangRolloutGenerator
    from skillev.training.performance_config import TrainingPerformanceConfig

    harness = make_training_harness()
    backbone = harness.backbone
    backbone.performance_config = TrainingPerformanceConfig()
    kernel = harness.loop._kernel
    with pytest.raises(ValueError):
        kernel.create_gradient_stream()
    generator = ExternalSGLangRolloutGenerator(
        config=ExternalSGLangRolloutConfig("http://localhost:1234"),
        tokenizer=backbone.tokenizer,
        gateway=None,
        snapshot_provider=harness.generator.snapshot,
    )
    try:
        kernel._generator = generator
        assert isinstance(kernel.create_gradient_stream(), LocalGradientStepStream)
        backbone.performance_config = replace(
            backbone.performance_config, pipeline_mode="sealed-batch"
        )
        assert kernel.create_gradient_stream() is None
    finally:
        generator.close()


def test_cancelled_seal_drains_cuda_owner_before_clearing_gradients(
    make_training_harness, monkeypatch
):
    import skillev.training.streaming_step as streaming

    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    optimizer, parameters = create_ttb_optimizer(harness.backbone, harness.config.optimizer)
    entered, release = threading.Event(), threading.Event()
    compute = streaming.compute_ttb_artifact_contribution
    zero = optimizer.zero_grad
    unsafe_clears = []

    def blocked(**kwargs):
        entered.set()
        assert release.wait(10)
        return compute(**kwargs)

    def checked_clear(*args, **kwargs):
        if entered.is_set() and not release.is_set():
            unsafe_clears.append(True)
        return zero(*args, **kwargs)

    monkeypatch.setattr(streaming, "compute_ttb_artifact_contribution", blocked)
    monkeypatch.setattr(optimizer, "zero_grad", checked_clear)
    plan = TrainingBatchPlan(
        batch.batch_id,
        batch.optimizer_step,
        batch.policy_snapshot_id,
        batch.library_version,
        tuple(
            PlannedRollout(
                i + 1,
                harness.task_provider.tasks[i],
                a.record.trajectory_id,
                harness.loop.decoding_snapshot,
            )
            for i, a in enumerate(batch.artifacts)
        ),
    )
    stream = LocalGradientStepStream(
        backbone=harness.backbone,
        parameters=parameters,
        optimizer=optimizer,
        temperature_beta=1.0,
        clock=lambda: "2026-09-08T00:00:00Z",
    )

    async def run():
        await stream.begin(plan, harness.generator.snapshot())
        for i, artifact in enumerate(batch.artifacts):
            await stream.accept(i, artifact)
        assert await asyncio.to_thread(entered.wait, 5)
        pending = asyncio.create_task(stream.seal(batch))
        await asyncio.sleep(0)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        assert not unsafe_clears
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.discard_uncommitted()

    asyncio.run(run())
    assert not unsafe_clears
    assert not optimizer.state
    assert all(p.grad is None for p in named_ttb_parameters(parameters).values())


@pytest.mark.parametrize("stage", ["prepare_edge_plan", "compute_ttb_artifact_contribution"])
def test_unexpected_owner_error_propagates_after_draining(
    make_training_harness, monkeypatch, stage
):
    import skillev.training.streaming_step as streaming
    from skillev.training.distributed_ttb import DistributedTTBError

    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    optimizer, parameters = create_ttb_optimizer(harness.backbone, harness.config.optimizer)

    def fail(*args, **kwargs):
        raise AssertionError("injected unexpected owner failure")

    monkeypatch.setattr(streaming, stage, fail)
    stream = LocalGradientStepStream(
        backbone=harness.backbone,
        parameters=parameters,
        optimizer=optimizer,
        temperature_beta=1.0,
        clock=lambda: "2026-09-08T00:00:00Z",
    )

    async def run():
        await stream.begin_collected(batch, harness.generator.snapshot())
        try:
            for position, artifact in enumerate(batch.artifacts):
                await stream.accept(position, artifact)
            await stream.seal(batch)
        finally:
            await stream.discard_uncommitted()

    with pytest.raises((RuntimeError, DistributedTTBError)):
        asyncio.run(run())
    assert not optimizer.state
    assert all(p.grad is None for p in named_ttb_parameters(parameters).values())
