import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from skillev.policy.interface import EncodedPolicyPrompt, ModelInputWindow
from skillev.policy.versions import TrainableVersions
from skillev.rollout.provisional import ProvisionalStep
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.training import provisional_preparation as module


def test_actual_forward_ids_reuse_skips_forward_tokenization(make_training_harness, monkeypatch):
    from skillev.training import provisional_math

    harness = make_training_harness()
    artifact = asyncio.run(harness.loop.collect_batch()).artifacts[0]
    value = value_for(artifact, 0)
    versions = TrainableVersions.from_backbone(harness.backbone)
    expected = provisional_math.prepare_provisional_step(harness.backbone, value, versions)
    ids = expected.forward.prefix_ids
    supplied = replace(value, forward_input=EncodedPolicyPrompt(ids, len(ids)))
    original = provisional_math.encode_policy_prompt
    calls = []

    def encode(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(provisional_math, "encode_policy_prompt", encode)
    actual = provisional_math.prepare_provisional_step(harness.backbone, supplied, versions)
    assert actual == expected
    assert len(calls) == 1  # Only hindsight needs another encoding.
    assert actual.forward.prefix_ids is ids


def value_for(artifact, index):
    return ProvisionalStep(
        artifact.record.trajectory_id,
        artifact.manifest.task_id,
        artifact.manifest.policy_snapshot,
        artifact.manifest.library_version,
        artifact.record.initial_context.query,
        artifact.initial_context.text,
        artifact.record.steps[:index],
        artifact.record.steps[index],
        ModelInputWindow.from_meta(artifact.record.initial_context.meta),
    )


def test_admission_does_not_wait_for_encoding_and_terminal_reuses_exact_edges(
    make_training_harness, monkeypatch
):
    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    artifact = batch.artifacts[0]
    expected = prepare_edge_plan(
        harness.backbone.tokenizer, artifact.record, artifact.initial_context.text
    )
    proceed = threading.Event()
    original = module.prepare_provisional_step
    calls, published, errors = [], [], []

    def slow(*args):
        assert proceed.wait(10)
        calls.append(args[1].step.index)
        return original(*args)

    monkeypatch.setattr(module, "prepare_provisional_step", slow)

    async def run():
        with ThreadPoolExecutor(max_workers=2) as workers:
            preparation = module.ProvisionalPreparation(
                harness.backbone,
                TrainableVersions.from_backbone(harness.backbone),
                workers,
                lambda p, edge, metrics: published.append(edge.step.index),
                errors.append,
                queue_bytes=10**8,
                prepared_bytes=10**8,
            )
            try:
                for index in range(artifact.record.horizon):
                    await asyncio.wait_for(preparation.accept(0, value_for(artifact, index)), 1)
                # Actor can start its next request while all tokenization is blocked.
                assert not calls
                assert not published
                assert preparation.snapshot()["queued_history_bytes"] > 0
            finally:
                proceed.set()
            actual = await asyncio.to_thread(preparation.plan, 0, artifact)
            assert actual == expected
            assert published == list(range(1, artifact.record.horizon + 1))
            before = list(calls)
            assert preparation.plan(0, artifact) == expected
            assert calls == before  # Terminal validation is not another encoding pass.
            preparation.forget_completed(0)
            assert preparation.snapshot()["retained_plan_bytes"] == 0
            assert not errors

    asyncio.run(run())


def test_preparation_failure_aborts_and_wakes_bounded_admission(make_training_harness, monkeypatch):
    harness = make_training_harness()
    artifact = asyncio.run(harness.loop.collect_batch()).artifacts[0]
    value = value_for(artifact, 0)
    proceed = threading.Event()
    complete_callback = threading.Event()
    errors = []

    class DelayedCallbackPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            # Reproduce capacity becoming available before Future callbacks
            # execute. Failure admission must not depend on callback scheduling.
            future.add_done_callback(lambda _future: complete_callback.wait(10))
            return future

    def broken(*args):
        assert proceed.wait(10)
        raise ValueError("controlled tokenizer failure")

    monkeypatch.setattr(module, "prepare_provisional_step", broken)

    async def run():
        with DelayedCallbackPool(max_workers=1) as workers:
            preparation = module.ProvisionalPreparation(
                harness.backbone,
                TrainableVersions.from_backbone(harness.backbone),
                workers,
                lambda *args: pytest.fail("must not publish a failed edge"),
                errors.append,
                queue_bytes=module.retained_bytes(value),
                prepared_bytes=10**8,
            )
            await preparation.accept(0, value)
            blocked = asyncio.create_task(preparation.accept(1, value))
            await asyncio.sleep(0)
            assert not blocked.done()
            proceed.set()
            try:
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(blocked, 2)
                with pytest.raises(ValueError):
                    await asyncio.to_thread(preparation.plan, 0, artifact)
                assert len(errors) == 1
            finally:
                complete_callback.set()

    asyncio.run(run())


def test_retained_plan_capacity_is_explicit_failure_not_deadlock(make_training_harness):
    harness = make_training_harness()
    artifact = asyncio.run(harness.loop.collect_batch()).artifacts[0]

    async def run():
        with ThreadPoolExecutor(max_workers=1) as workers:
            preparation = module.ProvisionalPreparation(
                harness.backbone,
                TrainableVersions.from_backbone(harness.backbone),
                workers,
                lambda *args: pytest.fail("oversized plans cannot be published"),
                lambda e: None,
                queue_bytes=10**8,
                prepared_bytes=1,
            )
            await preparation.accept(0, value_for(artifact, 0))
            with pytest.raises(MemoryError):
                await asyncio.wait_for(asyncio.to_thread(preparation.plan, 0, artifact), 2)

    asyncio.run(run())
