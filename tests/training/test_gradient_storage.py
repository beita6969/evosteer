from __future__ import annotations

import torch

from skillev.training.gradient_buckets import pack_gradients


def test_bucket_copy_owns_storage_and_keeps_dtype_and_values():
    source = {
        "forward.small": torch.arange(3, dtype=torch.float32),
        "backward.small": torch.arange(5, dtype=torch.float32),
        "z.double": torch.arange(2, dtype=torch.float64),
    }
    expected = {n: t.clone() for n, t in source.items()}
    packed = pack_gradients(source)
    for tensor in source.values():
        tensor.zero_()
    restored = packed.on_device(source)
    assert len(packed.buffers) == 2
    assert packed.nbytes == sum(t.numel() * t.element_size() for t in source.values())
    for name in expected:
        torch.testing.assert_close(restored[name], expected[name], rtol=0, atol=0)
        assert restored[name].data_ptr() != source[name].data_ptr()


def test_scaled_contribution_does_not_alias_discarded_parameter_gradients(
    make_training_harness, monkeypatch
):
    import asyncio

    from skillev.training import step_math

    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    groups = harness.backbone.parameter_groups()
    named = step_math.named_ttb_parameters(groups)
    unscaled = {}
    original = step_math.backward_trajectory_delta_streaming

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        unscaled.update({name: p.grad for name, p in named.items()})
        return result

    monkeypatch.setattr(step_math, "backward_trajectory_delta_streaming", observed)
    shard = step_math.compute_ttb_artifact_contribution(
        backbone=harness.backbone,
        parameters=groups,
        artifact=batch.artifacts[0],
        position=0,
        batch_id=batch.batch_id,
        optimizer_step=batch.optimizer_step,
        policy_snapshot_id=batch.policy_snapshot_id,
        library_version=batch.library_version,
        global_batch_size=len(batch.artifacts),
        temperature_beta=harness.config.method.temperature_beta,
    )
    expected = {n: g.clone() for n, g in shard.gradients.items()}
    assert all(p.grad is None for p in named.values())
    for name, gradient in unscaled.items():
        assert gradient.data_ptr() != shard.gradients[name].data_ptr()
        gradient.zero_()
    for name in expected:
        torch.testing.assert_close(shard.gradients[name], expected[name], rtol=0, atol=0)
