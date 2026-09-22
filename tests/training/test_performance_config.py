from dataclasses import replace

import pytest
import torch

from skillev.training.deadline import (
    TrainingDeadline,
    TrainingDeadlineExceededError,
    forecast_deadline,
)
from skillev.training.gradient_buckets import flatten_bucket, tensor_buckets, unpack_bucket
from skillev.training.performance_config import TrainingPerformanceConfig


def test_profile_roundtrip_is_execution_only():
    profile = TrainingPerformanceConfig.load("configs/training/protocol13_performance.yaml")
    assert TrainingPerformanceConfig.from_value(profile.to_value()) == profile
    assert profile.workflow().max_inflight_model_requests == 8
    for patch in ({"actor_requests": 0}, {"max_loras_per_batch": 1}, {"pipeline_mode": "stale"}):
        with pytest.raises(ValueError):
            replace(profile, **patch)


def test_current_training_requires_two_actual_gradient_owners():
    from skillev_private.experiments.protocol_v13_training_debug import (
        _require_two_gradient_workers,
    )

    profile = TrainingPerformanceConfig.load("configs/training/protocol13_performance.yaml")
    _require_two_gradient_workers(profile, 2)
    for world, participates in ((1, True), (2, False), (3, True)):
        with pytest.raises(ValueError):
            _require_two_gradient_workers(
                replace(profile, coordinator_participates=participates), world
            )


def test_deadline_distinguishes_evidence_from_capacity_and_finalization():
    args = {"completed_steps": 5, "elapsed_seconds": 4500, "future_extra_seconds": 28800}
    assert not forecast_deadline(**args, measured_step_seconds=(800, 800)).admissible
    fast = forecast_deadline(**args, measured_step_seconds=(800, 800, 800))
    slow = forecast_deadline(**args, measured_step_seconds=(1100, 1100, 1100))
    assert fast.admissible
    assert not slow.admissible
    assert fast.projected_total_seconds == 4500 + 245 * 800 + 28800
    final = forecast_deadline(
        completed_steps=250,
        elapsed_seconds=72 * 3600,
        future_extra_seconds=1,
        measured_step_seconds=(),
    )
    assert not final.admissible
    assert final.status == "finalizing"


def test_buckets_roundtrip_preserves_dtype_shape_and_bounds():
    values = {
        "c": torch.ones(3, dtype=torch.float64),
        "a": torch.arange(4.0),
        "b": torch.zeros((2, 2)),
    }
    layouts = tuple(tensor_buckets(values, maximum_bytes=16))
    assert len(layouts) == 3
    for names in layouts:
        restored = unpack_bucket(names, flatten_bucket(names, values), values)
        for name in names:
            torch.testing.assert_close(restored[name], values[name], rtol=0, atol=0)


def test_deadline_rejects_next_step_without_touching_training_state(make_training_harness):
    import asyncio
    import time

    harness = make_training_harness()
    loop = harness.loop
    before = loop.state
    loop.execution_deadline = TrainingDeadline(time.monotonic() - 2, 1)
    with pytest.raises(TrainingDeadlineExceededError):
        asyncio.run(loop.collect_batch())
    assert loop.state == before
    assert loop.optimizer_step == 0
    assert not loop.optimizer.state


def test_drain_margin_stops_next_batch_before_hard_timeout():
    deadline = TrainingDeadline(0, 100, drain_margin_seconds=10, active_step_seconds=30)
    deadline.require_next_step(89)
    with pytest.raises(TrainingDeadlineExceededError):
        deadline.require_next_step(90)
    with pytest.raises(ValueError):
        TrainingDeadline(0, 100, drain_margin_seconds=100)
