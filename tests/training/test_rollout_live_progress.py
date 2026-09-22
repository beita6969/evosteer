"""Operational clocks cannot affect rollout identities or invent server timings."""

import asyncio

import pytest

from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress, server_metrics
from skillev.training.rollout_workflow import (
    LONG_HORIZON_FIRST,
    AsyncResourceLimiter,
    RolloutBatchWorkflow,
    RolloutWorkflowBinding,
)


def test_waiting_is_visible_before_lease_finishes_and_cancel_cleans_it():
    async def run():
        limiter = AsyncResourceLimiter(1, name="model")
        row = RolloutProgress(trajectory_id="t", policy_snapshot_id="p")
        row.begin_phase("action", 19)

        async def waiting():
            with bind_progress(row):
                async with limiter.lease():
                    pytest.fail("blocked request acquired a slot")

        async with limiter.lease():
            task = asyncio.create_task(waiting())
            await asyncio.sleep(0.01)
            live = limiter.active_snapshot()
            assert live["waiting"] == 1
            assert live["serving"] == 1
            assert live["queue_ages_seconds"][0] > 0
            assert row.snapshot()["stage"] == "action-model-queue"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert limiter.active_snapshot()["waiting"] == 0
        assert limiter.active_snapshot()["serving"] == 0

    asyncio.run(run())


def test_environment_repetition_and_missing_backend_metrics_are_not_failures():
    row = RolloutProgress(trajectory_id="t")
    for _ in range(2):
        with row.environment_command("look"):
            row.observation("unchanged room")
    value = row.snapshot()
    assert value["environment_command_count"] == 2
    assert value["repeated_command_count"] == value["unchanged_observation_count"] == 1
    assert value["environment_errors"] == 0
    assert "unchanged room" not in str(value)
    assert "look" not in str(value)
    assert server_metrics({})["server_prefill_seconds"] is None
    assert server_metrics({"e2e_latency": 3})["server_e2e_seconds"] == 3
    assert server_metrics({"queue_time": float("nan")})["server_queue_seconds"] is None


def test_long_horizon_start_order_does_not_reorder_samples_or_results():
    async def run():
        binding = RolloutWorkflowBinding(
            max_resident_trajectories=16, scheduling_policy=LONG_HORIZON_FIRST
        )
        workflow = RolloutBatchWorkflow(binding)
        started = []
        gate = asyncio.Event()
        items = tuple((i, f"original-seed-coordinate-{i}") for i in range(32))

        async def execute(item):
            started.append(item[0])
            await gate.wait()
            return item

        result = asyncio.create_task(
            workflow.run(
                items, execute, declared_horizons=tuple(50 if i % 8 == 4 else 8 for i in range(32))
            )
        )
        while len(started) < 16:
            await asyncio.sleep(0)
        assert started[:4] == [4, 12, 20, 28]
        gate.set()
        assert await result == items

    asyncio.run(run())


def test_cache_prefill_counts_and_missing_decode_timing_remain_distinct():
    measured = server_metrics(
        {"prompt_tokens": 1000, "cached_tokens": 640, "decode_throughput": 1e7}
    )
    assert measured["server_prefill_tokens"] == 360
    assert measured["server_decode_tokens_per_second"] is None
    assert server_metrics({"prompt_tokens": 1000})["server_prefill_tokens"] is None
    assert server_metrics({"prompt_tokens": 3, "cached_tokens": 4})["server_prefill_tokens"] is None
    assert (
        server_metrics({"decode_time": 2.0, "decode_throughput": 30})[
            "server_decode_tokens_per_second"
        ]
        == 30
    )


def test_prefill_timestamp_does_not_validate_an_unmeasured_decode_rate():
    # Regression for the observed 30-token response: timestamps were present,
    # but the server still supplied a sentinel-like rate without decode_time.
    meta = {
        "completion_tokens": 30,
        "forward_entry_time": 100.0,
        "prefill_finished_time": 100.12307595,
        "request_finished_ts": 100.86737965,
        "e2e_latency": 0.86737965,
        "decode_throughput": 32588710.513867084,
    }
    original = dict(meta)
    measured = server_metrics(meta)
    assert measured["server_after_prefill_span_seconds"] == pytest.approx(0.7443037)
    assert measured["server_e2e_seconds"] == 0.86737965
    assert measured["server_decode_seconds"] is None
    assert measured["server_decode_tokens_per_second"] is None
    assert meta == original  # Never repair or rewrite the original server response.
    row = RolloutProgress(trajectory_id="synthetic")
    row.begin_phase("action", 20)
    row.phase_metrics(**measured)
    row.finish_phase(30, "stop")
    snapshot = row.snapshot()
    assert snapshot["phases"][0]["output_tokens"] == 30
    assert snapshot["phases"][0]["server_decode_tokens_per_second"] is None
    costs = snapshot["phase_summary"]["action"]["measurements"]
    assert costs["server_decode_seconds"]["measured_requests"] == 0
    assert costs["server_after_prefill_span_seconds"]["measured_requests"] == 1


@pytest.mark.parametrize("duration", [None, 0, -1, True, "0.5", float("nan"), float("inf")])
def test_invalid_or_missing_decode_duration_never_unlocks_reported_rate(duration):
    metrics = server_metrics(
        {
            "decode_time": duration,
            "decode_throughput": 123,
            "prefill_finished_time": 100,
            "request_finished_ts": 101,
        }
    )
    assert metrics["server_decode_tokens_per_second"] is None
    assert metrics["server_after_prefill_span_seconds"] == 1


def test_explicit_decode_timing_preserves_reported_rate_without_inventing_missing_rate():
    measured = server_metrics(
        {"decode_time": 2.0, "decode_throughput": 30, "completion_tokens": 60}
    )
    assert measured["server_decode_seconds"] == 2.0
    assert measured["server_decode_tokens_per_second"] == 30
    absent = server_metrics({"decode_time": 2.0, "completion_tokens": 60})
    assert absent["server_decode_tokens_per_second"] is None
    row = RolloutProgress(trajectory_id="no-server-measurements")
    row.begin_phase("reasoning", 10)
    row.finish_phase(5, "stop")
    assert row.snapshot()["phases"][0]["server_decode_tokens_per_second"] is None
