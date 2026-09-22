from skillev.diagnostics.rollout_progress import RolloutProgress, server_metrics
from skillev.training.step_timing import StepTiming


def test_fresh_process_step_three_is_cold_without_erasing_its_wall_time():
    timing = StepTiming(
        "batch",
        3,
        100,
        180,
        110,
        190,
        200,
        None,
        process_instance_id="new-process",
        process_step_index=1,
    )
    result = timing.to_value()
    assert result["warmup"] is True
    assert result["global_step_initial_warmup"] is False
    assert result["step_wall_seconds"] == 100
    assert result["service_instance_id"] is None


def test_unset_server_timestamps_are_not_zero_latency_or_client_ttft():
    empty = server_metrics({"queue_time": 0, "id": "request"})
    assert empty["server_queue_seconds"] is None
    assert empty["server_prefill_span_seconds"] is None
    measured = server_metrics(
        {
            "queue_time": 0.1,
            "forward_entry_time": 100,
            "prefill_finished_time": 101,
            "request_finished_ts": 110,
        }
    )
    assert measured["server_prefill_span_seconds"] == 1
    assert measured["server_after_prefill_span_seconds"] == 9
    assert measured["server_prefill_seconds"] is None
    row = RolloutProgress(max_turns=50)
    row.begin_phase("reasoning", 100)
    row.phase_metrics(**measured)
    row.finish_phase(10, "stop")
    snapshot = row.snapshot()
    assert snapshot["phases"][0]["client_first_token_seconds"] is None
    summary = snapshot["phase_summary"]["reasoning"]["measurements"]
    assert summary["input_tokens"]["sum"] == 100
    assert summary["server_decode_seconds"]["sum"] is None
