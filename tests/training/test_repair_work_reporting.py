from skillev.training.gradient_work_log import GradientWorkLog
from skillev.training.step_timing import StepTiming


def test_work_log_keeps_every_edge_and_never_claims_host_time_is_cuda_time():
    log = GradientWorkLog()
    for index in (1, 2):
        for direction in ("forward-policy", "backward-policy"):
            identity = {"step_index": index, "direction": direction}
            log.observe(
                0, {**identity, "stage": "score-and-backward", "prefix_tokens": index * 10}, index
            )
            log.observe(0, {**identity, "stage": "edge-backward-returned"}, index + 0.5)
    snapshot = log.snapshot()
    assert len(snapshot) == 4
    assert all(row["host_finished"] > row["host_started"] for row in snapshot)
    assert all(row["cuda_seconds"] is None for row in snapshot)
    snapshot[0]["host_finished"] = 999
    assert log.snapshot()[0]["host_finished"] != 999


def test_final_work_sidecar_not_added_as_concurrent_wall_time():
    timing = StepTiming(
        "batch",
        1,
        0,
        7,
        2,
        9,
        10,
        None,
        gradient_detail={"dispatch_wait_seconds_by_rank": {"0": 100}},
    )
    assert timing.to_value()["step_wall_seconds"] == 10
    assert timing.to_value()["gradient_tail_seconds"] == 2
