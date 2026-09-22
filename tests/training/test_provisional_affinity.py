from types import SimpleNamespace

import pytest

from skillev.training.step_math import create_ttb_optimizer
from skillev.training.streaming_step import GradientStepStream


def test_remote_dispatch_wait_is_reported_without_inflating_local_buffer_wait(
    make_training_harness, monkeypatch
):
    harness = make_training_harness()
    optimizer, parameters = create_ttb_optimizer(harness.backbone, harness.config.optimizer)
    stream = GradientStepStream(
        backbone=harness.backbone,
        parameters=parameters,
        optimizer=optimizer,
        temperature_beta=1.0,
        clock=lambda: "2026-09-08T00:00:00Z",
    )
    stream._wait_reasons[1] = "other-owner"
    times = iter((10.0, 12.0))
    monkeypatch.setattr("skillev.training.streaming_step.time.perf_counter", lambda: next(times))
    monkeypatch.setattr(stream._condition, "wait", lambda: None)
    stream._wait(local_owner=False, rank=1)
    assert stream._wait_totals_by_rank[1]["other-owner"] == 2.0
    assert stream._buffer_wait_seconds == stream._queue_seconds == 0.0


@pytest.mark.parametrize(
    ("world", "participates", "expected"),
    [(1, True, [0] * 4), (2, True, [0, 1, 0, 1]), (2, False, [1] * 4), (3, False, [1, 2, 1, 2])],
)
def test_sparse_live_trajectories_do_not_all_pin_to_first_idle_worker(
    make_training_harness, world, participates, expected
):
    harness = make_training_harness()
    optimizer, parameters = create_ttb_optimizer(harness.backbone, harness.config.optimizer)
    stream = GradientStepStream(
        backbone=harness.backbone,
        parameters=parameters,
        optimizer=optimizer,
        temperature_beta=1.0,
        clock=lambda: "2026-09-08T00:00:00Z",
        provisional_edges=True,
    )
    # Exercise the actual claim path without a CUDA owner or a transport thread.
    stream._world, stream._participates = world, participates
    stream._assignments = [[] for _ in range(world)]
    stream._unassigned = set(range(5))
    stream._progress = {p: {} for p in range(5)}
    stream._contribution_bytes = 1
    for position, owner in enumerate(expected):
        stream._route_provisional(position)
        assert stream._progress[position]["provisional_worker_affinity"] == owner
        edge = SimpleNamespace(token_cost=100 + position)
        stream._provisional_queue[position] = [edge]
        # An available rank must not steal a newly arriving trajectory assigned
        # to a busy rank. All previous actors are waiting for their next step.
        for rank in range(0 if participates else 1, world):
            if rank != owner:
                assert stream._claim(rank) is None
                assert stream._wait_reasons[rank] == "other-owner"
                assert stream._buffer_wait_seconds == 0
        assert stream._claim(owner) == (position, None, edge)
        assert not any(stream._provisional_queue.values())
        stream._route_provisional(position)  # The next edge cannot migrate state.
        assert stream._provisional_affinity[position] == owner
    assert [stream._owners[p] for p in range(4)] == expected

    # A finalized trajectory no longer reserves future scheduling load.
    stream._unassigned.remove(0)
    stream._route_provisional(4)
    assert stream._provisional_affinity[4] == expected[0]


def test_initial_assignment_prefers_less_queued_work_not_equal_actor_count(make_training_harness):
    harness = make_training_harness()
    optimizer, parameters = create_ttb_optimizer(harness.backbone, harness.config.optimizer)
    stream = GradientStepStream(
        backbone=harness.backbone,
        parameters=parameters,
        optimizer=optimizer,
        temperature_beta=1.0,
        clock=lambda: "now",
        provisional_edges=True,
    )
    stream._world, stream._participates = 2, True
    stream._unassigned = {0, 1, 2}
    stream._progress = {p: {} for p in range(3)}
    stream._provisional_affinity = {0: 0, 1: 1}
    stream._provisional_edge_cost = {0: 60000, 1: 1000}
    stream._provisional_queue = {0: [SimpleNamespace(token_cost=60000)]}
    stream._route_provisional(2)
    assert stream._provisional_affinity[2] == 1
    assert stream._provisional_affinity[0] == 0  # No migration without a state-transfer protocol.
