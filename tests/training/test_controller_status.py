from datetime import UTC, datetime, timedelta

import pytest

from skillev.training.controller_status import ControllerStatusWriter, controller_observation


def test_controller_progress_is_not_heartbeat_or_upload_ack(tmp_path):
    writer = ControllerStatusWriter(
        tmp_path / "controller.json",
        run_id="fresh",
        process_instance_id="p1",
        resource_roles={"actor": 1},
    )
    start = datetime(2026, 9, 13, tzinfo=UTC)

    def emit(second, **kwargs):
        return writer.write(
            state="collecting",
            last_committed_step=1,
            commit_id="c1",
            committed_at=start.isoformat(),
            monotonic_now=second,
            observed_at=(start + timedelta(seconds=second)).isoformat(),
            **kwargs,
        )

    first = emit(0)
    second = emit(70)
    assert second["phase_progress_monotonic"] == first["phase_progress_monotonic"]
    observed = controller_observation(
        second,
        now=start + timedelta(seconds=71),
        heartbeat_stale_after=10,
        progress_stale_after=60,
        expected_run_id="fresh",
    )
    assert observed["state"] == "collecting"
    assert observed["freshness"] == "no-phase-progress"
    stale = controller_observation(
        second,
        now=start + timedelta(seconds=90),
        heartbeat_stale_after=10,
        progress_stale_after=60,
        expected_run_id="fresh",
    )
    assert stale["state"] == "unknown"
    progressed = emit(95, gradient={"gradient_contributions_completed": 1})
    assert progressed["phase_progress_monotonic"] == 95
    restart = ControllerStatusWriter(
        tmp_path / "controller.json", run_id="fresh", process_instance_id="p2", resource_roles={}
    )
    receipt = restart.write(
        state="paused",
        last_committed_step=1,
        commit_id="c1",
        committed_at=start.isoformat(),
        monotonic_now=1,
        observed_at=start.isoformat(),
    )
    assert (
        controller_observation(
            receipt,
            now=start + timedelta(days=1),
            heartbeat_stale_after=10,
            progress_stale_after=60,
            expected_run_id="fresh",
        )["state"]
        == "paused"
    )
    with pytest.raises(ValueError):
        controller_observation(
            receipt,
            now=start,
            heartbeat_stale_after=10,
            progress_stale_after=60,
            expected_run_id="old",
        )
