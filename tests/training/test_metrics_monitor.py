import copy
import json
from types import SimpleNamespace

import pytest
from skillev_private.experiments.bayesian_training_setup import _start_progress_monitor

from skillev.training.metrics_export import MetricsStore, export_available
from tests.training.test_metrics_contract import event


def test_metrics_failure_requests_cooperative_stop_instead_of_silent_thread_death(tmp_path):
    application = SimpleNamespace(
        training_loop=SimpleNamespace(
            optimizer_step=0, finalized_timings=(), rollout_progress={}, gradient_progress=None
        )
    )
    (tmp_path / "events.jsonl").write_text("not-json\n")
    stop, thread = _start_progress_monitor(
        application,
        total_steps=250,
        performance_path=tmp_path / "performance.jsonl",
        run_started=0.0,
        metrics_condition_id="raw",
    )
    thread.join(timeout=3)
    stop.set()
    assert not thread.is_alive()
    assert (tmp_path / "STOP_AFTER_CHECKPOINT").exists()
    assert (tmp_path / "monitor-failure.json").exists()
    assert not (tmp_path / "committed-metrics.jsonl").exists()


@pytest.mark.parametrize("existing_store", [True, False])
def test_prompt_continuation_monitor_keeps_old_labels_and_does_not_pause(tmp_path, existing_store):
    first = event()
    second = copy.deepcopy(first)
    second["event_id"] = "second-commit"
    second["payload"].update(optimizer_step=2, batch_id="second-batch")
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    store = MetricsStore(tmp_path / "committed-metrics.sqlite3")
    original = None
    if existing_store:
        export_available(events, store, condition_id="original", committed_through=1)
        original = store.values()[0]
    store.close()
    application = SimpleNamespace(
        training_loop=SimpleNamespace(
            optimizer_step=2, finalized_timings=(), rollout_progress={}, gradient_progress=None
        )
    )
    stop, thread = _start_progress_monitor(
        application,
        total_steps=250,
        performance_path=tmp_path / "performance.jsonl",
        run_started=0.0,
        metrics_condition_id="proactive",
        metrics_condition_starts={1: "original", 2: "proactive"},
    )
    stop.set()  # The monitor still completes its first export before exiting.
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert not (tmp_path / "STOP_AFTER_CHECKPOINT").exists()
    assert not (tmp_path / "monitor-failure.json").exists()
    rows = [
        json.loads(line) for line in (tmp_path / "committed-metrics.jsonl").read_text().splitlines()
    ]
    assert [row["condition_id"] for row in rows] == ["original", "proactive"]
    if original is not None:
        assert rows[0] == original
