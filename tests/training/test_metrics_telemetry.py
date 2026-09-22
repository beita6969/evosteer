"""Synthetic committed evidence only: no model calls or task-bank contents."""

import json

import pytest

from skillev.training.metrics_contract import TrainingMetricsSnapshot
from skillev.training.metrics_export import (
    MetricsStore,
    export_available,
    export_enriched_available,
)
from skillev.training.metrics_telemetry import CommittedTelemetry, performance_rows
from tests.training.test_metrics_contract import event


def commit(step=1, cells=("one", "one")):
    value = event()
    value["event_id"] = f"commit-{step}"
    value["payload"]["batch_id"] = f"batch-{step}"
    value["payload"]["optimizer_step"] = step
    value["payload"]["posterior_batch"] = {
        "batch_id": f"batch-{step}",
        "updates": [{"skill_id": "PRIVATE_SKILL", "z": {"domain": cell}} for cell in cells],
    }
    for record in value["payload"]["records"]:
        record["steps"][0]["invoked_skill_ids"] = ["PRIVATE_SKILL"]
    return value


def phase(step, reason="phase-detected", kind="phase_detection_recorded", **payload):
    return {
        "run_id": "synthetic-run",
        "event_id": f"{kind}-{step}",
        "event_type": kind,
        "payload": {
            "optimizer_step": step,
            "batch_id": f"batch-{step}",
            "reason": reason,
            **payload,
        },
    }


def timing(step, process="process-a", elapsed=None):
    return {
        "committed": True,
        "optimizer_step": step,
        "batch_id": f"batch-{step}",
        "process_instance_id": process,
        "process_elapsed_seconds": elapsed if elapsed is not None else step * 200,
        "run_elapsed_seconds": 10000000,
        "step_wall_seconds": 100,
        "rollout_span_seconds": 80,
        "gradient_span_seconds": 70,
        "gradient_tail_seconds": 15,
        "overlap_seconds": 60,
        "durability_seconds": 5,
        "rollout": {
            "batch_id": f"batch-{step}",
            "prompt_tokens": 700,
            "completion_tokens": 90,
            "model_calls": 28,
        },
        "physical_generation_usage_cumulative": {"server_generated_tokens": step * 150},
    }


def write(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def setup_export(tmp_path, **kwargs):
    store = MetricsStore(tmp_path / "enriched.sqlite3", enriched=True)
    collector = CommittedTelemetry(
        store, run_id="synthetic-run", condition_id="synthetic-condition", **kwargs
    )
    return store, collector


def test_real_committed_fields_and_native_success_are_not_reward_or_private_text(tmp_path):
    store, collector = setup_export(tmp_path)
    source = commit()
    collector.observe(source)
    collector.observe(phase(1, reason="insufficient-window"))
    assert collector.flush({}) == 1
    row = store.values()[0]
    telemetry = row["telemetry"]
    assert row["format"].endswith("@2")
    assert (
        row["metrics"]
        == store.actions.enrich(
            TrainingMetricsSnapshot.from_event(source, condition_id="synthetic-condition"), source
        ).to_value()["metrics"]
    )
    assert telemetry["skill_invocation_count"] == 28
    assert telemetry["posterior_update_event_count"] == 2
    assert telemetry["posterior_cells_touched_count"] == 1
    assert telemetry["native_success_counts"]["hotpotqa"] == {
        "success_count": 7,
        "trajectory_count": 28,
    }
    assert telemetry["phase_check_count"] == 1
    assert telemetry["phase_trigger_count"] == telemetry["library_mutation_count"] == 0
    assert "PRIVATE_" not in json.dumps(row)
    assert telemetry["logical_input_tokens"] is None
    assert telemetry["reserved_gpu_count"] is None
    store.close()


def test_waits_for_complete_performance_and_later_phase_events_then_restart_is_idempotent(tmp_path):
    events, performance = tmp_path / "events.jsonl", tmp_path / "performance.jsonl"
    write(events, commit())
    performance.write_text(json.dumps(timing(1)))  # unterminated write is not published
    store, collector = setup_export(tmp_path, performance=performance)
    offset, added = export_enriched_available(events, collector)
    assert added == 0
    assert not store.values()
    with events.open("a") as stream:
        stream.write(json.dumps(phase(1)) + "\n")
        stream.write(json.dumps(phase(1, kind="evolution_no_op_committed")) + "\n")
    write(performance, timing(1))
    offset, added = export_enriched_available(events, collector, offset=offset)
    assert added == 1
    value = store.values()[0]
    assert value["telemetry"]["phase_no_op_count"] == 1
    assert value["telemetry"]["library_mutation_count"] == 0
    assert value["telemetry"]["logical_input_tokens"] == 700
    assert value["telemetry"]["server_generated_tokens_delta"] is None
    store.close()
    store, collector = setup_export(tmp_path, performance=performance)
    assert export_enriched_available(events, collector)[1] == 0
    assert store.values() == [value]
    store.close()


@pytest.mark.parametrize(
    ("reason", "check", "trigger"),
    [(None, None, None), ("slot-does-not-allow-phase-detection", 0, 0), ("phase-detected", 1, 1)],
)
def test_detection_check_trigger_are_distinct_and_unresolved_is_unknown(
    tmp_path, reason, check, trigger
):
    store, collector = setup_export(tmp_path)
    collector.observe(commit())
    collector.observe(phase(1, reason=reason))
    collector.flush({})
    result = store.values()[0]["telemetry"]
    assert result["phase_detection_record_count"] == 1
    assert result["phase_check_count"] == check
    assert result["phase_trigger_count"] == trigger
    assert result["phase_no_op_count"] == (0 if trigger == 0 else None)
    store.close()


def test_actual_library_mutation_and_authoring_usage_not_detection_count(tmp_path):
    store, collector = setup_export(tmp_path)
    collector.observe(commit())
    collector.observe(phase(1))
    collector.observe(
        phase(
            1,
            kind="evolution_cycle_committed",
            mutation={"actions": [{"kind": "generate"}, {"kind": "refine"}]},
            library_version_before="old",
            library_version_after="new",
            authoring_usage={"input_tokens": 42, "output_tokens": 11, "model_calls": 1},
        )
    )
    collector.flush({})
    result = store.values()[0]["telemetry"]
    assert result["library_mutation_count"] == 1
    assert result["library_mutation_action_count"] == 2
    assert result["phase_no_op_count"] == 0
    assert result["authoring_input_tokens"] == 42
    store.close()


def test_posterior_full_history_and_condition_segment_have_separate_scopes(tmp_path):
    store, collector = setup_export(tmp_path, committed_after=2)
    for step in range(1, 4):
        collector.observe(commit(step, ("one", str(step))))
    collector.flush({})
    assert len(store.values()) == 1
    result = store.values()[0]["telemetry"]
    assert result["posterior_update_event_count_cumulative"] == 6
    assert result["posterior_cells_touched_cumulative"] == 4
    assert result["posterior_update_event_count_segment_cumulative"] == 2
    assert result["posterior_cells_touched_segment_cumulative"] == 2
    store.close()


def test_missing_history_or_missing_fields_are_not_zero(tmp_path):
    store, collector = setup_export(tmp_path, committed_after=23)
    source = commit(24)
    del source["payload"]["posterior_batch"]
    del source["payload"]["records"][0]["steps"][0]["invoked_skill_ids"]
    collector.observe(source)
    collector.flush({})
    result = store.values()[0]["telemetry"]
    for key in (
        "skill_invocation_count",
        "posterior_update_event_count",
        "posterior_cells_touched_count",
        "posterior_update_event_count_cumulative",
        "posterior_update_event_count_segment_cumulative",
        "phase_check_count",
        "library_mutation_count",
    ):
        assert result[key] is None
    store.close()


def test_process_counter_and_gpu_reservations_never_use_run_elapsed_or_sum_cumulative(tmp_path):
    store, collector = setup_export(
        tmp_path, gpu_count=3, gpu_process_ids=("process-a", "process-b")
    )
    rows = {}
    for step, process, elapsed in (
        (1, "process-a", 200),
        (2, "process-a", 500),
        (3, "process-b", 150),
        (4, "undeclared", 999),
    ):
        collector.observe(commit(step))
        rows[step, f"batch-{step}"] = timing(step, process, elapsed)
    rows[2, "foreign-batch"] = {**timing(2, elapsed=999999), "batch_id": "foreign-batch"}
    assert collector.flush(rows) == 4
    one, two, three, four = [row["telemetry"] for row in store.values()]
    assert one["server_generated_tokens_delta"] is None
    assert two["server_generated_tokens_delta"] == 150
    assert three["server_generated_tokens_delta"] is None
    assert two["reserved_gpu_hours_step"] == pytest.approx(300 / 3600)
    assert two["reserved_gpu_hours_observed_cumulative"] == pytest.approx(600 / 3600)
    assert two["reserved_gpu_hours_process_elapsed"] == pytest.approx(1500 / 3600)
    assert three["reserved_gpu_hours_processes_observed_cumulative"] == pytest.approx(1950 / 3600)
    assert three["gpu_processes_observed_count"] == 2
    assert four["reserved_gpu_hours_step"] is None
    assert four["reserved_gpu_hours_process_elapsed"] is None
    assert four["reserved_gpu_hours_observed_cumulative"] == pytest.approx(900 / 3600)
    assert four["measured_server_input_tokens"] is None
    store.close()


def test_counter_decrease_and_absent_counter_are_unknown(tmp_path):
    store, collector = setup_export(tmp_path)
    collector.observe(commit())
    collector.observe(commit(2))
    first, second = timing(1), timing(2)
    second["physical_generation_usage_cumulative"]["server_generated_tokens"] = 2
    collector.flush({(1, "batch-1"): first, (2, "batch-2"): second})
    result = store.values()[1]["telemetry"]
    assert result["server_generated_tokens_delta"] is None
    assert result["admitted_content_tokens_delta"] is None
    store.close()


def test_explicit_unknown_freezes_once_but_different_declaration_requires_new_store(tmp_path):
    performance = tmp_path / "performance.jsonl"
    store, collector = setup_export(
        tmp_path, performance=performance, allow_missing_performance=True
    )
    collector.observe(commit())
    assert collector.flush({}) == 1
    old = store.values()
    assert collector.flush({(1, "batch-1"): timing(1)}) == 0
    assert store.values() == old
    with pytest.raises(ValueError):
        CommittedTelemetry(store, run_id="synthetic-run", condition_id="different")
    store.close()


def test_old_store_and_export_stay_v1_and_cannot_be_repurposed(tmp_path):
    events, path = tmp_path / "events.jsonl", tmp_path / "metrics.sqlite3"
    write(events, commit())
    store = MetricsStore(path)
    assert export_available(events, store, condition_id="old")[1] == 1
    assert store.values()[0]["format"].endswith("@1")
    assert "telemetry" not in store.values()[0]
    store.close()
    with pytest.raises(ValueError):
        MetricsStore(path, enriched=True)
    store, _ = setup_export(tmp_path)
    store.close()
    with pytest.raises(ValueError):
        MetricsStore(tmp_path / "enriched.sqlite3")


def test_wrong_run_and_foreign_sidecar_rejected(tmp_path):
    events = tmp_path / "events.jsonl"
    write(events, commit())
    store, collector = setup_export(tmp_path, performance=tmp_path / "other" / "performance.jsonl")
    with pytest.raises(ValueError):
        export_enriched_available(events, collector)
    source = commit()
    source["run_id"] = "other-run"
    with pytest.raises(ValueError):
        collector.observe(source)
    store.close()


def test_conflicting_performance_rejected_and_missing_batch_waits(tmp_path):
    path = tmp_path / "performance.jsonl"
    write(path, timing(1), {**timing(1), "step_wall_seconds": 99})
    with pytest.raises(ValueError):
        performance_rows(path)
    store, collector = setup_export(tmp_path, performance=path)
    collector.observe(commit())
    assert collector.flush({(1, "different-batch"): timing(1)}) == 0
    store.close()


def test_cli_explicit_enrichment_writes_new_format_and_refuses_legacy_output(tmp_path, monkeypatch):
    from skillev.training.metrics_export import main

    events, performance = tmp_path / "events.jsonl", tmp_path / "performance.jsonl"
    output = tmp_path / "telemetry-v2.jsonl"
    write(events, commit())
    write(performance, timing(1))
    args = [
        "metrics-export",
        "--events",
        str(events),
        "--performance",
        str(performance),
        "--enrich-telemetry",
        "--observed-run-id",
        "synthetic-run",
        "--condition-id",
        "synthetic-condition",
        "--output",
        str(output),
        "--gpu-count",
        "3",
        "--gpu-process-id",
        "process-a",
    ]
    monkeypatch.setattr("sys.argv", args)
    main()
    saved = output.read_bytes()
    result = json.loads(saved)
    assert result["telemetry"]["reserved_gpu_count"] == 3
    assert result["telemetry"]["gradient_tail_seconds"] == 15
    main()
    assert output.read_bytes() == saved
    legacy = TrainingMetricsSnapshot.from_event(commit(), condition_id="old").to_value()
    write(output, legacy)
    old_bytes = output.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes() == old_bytes


def test_no_declared_process_measurement_does_not_invent_zero_gpu_hours(tmp_path):
    store, collector = setup_export(tmp_path, gpu_count=3, gpu_process_ids=("later-process",))
    collector.observe(commit())
    collector.flush({(1, "batch-1"): timing(1)})
    result = store.values()[0]["telemetry"]
    assert result["gpu_processes_observed_count"] is None
    assert result["reserved_gpu_hours_processes_observed_cumulative"] is None
    assert result["reserved_gpu_hours_observed_cumulative"] is None
    store.close()
