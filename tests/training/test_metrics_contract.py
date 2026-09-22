import copy
import json

import pytest

from skillev.evaluation.format_content_review import FORMAT_REVIEW_PROFILE
from skillev.training.metrics_contract import TrainingMetricsSnapshot
from skillev.training.metrics_export import MetricsStore, export_available


def event():
    records, residuals = [], []
    for i in range(28):
        records.append(
            {
                "trajectory_id": f"trajectory-{i}",
                "query": "PRIVATE_QUESTION_DO_NOT_EXPORT",
                "steps": [
                    {
                        "action_text": '{"kind":"complete","name":"final",'
                        '"arguments":{"value":{"answer":"PRIVATE_ANSWER"}}}',
                        "action_token_count": 7,
                    }
                ],
                "reward": {
                    "value": 0.5,
                    "success": i < 7,
                    "native_payload": {
                        "benchmark_id": "hotpotqa",
                        "public_metrics": {"answer-f1": 0.5, "answer-exact-match": float(i < 7)},
                        "training_evidence_source": {
                            "benchmark_id": "hotpotqa",
                            "population_id": "synthetic",
                            "source_question_id": f"source-{i % 7}",
                        },
                    },
                },
            }
        )
        residuals.append(
            {
                "trajectory_id": f"trajectory-{i}",
                "delta": 2.0,
                "horizon": 1,
                "log_z": 0.0,
                "sum_forward": -1.0,
                "sum_backward": -2.0,
            }
        )
    return {
        "event_type": "training_step_committed",
        "run_id": "synthetic-run",
        "event_id": "commit-one",
        "occurred_at": "2026-09-09T00:00:00Z",
        "payload": {
            "records": records,
            "batch_id": "batch-one",
            "optimizer_step": 1,
            "policy_snapshot_before": "opaque-forward-initial",
            "stats": {"residuals": residuals, "batch_loss": 4.0, "mean_reward": 0.5},
        },
    }


def test_metrics_keep_distinct_denominators_native_scores_and_private_boundaries():
    snapshot = TrainingMetricsSnapshot.from_event(event(), condition_id="raw-json")
    assert snapshot.metrics["batch_success_fraction"] == 0.25
    assert snapshot.metrics["reward_mean"] == 0.5
    assert snapshot.metrics["unique_source_question_count"] == 7
    assert snapshot.metrics["action_structure_valid_fraction"] == 1
    assert snapshot.metrics["admitted_count"] is None
    assert snapshot.sampled_policy_step == 0
    assert snapshot.optimizer_step == 1
    assert snapshot.native_metrics_by_domain["hotpotqa"]["answer-exact-match"] == 0.25
    assert snapshot.native_metrics_by_domain["hotpotqa"]["answer-f1"] == 0.5
    assert "PRIVATE_" not in json.dumps(snapshot.to_value())
    assert "hotpotqa" not in json.dumps(snapshot.wandb_values())


def test_raw_residual_is_not_length_normalized_loss_and_first_turn_has_batch_denominator():
    source = event()
    record = source["payload"]["records"][0]
    record["steps"].append({"action_text": "not an action", "action_token_count": 3})
    source["payload"]["stats"]["residuals"][0]["horizon"] = 2
    source["payload"]["stats"]["batch_loss"] = (27 * 4 + 1) / 28
    metrics = TrainingMetricsSnapshot.from_event(source, condition_id="raw").metrics
    assert metrics["raw_delta_sq_mean"] == 4
    assert metrics["ttb_loss"] == pytest.approx(109 / 28)
    assert metrics["action_structure_valid_fraction"] == pytest.approx(28 / 29)
    assert metrics["first_turn_structure_valid_fraction"] == 1
    assert metrics["parse_error_count"] == 1


def test_missing_native_score_is_not_zero_or_partial_mean():
    source = event()
    del source["payload"]["records"][0]["reward"]["native_payload"]["public_metrics"]["answer-f1"]
    snapshot = TrainingMetricsSnapshot.from_event(source, condition_id="raw")
    assert snapshot.native_metrics_by_domain["hotpotqa"]["answer-f1"] is None
    assert snapshot.native_metrics_by_domain["hotpotqa"]["answer-f1/observed_count"] == 27


def test_content_review_does_not_relabel_native_scores_or_action_validity():
    source = event()
    record = source["payload"]["records"][0]
    record["reward"]["value"] = 1.0
    record["steps"][0]["action_text"] = "malformed submission"
    original = copy.deepcopy(record["reward"])
    original.update(value=0.0, success=False)
    record["reward"]["native_payload"]["format_content_review"] = {
        "approved": True,
        "native_result": original,
        "verdict": {"explanation": "PRIVATE_REVIEW_DO_NOT_EXPORT"},
    }
    source["payload"]["stats"]["mean_reward"] = 14.5 / 28
    snapshot = TrainingMetricsSnapshot.from_event(source, condition_id=FORMAT_REVIEW_PROFILE)
    assert snapshot.metrics["format_review_count"] == 1
    assert snapshot.metrics["format_review_approved_count"] == 1
    assert snapshot.metrics["native_reward_mean"] == 13.5 / 28
    assert snapshot.metrics["reward_mean"] == 14.5 / 28
    assert snapshot.metrics["native_success_fraction"] == 6 / 28
    assert snapshot.metrics["batch_success_fraction"] == 7 / 28
    assert snapshot.metrics["parse_error_count"] == 1
    assert "PRIVATE_" not in json.dumps(snapshot.to_value())


@pytest.mark.parametrize("kind", ["empty", "duplicate", "reorder", "wrong-loss", "preview"])
def test_incomplete_or_inconsistent_sources_are_not_exported(kind):
    source = event()
    payload = source["payload"]
    if kind == "empty":
        payload["records"] = []
    elif kind == "duplicate":
        payload["records"][1] = payload["records"][0]
    elif kind == "reorder":
        payload["stats"]["residuals"].reverse()
    elif kind == "wrong-loss":
        payload["stats"]["batch_loss"] = 0.0
    else:
        source["event_type"] = "preview"
    with pytest.raises(ValueError):
        TrainingMetricsSnapshot.from_event(source, condition_id="raw")


def test_store_restart_deduplicates_and_partial_event_waits(tmp_path):
    path, output = tmp_path / "events.jsonl", tmp_path / "metrics.jsonl"
    source = event()
    raw = json.dumps(source)
    path.write_text(raw[:20])
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    assert export_available(path, store, condition_id="raw") == (0, 0)
    path.write_text(raw + "\n" + raw + "\n")
    _, added = export_available(path, store, condition_id="raw")
    assert added == 1
    store.publish(output)
    store.close()
    recovered = MetricsStore(tmp_path / "metrics.sqlite3")
    _, added = export_available(path, recovered, condition_id="raw")
    assert added == 0
    assert len(output.read_text().splitlines()) == 1
    changed = copy.deepcopy(source)
    changed["event_id"] = "conflicting-commit"
    with pytest.raises(ValueError):
        recovered.add(TrainingMetricsSnapshot.from_event(changed, condition_id="raw"))
    recovered.close()


def test_event_publication_waits_for_finalized_transaction_timing(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event()) + "\n")
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    assert export_available(path, store, condition_id="raw", committed_through=0) == (0, 0)
    offset, added = export_available(path, store, condition_id="raw", committed_through=1)
    assert offset > 0
    assert added == 1
    store.close()


def execution_events(source):
    events = []
    for i, record in enumerate(source["payload"]["records"]):
        record["steps"][0].update(action_token_ids=[i + 1], observation_text="public observation")
        events.append(
            {
                "event_type": "agent_step_recorded",
                "run_id": source["run_id"],
                "payload": {
                    "trajectory_id": record["trajectory_id"],
                    "turn": 1,
                    "action_token_ids": [i + 1],
                    "observation_text": "public observation",
                    "assessment": {
                        "parse_status": "valid",
                        "admitted": i < 14,
                        "executed": i < 7,
                        "accepted_submission": 7 <= i < 14,
                        "environment_terminal": i == 0,
                        "execution_status": "success" if i < 7 else None,
                        "terminal_task_success": None,
                    },
                },
            }
        )
    return events


def test_explicit_assessments_follow_only_committed_edges_and_survive_restart(tmp_path):
    source = event()
    rows = execution_events(source)
    abandoned = copy.deepcopy(rows[14])
    abandoned["payload"]["action_token_ids"] = [999]
    abandoned["payload"]["assessment"]["admitted"] = True
    rows.insert(0, abandoned)
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [*rows, source]))
    store_path = tmp_path / "metrics.sqlite3"
    store = MetricsStore(store_path)
    _, added = export_available(path, store, condition_id="raw")
    assert added == 1
    metrics = store.values()[0]["metrics"]
    assert metrics["assessed_action_count"] == 28
    assert metrics["structural_valid_count"] == 28
    assert metrics["admitted_count"] == 14
    assert metrics["executed_count"] == 7
    assert metrics["accepted_submission_count"] == 7
    assert metrics["environment_terminal_count"] == 1
    assert metrics["success_count"] == 7  # Native labels, never execution status.
    store.close()
    store = MetricsStore(store_path)
    assert export_available(path, store, condition_id="raw")[1] == 0
    assert store.values()[0]["metrics"] == metrics
    store.close()


def test_partial_assessments_are_coverage_not_a_smaller_denominator(tmp_path):
    source = event()
    rows = execution_events(source)
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [rows[0], source]))
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    export_available(path, store, condition_id="raw")
    metrics = store.values()[0]["metrics"]
    assert metrics["assessed_action_count"] == 1
    assert metrics["admitted_count"] is None
    assert metrics["executed_count"] is None
    assert metrics["accepted_submission_count"] is None
    store.close()


def test_branch_export_keeps_parent_events_but_never_relabels_parent_metrics(tmp_path):
    first = event()
    third = copy.deepcopy(first)
    third["event_id"] = "new-condition-third"
    third["payload"].update(optimizer_step=3, batch_id="batch-three")
    path = tmp_path / "events.jsonl"
    original = json.dumps(first) + "\n" + json.dumps(third) + "\n"
    path.write_text(original)
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    try:
        _, added = export_available(path, store, condition_id="new", committed_after=2)
        assert added == 1
        assert [p["optimizer_step"] for p in store.values()] == [3]
        assert path.read_text() == original
    finally:
        store.close()
