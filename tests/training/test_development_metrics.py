"""Live telemetry uses canonical sources and real manifests/assessment evidence."""

import json

import pytest
from skillev_private.experiments.development_metrics import snapshot

from tests.training.test_development_collection import write
from tests.training.test_development_comparison import pair_fixture


def test_launch_order_is_not_domain_order_and_missing_timing_not_zero(tmp_path):
    root, _ = pair_fixture(tmp_path)
    frozen = json.loads((root / "selection-private.json").read_text())
    frozen["source_manifest"]["ordered_sources"][1]["report_domain"] = "alfworld"
    write(root / "selection-private.json", frozen)
    controls = json.loads((root / "controls-private.json").read_text())
    controls["selection"] = frozen
    write(root / "controls-private.json", controls)
    progress = root / "collection/episodes/rollout-progress.json"
    traces = json.loads(progress.read_text())
    traces["trajectories"][1]["phases"][0]["input_tokens"] = 99
    traces["trajectories"].reverse()
    write(progress, traces)
    value = snapshot(root)
    metrics = value["metrics"]
    assert metrics["domain/alfworld/completed/phase/reasoning/input_tokens/measured_sum"] == 99
    assert metrics["domain/healthbench/completed/phase/reasoning/input_tokens/measured_sum"] == 12
    assert metrics["completed/reasoning_output_tokens"] > 0
    assert metrics["completed/admitted_count"] is None
    assert metrics["completed/phase/reasoning/server_decode_seconds/measured_sum"] is None
    assert metrics["completed/phase/reasoning/server_decode_seconds/measured_requests"] == 0
    assert "synthetic-source" not in json.dumps(value)
    assert not any("loss" in name for name in metrics)


def test_live_partial_denominator_and_partial_event_do_not_fabricate_completion(tmp_path):
    root, _ = pair_fixture(tmp_path)
    (root / "summary.json").unlink()
    (root / "collection/episodes/episode-000001-private.json").unlink()
    (root / "collection/events.jsonl").write_text('{"run_id":')
    metrics = snapshot(root, elapsed_seconds=20)["metrics"]
    assert metrics["planned_sources"] == 2
    assert metrics["completed_sources"] == 1
    assert metrics["pending_sources"] == 1
    assert metrics["completed/reward_mean"] == pytest.approx(0.2)
    assert metrics["completed/success_count"] == 0
    assert metrics["completed_sources_per_minute"] == 3
    assert metrics["rough_remaining_seconds"] == 20
    assert not metrics["complete"]
    assert metrics["completed/admitted_count"] is None
    with pytest.raises(FileNotFoundError):
        snapshot(root)


def test_native_success_submission_and_environment_execution_stay_separate(tmp_path):
    root, _ = pair_fixture(tmp_path)
    events = []
    expected_submissions = 0
    for position in range(2):
        artifact = json.loads(
            (root / f"collection/episodes/episode-{position:06d}-private.json").read_text()
        )["artifact"]
        record = artifact["record"]
        for turn, step in enumerate(record["steps"], 1):
            expected_submissions += 1
            events.append(
                {
                    "run_id": "collection",
                    "payload": {
                        "trajectory_id": record["trajectory_id"],
                        "turn": turn,
                        "action_token_ids": step["action_token_ids"],
                        "observation_text": step["observation_text"],
                        "assessment": {
                            "admitted": True,
                            "executed": False,
                            "accepted_submission": True,
                            "environment_terminal": False,
                            "execution_status": "success",
                        },
                    },
                }
            )
    (root / "collection/events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    metrics = snapshot(root)["metrics"]
    assert metrics["completed/success_count"] == 0
    assert metrics["completed/accepted_submission_count"] == expected_submissions
    assert metrics["completed/executed_count"] == 0
    assert metrics["completed/execution_returned_success_count"] == 0
    assert metrics["completed/admission_validity"] == 1


def test_empty_or_failed_collection_is_unknown_not_zero_task_success(tmp_path):
    root, _ = pair_fixture(tmp_path)
    (root / "summary.json").unlink()
    for position in range(2):
        path = root / f"collection/episodes/episode-{position:06d}-private.json"
        outcome = json.loads(path.read_text())
        outcome.update(
            artifact=None, infrastructure_error="SyntheticFailure", execution_status="failed"
        )
        write(path, outcome)
    metrics = snapshot(root, elapsed_seconds=3)["metrics"]
    assert metrics["infrastructure_failed_sources"] == 2
    assert metrics["completed_sources"] == 0
    assert metrics["completed/success_count"] is None
    assert metrics["completed/reasoning_output_tokens"] is None
    assert metrics["rough_remaining_seconds"] is None
    assert not metrics["complete"]
