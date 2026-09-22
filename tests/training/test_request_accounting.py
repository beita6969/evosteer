import copy
import json
import sqlite3
import zlib

import pytest

from skillev.training.request_accounting import committed_request_accounting
from skillev.training.step_timing import StepTiming
from tests.training.test_committed_evidence import evidence
from tests.training.test_run_observer import observer


def detail(source):
    payload = source["payload"]
    record = payload["records"][0]
    return {
        "trajectories": [
            {
                "trajectory_id": record["trajectory_id"],
                "batch_id": payload["batch_id"],
                "policy_snapshot_id": payload["policy_snapshot_before"],
                "library_version": payload["library_version"],
                "phases": [
                    {
                        "phase": "reasoning",
                        "turn_index": 1,
                        "input_tokens": 10,
                        "output_tokens": 4096,
                        "elapsed_seconds": 9,
                        "client_response_seconds": 8,
                        "server_decode_seconds": None,
                        "server_cached_tokens": 3,
                        "PRIVATE_prompt": "PRIVATE_ANSWER",
                    }
                ],
            }
        ]
    }


def judge_row(db, task="same-task"):
    identity = json.dumps(["judge", task, json.dumps({"rubric": "PRIVATE_RUBRIC"})])
    response = json.dumps(
        {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
            "choices": [{"content": "PRIVATE_ANSWER"}],
        }
    )
    values = (
        identity,
        zlib.compress(b'{"prompt":"PRIVATE_QUESTION"}'),
        "COMPLETED",
        200,
        zlib.compress(json.dumps(response).encode()),
    )
    db.execute("INSERT INTO requests VALUES(?,?,?,?,?)", values)
    return values


def test_thinking_phase_and_actual_journal_measurements_without_double_counting(tmp_path):
    source, args = evidence(tmp_path)
    record = source["payload"]["records"][0]
    record["reward"]["native_payload"]["benchmark_id"] = "aime2026"
    phases = detail(source)
    phases["trajectories"] *= 2  # Same observed episode, not a second request.
    before = copy.deepcopy(phases)
    report = committed_request_accounting(source, rollout_detail=phases, journal=args["requests"])
    assert phases == before
    row = next(
        row
        for row in report["actor_requests"]
        if row["trajectory_id"] == record["trajectory_id"] and row["phase"] == "reasoning"
    )
    assert row["output_tokens"] == 4096
    assert row["elapsed_seconds"] == 9
    assert row["server_decode_seconds"] is None
    assert row["server_cached_tokens"] == 3
    assert len(report["actor_requests"]) == 56
    assert (
        sum(domain.get("distinct_journal_request_count", 0) or 0 for domain in report["domains"])
        == 56
    )
    assert "PRIVATE_" not in json.dumps(report)
    assert all(d.get("physical_http_attempt_count") is None for d in report["domains"])


def test_missing_measurements_are_null_not_zero_costs(tmp_path):
    source, _ = evidence(tmp_path)
    report = committed_request_accounting(source)
    assert report["actor_requests"] == []
    for row in report["domains"]:
        if row["role"] == "actor":
            assert row["phases"] is None
            assert row["logical_phase_count"] is None
            assert row["distinct_journal_request_count"] is None
        else:
            assert row["request_count"] is None
            assert row["seconds"] is None
            assert row["output_tokens"] is None


def test_same_task_judge_rows_are_deduplicated_but_never_assigned_to_rollouts(tmp_path):
    source, args = evidence(tmp_path)
    for record in source["payload"]["records"][:4]:
        record["initial_context"] = {"meta": {"task_id": "same-task"}}
    with sqlite3.connect(args["requests"]) as db:
        values = judge_row(db)
        db.execute("INSERT INTO requests VALUES(?,?,?,?,?)", values)
        judge_row(db, "different-task")
    report = committed_request_accounting(source, journal=args["requests"])
    candidates = report["unassigned_same_task_judge_rows"]
    assert len(candidates) == 1
    assert candidates[0]["input_tokens"] == 21
    assert candidates[0]["cached_tokens"] == 2
    assert candidates[0]["assigned_optimizer_step"] is None
    assert candidates[0]["seconds"] is None
    assert "PRIVATE_" not in json.dumps(report)
    later = copy.deepcopy(source)
    later["payload"]["optimizer_step"] = 2
    # Task-only judge candidate ownership remains unknown even in another step;
    # costs are never added to step totals or multiplied by four trajectories.
    assert (
        committed_request_accounting(later, journal=args["requests"])[
            "unassigned_same_task_judge_rows"
        ]
        == candidates
    )


@pytest.mark.parametrize("field", ["policy_snapshot_id", "library_version", "batch_id"])
def test_phase_cross_policy_or_library_never_silently_mixed(tmp_path, field):
    source, _ = evidence(tmp_path)
    phases = detail(source)
    phases["trajectories"][0][field] = "wrong-original"
    with pytest.raises(ValueError):
        committed_request_accounting(source, rollout_detail=phases)


def test_journal_cross_policy_never_assigned_by_episode_alone(tmp_path):
    source, args = evidence(tmp_path)
    source["payload"]["policy_snapshot_before"] = "other-policy"
    with pytest.raises(ValueError):
        committed_request_accounting(source, journal=args["requests"])


def test_finalized_phases_reach_original_commit_and_persistent_mirror_before_monitor(tmp_path):
    obs, source, args = observer(tmp_path)
    phases = detail(source)
    timing = StepTiming(
        batch_id=source["payload"]["batch_id"],
        optimizer_step=1,
        started=0,
        rollout_finished=10,
        gradient_started=1,
        gradient_finished=12,
        committed=13,
        rollout=None,
        rollout_detail=phases,
    )
    with obs:
        obs.record_timings((timing,))
        assert obs.status()["last_indexed_step"] == 0
        assert not (tmp_path / "performance.jsonl").exists()
        obs.observe_commits(1)
        assert obs.drain()["mirror_counts"]["mirrored"] == 1
        target = tmp_path / "persistent/step-00000001"
        assert json.loads((target / "rollout-phases.json").read_text()) == phases
        assert json.loads((target / "finalized-timing.json").read_text()) == timing.to_value()
        report = json.loads((target / "request-accounting.json").read_text())
        assert report["source_commit_id"] == source["event_id"]
        assert report["actor_requests"][0]["output_tokens"] == 4096
        assert "PRIVATE_" not in (target / "request-accounting.json").read_text()


def test_mirror_preserves_exact_unassigned_judge_rows_not_payload_in_report(tmp_path):
    from skillev.training.evidence_mirror import copy_request_rows

    source, args = evidence(tmp_path)
    with sqlite3.connect(args["requests"]) as db:
        original = judge_row(db)
        source_rowid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        judge_row(db, "unrelated-task")
    target = tmp_path / "subset.sqlite"
    counts = copy_request_rows(
        args["requests"],
        target,
        [source["payload"]["records"][0]["trajectory_id"]],
        additional_request_rowids=(source_rowid,),
    )
    assert counts["requests"] == 3  # R/A plus one distinct original judge row.
    with sqlite3.connect(target) as db:
        saved = db.execute(
            "SELECT * FROM requests WHERE json_extract(identity,'$[0]')='judge'"
        ).fetchall()
        assert saved == [original]
