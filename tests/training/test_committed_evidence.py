import copy
import json
import sqlite3
import zlib

import pytest

from skillev.training.committed_evidence import (
    export_committed_step_evidence,
    index_committed_event,
    mirror_backlog_status,
    write_committed_index,
)
from tests.training.test_metrics_contract import event


def evidence(tmp_path):
    source = event()
    source["payload"]["library_version"] = "library-original"
    for record in source["payload"]["records"]:
        record["decoding_snapshot_id"] = "decoding-original"
        record["steps"][0].update(action_token_ids=[4, 5], observation_text="public result")
    events = tmp_path / "events.jsonl"
    action_events = [
        {
            "event_type": "agent_step_recorded",
            "event_id": f"action-{i}",
            "run_id": source["run_id"],
            "payload": {
                "trajectory_id": r["trajectory_id"],
                "turn": 1,
                "action_token_ids": [4, 5],
                "observation_text": "public result",
                "assessment": {"admitted": True, "executed": True},
            },
        }
        for i, r in enumerate(source["payload"]["records"])
    ]
    events.write_text("".join(json.dumps(e) + "\n" for e in [*action_events, source]))
    directory = tmp_path / "inflight" / "step-00000001"
    directory.mkdir(parents=True)
    dbfile = tmp_path / "requests.sqlite"
    with sqlite3.connect(dbfile) as db:
        db.execute(
            "CREATE TABLE requests(identity TEXT, payload BLOB, state TEXT, "
            "status INT, response BLOB)"
        )
        for position, record in enumerate(source["payload"]["records"], 1):
            artifact = {
                "record": record,
                "manifest": {
                    "policy_snapshot": {"snapshot_id": source["payload"]["policy_snapshot_before"]},
                    "library_version": "library-original",
                },
            }
            (directory / f"trajectory-{position:06d}.json").write_text(
                json.dumps({"artifact": artifact})
            )
            for phase in ("reasoning", "action"):
                identity = [
                    record["trajectory_id"],
                    "1",
                    phase,
                    source["payload"]["policy_snapshot_before"],
                    "library-original",
                    "decoding-original",
                ]
                db.execute(
                    "INSERT INTO requests VALUES(?,?,?,?,?)",
                    (
                        json.dumps(identity),
                        zlib.compress(json.dumps({"input_ids": [1, 2, 3]}).encode()),
                        "COMPLETED",
                        200,
                        zlib.compress(
                            json.dumps(
                                {"output_ids": [4, 5], "meta_info": {"finish_reason": "length"}}
                            ).encode()
                        ),
                    ),
                )
    return source, {
        "events": events,
        "event_offset": None,
        "inflight": directory.parent,
        "requests": dbfile,
        "condition_id": "new-condition",
        "expected_batch_size": 28,
    }


def test_original_b28_raw_responses_indexed_without_parsing_or_mirroring(tmp_path):
    source, args = evidence(tmp_path)
    result = index_committed_event(source, **args)
    assert result["status"] == "RAW_AVAILABLE"
    assert result["trajectory_count"] == 28
    assert result["action_count"] == 28
    assert sum(len(t["request_records"]) for t in result["trajectories"]) == 56
    assert "PRIVATE_" not in json.dumps(result)
    assert result["mirror_status"] == "not-observed"
    root = tmp_path / "index"
    path = write_committed_index(result, root)
    receipt = root / "mirror-queue" / path.name
    receipt.unlink()  # crash after index, before mirror enqueue
    write_committed_index(result, root)
    assert receipt.exists()
    assert mirror_backlog_status(root, max_pending_steps=0)["pause_before_next_batch"]
    raw = json.loads(receipt.read_text())
    raw["state"] = "mirrored"
    receipt.write_text(json.dumps(raw))
    write_committed_index(result, root)  # must not reset successful mirror receipt
    assert not mirror_backlog_status(root, max_pending_steps=0)["pause_before_next_batch"]
    assert mirror_backlog_status(root, minimum_free_bytes=1)["pause_before_next_batch"]


def test_aggregate_only_missing_objects_and_unknown_response_are_not_substituted(tmp_path):
    source, args = evidence(tmp_path)
    (args["inflight"] / "step-00000001" / "trajectory-000001.json").unlink()
    with sqlite3.connect(args["requests"]) as db:
        db.execute("UPDATE requests SET state='DISPATCHED',response=NULL WHERE rowid=1")
    result = index_committed_event(source, **args)
    assert result["status"] == "RAW_MISSING"
    assert result["trajectories"][0]["request_records"][1]["status"] == "RAW_AVAILABLE"
    assert result["trajectories"][0]["request_records"][0]["status"] == "OUTCOME_UNKNOWN"
    args.pop("event_offset")
    exported = export_committed_step_evidence(**args, first_step=1, last_step=2)
    assert exported["steps"][1]["status"] == "RAW_MISSING"
    assert "trajectories" not in exported["steps"][1]


@pytest.mark.parametrize(
    "bad", ["snapshot", "duplicate-response", "duplicate-trajectory", "population"]
)
def test_wrong_original_identity_and_duplicate_response_fail(tmp_path, bad):
    source, args = evidence(tmp_path)
    if bad == "snapshot":
        source["payload"]["policy_snapshot_before"] = "another-policy"
    elif bad == "duplicate-trajectory":
        source["payload"]["records"][1] = copy.deepcopy(source["payload"]["records"][0])
    elif bad == "population":
        args["expected_batch_size"] = 27
    else:
        with sqlite3.connect(args["requests"]) as db:
            db.execute("INSERT INTO requests SELECT * FROM requests WHERE rowid=1")
    with pytest.raises(ValueError):
        index_committed_event(source, **args)


def test_missing_source_file_reports_raw_missing_not_examples(tmp_path):
    result = export_committed_step_evidence(
        events=tmp_path / "absent",
        inflight=tmp_path,
        requests=None,
        condition_id="new",
        first_step=45,
        last_step=46,
    )
    assert [r["optimizer_step"] for r in result["steps"]] == [45, 46]
    assert all(r["status"] == "RAW_MISSING" and "trajectories" not in r for r in result["steps"])


def test_no_action_ids_cannot_claim_execution_evidence(tmp_path):
    source, args = evidence(tmp_path)
    record = source["payload"]["records"][0]
    del record["steps"][0]["action_token_ids"]
    path = args["inflight"] / "step-00000001" / "trajectory-000001.json"
    raw = json.loads(path.read_text())
    raw["artifact"]["record"] = record
    path.write_text(json.dumps(raw))
    result = index_committed_event(source, **args)
    assert result["status"] == "RAW_MISSING"
    assert result["trajectories"][0]["action_records"][0]["assessment_reference"] is None
