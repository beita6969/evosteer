import copy
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from skillev.training import evidence_mirror
from skillev.training.run_observer import CommittedRunObserver
from tests.training.test_committed_evidence import evidence


def observer(tmp_path, *, mirror_root=None, **kwargs):
    source, args = evidence(tmp_path)
    instance = CommittedRunObserver(
        tmp_path,
        run_id=source["run_id"],
        condition_id="new-condition",
        mirror_root=mirror_root or tmp_path / "persistent",
        events=args["events"],
        inflight=args["inflight"],
        requests=args["requests"],
        expected_batch_size=28,
        **kwargs,
    )
    return instance, source, args


def test_only_completed_boundary_indexes_and_copies_exact_original_rows(tmp_path):
    obs, source, args = observer(tmp_path)
    future = copy.deepcopy(source)
    future["event_id"] = "future-commit"
    future["payload"]["optimizer_step"] = 2
    with args["events"].open("a") as stream:
        stream.write(json.dumps(future) + "\n")
    with sqlite3.connect(args["requests"]) as db:
        original = db.execute("SELECT * FROM requests ORDER BY identity").fetchall()
        unrelated = list(original[0])
        unrelated[0] = json.dumps(["unrelated", "1", "action", "policy", "library", "decoding"])
        db.execute("INSERT INTO requests VALUES(?,?,?,?,?)", unrelated)
    with obs:
        before = obs.observe_commits(0)
        assert before["last_indexed_step"] == 0
        assert before["source_commit_id"] is None
        assert not list((tmp_path / "evidence").glob("step-*.json"))
        obs.observe_commits(1)
        status = obs.drain()
        assert status["last_indexed_step"] == 1
        assert status["source_commit_id"] == source["event_id"]
        assert status["committed_at"] == source["occurred_at"]
        assert status["mirror_counts"]["mirrored"] == 1
        assert not status["pause_required"]
        target = tmp_path / "persistent" / "step-00000001"
        assert not (tmp_path / "persistent" / "step-00000002").exists()
        with sqlite3.connect(target / "requests.sqlite3") as db:
            assert db.execute("SELECT * FROM requests ORDER BY identity").fetchall() == original
        source_lines = args["events"].read_bytes().splitlines(keepends=True)
        assert (target / "events.jsonl").read_bytes() == b"".join(source_lines[:-1])
        for artifact in args["inflight"].glob("step-00000001/trajectory-*.json"):
            assert (
                target / "inflight" / "step-00000001" / artifact.name
            ).read_bytes() == artifact.read_bytes()
        assert "PRIVATE_" not in (target / "learning.json").read_text()
        assert "PRIVATE_" not in (target / "actions.json").read_text()
        assert json.loads((target / "mirror.json").read_text())["state"] == "complete"


@pytest.mark.parametrize("already_indexed", [False, True])
def test_declared_prompt_continuation_preserves_historical_evidence(tmp_path, already_indexed):
    obs, source, args = observer(tmp_path)
    with obs:
        if already_indexed:
            obs.observe_commits(1)
            obs.drain()
    historical = tmp_path / "evidence" / "prepared" / "step-00000001" / "learning.json"
    before = historical.read_bytes() if already_indexed else None
    if already_indexed:
        with pytest.raises(ValueError):
            CommittedRunObserver(
                tmp_path,
                run_id=source["run_id"],
                condition_id="proactive-prompt",
                mirror_root=tmp_path / "persistent",
            )
    future = copy.deepcopy(source)
    future["event_id"] = "second-commit"
    future["payload"]["optimizer_step"] = 2
    with args["events"].open("a") as stream:
        stream.write(json.dumps(future) + "\n")
    shutil.copytree(args["inflight"] / "step-00000001", args["inflight"] / "step-00000002")
    with CommittedRunObserver(
        tmp_path,
        run_id=source["run_id"],
        condition_id="proactive-prompt",
        condition_starts={1: "new-condition", 2: "proactive-prompt"},
        mirror_root=tmp_path / "persistent",
        requests=args["requests"],
        expected_batch_size=28,
    ) as resumed:
        assert resumed.observe_commits(2)["last_indexed_step"] == 2
        resumed.drain()
    for step, expected in ((1, "new-condition"), (2, "proactive-prompt")):
        index = json.loads((tmp_path / "evidence" / f"step-{step:08d}.json").read_text())
        learning = json.loads(
            (tmp_path / "evidence" / "prepared" / f"step-{step:08d}" / "learning.json").read_text()
        )
        assert index["condition_id"] == learning["condition_id"] == expected
    if before is not None:
        assert historical.read_bytes() == before


def test_low_space_is_pending_not_mirrored_and_status_does_not_scan_events(tmp_path, monkeypatch):
    obs, _, args = observer(tmp_path, minimum_free_bytes=10)
    actual_usage = evidence_mirror.shutil.disk_usage
    monkeypatch.setattr(
        evidence_mirror.shutil, "disk_usage", lambda _: type(actual_usage(tmp_path))(100, 100, 0)
    )
    with obs:
        obs.observe_commits(1)
        status = obs.drain()
        assert status["mirror_counts"]["pending"] == 1
        assert status["mirror_counts"]["mirrored"] == 0
        assert status["pause_required"]
        original_open = Path.open

        def no_event_reads(path, *a, **kw):
            if path == args["events"]:
                raise AssertionError("monitor status must not rescan events")
            return original_open(path, *a, **kw)

        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "open", no_event_reads)
            assert obs.status()["last_indexed_step"] == 1
        monkeypatch.setattr(evidence_mirror.shutil, "disk_usage", actual_usage)
        obs.retry_mirrors()
        assert obs.drain()["mirror_counts"]["mirrored"] == 1


def test_copy_failure_keeps_partial_and_explicit_retry_preserves_source(tmp_path, monkeypatch):
    obs, _, args = observer(tmp_path)
    source_before = args["events"].read_bytes()
    real_copy = evidence_mirror.copy_file

    def fail(source, destination):
        if source.name == "learning.json":
            raise OSError("synthetic disk failure")
        real_copy(source, destination)

    monkeypatch.setattr(evidence_mirror, "copy_file", fail)
    with obs:
        obs.observe_commits(1)
        status = obs.drain()
        assert status["mirror_counts"]["failed"] == 1
        assert status["pause_required"]
        assert not (tmp_path / "persistent" / "step-00000001").exists()
        partials = list((tmp_path / "persistent").glob(".step-*.partial-*"))
        assert len(partials) == 1
        monkeypatch.setattr(evidence_mirror, "copy_file", real_copy)
        obs.retry_mirrors()
        assert obs.drain()["mirror_counts"]["mirrored"] == 1
        assert partials[0].exists()
        assert args["events"].read_bytes() == source_before


def test_published_destination_without_local_ack_recovers_without_copy(tmp_path, monkeypatch):
    obs, source, args = observer(tmp_path)
    with obs:
        obs.observe_commits(1)
        assert obs.drain()["mirror_counts"]["mirrored"] == 1
    queue = tmp_path / "evidence" / "mirror-queue" / "step-00000001.json"
    row = json.loads(queue.read_text())
    row["state"] = "copying"
    queue.write_text(json.dumps(row))

    def no_copy(*args):
        raise AssertionError("already published complete copy must be reused")

    monkeypatch.setattr(evidence_mirror, "copy_file", no_copy)
    with CommittedRunObserver(
        tmp_path,
        run_id=source["run_id"],
        condition_id="new-condition",
        mirror_root=tmp_path / "persistent",
        requests=args["requests"],
    ) as resumed:
        resumed.observe_commits(1)
        assert resumed.drain()["mirror_counts"]["mirrored"] == 1


def test_missing_raw_artifact_pauses_without_manufacturing_mirror(tmp_path):
    obs, _, args = observer(tmp_path)
    (args["inflight"] / "step-00000001" / "trajectory-000001.json").unlink()
    with obs:
        obs.observe_commits(1)
        status = obs.drain()
        assert status["raw_missing_steps"] == [1]
        assert status["pause_required"]
        assert status["mirror_counts"]["failed"] == 1
        assert not (tmp_path / "persistent" / "step-00000001").exists()
        index = json.loads((tmp_path / "evidence" / "step-00000001.json").read_text())
        assert index["status"] == "RAW_MISSING"


def test_no_complete_commit_is_not_inferred_and_second_worker_is_rejected(tmp_path):
    obs, _, args = observer(tmp_path)
    args["events"].unlink()
    with obs:
        status = obs.observe_commits(1)
        assert status["pause_required"]
        assert status["last_indexed_step"] == 0
        assert status["committed_at"] is None
        with pytest.raises(OSError):
            CommittedRunObserver(
                tmp_path,
                run_id="synthetic-run",
                condition_id="new-condition",
                mirror_root=tmp_path / "persistent",
            )


def test_declared_backlog_limit_can_pause_but_queue_alone_does_not(tmp_path, monkeypatch):
    obs, _, _ = observer(tmp_path, max_pending_steps=0)
    with obs:
        # Deterministically keep a real queue pending until explicitly notified.
        monkeypatch.setattr(obs.mirror, "notify", lambda: None)
        status = obs.observe_commits(1)
        assert status["mirror_counts"]["pending"] == 1
        assert "declared-backlog-limit" in status["pause_reasons"]
        obs.pending_limit = None
        assert not obs.status()["pause_required"]


def test_official_journal_subset_retains_identity_endpoint_blobs_and_route(tmp_path):
    from skillev.runtime.request_journal import DurableRequestJournal

    source, destination = tmp_path / "official.sqlite3", tmp_path / "copy.sqlite3"
    journal = DurableRequestJournal(source)
    for episode in ("selected", "unrelated"):
        identity = (episode, "1", "action", "policy", "library", "decode")
        journal.request(
            identity=identity,
            endpoint="https://example.invalid/generate",
            payload={"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 4}},
            send=lambda: (200, {"output_ids": [3, 4], "meta_info": {"finish_reason": "stop"}}),
        )
        journal.save_episode_route(episode, "policy", "https://example.invalid/generate")
    counts = evidence_mirror.copy_request_rows(source, destination, ["selected"])
    assert counts["requests"] == counts["episode_routes"] == 1
    with sqlite3.connect(source) as original, sqlite3.connect(destination) as copied:
        expected = original.execute(
            "SELECT * FROM requests WHERE json_extract(identity,'$[0]')='selected'"
        ).fetchall()
        assert copied.execute("SELECT * FROM requests").fetchall() == expected
        assert copied.execute("SELECT episode FROM episode_routes").fetchall() == [("selected",)]
        assert original.execute("SELECT count(*) FROM requests").fetchone()[0] == 2


def test_queue_never_says_mirrored_before_atomic_publish(tmp_path, monkeypatch):
    obs, _, _ = observer(tmp_path)
    actual_rename = evidence_mirror.os.rename
    inspected = []

    def rename(stage, final):
        queue = tmp_path / "evidence" / "mirror-queue" / "step-00000001.json"
        assert json.loads(queue.read_text())["state"] == "copying"
        assert json.loads((stage / "mirror.json").read_text())["state"] == "complete"
        assert not final.exists()
        inspected.append(True)
        return actual_rename(stage, final)

    monkeypatch.setattr(evidence_mirror.os, "rename", rename)
    with obs:
        obs.observe_commits(1)
        assert obs.drain()["mirror_counts"]["mirrored"] == 1
        assert inspected == [True]


def test_missing_queue_after_index_commit_is_recreated_on_resume(tmp_path, monkeypatch):
    obs, source, args = observer(tmp_path)
    with obs:
        monkeypatch.setattr(obs.mirror, "notify", lambda: None)
        obs.observe_commits(1)
        (tmp_path / "evidence" / "mirror-queue" / "step-00000001.json").unlink()
    with CommittedRunObserver(
        tmp_path,
        run_id=source["run_id"],
        condition_id="new-condition",
        mirror_root=tmp_path / "persistent",
        requests=args["requests"],
    ) as resumed:
        resumed.observe_commits(1)
        assert resumed.drain()["state"] == "mirrored"


def test_restored_original_artifact_can_be_mirrored_without_rewriting_initial_missing_index(
    tmp_path,
):
    obs, _, args = observer(tmp_path)
    path = args["inflight"] / "step-00000001" / "trajectory-000001.json"
    original = path.read_bytes()
    path.unlink()
    with obs:
        obs.observe_commits(1)
        assert obs.drain()["state"] == "RAW_MISSING"
        initial = (tmp_path / "evidence" / "step-00000001.json").read_bytes()
        path.write_bytes(original)  # restore the original bytes, not a regenerated rollout
        obs.retry_mirrors()
        status = obs.drain()
        assert status["state"] == "mirrored"
        assert status["initial_index_raw_missing_steps"] == [1]
        assert status["raw_missing_steps"] == []
        assert not status["pause_required"]
        assert (tmp_path / "evidence" / "step-00000001.json").read_bytes() == initial


def test_smaller_current_batch_recovers_original_full_batch(tmp_path):
    source, args = evidence(tmp_path)
    with CommittedRunObserver(
        tmp_path,
        run_id=source["run_id"],
        condition_id="six-domain",
        condition_starts={1: "original-seven", 2: "six-domain"},
        batch_size_starts={1: 28, 2: 24},
        expected_batch_size=24,
        mirror_root=tmp_path / "persistent",
        events=args["events"],
        inflight=args["inflight"],
        requests=args["requests"],
    ) as obs:
        assert obs.observe_commits(1)["last_indexed_step"] == 1
        assert not obs.drain()["pause_required"]
    index = json.loads((tmp_path / "evidence/step-00000001.json").read_text())
    assert index["trajectory_count"] == 28
    assert index["condition_id"] == "original-seven"
