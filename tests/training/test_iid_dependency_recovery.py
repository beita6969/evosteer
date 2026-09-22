import json
import sqlite3

import pytest
from skillev_private.evaluation.iid_dependency_recovery import prepare_dependency_recovery

from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_run(tmp_path):
    source = tmp_path / "original"
    episodes = source / "episodes"
    controls = {
        "frozen": "synthetic",
        "panel": {"records": [{"source": {"benchmark": d}} for d in ("hotpotqa", "healthbench")]},
    }
    write(source / "expanded-controls-private.json", controls)
    write(episodes / "plan-private.json", {"chunk_size": 32})
    rows = [
        {
            "canonical_position": i,
            "trajectory_id": f"episodes-eval-{i:06d}",
            "stage": "artifact-ready" if i == 0 else "failed",
            "task_domain": "hotpotqa/multi-hop-qa" if i == 0 else "healthbench/health-dialogue",
        }
        for i in range(2)
    ]
    write(episodes / "rollout-progress.json", {"status": "failed", "trajectories": rows})
    for row in rows:
        write(
            episodes / f"started-episode-{row['canonical_position']:06d}.json",
            {"position": row["canonical_position"], "trajectory_id": row["trajectory_id"]},
        )
    write(
        episodes / "failure-episode-000001-private.json",
        {
            "position": 1,
            "task_id": "healthbench/synthetic",
            "error_type": "TerminalEvaluatorError",
            "traceback": "_load_official\nModuleNotFoundError: No module named 'blobfile'",
        },
    )
    ledger = tmp_path / "original-evaluation-requests-healthbench-criteria" / "case.json"
    write(
        ledger,
        {
            "binding": {"task_id": "healthbench/synthetic"},
            "status": "incomplete-grading",
            "error_type": "ModuleNotFoundError",
            "requests": [],
            "result": None,
        },
    )
    journal = DurableRequestJournal(tmp_path / "original-evaluation-requests.sqlite3")
    for row in rows:
        journal.request(
            identity=(row["trajectory_id"], "0", "action"),
            endpoint="synthetic://owner",
            payload={"input": row["trajectory_id"]},
            send=lambda: (200, {"text": "original model response"}),
        )
    return source, controls, ledger, journal.path


def test_recovery_restores_original_response_and_only_unstarted_can_dispatch(tmp_path):
    source, controls, _, original = source_run(tmp_path)
    journal = prepare_dependency_recovery(
        source, tmp_path / "recovered", expanded=controls, chunk_size=32
    )
    calls = []

    def dispatch():
        calls.append("new")
        return 200, {"text": "new response"}

    kwargs = {
        "identity": ("episodes-eval-000001", "0", "action"),
        "endpoint": "synthetic://owner",
        "payload": {"input": "episodes-eval-000001"},
        "send": dispatch,
    }
    _, restored = journal.request(**kwargs)
    assert restored["text"] == "original model response"
    assert restored["skillev_restored_response"] is True
    assert not calls
    with pytest.raises(ValueError):
        journal.request(**{**kwargs, "payload": {"input": "changed"}})
    with pytest.raises(UnknownRequestOutcomeError):
        journal.request(**{**kwargs, "identity": ("episodes-eval-000001", "1", "action")})
    assert not calls
    journal.request(**{**kwargs, "identity": ("episodes-eval-000002", "0", "action")})
    assert calls == ["new"]
    with sqlite3.connect(original) as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 2


@pytest.mark.parametrize("bad_case", ["judge-sent", "owner-unknown", "environment", "controls"])
def test_recovery_rejects_ambiguous_or_changed_source(tmp_path, bad_case):
    source, controls, ledger, original = source_run(tmp_path)
    if bad_case == "judge-sent":
        value = json.loads(ledger.read_text())
        value["requests"] = [{"status": "DISPATCHED"}]
        write(ledger, value)
    elif bad_case == "owner-unknown":
        with sqlite3.connect(original) as db:
            db.execute("UPDATE requests SET state='DISPATCHED' WHERE rowid=1")
    elif bad_case == "environment":
        path = source / "episodes/rollout-progress.json"
        value = json.loads(path.read_text())
        value["trajectories"][0]["task_domain"] = "alfworld/embodied-navigation"
        controls["panel"]["records"][0]["source"]["benchmark"] = "alfworld"
        write(source / "expanded-controls-private.json", controls)
        write(path, value)
    else:
        controls = {"frozen": "changed"}
    with pytest.raises(ValueError):
        prepare_dependency_recovery(
            source, tmp_path / "recovered", expanded=controls, chunk_size=32
        )
