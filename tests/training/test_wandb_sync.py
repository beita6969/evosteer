import importlib.util
import json
from pathlib import Path
from typing import ClassVar

import pytest

from skillev.training.metrics_contract import TrainingMetricsSnapshot
from tests.training.test_metrics_contract import event


@pytest.fixture
def uploader(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "training_wandb_sync",
        Path(__file__).resolve().parents[2] / "scripts/sync_training_wandb.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wandb_allowlist_excludes_cases_native_single_question_results_and_unknown_fields(
    tmp_path, uploader
):
    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    row["metrics"]["secret"] = "PRIVATE_ANSWER"
    row["native_metrics_by_domain"]["hotpotqa"]["secret"] = "PRIVATE_ANSWER"
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    points = uploader.load_points(path)
    assert len(points) == 1
    assert points[0]["train/batch_success_fraction"] == 0.25
    assert "PRIVATE_ANSWER" not in json.dumps(points)
    assert "hotpotqa" not in json.dumps(points)


def test_wandb_refuses_mixed_runs_and_conflicting_commits(tmp_path, uploader):
    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    path = tmp_path / "metrics.jsonl"
    other = {**row, "run_id": "different"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError):
        uploader.load_points(path)
    other = {**row, "commit_id": "different"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError):
        uploader.load_points(path)


def test_wandb_source_cannot_switch_between_polls_or_restarts(tmp_path, uploader):
    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    path, state = tmp_path / "metrics.jsonl", tmp_path / "source.json"
    path.write_text(json.dumps(row) + "\n")
    uploader.load_points(path, source_state=state)
    assert uploader.load_points(path, source_state=state)
    other = {**row, "run_id": "another-run", "optimizer_step": 2, "sampled_policy_step": 1}
    path.write_text(json.dumps(other) + "\n")
    with pytest.raises(ValueError):
        uploader.load_points(path, source_state=state)


def test_scientific_step_is_published_before_ack_and_resume_does_not_resend(
    tmp_path, uploader, monkeypatch
):
    import sys
    from types import SimpleNamespace

    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(row) + "\n")
    state = tmp_path / "state"
    state.mkdir()
    published, calls = [], []

    class Run:
        url = "https://example.invalid/run"
        summary: ClassVar[dict[str, bool]] = {}

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, point, *, commit=False):
            calls.append(point["optimizer_step"])
            # Commit immediately; SDK's append index is separate from optimizer_step.
            if commit:
                published.append(dict(point))

        def finish(self):
            pass

    remote = SimpleNamespace(scan_history=lambda **kwargs: iter(published))
    sdk = SimpleNamespace(
        Api=lambda: SimpleNamespace(run=lambda path: remote),
        init=lambda **kwargs: Run(),
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    args = SimpleNamespace(
        entity="test",
        project="test",
        run_id="test",
        state=state,
        new=True,
        metrics=metrics,
        follow=False,
        ack_timeout=0,
        quality_root=None,
    )
    uploader.sync(args)
    assert calls == [row["optimizer_step"]]
    assert published[0]["train/ttb_loss"] == row["metrics"]["ttb_loss"]
    assert not (state / "pending.json").exists()

    # A lost local acknowledgment is resolved by the original remote point.
    (state / "pending.json").write_text(json.dumps(published[0]))
    args.new = False
    uploader.sync(args)
    assert calls == [row["optimizer_step"]]
    assert not (state / "pending.json").exists()


def test_remote_ack_accepts_only_identical_repeated_history_rows(uploader):
    from types import SimpleNamespace

    first = {
        "optimizer_step": 1,
        "source_commit_id": "first",
        "_step": 1,
        "_timestamp": 10,
        "train/ttb_loss": 0.7,
    }
    second = {**first, "optimizer_step": 2, "source_commit_id": "second", "_step": 2}
    rows = [first, dict(first), second]
    sdk = SimpleNamespace(
        Api=lambda: SimpleNamespace(
            run=lambda path: SimpleNamespace(scan_history=lambda **kwargs: iter(rows))
        )
    )
    assert uploader.remote_points(sdk, "run") == {1: "first", 2: "second"}
    for changed in ({"train/ttb_loss": 0.8}, {"source_commit_id": "other"}, {"_step": 3}):
        rows[1] = {**first, **changed}
        with pytest.raises(RuntimeError):
            uploader.remote_points(sdk, "run")


def quality_fixture(root, row):
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(
        json.dumps(
            {
                "run_id": row["run_id"],
                "condition_id": row["condition_id"],
                "reference_scope": "owner-selected-composite",
            }
        )
    )
    step = row["optimizer_step"]
    probe = {
        "policy_step": step,
        "policy_snapshot_id": "policy",
        "evidence_id": "probe",
        "condition_id": row["condition_id"],
        "metrics": {"panel/success_fraction": 0.5, "secret": "PRIVATE_ANSWER"},
        "cases": ["PRIVATE_ANSWER"],
    }
    decision = {
        "policy_step": step,
        "policy_snapshot_id": "policy",
        "evidence_ids": ["probe"],
        "status": "warning",
        "action": "continue",
        "triggered_rules": [],
        "metrics_unavailable": [],
    }
    (root / f"probe-{step:08d}.json").write_text(json.dumps(probe))
    (root / f"decisions-{step:08d}.json").write_text(json.dumps([decision]))
    return probe, decision


def test_quality_summary_recovers_when_mirrored_decision_disappears(
    tmp_path, uploader, monkeypatch
):
    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    root = tmp_path / "quality"
    quality_fixture(root, row)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"run_id": row["run_id"], "condition_id": "raw"}))
    step = row["optimizer_step"]
    decision = root / f"decisions-{step:08d}.json"
    original = type(decision).read_text

    def disappearing_read(path, *args, **kwargs):
        if path == decision:
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(decision), "read_text", disappearing_read)
        assert not uploader.load_quality_summaries(
            root, source_state=source, committed_steps={step}
        )
    summary = uploader.load_quality_summaries(root, source_state=source, committed_steps={step})
    assert summary[f"quality/step{step}/success_fraction"] == 0.5
    decision.write_text("{unfinished")
    with pytest.raises(json.JSONDecodeError):
        uploader.load_quality_summaries(root, source_state=source, committed_steps={step})


def test_quality_summary_is_scoped_complete_and_aggregate_only(tmp_path, uploader):
    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    root = tmp_path / "quality"
    probe, decision = quality_fixture(root, row)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"run_id": row["run_id"], "condition_id": "raw"}))
    step = row["optimizer_step"]
    summary = uploader.load_quality_summaries(root, source_state=source, committed_steps={step})
    assert summary[f"quality/step{step}/success_fraction"] == 0.5
    assert summary[f"quality/step{step}/status"] == "warning"
    assert "PRIVATE_ANSWER" not in json.dumps(summary)
    assert not any(k.startswith("train/") for k in summary)
    assert not uploader.load_quality_summaries(root, source_state=source, committed_steps=set())
    path = root / f"decisions-{step:08d}.json"
    path.unlink()
    assert not uploader.load_quality_summaries(root, source_state=source, committed_steps={step})
    decision["policy_snapshot_id"] = "different"
    path.write_text(json.dumps([decision]))
    with pytest.raises(ValueError):
        uploader.load_quality_summaries(root, source_state=source, committed_steps={step})
    source.write_text(json.dumps({"run_id": "other", "condition_id": "raw"}))
    with pytest.raises(ValueError):
        uploader.load_quality_summaries(root, source_state=source, committed_steps={step})


def test_same_writer_keeps_quality_when_next_training_point_is_logged(
    tmp_path, uploader, monkeypatch
):
    import sys
    from types import SimpleNamespace

    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    step = row["optimizer_step"]
    root, state, metrics = tmp_path / "quality", tmp_path / "state", tmp_path / "metrics.jsonl"
    state.mkdir()
    quality_fixture(root, row)
    metrics.write_text(json.dumps(row) + "\n")
    history, server_summary = [], {}

    class Run:
        url = "https://example.invalid/run"

        def __init__(self):
            # Simulate a resumed writer with a stale/empty summary cache.
            self.summary = {}

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, point, *, commit):
            assert commit
            assert not any(k.startswith("quality/") for k in point)
            history.append(dict(point))
            server_summary.clear()
            server_summary.update(self.summary)

        def finish(self):
            server_summary.clear()
            server_summary.update(self.summary)

    sdk = SimpleNamespace(
        Api=lambda: SimpleNamespace(
            run=lambda path: SimpleNamespace(scan_history=lambda **kwargs: iter(history))
        ),
        init=lambda **kwargs: Run(),
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    args = SimpleNamespace(
        entity="test",
        project="test",
        run_id="test",
        state=state,
        new=True,
        metrics=metrics,
        follow=False,
        ack_timeout=0,
        quality_root=root,
    )
    uploader.sync(args)
    key = f"quality/step{step}/success_fraction"
    assert server_summary[key] == 0.5
    next_row = {
        **row,
        "optimizer_step": step + 1,
        "sampled_policy_step": step,
        "commit_id": "next-commit",
    }
    metrics.write_text(json.dumps(row) + "\n" + json.dumps(next_row) + "\n")
    args.new = False
    uploader.sync(args)
    assert server_summary[key] == 0.5
    assert [p["optimizer_step"] for p in history] == [step, step + 1]


def test_ack_uses_fresh_history_and_refuses_a_truncated_scan(uploader):
    from types import SimpleNamespace

    rows = [{"optimizer_step": 1, "source_commit_id": "first"}]
    calls = []

    def scan_history(*, use_cache):
        calls.append(use_cache)
        return iter(rows)

    remote = SimpleNamespace(scan_history=scan_history, summary={"optimizer_step": 2})
    sdk = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda path: remote))
    with pytest.raises(RuntimeError):
        uploader.remote_points(sdk, "run")
    assert calls == [False]
    rows.append({"optimizer_step": 2, "source_commit_id": "second"})
    assert uploader.remote_points(sdk, "run") == {1: "first", 2: "second"}


def test_ack_waits_for_index_lag_without_republishing(uploader, monkeypatch):
    reads = []

    def delayed_history(wandb, path):
        reads.append(path)
        if len(reads) == 1:
            raise uploader.IncompleteRemoteHistoryError("index lag")
        return {1: "first", 2: "second"}

    monkeypatch.setattr(uploader, "remote_points", delayed_history)
    monkeypatch.setattr(uploader.time, "sleep", lambda seconds: None)
    uploader.await_visible(object(), "run", 2, "second", 10)
    assert len(reads) == 2

    def unavailable(wandb, path):
        raise uploader.IncompleteRemoteHistoryError("still incomplete")

    monkeypatch.setattr(uploader, "remote_points", unavailable)
    with pytest.raises(uploader.IncompleteRemoteHistoryError):
        uploader.await_remote_points(object(), "run", 0)

    def conflict(wandb, path):
        raise RuntimeError("conflicting committed evidence")

    monkeypatch.setattr(uploader, "remote_points", conflict)
    with pytest.raises(RuntimeError):
        uploader.await_remote_points(object(), "run", 10)


def test_summary_heartbeat_never_logs_a_training_point(uploader, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(uploader.time, "time", lambda: 1000.0)
    run = SimpleNamespace(summary={})  # No log() method: heartbeat must not use it.
    known = {1: "committed-source"}
    sent = uploader.publish_uploader_heartbeat(run, known=known, last_sent=None, monotonic_now=10.0)
    assert run.summary["telemetry/last_acknowledged_optimizer_step"] == 1
    assert run.summary["telemetry/heartbeat_scope"] == "uploader-not-training-process"
    assert not any(k.startswith("train/") or k == "optimizer_step" for k in run.summary)
    monkeypatch.setattr(uploader.time, "time", lambda: 1100.0)
    sent = uploader.publish_uploader_heartbeat(run, known=known, last_sent=sent, monotonic_now=69.0)
    assert run.summary["telemetry/uploader_heartbeat_unix"] == 1000.0
    sent = uploader.publish_uploader_heartbeat(run, known=known, last_sent=sent, monotonic_now=70.0)
    assert sent == 70.0
    assert run.summary["telemetry/uploader_heartbeat_unix"] == 1100.0
    assert known == {1: "committed-source"}
    empty = SimpleNamespace(summary={})
    uploader.publish_uploader_heartbeat(empty, known={}, last_sent=None, monotonic_now=0)
    assert empty.summary["telemetry/last_acknowledged_optimizer_step"] == 0


def test_declared_branch_remote_history_starts_after_parent_step(uploader):
    from types import SimpleNamespace

    rows = [{"optimizer_step": 3, "source_commit_id": "third"}]
    run = SimpleNamespace(
        config={"first_optimizer_step": 3},
        summary={"optimizer_step": 3},
        scan_history=lambda **kwargs: iter(rows),
    )
    sdk = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda path: run))
    assert uploader.remote_points(sdk, "branch") == {3: "third"}
    rows.append({"optimizer_step": 5, "source_commit_id": "fifth"})
    with pytest.raises(uploader.IncompleteRemoteHistoryError):
        uploader.remote_points(sdk, "branch")
    rows[:] = [{"optimizer_step": 1, "source_commit_id": "parent"}]
    with pytest.raises(uploader.IncompleteRemoteHistoryError):
        uploader.remote_points(sdk, "branch")


def test_expanded_metrics_are_allowlisted_and_unknown_execution_stays_unknown(tmp_path, uploader):
    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    row["format"] = "skillev-training-metrics@2"
    row["telemetry"] = {"posterior_update_event_count": 3, "secret": "PRIVATE_ANSWER"}
    row["metrics"].update(admitted_count=None, executed_count=0, execution_returned_success_count=0)
    row["native_metrics_by_domain"]["hotpotqa"]["secret"] = "PRIVATE_ANSWER"
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row) + "\n")
    ordinary = uploader.load_points(path)[0]
    extra = uploader.training_wandb_telemetry.load_points(path)[0]
    assert ordinary["source_commit_id"] == extra["telemetry/source_commit_id"]
    assert "source_commit_id" not in extra
    assert "optimizer_step" not in extra
    assert extra["posterior/update_event_count"] == 3
    assert extra["train/action_admitted_fraction"] is None
    assert extra["train/action_execution_valid_fraction"] is None
    assert (
        extra["train/domain/hotpotqa/reward_mean"]
        == row["native_metrics_by_domain"]["hotpotqa"]["reward_mean"]
    )
    assert "PRIVATE_ANSWER" not in json.dumps(extra)
    assert extra["phase/trigger_count"] is None
    assert extra["skills/two_distinct_met_fraction"] is None


def test_distinct_skill_target_is_uploaded_without_redefining_action_count(tmp_path, uploader):
    from skillev.training.metrics_telemetry import step_facts
    from tests.training.test_skill_target_metrics import source

    committed = source()
    row = TrainingMetricsSnapshot.from_event(committed, condition_id="two-skills").to_value()
    row["format"] = "skillev-training-metrics@2"
    row["telemetry"] = step_facts(committed)
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert uploader.load_points(path)[0]["train/action_count"] == 84
    extra = uploader.training_wandb_telemetry.load_points(path)[0]
    assert extra["skills/two_distinct_met_fraction"] == 0.5
    assert extra["skills/two_distinct_met_trajectory_count"] == 14
    assert extra["skills/invocation_count"] == 56
    assert "PRIVATE_" not in json.dumps(extra)


def test_telemetry_backfill_resume_and_later_commit_never_repeat_an_optimizer_row(
    tmp_path, uploader, monkeypatch
):
    import sys
    from types import SimpleNamespace

    row = TrainingMetricsSnapshot.from_event(event(), condition_id="raw").to_value()
    step = row["optimizer_step"]
    path, state = tmp_path / "metrics.jsonl", tmp_path / "state"
    state.mkdir()
    path.write_text(json.dumps(row) + "\n")
    history = []

    class Run:
        url = "https://example.invalid/run"
        summary: ClassVar[dict] = {}

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, point, *, commit):
            assert commit
            history.append({**point, "_step": len(history)})

        def finish(self):
            pass

    sdk = SimpleNamespace(
        Api=lambda: SimpleNamespace(
            run=lambda path: SimpleNamespace(scan_history=lambda **kwargs: iter(history))
        ),
        init=lambda **kwargs: Run(),
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    args = SimpleNamespace(
        entity="test",
        project="test",
        run_id="test",
        state=state,
        new=True,
        metrics=path,
        follow=False,
        ack_timeout=0,
        quality_root=None,
    )
    uploader.sync(args)
    row.update(format="skillev-training-metrics@2", telemetry={"skill_invocation_count": 2})
    path.write_text(json.dumps(row) + "\n")
    args.new = False
    uploader.sync(args)
    assert len(history) == 2
    assert uploader.remote_points(sdk, "run") == {step: row["commit_id"]}
    extra = uploader.training_wandb_telemetry.load_points(path)[0]
    (state / "telemetry-pending.json").write_text(json.dumps(extra))
    uploader.sync(args)
    assert len(history) == 2
    assert not (state / "telemetry-pending.json").exists()
    following = {
        **row,
        "optimizer_step": step + 1,
        "sampled_policy_step": step,
        "commit_id": "next",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(following) + "\n")
    uploader.sync(args)
    assert [p["optimizer_step"] for p in history if "source_commit_id" in p] == [step, step + 1]
    assert [
        p["telemetry/optimizer_step"] for p in history if "telemetry/source_commit_id" in p
    ] == [step, step + 1]
    assert [p["_step"] for p in history] == list(range(4))
