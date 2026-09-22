import json
from types import SimpleNamespace

import pytest

from skillev.training.metric_events import EVENT_CONTRACT, native_fields, supplemental_point
from skillev.training.metrics_contract import TrainingMetricsSnapshot
from tests.training.test_metrics_contract import event

pytest_plugins = ("tests.training.test_wandb_sync",)


def test_native_allowlist_aliases_unknown_and_negative_rubric():
    source = event()
    first = source["payload"]["records"][0]["reward"]["native_payload"]
    first["public_metrics"]["answer-f1"] = None
    first["public_metrics"]["f1"] = 1.0  # not substituted for explicit null
    first["public_metrics"]["secret"] = "PRIVATE_ANSWER"
    hb = source["payload"]["records"][1]["reward"]["native_payload"]
    hb["benchmark_id"] = "healthbench"
    hb["public_metrics"] = {"luna-medium-api-rubric-score": -0.2}
    fields = native_fields(source)
    assert fields["train/domain/hotpotqa/f1"] is None
    assert fields["train/domain/hotpotqa/f1_observed_count"] == 26
    assert fields["train/domain/healthbench/luna_medium_rubric_score"] == -0.2
    assert fields["train/domain/healthbench/qwen_local_rubric_score"] is None
    assert fields["train/domain/triviaqa/em"] is None
    assert "PRIVATE_" not in json.dumps(fields)


def test_new_contract_groups_source_aliases_and_preserves_null_metric_objects():
    source = event()
    records = source["payload"]["records"]
    for i, record in enumerate(records):
        native = record["reward"]["native_payload"]
        native["public_metrics"] = None
        native["training_evidence_source"] = {
            "benchmark_id": "hotpotqa",
            "source_question_id": "one-original-source",
            "population_id": "first-alias" if i % 2 else "second-alias",
        }
    fields = native_fields(source)
    assert fields["train/unique_source_question_count"] == 1
    assert fields["train/source_population_membership_count"] == 2
    assert fields["train/unknown_source_trajectory_count"] == 0
    assert fields["train/domain/hotpotqa/f1"] is None
    del records[0]["reward"]["native_payload"]["training_evidence_source"]
    assert native_fields(source)["train/unknown_source_trajectory_count"] == 1


def quality_fixture(root):
    root.mkdir()
    (root / "source.json").write_text(
        json.dumps(
            {"run_id": "synthetic-run", "condition_id": "raw", "reference_scope": "fixed-held-out"}
        )
    )
    (root / "probe-00000000.json").write_text(
        json.dumps(
            {
                "policy_step": 0,
                "condition_id": "raw",
                "policy_snapshot_id": "initial",
                "evidence_id": "quality0",
                "metrics": {
                    "panel/success_fraction": 0.5,
                    "hotpotqa/answer-exact-match": 0,
                    "hotpotqa/answer-f1": None,
                    "secret": "PRIVATE_ANSWER",
                },
            }
        )
    )
    (root / "decisions-00000000.json").write_text(
        json.dumps(
            [
                {
                    "policy_step": 0,
                    "policy_snapshot_id": "initial",
                    "evidence_ids": ["quality0"],
                    "status": "verified",
                    "action": "continue",
                    "triggered_rules": [],
                    "metrics_unavailable": [],
                }
            ]
        )
    )


def fake_sdk(monkeypatch, *, published, definitions):
    import sys

    class Run:
        url = "https://example.invalid/run"

        def __init__(self):
            self.summary = {}

        def define_metric(self, name, **kwargs):
            definitions.append((name, kwargs))

        def log(self, point, *, commit):
            assert commit
            published.append(dict(point, _step=69 + len(published)))

        def finish(self):
            pass

    run = Run()
    remote = SimpleNamespace(scan_history=lambda **kwargs: iter(published), config={})

    def initialize(**kwargs):
        remote.config.update(kwargs.get("config", {}))
        return run

    sdk = SimpleNamespace(
        Api=lambda: SimpleNamespace(run=lambda path: remote),
        init=initialize,
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk, run


def test_quality_zero_before_first_commit_combined_metrics_and_resume(
    tmp_path, uploader, monkeypatch
):
    published, definitions = [], []
    fake_sdk(monkeypatch, published=published, definitions=definitions)
    quality_fixture(tmp_path / "quality")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")
    state = tmp_path / "state"
    state.mkdir()
    args = SimpleNamespace(
        entity="test",
        project="test",
        run_id="new-run",
        state=state,
        new=True,
        metrics=metrics,
        follow=False,
        ack_timeout=0,
        quality_root=tmp_path / "quality",
        event_contract=EVENT_CONTRACT,
        source_run_id="synthetic-run",
        condition_id="raw",
        events=None,
    )
    uploader.sync(args)
    assert len(published) == 1
    assert published[0]["record_kind"] == "quality_evaluation"
    assert published[0]["quality/policy_step"] == 0
    assert "optimizer_step" not in published[0]
    assert published[0]["quality/hotpotqa_em"] == 0
    assert published[0]["quality/hotpotqa_f1"] is None
    assert "PRIVATE_" not in json.dumps(published)
    # The next poll sees a complete first commit with timing already available.
    source = event()
    row = TrainingMetricsSnapshot.from_event(source, condition_id="raw").to_value()
    row.update(format="skillev-training-metrics@2", telemetry={"step_wall_seconds": 12.0})
    metrics.write_text(json.dumps(row) + "\n")
    args.events = tmp_path / "events.jsonl"
    args.events.write_text(json.dumps(source) + "\n")
    args.new = False
    uploader.sync(args)
    assert len(published) == 2
    assert published[1]["record_kind"] == "committed_training_step"
    assert published[1]["train/optimizer_step"] == 1
    assert published[1]["timing/step_wall_seconds"] == 12
    assert published[1]["train/domain/hotpotqa/em"] == 0.25
    assert published[1]["_step"] == 70  # SDK index does not change scientific step.
    uploader.sync(args)
    assert len(published) == 2
    assert all(
        options.get("step_sync") is False for _, options in definitions if "step_metric" in options
    )
    assert ("*", {"step_metric": "optimizer_step"}) not in definitions


def test_late_step45_metrics_use_backfill_names_not_train_axis(tmp_path, uploader, monkeypatch):
    published, definitions = [], []
    sdk, run = fake_sdk(monkeypatch, published=published, definitions=definitions)
    import training_wandb_telemetry as telemetry

    point = supplemental_point(
        {
            "telemetry/optimizer_step": 45,
            "telemetry/source_commit_id": "c45",
            "train/reward_mean": 0.5,
            "timing/step_wall_seconds": 90,
        }
    )
    publisher = telemetry.Publisher(sdk, "test/run", tmp_path, 0)
    publisher.publish(run, [point], {45: "c45"})
    assert published[0]["_step"] == 69
    assert published[0]["telemetry/optimizer_step"] == 45
    assert "optimizer_step" not in published[0]
    assert "train/reward_mean" not in published[0]
    assert published[0]["telemetry/train/reward_mean"] == 0.5
    assert all(name.startswith("telemetry/") for name, _ in definitions)
    publisher.publish(run, [point], {45: "c45"})
    assert len(published) == 1


def test_quality_unknown_ack_retains_pending_and_never_resends(tmp_path, uploader, monkeypatch):
    import training_wandb_events as events

    published, definitions = [], []
    sdk, run = fake_sdk(monkeypatch, published=published, definitions=definitions)
    point = {
        "record_kind": "quality_evaluation",
        "quality/policy_step": 0,
        "quality/evidence_id": "q0",
    }
    publisher = events.QualityPublisher(sdk, "run", tmp_path, 0)
    monkeypatch.setattr(publisher, "remote", dict)
    with pytest.raises(RuntimeError):
        publisher.publish(run, [point])
    assert (tmp_path / "quality-pending.json").exists()
    resumed = events.QualityPublisher(sdk, "run", tmp_path, 0)
    resumed.publish(run, [point])
    assert len(published) == 1


def test_uploader_heartbeat_does_not_mark_missing_controller_alive(tmp_path, uploader, monkeypatch):
    import training_wandb_events as events

    published, definitions = [], []
    _, run = fake_sdk(monkeypatch, published=published, definitions=definitions)
    events.heartbeat(
        run,
        known={1: "c1"},
        controller=tmp_path / "absent",
        source_run_id="run",
        stale_after=30,
        progress_after=300,
        last_sent=None,
        monotonic_now=0,
    )
    row = published[0]
    assert row["record_kind"] == "uploader_heartbeat"
    assert row["heartbeat/controller_state"] == "unknown"
    assert row["heartbeat/upload_lag_steps"] is None
    assert "optimizer_step" not in row
    assert "train/optimizer_step" not in row


def test_declared_native_source_waits_for_complete_matching_commit(tmp_path, uploader, monkeypatch):
    published, definitions = [], []
    fake_sdk(monkeypatch, published=published, definitions=definitions)
    source = event()
    row = TrainingMetricsSnapshot.from_event(source, condition_id="raw").to_value()
    metrics, events = tmp_path / "metrics.jsonl", tmp_path / "events.jsonl"
    metrics.write_text(json.dumps(row) + "\n")
    events.write_text(json.dumps(source))  # last line not durable/complete yet
    state = tmp_path / "state"
    state.mkdir()
    args = SimpleNamespace(
        entity="test",
        project="test",
        run_id="new",
        state=state,
        new=True,
        metrics=metrics,
        follow=False,
        ack_timeout=0,
        quality_root=None,
        event_contract=EVENT_CONTRACT,
        source_run_id="synthetic-run",
        condition_id="raw",
        events=events,
    )
    uploader.sync(args)
    assert published == []
    events.write_text(json.dumps(source) + "\n")
    args.new = False
    uploader.sync(args)
    assert len(published) == 1
    assert published[0]["train/domain/hotpotqa/f1"] == 0.5


def test_new_axes_cannot_migrate_a_legacy_remote_run(tmp_path, uploader, monkeypatch):
    published, definitions = [], []
    fake_sdk(monkeypatch, published=published, definitions=definitions)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")
    state = tmp_path / "state"
    state.mkdir()
    args = SimpleNamespace(
        entity="test",
        project="test",
        run_id="old",
        state=state,
        new=False,
        metrics=metrics,
        follow=False,
        ack_timeout=0,
        quality_root=None,
        event_contract=EVENT_CONTRACT,
        source_run_id="synthetic-run",
        condition_id="new",
        events=None,
    )
    with pytest.raises(ValueError):
        uploader.sync(args)
    assert published == []
