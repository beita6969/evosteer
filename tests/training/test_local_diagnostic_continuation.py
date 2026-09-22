"""Append-only declaration/ledger tests using synthetic private evidence."""

import asyncio
import json
import sqlite3
import zlib
from dataclasses import replace

import pytest
from skillev_private.experiments import local_diagnostic_continuation as continuation
from skillev_private.experiments import local_training_diagnostic as diagnostic
from skillev_private.experiments.training_data_condition import data_condition_scientific

from skillev.runtime import BudgetLedger, BudgetVector
from tests.training.test_local_training_diagnostic import CONFIG, bindings_file


def budget_source(root):
    directory = root / "inflight/step-00000001"
    directory.mkdir(parents=True)
    maximum = BudgetVector(input_tokens=40, output_tokens=20, model_calls=1).to_value()
    actual = BudgetVector(input_tokens=10, output_tokens=5, model_calls=1).to_value()
    charge = {
        "reservation_id": "actor-call",
        "invocation_id": "trajectory",
        "maximum": maximum,
        "actual": actual,
        "run_id": root.name,
        "attempt_id": "diagnostic-seed0",
    }
    (directory / "batch.json").write_text(
        json.dumps({"rollouts": [{"position": 0, "trajectory_id": "trajectory"}]})
    )
    (directory / "trajectory-000000.json").write_text(
        json.dumps(
            {"artifact": {"record": {"trajectory_id": "trajectory"}}, "budget_entries": [charge]}
        )
    )
    db = sqlite3.connect(root / "requests.sqlite3")
    db.execute("CREATE TABLE requests(identity TEXT, payload BLOB, state TEXT)")
    for identity in (["actor", "trajectory", "actor-call"], ["author", root.name, "author-call"]):
        db.execute(
            "INSERT INTO requests VALUES(?,?,?)",
            (
                json.dumps(identity),
                zlib.compress(json.dumps({"sampling_params": {"sampling_seed": 17}}).encode()),
                "COMPLETED",
            ),
        )
    db.commit()
    db.close()
    events = []
    for rid in ("actor-call", "author-call"):
        for kind, payload in (
            ("budget_reserved", {"reservation_id": rid, "maximum": maximum}),
            ("budget_settled", {"reservation_id": rid, "actual": actual}),
        ):
            events.append(
                {
                    "run_id": root.name,
                    "attempt_id": "diagnostic-seed0",
                    "event_type": kind,
                    "payload": payload,
                }
            )
    path = root / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path, events


def test_original_actor_and_author_charges_restore_exactly_once_without_republication(tmp_path):
    path, _ = budget_source(tmp_path)
    original = path.read_bytes()
    entries = continuation.completed_budget(tmp_path, 1)
    assert [e.reservation.invocation_id for e in entries] == ["trajectory", "phi-authoring:17"]
    ledger = BudgetLedger(
        run_id=tmp_path.name,
        attempt_id="diagnostic-seed0",
        cap=BudgetVector(input_tokens=100, output_tokens=100, model_calls=10),
    )
    ledger.restore_completed(entries)
    assert ledger.settled == BudgetVector(input_tokens=20, output_tokens=10, model_calls=2)
    with pytest.raises(RuntimeError):
        ledger.restore_completed(entries)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "problem", ["unknown-request", "unsettled", "later-commit", "changed-charge"]
)
def test_ambiguous_old_work_is_rejected_not_reexecuted(tmp_path, problem):
    path, events = budget_source(tmp_path)
    if problem == "unknown-request":
        with sqlite3.connect(tmp_path / "requests.sqlite3") as db:
            db.execute("UPDATE requests SET state='DISPATCHED'")
    elif problem == "unsettled":
        events.pop()
    elif problem == "later-commit":
        events.append(
            {**events[0], "event_type": "training_step_committed", "payload": {"optimizer_step": 2}}
        )
    else:
        events[1]["payload"]["actual"]["output_tokens"] = 99
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    original = path.read_bytes()
    with pytest.raises(ValueError):
        continuation.completed_budget(tmp_path, 1)
    assert path.read_bytes() == original


def test_task_prefix_and_method_changes_reject_before_checkpoint_loading(tmp_path, monkeypatch):
    from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records

    from tests.benchmarks.test_protocol_v13_training import _sources

    config = diagnostic.diagnostic_config(diagnostic.load_fresh_config(CONFIG), total_steps=10)
    bindings = diagnostic.load_bindings(bindings_file(tmp_path))
    rows = build_protocol13_training_records(_sources())
    selected = diagnostic.seven_domain_training_trajectories(rows, steps=10)
    prefix = selected[:84]
    root = tmp_path / "original"
    root.mkdir()
    declaration = {
        "format": diagnostic.FORMAT,
        "formal": replace(config, steps=3).to_value(),
        **data_condition_scientific(bindings.data_condition, prefix),
    }
    (root / "diagnostic-condition.json").write_text(json.dumps(declaration))
    called = []
    monkeypatch.setattr(
        continuation, "FilesystemTrainingCheckpointStore", lambda **_: called.append("checkpoint")
    )
    for changed_config, changed_records in (
        (replace(config, max_action_tokens=config.max_action_tokens + 1), selected),
        (config, (selected[4], *selected[1:])),
    ):
        with pytest.raises(ValueError):
            continuation.require_append(
                root=root,
                resume=root / "checkpoints/step3",
                config=changed_config,
                bindings=bindings,
                selected=changed_records,
            )
    assert not called


@pytest.mark.parametrize(("resume", "steps"), [(None, 10), ("old46", 3), ("old8", 5)])
def test_cli_never_turns_append_authorization_into_a_fresh_ten_step_run(tmp_path, resume, steps):
    with pytest.raises(ValueError):
        asyncio.run(
            diagnostic.run(
                config_path=tmp_path / "absent",
                bindings_path=tmp_path / "absent",
                root=tmp_path / "absent",
                resume=resume,
                total_steps=steps,
            )
        )


def test_local_resume_uses_exact_application_restore_and_real_journal(tmp_path, monkeypatch):
    from tests.training.test_local_training_diagnostic import local_runtime

    runtime = local_runtime(tmp_path)
    captured = {}
    monkeypatch.setattr(diagnostic.SKILLEVApplication, "resume", lambda **kw: captured.update(kw))
    snapshot = tmp_path / "snapshots/step3"
    diagnostic.build_application(
        runtime=runtime,
        storage=diagnostic.PrivateCheckpointStorageBinding(directory=str(snapshot.parent)),
        resume=snapshot,
        plan_continuation="explicit-fixture",
    )
    assert captured["snapshot_directory"] == snapshot
    assert captured["gradient_preparer"] is None
    assert captured["plan_continuation"] == "explicit-fixture"
    assert captured["step_transaction_journal"].directory == snapshot.parent / "step-transactions"


def test_loading_controller_reports_proven_step_three_not_fresh_zero(tmp_path):
    from skillev_private.experiments.training_controller import TrainingController

    controller = TrainingController(
        tmp_path, resource_roles={}, gpu_uuids=(), initial_committed_step=3
    )
    try:
        status = json.loads((tmp_path / "controller-status.json").read_text())
        assert status["last_committed_step"] == 3
        assert status["state"] == "starting"
    finally:
        controller.close_resources()
