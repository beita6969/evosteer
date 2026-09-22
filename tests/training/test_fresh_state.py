"""Actual tiny application state; no fresh labels can hide trained tensors."""

import asyncio
import json
import sqlite3
from dataclasses import replace

import pytest
import torch

from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training.fresh_state import (
    FreshNamespaces,
    FreshStateError,
    inspect_fresh_state,
    require_fresh_state,
    save_fresh_state_report,
)
from tests.application.test_full_vertical_loop import build_application_fixture


@pytest.fixture
def fresh(tmp_path, training_backbone_config):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    namespaces = FreshNamespaces(
        (tmp_path / "requests.sqlite3",), (tmp_path / "evidence", tmp_path / "inflight")
    )
    DurableRequestJournal(namespaces.request_journals[0])
    # Independent reference captured before the candidate can be exercised or
    # deliberately contaminated below; no frozen model state is copied.
    inputs = {
        "initial_library": app.library.state,
        "namespaces": namespaces,
        "preparation_state": read_policy_checkpoint_state(fixture.initial_policy_directory),
        "initial_parameters": {
            name: p.detach().cpu().clone()
            for name, p in app.backbone.named_trainable_parameters().items()
        },
    }
    return fixture, inputs


def test_actual_fresh_start_is_readonly_and_saved_exclusively(fresh, tmp_path):
    fixture, inputs = fresh
    app = fixture.application
    before = app.projections.runtime_state().to_value()
    rng = torch.random.get_rng_state().clone()
    report = inspect_fresh_state(app, **inputs)
    require_fresh_state(report)
    assert report["state"] == "verified"
    assert report["checks"]["forward_default_lora_b_zero"] is True
    assert torch.equal(rng, torch.random.get_rng_state())
    assert before == app.projections.runtime_state().to_value()
    assert not app.training_loop.optimizer.state
    assert not any(p.grad is not None for p in app.backbone.named_trainable_parameters().values())
    target = tmp_path / "fresh-start.json"
    save_fresh_state_report(target, report)
    assert json.loads(target.read_text()) == report
    with pytest.raises(FileExistsError):
        save_fresh_state_report(target, report)


def test_missing_independent_reference_is_unverified_not_assumed_fresh(fresh):
    fixture, inputs = fresh
    inputs.update(initial_parameters=None, preparation_state=None)
    report = inspect_fresh_state(fixture.application, **inputs)
    assert report["checks"]["preparation_tensors_match"] is None
    assert report["checks"]["forward_default_lora_b_zero"] is True
    with pytest.raises(FreshStateError):
        require_fresh_state(report)
    require_fresh_state(report, allow_unverified=True)


@pytest.mark.parametrize("component", ["forward", "backward", "z_head"])
def test_changed_trainables_with_empty_adam_and_step_zero_are_rejected(fresh, component):
    fixture, inputs = fresh
    app = fixture.application
    group = getattr(app.backbone.parameter_groups(), component)
    with torch.no_grad():
        group[0].flatten()[0] += 1
    report = inspect_fresh_state(app, **inputs)
    assert report["optimizer_step"] == 0
    assert report["optimizer_state_entries"] == 0
    assert report["checks"]["preparation_tensors_match"] is False
    with pytest.raises(FreshStateError):
        require_fresh_state(report, allow_unverified=True)


def test_old_forward_even_redeclared_as_preparation_is_not_base_equivalent(fresh):
    fixture, inputs = fresh
    app = fixture.application
    forward_ids = {id(p) for p in app.backbone.parameter_groups().forward}
    b = next(
        p
        for n, p in app.backbone.named_trainable_parameters().items()
        if id(p) in forward_ids and ".lora_B." in n
    )
    with torch.no_grad():
        b.fill_(0.1)
    inputs["initial_parameters"] = {
        n: p.detach().cpu().clone() for n, p in app.backbone.named_trainable_parameters().items()
    }
    report = inspect_fresh_state(app, **inputs)
    assert report["checks"]["preparation_tensors_match"] is True
    assert report["checks"]["forward_default_lora_b_zero"] is False
    with pytest.raises(FreshStateError):
        require_fresh_state(report)


def test_real_completed_step_is_not_fresh_and_history_is_not_erased(fresh):
    fixture, inputs = fresh
    app = fixture.application
    asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    state = app.projections.runtime_state().to_value()
    report = inspect_fresh_state(app, **inputs)
    for name in (
        "optimizer_step_zero",
        "task_cursor_zero",
        "run_cursor_zero",
        "optimizer_state_empty",
        "posterior_empty",
        "projection_fresh",
        "detector_fresh",
        "no_prior_execution_events",
    ):
        assert report["checks"][name] is False
    assert app.projections.runtime_state().to_value() == state
    with pytest.raises(FreshStateError):
        require_fresh_state(report)


@pytest.mark.parametrize("kind", ["request", "route", "evidence", "broken-journal"])
def test_prior_namespace_contents_rejected_without_mutation(fresh, kind):
    fixture, inputs = fresh
    namespaces = inputs["namespaces"]
    journal = namespaces.request_journals[0]
    if kind in {"request", "route"}:
        with sqlite3.connect(journal) as db:
            if kind == "route":
                db.execute(
                    "INSERT INTO episode_routes VALUES ('old-episode','old-policy','synthetic')"
                )
            else:
                db.execute(
                    "INSERT INTO requests (identity,endpoint,payload,state) "
                    "VALUES ('old','synthetic',X'00','DISPATCHED')"
                )
    elif kind == "broken-journal":
        journal.write_bytes(b"not a database")
    else:
        directory = namespaces.evidence_directories[0]
        directory.mkdir()
        (directory / "original.json").write_text('{"old":true}')
    original = journal.read_bytes()
    report = inspect_fresh_state(fixture.application, **inputs)
    assert (
        report["checks"][
            "evidence_namespace_empty" if kind == "evidence" else "request_namespace_empty"
        ]
        is False
    )
    with pytest.raises(FreshStateError):
        require_fresh_state(report)
    assert journal.read_bytes() == original


def test_cursor_optimizer_preparation_and_library_checks_are_independent(fresh):
    fixture, inputs = fresh
    app = fixture.application
    # An Adam state entry with step zero is still initialized state, not fresh.
    p = next(iter(app.backbone.named_trainable_parameters().values()))
    app.training_loop.optimizer.state[p] = {"step": torch.tensor(0.0)}
    inputs["preparation_state"] = replace(inputs["preparation_state"], optimizer_step=46)
    inputs["namespaces"] = FreshNamespaces((), ())
    report = inspect_fresh_state(app, **inputs)
    assert report["checks"]["optimizer_state_empty"] is False
    assert report["checks"]["preparation_step_zero"] is False
    assert report["checks"]["request_namespace_empty"] is None
    assert report["checks"]["evidence_namespace_empty"] is None
    assert report["checks"]["initial_library_matches"] is True


def test_wrong_initial_library_and_preparation_backbone_are_not_accepted(fresh):
    from skillev.runtime import SkillLibraryState

    fixture, inputs = fresh
    original = inputs["initial_library"]
    inputs["initial_library"] = SkillLibraryState.from_seed_documents(
        (next(iter(original.documents.values())),)
    )
    inputs["preparation_state"] = replace(inputs["preparation_state"], backbone_id="other-base")
    report = inspect_fresh_state(fixture.application, **inputs)
    assert report["checks"]["initial_library_matches"] is False
    assert report["checks"]["preparation_backbone_matches"] is False
    assert fixture.application.library.state == original
    assert report["declared_z_initialization"] is not None
    with pytest.raises(FreshStateError):
        require_fresh_state(report)


def test_unsettled_ledger_and_live_gradient_stream_are_observed_not_cleared(fresh):
    from types import SimpleNamespace

    from skillev.runtime import BudgetReservation, BudgetVector

    fixture, inputs = fresh
    loop = fixture.application.training_loop
    ledger = loop.ledger
    reservation = BudgetReservation(
        reservation_id="unfinished",
        run_id=ledger.run_id,
        attempt_id=ledger.attempt_id,
        invocation_id="synthetic-call",
        maximum=BudgetVector(model_calls=1),
    )
    ledger.reserve(reservation)
    loop._live_gradient_stream = SimpleNamespace(progress_snapshot=lambda: {"active": True})
    report = inspect_fresh_state(fixture.application, **inputs)
    assert report["checks"]["ledger_fully_settled"] is False
    assert report["checks"]["no_active_gradient_stream"] is False
    assert ledger.reserved.model_calls == 1
    assert loop.gradient_progress["active"] is True
    with pytest.raises(FreshStateError):
        require_fresh_state(report)


@pytest.mark.parametrize("contents", ["empty", "old-file", "deep", "symlink"])
def test_bounded_empty_management_directories(fresh, tmp_path, contents):
    fixture, inputs = fresh
    root = inputs["namespaces"].evidence_directories[0]
    managed = root / "indexes" / "pending"
    managed.mkdir(parents=True)
    if contents == "old-file":
        (managed / "old.json").write_text("{}")
    elif contents == "deep":
        (managed / "beyond-declared-depth").mkdir()
    elif contents == "symlink":
        (managed / "external").symlink_to(tmp_path)
    report = inspect_fresh_state(fixture.application, **inputs)
    assert report["checks"]["evidence_namespace_empty"] is (contents == "empty")
    if contents == "empty":
        require_fresh_state(report)
    else:
        with pytest.raises(FreshStateError):
            require_fresh_state(report)
