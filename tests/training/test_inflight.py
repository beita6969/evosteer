import asyncio
from dataclasses import replace

import pytest

from skillev.runtime import BudgetLedger, BudgetReservation, BudgetSettlement, BudgetVector
from skillev.runtime.budget_ledger import DuplicateBudgetReservationError, LedgerEntry


def test_completed_batch_is_restored_without_generation_or_double_budget(
    make_training_harness, tmp_path
):
    original = make_training_harness()
    directory = tmp_path / "inflight"
    original.loop.configure_inflight(directory, condition={"caps": [5, 25]})
    batch = asyncio.run(original.loop.collect_batch())
    restored = make_training_harness(fail_on_generation_call=1)
    restored.loop.configure_inflight(directory, condition={"caps": [5, 25]})
    actual = asyncio.run(restored.loop.collect_batch())
    assert actual == batch
    assert restored.loop.ledger.entries == original.loop.ledger.entries
    assert restored.loop.optimizer_step == 0
    assert not restored.projections.posterior_provenance.batches


def test_partial_batch_retains_only_successfully_cleaned_artifacts(
    make_training_harness, tmp_path, monkeypatch
):
    original = make_training_harness()
    directory = tmp_path / "inflight"
    original.loop.configure_inflight(directory, condition={})
    create = original.session_factory.create
    count = 0

    def fail_second(task):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("environment infrastructure unavailable")
        return create(task)

    monkeypatch.setattr(
        type(original.session_factory), "create", lambda self, task: fail_second(task)
    )
    with pytest.raises(RuntimeError):
        asyncio.run(original.loop.collect_batch())
    # With the fixture's serial workflow, first artifact is durable, the failed
    # second environment never acquires a fabricated terminal result.
    paths = list(directory.glob("step-*/trajectory-*.json"))
    assert len(paths) == 1
    assert original.loop.optimizer_step == 0


def test_saved_batch_cannot_be_reused_after_condition_change(make_training_harness, tmp_path):
    original = make_training_harness()
    original.loop.configure_inflight(tmp_path, condition={"caps": [2, 50]})
    asyncio.run(original.loop.collect_batch())
    changed = make_training_harness(fail_on_generation_call=1)
    changed.loop.configure_inflight(tmp_path, condition={"caps": [5, 25]})
    with pytest.raises(ValueError):
        asyncio.run(changed.loop.collect_batch())
    assert not changed.loop.ledger.entries


def test_budget_import_rejects_entire_duplicate_or_wrong_attempt():
    ledger = BudgetLedger(run_id="run", attempt_id="attempt", cap=BudgetVector(model_calls=4))
    reservation = BudgetReservation(
        "call", "run", "attempt", "trajectory", BudgetVector(model_calls=2)
    )
    entry = LedgerEntry.reserved(reservation).settled(
        BudgetSettlement("call", BudgetVector(model_calls=1))
    )
    with pytest.raises(DuplicateBudgetReservationError):
        ledger.restore_completed([entry, entry])
    assert not ledger.entries
    wrong = replace(entry, reservation=replace(reservation, attempt_id="other"))
    with pytest.raises(ValueError):
        ledger.restore_completed([entry, wrong])
    assert not ledger.entries
    ledger.restore_completed([entry])
    with pytest.raises(DuplicateBudgetReservationError):
        ledger.restore_completed([entry])
    assert ledger.settled.model_calls == 1


def test_full_durable_batch_must_be_readable_before_optimizer(
    make_training_harness, tmp_path, monkeypatch
):
    import json

    from skillev.training.inflight import InFlightBatchStore

    captured = []
    original_begin = InFlightBatchStore.begin

    def begin(self, plan):
        batch = original_begin(self, plan)
        captured.append(batch)
        return batch

    monkeypatch.setattr(InFlightBatchStore, "begin", begin)
    harness = make_training_harness()
    harness.loop.configure_inflight(tmp_path, condition={})
    batch = asyncio.run(harness.loop.collect_batch())
    durable = captured[0]
    result = durable.require_complete(batch.artifacts, tokenizer=harness.generator.tokenizer)
    assert result["trajectory_count"] == len(batch.artifacts)
    assert harness.loop.optimizer_step == 0
    assert (
        durable.require_complete(batch.artifacts, tokenizer=harness.generator.tokenizer) == result
    )
    path = durable.directory / "trajectory-000001.json"
    original = path.read_text()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        durable.require_complete(batch.artifacts, tokenizer=harness.generator.tokenizer)
    raw = json.loads(original)
    raw["artifact"]["manifest"]["library_version"] = "different-library"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        durable.require_complete(batch.artifacts, tokenizer=harness.generator.tokenizer)
