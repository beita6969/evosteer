import asyncio
import json
import signal

import pytest

from skillev.training.stopping import StopAfterCheckpoint, TrainingPausedError
from tests.application.test_full_vertical_loop import build_application_fixture


def test_stop_during_collection_finishes_transaction_before_pause(
    tmp_path, training_backbone_config, monkeypatch
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    request = StopAfterCheckpoint(tmp_path / "STOP_AFTER_CHECKPOINT")
    collect = app.training_loop.collect_batch
    calls = 0

    async def collect_then_request():
        nonlocal calls
        calls += 1
        batch = await collect()
        request.request()
        return batch

    monkeypatch.setattr(app.training_loop, "collect_batch", collect_then_request)
    with pytest.raises(TrainingPausedError) as stopped:
        asyncio.run(app.evolution_loop.run(fixture.run_plan, stop_requested=request))
    assert calls == 1
    assert stopped.value.optimizer_step == app.training_loop.optimizer_step == 1
    checkpoint = stopped.value.checkpoint
    assert (checkpoint / "COMPLETE").is_file()
    state = json.loads((checkpoint / "runtime_state.json").read_text())
    assert state["execution_state"]["run_cursor"]["completed_training_steps"] == 1
    assert len(app.projections.posterior_provenance.batches) == 1
    journal = app.evolution_loop.step_transaction_journal
    assert journal is not None
    assert journal.load(1).state.value == "committed"
    assert not app.evolution_loop._failed
    # A pause must not pretend that the full run's final product exists.
    with pytest.raises(RuntimeError):
        _ = app.evolution_loop.final_training_snapshot_directory


def test_stop_file_and_signal_are_requests_only_and_handlers_restore(tmp_path):
    path = tmp_path / "STOP_AFTER_CHECKPOINT"
    request = StopAfterCheckpoint(path)
    before = signal.getsignal(signal.SIGUSR1)
    assert not request()
    with request.signals():
        signal.raise_signal(signal.SIGUSR1)
        assert request()
        assert not path.exists()
    assert signal.getsignal(signal.SIGUSR1) == before
    path.touch()
    assert StopAfterCheckpoint(path)()
    assert path.exists()


def test_quality_failure_preserves_full_commit_before_pausing(tmp_path, training_backbone_config):
    from dataclasses import asdict

    from skillev.training.quality_gate import ProtocolProbe, QualityGatePolicy, QualityRule
    from skillev.training.quality_monitor import QualityCheckpointStop

    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    root = tmp_path / "quality"
    monitor = QualityCheckpointStop(
        root,
        QualityGatePolicy(
            "synthetic@1", "fixed", "raw", 1, 10, 1, (QualityRule("structure", 0.8, 0.1),)
        ),
    )

    async def checked_boundary():
        # Simulate awaited read-only model work, not a synchronously available file.
        await asyncio.sleep(0)
        step = app.training_loop.optimizer_step
        probe = ProtocolProbe(
            f"probe-{step}",
            "fixed",
            "raw",
            app.training_loop.policy_snapshot_id,
            step,
            12,
            {"structure": 1.0 if step == 0 else 0.5},
        )
        (root / f"probe-{step:08d}.json").write_text(json.dumps(asdict(probe)))
        return monitor.check(policy_step=step, policy_snapshot_id=probe.policy_snapshot_id)

    with pytest.raises(TrainingPausedError) as paused:
        asyncio.run(app.evolution_loop.run(fixture.run_plan, stop_requested=checked_boundary))
    assert paused.value.optimizer_step == 1
    assert (paused.value.checkpoint / "COMPLETE").is_file()
    assert len(app.projections.posterior_provenance.batches) == 1
    assert app.evolution_loop.step_transaction_journal.load(1).state.value == "committed"
