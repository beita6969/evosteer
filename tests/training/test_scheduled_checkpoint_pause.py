"""Execution handoffs do not shorten the frozen training/evolution plan."""

import asyncio
from types import SimpleNamespace

import pytest
from skillev_private.experiments.bayesian_improve_training import (
    _stop_at_committed_boundary,
    run_coordinator,
)

from skillev.training.stopping import StopAfterCheckpoint


@pytest.mark.parametrize(("step", "expected"), [(0, False), (1, False), (2, True), (3, True)])
def test_scheduled_pause_preserves_due_probe_and_uses_committed_step(tmp_path, step, expected):
    request = StopAfterCheckpoint(tmp_path / "STOP_AFTER_CHECKPOINT")
    calls = []

    async def check(**coordinates):
        calls.append(coordinates)
        return False

    probes = SimpleNamespace(check=check)
    assert (
        asyncio.run(
            _stop_at_committed_boundary(
                request, probes, optimizer_step=step, policy_snapshot_id="fixed", pause_at_step=2
            )
        )
        is expected
    )
    assert calls == [{"policy_step": step, "policy_snapshot_id": "fixed"}]
    assert not request.request_file.exists()  # no mutation of operator intent


def test_quality_stop_works_before_scheduled_pause_and_resume_uses_new_handoff(tmp_path):
    request = StopAfterCheckpoint(tmp_path / "STOP_AFTER_CHECKPOINT")

    async def failed(**_coordinates):
        return True

    assert asyncio.run(
        _stop_at_committed_boundary(
            request,
            SimpleNamespace(check=failed),
            optimizer_step=1,
            policy_snapshot_id="current",
            pause_at_step=25,
        )
    )
    for target, expected in ((4, False), (None, False), (2, True)):
        assert (
            asyncio.run(
                _stop_at_committed_boundary(
                    request,
                    None,
                    optimizer_step=2,
                    policy_snapshot_id="restored",
                    pause_at_step=target,
                )
            )
            is expected
        )


def test_explicit_operator_stop_does_not_wait_for_new_probe(tmp_path):
    request = StopAfterCheckpoint(tmp_path / "STOP_AFTER_CHECKPOINT")
    request.request()

    async def not_called(**_coordinates):
        raise AssertionError("operator requested immediate boundary stop")

    assert asyncio.run(
        _stop_at_committed_boundary(
            request,
            SimpleNamespace(check=not_called),
            optimizer_step=1,
            policy_snapshot_id="current",
            pause_at_step=25,
        )
    )


def test_operator_stop_arriving_during_probe_prevents_next_batch(tmp_path):
    request = StopAfterCheckpoint(tmp_path / "STOP_AFTER_CHECKPOINT")
    completed = []

    async def check(**_coordinates):
        request.request_file.touch()
        await asyncio.sleep(0)
        completed.append(True)
        return False  # A passing probe must not erase the operator's request.

    assert asyncio.run(
        _stop_at_committed_boundary(
            request,
            SimpleNamespace(check=check),
            optimizer_step=10,
            policy_snapshot_id="current",
            pause_at_step=None,
        )
    )
    assert completed


@pytest.mark.parametrize("target", [0, -1, 250, 251, True, 2.5])
def test_invalid_handoff_rejected_before_creating_run_or_resources(tmp_path, target):
    root = tmp_path / "not-created"
    with pytest.raises(ValueError):
        asyncio.run(
            run_coordinator(
                config=SimpleNamespace(steps=250),
                bindings=None,
                profile=None,
                root=root,
                resume=None,
                topology=None,
                pause_at_step=target,
            )
        )
    assert not root.exists()
