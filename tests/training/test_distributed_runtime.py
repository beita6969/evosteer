from __future__ import annotations

import asyncio

import pytest

from skillev.training import distributed_runtime
from skillev.training.distributed_ttb import (
    DistributedTTBError,
    DistributedTTBTopology,
)


def _topology(rank: int) -> DistributedTTBTopology:
    return DistributedTTBTopology(rank=rank, world_size=2, local_rank=rank, backend="nccl")


def _patch_process_group(monkeypatch: pytest.MonkeyPatch, *, rank: int) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(
        distributed_runtime,
        "initialize_distributed_ttb",
        lambda **kwargs: _topology(rank),
    )
    monkeypatch.setattr(distributed_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        distributed_runtime.dist,
        "destroy_process_group",
        lambda: events.append("destroy"),
    )
    monkeypatch.setattr(
        distributed_runtime,
        "_gather_worker_readiness",
        lambda topology, worker_error: (
            {"rank": 0, "status": "ready"},
            {"rank": 1, "status": "ready"},
        ),
    )
    return events


def test_rank_zero_builds_only_coordinator_and_stops_workers(monkeypatch) -> None:
    events = _patch_process_group(monkeypatch, rank=0)

    class _Coordinator:
        def __init__(self, topology) -> None:
            events.append("coordinator-created")

        def close(self) -> None:
            events.append("stop-workers")

    monkeypatch.setattr(distributed_runtime, "DistributedTTBGradientCoordinator", _Coordinator)

    async def coordinator(topology, gradient_coordinator) -> str:
        events.append("application")
        return "complete"

    def forbidden_worker(topology):
        raise AssertionError("rank zero must not construct a worker backbone")

    result = asyncio.run(
        distributed_runtime.run_distributed_ttb_role(
            worker_backbone_factory=forbidden_worker,
            coordinator_entrypoint=coordinator,
        )
    )

    assert result.coordinator_result == "complete"
    assert events == ["coordinator-created", "application", "stop-workers", "destroy"]


def test_worker_rank_builds_only_worker_backbone(monkeypatch) -> None:
    events = _patch_process_group(monkeypatch, rank=1)
    worker = object()
    monkeypatch.setattr(
        distributed_runtime,
        "serve_distributed_ttb_worker",
        lambda **kwargs: events.append("worker-loop"),
    )

    async def forbidden_coordinator(topology, gradient_coordinator):
        raise AssertionError("worker rank must not construct the application")

    result = asyncio.run(
        distributed_runtime.run_distributed_ttb_role(
            worker_backbone_factory=lambda topology: worker,  # type: ignore[return-value]
            coordinator_entrypoint=forbidden_coordinator,
        )
    )

    assert result.coordinator_result is None
    assert events == ["worker-loop", "destroy"]


def test_worker_construction_failure_reaches_coordinator(monkeypatch) -> None:
    events = _patch_process_group(monkeypatch, rank=0)
    monkeypatch.setattr(
        distributed_runtime,
        "_gather_worker_readiness",
        lambda topology, worker_error: (
            {"rank": 0, "status": "ready"},
            {"error_class": "RuntimeError", "rank": 1, "status": "error"},
        ),
    )

    async def coordinator(topology, gradient_coordinator):
        raise AssertionError("coordinator application must not start")

    with pytest.raises(DistributedTTBError):
        asyncio.run(
            distributed_runtime.run_distributed_ttb_role(
                worker_backbone_factory=lambda topology: object(),  # type: ignore[return-value]
                coordinator_entrypoint=coordinator,
            )
        )
    assert events == ["destroy"]
