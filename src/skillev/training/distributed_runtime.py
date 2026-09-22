"""Process-role boundary for the formal distributed TTB runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import torch.distributed as dist

from skillev.policy import PolicyBackbone

from .distributed_ttb import (
    DistributedTTBError,
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
    initialize_distributed_ttb,
    serve_distributed_ttb_worker,
)

ResultT = TypeVar("ResultT")
WorkerBackboneFactory = Callable[[DistributedTTBTopology], PolicyBackbone]
CoordinatorEntrypoint = Callable[
    [DistributedTTBTopology, DistributedTTBGradientCoordinator],
    Awaitable[ResultT],
]


@dataclass(frozen=True, slots=True)
class DistributedRoleResult(Generic[ResultT]):
    topology: DistributedTTBTopology
    coordinator_result: ResultT | None


async def run_distributed_ttb_role(
    *,
    worker_backbone_factory: WorkerBackboneFactory,
    coordinator_entrypoint: CoordinatorEntrypoint[ResultT],
    timeout_minutes: int = 30,
) -> DistributedRoleResult[ResultT]:
    """Run rank 0 as coordinator and every rank > 0 as a gradient worker.

    The NCCL group is initialized before any rank constructs a model.  Rank 0
    never constructs a second worker backbone, and worker ranks never construct
    the application, evaluator catalog, optimizer, or checkpoint writer.
    """

    if not callable(worker_backbone_factory) or not callable(coordinator_entrypoint):
        raise TypeError("distributed role runtime requires fixed entrypoints")
    topology = initialize_distributed_ttb(timeout_minutes=timeout_minutes)
    coordinator: DistributedTTBGradientCoordinator | None = None
    worker_backbone: PolicyBackbone | None = None
    try:
        worker_error: BaseException | None = None
        if topology.rank > 0:
            try:
                worker_backbone = worker_backbone_factory(topology)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                worker_error = error
        readiness = _gather_worker_readiness(topology, worker_error)
        failures = tuple(item for item in readiness if item.get("status") != "ready")
        if failures:
            raise DistributedTTBError(
                "a distributed TTB worker failed during model construction",
                diagnostics=failures,
            )
        if topology.rank > 0:
            if worker_backbone is None:  # pragma: no cover - readiness guarantees this
                raise RuntimeError("ready worker has no policy backbone")
            serve_distributed_ttb_worker(topology=topology, backbone=worker_backbone)
            return DistributedRoleResult(topology, None)

        coordinator = DistributedTTBGradientCoordinator(topology)
        result = await coordinator_entrypoint(topology, coordinator)
        return DistributedRoleResult(topology, result)
    finally:
        if coordinator is not None:
            coordinator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _gather_worker_readiness(
    topology: DistributedTTBTopology,
    worker_error: BaseException | None,
) -> tuple[dict[str, object], ...]:
    status: dict[str, object]
    if topology.rank == 0 or worker_error is None:
        status = {"rank": topology.rank, "status": "ready"}
    else:
        status = {
            "error_class": type(worker_error).__name__,
            "rank": topology.rank,
            "status": "error",
        }
    gathered: list[object] = [None] * topology.world_size
    dist.all_gather_object(gathered, status)
    if any(not isinstance(item, dict) for item in gathered):
        raise DistributedTTBError("distributed readiness response is malformed")
    return cast(tuple[dict[str, object], ...], tuple(gathered))


__all__ = [
    "CoordinatorEntrypoint",
    "DistributedRoleResult",
    "WorkerBackboneFactory",
    "run_distributed_ttb_role",
]
