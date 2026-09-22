from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from skillev.training.gradient_worker import (
    CostBalancedGradientWorkerPool,
    GradientRequest,
    GradientResult,
    GradientTensorSummary,
    GradientWorkItem,
    SealedGradientBatch,
)
from skillev.training.oom_recovery import OOMRecoveryCoordinator, OOMRecoveryPhase


@dataclass(slots=True)
class FakeWorker:
    worker_id: str
    oom: bool = False
    requests: list[GradientRequest] = field(default_factory=list)

    def compute_weighted_grads(self, request: GradientRequest) -> GradientResult:
        self.requests.append(request)
        if self.oom:
            raise torch.cuda.OutOfMemoryError("injected")
        value = float(sum(item.token_cost for item in request.items))
        tensor = torch.tensor([value], dtype=torch.float32)
        return GradientResult(
            worker_id=self.worker_id,
            profile=request.profile,
            attempt=request.attempt,
            processed_item_ids=tuple(item.item_id for item in request.items),
            loss_contribution=value,
            gradients={"adapter.weight": tensor},
            summaries={
                "adapter.weight": GradientTensorSummary(value, value, finite=True),
            },
            peak_allocated_bytes=100,
            peak_reserved_bytes=200,
        )


@dataclass(slots=True)
class FakeHooks:
    expanded: CostBalancedGradientWorkerPool
    calls: list[str] = field(default_factory=list)

    def abort_current_step(self, batch: SealedGradientBatch) -> None:
        self.calls.append(f"abort:{batch.batch_id}")

    def discard_uncommitted_gradients(self) -> None:
        self.calls.append("discard")

    def persist_oom_event_and_batch(self, error: Exception, batch: SealedGradientBatch) -> None:
        self.calls.append(f"persist:{batch.batch_id}:{type(error).__name__}")

    def restore_last_completed_step(self) -> None:
        self.calls.append("restore")

    def start_expanded_profile(self) -> CostBalancedGradientWorkerPool:
        self.calls.append("expanded")
        return self.expanded


def _batch() -> SealedGradientBatch:
    return SealedGradientBatch(
        batch_id="batch-2",
        optimizer_step=2,
        items=tuple(
            GradientWorkItem(f"item-{index}", cost, f"private://item-{index}")
            for index, cost in enumerate((40, 30, 20, 10), start=1)
        ),
    )


def test_typed_cuda_oom_discards_and_retries_the_identical_batch_on_two_workers() -> None:
    primary = FakeWorker("gpu3", oom=True)
    expanded_primary = FakeWorker("gpu3-expanded")
    standby = FakeWorker("gpu4-standby")
    hooks = FakeHooks(
        CostBalancedGradientWorkerPool(
            (expanded_primary, standby),
            profile="expanded_after_oom",
        )
    )
    coordinator = OOMRecoveryCoordinator(hooks)

    result = coordinator.execute(
        batch=_batch(),
        steady_pool=CostBalancedGradientWorkerPool((primary,), profile="steady"),
    )

    assert hooks.calls == [
        "abort:batch-2",
        "discard",
        "persist:batch-2:GradientWorkerOOMError",
        "restore",
        "expanded",
    ]
    assert result.attempt == 2
    assert result.profile == "expanded_after_oom"
    assert result.processed_item_ids == tuple(item.item_id for item in _batch().items)
    assert expanded_primary.requests[0].items
    assert standby.requests[0].items
    assert coordinator.phase is OOMRecoveryPhase.READY_TO_COMMIT


def test_steady_success_never_starts_the_cold_standby() -> None:
    steady = FakeWorker("gpu3")
    hooks = FakeHooks(
        CostBalancedGradientWorkerPool(
            (FakeWorker("gpu3-expanded"), FakeWorker("gpu4-standby")),
            profile="expanded_after_oom",
        )
    )

    result = OOMRecoveryCoordinator(hooks).execute(
        batch=_batch(),
        steady_pool=CostBalancedGradientWorkerPool((steady,), profile="steady"),
    )

    assert result.attempt == 1
    assert result.worker_ids == ("gpu3",)
    assert hooks.calls == []


def test_non_oom_worker_failure_is_not_misclassified_or_retried() -> None:
    @dataclass(slots=True)
    class BrokenWorker:
        worker_id: str = "gpu3"

        def compute_weighted_grads(self, request: GradientRequest) -> GradientResult:
            del request
            raise RuntimeError("NCCL transport failed")

    hooks = FakeHooks(
        CostBalancedGradientWorkerPool(
            (FakeWorker("gpu3-expanded"), FakeWorker("gpu4-standby")),
            profile="expanded_after_oom",
        )
    )

    with pytest.raises(RuntimeError):
        OOMRecoveryCoordinator(hooks).execute(
            batch=_batch(),
            steady_pool=CostBalancedGradientWorkerPool((BrokenWorker(),), profile="steady"),
        )
    assert hooks.calls == []


def test_cost_balancer_preserves_original_order_within_each_partition() -> None:
    first = FakeWorker("rank-0")
    second = FakeWorker("rank-1")
    pool = CostBalancedGradientWorkerPool((first, second), profile="expanded")

    result = pool.compute(_batch(), attempt=1)

    assert result.processed_item_ids == tuple(item.item_id for item in _batch().items)
    assert sum(len(worker.requests[0].items) for worker in (first, second)) == 4
    assert float(result.gradients["adapter.weight"].item()) == pytest.approx(100.0)
