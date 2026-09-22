import torch

from skillev.policy.activation_pool import PinnedActivationPool


class Event:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


def test_pool_never_reuses_or_unpins_a_pending_dma_buffer(monkeypatch):
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **{**kw, "pin_memory": False}))
    pool = PinnedActivationPool(16)
    source = torch.zeros(4)
    saved = pool.acquire(source)
    assert saved is not None
    assert pool.live_bytes == pool.allocated_bytes == 16
    d2h, h2d = Event(True), Event(False)
    pool.release(saved, d2h, [h2d])
    assert pool.acquire(source) is None
    assert pool.live_bytes == 16
    h2d.ready = True
    assert pool.acquire(source) is saved
    assert pool.reuses == 1
    pool.release(saved, d2h, [])
    pool.close()
    assert pool.live_bytes == pool.allocated_bytes == 0


def test_pool_can_evict_idle_wrong_shape_but_not_active_storage(monkeypatch):
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **{**kw, "pin_memory": False}))
    pool = PinnedActivationPool(16)
    first = pool.acquire(torch.zeros(4))
    pool.release(first, Event(True), [])
    second = pool.acquire(torch.zeros(2, dtype=torch.float64))
    assert second.shape == (2,)
    assert pool.allocated_bytes == 16
    assert pool.acquire(torch.zeros(1)) is None
    pool.release(second, Event(True), [])
    pool.close()


def test_pinned_allocation_failure_requests_safe_pageable_fallback(monkeypatch):
    def fail(*_a, **_kw):
        raise RuntimeError("locked memory exhausted")

    monkeypatch.setattr(torch, "empty", fail)
    pool = PinnedActivationPool(1024)
    assert pool.acquire(torch.zeros(4)) is None
    assert pool.allocated_bytes == pool.live_bytes == 0
    assert pool.fallbacks == 1


# Separate opt-in from CPU make check: CVD is required before touching CUDA.
def test_cuda_saved_activations_retain_graph_and_pool_lifetime():
    import gc
    import os

    import pytest

    from skillev.policy.scoring_execution import ActivationOffload

    if os.environ.get("SKILLEV_CUDA_OFFLOAD_TEST") != "1":
        pytest.skip("explicit CUDA activation-transport qualification only")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    torch.manual_seed(0)
    source = torch.randn(257, 129, device="cuda", requires_grad=True)
    expected = source.detach().clone().requires_grad_()
    offload = ActivationOffload(2 * source.numel() * source.element_size())
    for _attempt in range(3):
        source.grad = expected.grad = None
        with offload.context():
            loss = (source.transpose(0, 1).sin() ** 2).sum()
        baseline = (expected.transpose(0, 1).sin() ** 2).sum()
        for retain in [True, False]:
            loss.backward(retain_graph=retain)
            baseline.backward(retain_graph=retain)
            torch.testing.assert_close(source.grad, expected.grad, rtol=0, atol=0)
        del loss, baseline
        gc.collect()
    offload.pool.close()
    assert offload.peak_pinned_bytes <= offload.limit
    assert offload.pool.allocated_bytes == offload.pool.live_bytes == 0
    assert offload.pool.reuses > 0
