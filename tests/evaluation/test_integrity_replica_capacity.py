from contextlib import ExitStack

import pytest
from skillev_private.evaluation.integrity_replicas import ReplicaLoadBalancer


def test_long_continuations_do_not_reach_a_short_context_replica() -> None:
    pool = ReplicaLoadBalancer(3, context_lengths=(67584, 98304, 67584))

    with ExitStack() as active:
        # This models the actual three-service layout, including concurrent
        # long continuations. A busier eligible replica is safer than an idle
        # replica that would reject the unchanged input/output allowance.
        for _ in range(4):
            selected = active.enter_context(pool.acquire(required_context_tokens=81920))
            assert selected == 1
        short = [
            active.enter_context(pool.acquire(required_context_tokens=16384)) for _ in range(4)
        ]
        assert short.count(0) == short.count(2) == 2
    assert not any(pool.inflight)


def test_short_requests_still_use_all_compatible_replicas() -> None:
    pool = ReplicaLoadBalancer(3, context_lengths=(67584, 98304, 67584))
    with ExitStack() as active:
        selected = [
            active.enter_context(pool.acquire(required_context_tokens=67584)) for _ in range(6)
        ]
    assert all(selected.count(index) == 2 for index in range(3))


def test_unsupported_capacity_never_reserves_or_reduces_the_request() -> None:
    pool = ReplicaLoadBalancer(2, context_lengths=(16, 32))
    with pytest.raises(ValueError), pool.acquire(required_context_tokens=33):
        pytest.fail("an oversized request must not be dispatched")
    assert not any(pool.inflight)

    def failed_transport():
        with pool.acquire(required_context_tokens=32) as selected:
            assert selected == 1
            raise RuntimeError("synthetic transport failure")

    with pytest.raises(RuntimeError):
        failed_transport()
    assert not any(pool.inflight)


def test_broker_includes_output_reservation_when_selecting_replica(tmp_path) -> None:
    import asyncio

    from skillev.evaluation.input_metric_contracts import PublicTaskView
    from skillev.evaluation.step0_integrity import InferenceArm
    from tests.evaluation.test_integrity_broker_boundary import ServingFixture, runtime

    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, [])

    class RecordingPool(ReplicaLoadBalancer):
        def __init__(self):
            super().__init__(2)
            self.reservations = []

        def acquire(self, *, required_context_tokens=0):
            self.reservations.append(required_context_tokens)
            # Force this real broker request to fit only the second replica.
            self.context_lengths = (required_context_tokens - 1, required_context_tokens)
            return super().acquire(required_context_tokens=required_context_tokens)

    pool = RecordingPool()
    instance.replicas = pool
    first, second = ServingFixture([]), ServingFixture(["Final answer: 42"])
    instance.generators = [first, second]
    try:
        final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
        assert final.text == r"\boxed{42}"
        assert not first.profiles
        assert len(second.profiles) == 1
        assert pool.reservations == [final.prompt_tokens + second.profiles[0].max_new_tokens]
        assert not any(pool.inflight)
    finally:
        instance.journal.close()
