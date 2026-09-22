"""Regress the actual long-request affinity and implicit replay failures."""

from contextlib import ExitStack
from pathlib import Path

import pytest
from skillev_private.evaluation.integrity_replicas import ReplicaLoadBalancer
from skillev_private.evaluation.integrity_runtime import PrivateIntegrityRuntime


def test_inflight_calls_balance_independently_of_episode_affinity() -> None:
    balancer = ReplicaLoadBalancer(2)
    with ExitStack() as stack:
        assert [stack.enter_context(balancer.acquire()) for _ in range(4)] == [0, 1, 0, 1]
        assert balancer.inflight == [2, 2]
    assert balancer.inflight == [0, 0]


def test_new_work_uses_the_free_replica_while_a_long_call_runs() -> None:
    balancer = ReplicaLoadBalancer(2)
    with balancer.acquire() as long_call:
        for _ in range(5):
            with balancer.acquire() as short_call:
                assert short_call != long_call
                assert balancer.inflight == [1, 1]


def test_failure_releases_reservation() -> None:
    balancer = ReplicaLoadBalancer(2)
    with pytest.raises(RuntimeError), balancer.acquire():
        raise RuntimeError("synthetic serving failure")
    assert balancer.inflight == [0, 0]


def test_unknown_usage_cannot_silently_replay_a_clean_model_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PrivateIntegrityRuntime({"transport_attempts": 3}, None, tmp_path)  # type: ignore[arg-type]
