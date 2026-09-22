"""Per-process replica admission, separate from physical server/shared-arm limits."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from skillev_private.experiments import development_collection as dev
from skillev_private.experiments.readonly_replicas import (
    configure_replica_admission,
    measured_capacity,
    replica_declaration,
    require_capacity,
)

from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from tests.training.test_development_collection import request_fixture, write
from tests.training.test_readonly_replicas import ENDPOINTS


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.5, "32"])
def test_explicit_admission_requires_positive_integer_and_a_pool(limit):
    with pytest.raises(ValueError):
        replica_declaration({"actor_endpoints": list(ENDPOINTS), "replica_actor_requests": limit})
    with pytest.raises(ValueError):
        replica_declaration({"replica_actor_requests": 32})


def test_admission_is_frozen_and_legacy_pool_is_unchanged(tmp_path):
    request, _, _ = request_fixture(tmp_path)
    request["actor_endpoints"] = list(ENDPOINTS)
    _, old = dev.selection(request)
    assert old["actor_replica_pool"] == {
        "format": "adapter-free-readonly-replicas@1",
        "actor_endpoints": list(ENDPOINTS),
        "allocation": "canonical-position-modulo@1",
        "episode_sticky": True,
        "failure_reassignment": False,
    }
    request["replica_actor_requests"] = 32
    _, new = dev.selection(request)
    assert new["actor_replica_pool"]["admission"]["actor_requests_per_endpoint"] == 32
    assert not new["actor_replica_pool"]["admission"]["cross_process_shared_limiter"]
    request["comparison_reference"] = write(tmp_path / "old.json", old)
    with pytest.raises(ValueError):
        dev.selection(request)


def test_endpoint_permits_are_independent_and_legacy_global_gate_is_retained():
    async def exercise():
        resources = RolloutWorkflowResources(RolloutWorkflowBinding(max_inflight_model_requests=1))
        legacy = replica_declaration({"actor_endpoints": list(ENDPOINTS)})
        configure_replica_admission(resources, legacy)
        assert all(resources.model_limiter(e) is resources.model_requests for e in ENDPOINTS)
        explicit = replica_declaration(
            {"actor_endpoints": list(ENDPOINTS), "replica_actor_requests": 2}
        )
        configure_replica_admission(resources, explicit)
        first, second = (resources.model_limiter(e) for e in ENDPOINTS[:2])
        assert first is not second
        async with AsyncExitStack() as held:
            # Three simultaneous actor leases exceed the old global one-permit gate.
            await asyncio.wait_for(held.enter_async_context(first.lease(role="actor")), 1)
            await asyncio.wait_for(held.enter_async_context(first.lease(role="actor")), 1)
            await asyncio.wait_for(held.enter_async_context(second.lease(role="actor")), 1)
            assert resources.model_requests.timing.high_water_mark == 3
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(first.lease(role="actor").__aenter__(), 0.01)
            # Existing global limiter was never acquired by actor endpoints.
            async with resources.model_requests.lease(role="terminal-judge"):
                assert resources.model_requests.timing.high_water_mark == 4
        assert first.timing.calls == 2
        assert second.timing.calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("limit", [None, True, "32", 0, 31, 32, 40])
def test_observed_request_capacity_not_server_option(limit):
    raw = {
        "max_req_input_len": 2048,
        "max_total_num_tokens": 4096,
        "max_running_requests": limit,
        "server_args": {"max_running_requests": 32},
    }
    legacy = measured_capacity(raw)
    assert "max_running_requests" not in legacy
    observed = measured_capacity(raw, include_request_limit=True)
    if type(limit) is int and limit >= 32:
        require_capacity(observed, input_tokens=2048, output_tokens=1024, actor_requests=32)
    else:
        with pytest.raises(ValueError):
            require_capacity(observed, input_tokens=2048, output_tokens=1024, actor_requests=32)


def test_request_limit_uses_complete_actual_worker_measurements():
    raw = {"internal_states": [{"max_running_requests": 40}, {"max_running_requests": 32}]}
    assert measured_capacity(raw, include_request_limit=True)["max_running_requests"] == 32
    raw["internal_states"][1] = {"server_args": {"max_running_requests": 32}}
    assert measured_capacity(raw, include_request_limit=True)["max_running_requests"] is None
