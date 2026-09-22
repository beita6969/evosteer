import asyncio

from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress
from skillev.training import request_scheduling as module


def test_long_followups_and_action_are_prioritized_but_age_prevents_starvation(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])

    async def run():
        gate = module.FairRequestGate(1)
        await gate.__aenter__()
        order = []

        async def request(name, horizon, phase):
            row = RolloutProgress(max_turns=horizon)
            row.begin_phase(phase, 10)
            with bind_progress(row):
                async with gate:
                    order.append(name)

        oldest = asyncio.create_task(request("old-grader", 1, "grader"))
        await asyncio.sleep(0)
        now[0] += 7
        short = asyncio.create_task(request("short", 1, "reasoning"))
        long = asyncio.create_task(request("long", 50, "reasoning"))
        action = asyncio.create_task(request("action", 50, "action"))
        await asyncio.sleep(0)
        await gate.__aexit__()
        await asyncio.gather(oldest, short, long, action)
        assert order == ["old-grader", "action", "long", "short"]
        assert gate.active == 0

    asyncio.run(run())


def test_cancellation_before_and_after_grant_returns_capacity():
    async def run():
        gate = module.FairRequestGate(1)
        await gate.__aenter__()
        before = asyncio.create_task(gate.__aenter__())
        await asyncio.sleep(0)
        before.cancel()
        await asyncio.gather(before, return_exceptions=True)
        after = asyncio.create_task(gate.__aenter__())
        await asyncio.sleep(0)
        await gate.__aexit__()
        after.cancel()
        await asyncio.gather(after, return_exceptions=True)
        assert gate.active == 0
        async with gate:
            assert gate.active == 1

    asyncio.run(run())


def test_token_capacity_drains_for_aged_large_request_and_cancellation_releases_tokens():
    async def run():
        gate = module.FairRequestGate(3, token_capacity=10, aging=False)
        await gate.acquire(6)
        large = asyncio.create_task(gate.acquire(8))
        await asyncio.sleep(0)
        small = asyncio.create_task(gate.acquire(2))
        await asyncio.sleep(0)
        assert not large.done()
        assert not small.done()
        assert gate.active_tokens == 6
        gate.release(6)
        # Both fit after the first lease releases; cancellation after dispatch
        # must return the reservation even before the caller resumes.
        large.cancel()
        await asyncio.gather(large, return_exceptions=True)
        await small
        assert gate.active == 1
        assert gate.active_tokens == 2
        gate.release(2)
        assert gate.active == 0
        assert gate.active_tokens == 0

    asyncio.run(run())


def test_role_pools_share_only_their_physical_endpoint_capacity():
    from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources

    async def run():
        resources = RolloutWorkflowResources(RolloutWorkflowBinding())
        resources.configure_model_endpoint("http://actor", capacity=2, token_capacity=12)
        resources.configure_model_endpoint("http://judge", capacity=1, token_capacity=12)
        actor = resources.model_limiter("http://actor/v1")
        same_service_judge = resources.model_limiter("http://actor")
        independent_judge = resources.model_limiter("http://judge")
        assert actor is same_service_judge
        started = asyncio.Event()

        async def shared_judge():
            async with same_service_judge.lease(role="judge"):
                started.set()

        async with actor.lease(token_cost=8, role="actor"):
            task = asyncio.create_task(shared_judge())
            await asyncio.sleep(0)
            assert not started.is_set()
            async with independent_judge.lease(role="judge"):
                assert resources.model_requests.timing.high_water_mark == 2
        await task
        assert started.is_set()
        assert resources.model_requests.timing.calls == 3
        assert actor.active_snapshot()["active_reserved_tokens"] == 0
        assert actor.active_snapshot()["calls_by_role"] == {"actor": 1, "judge": 1}
        resources.begin_batch_window()
        assert resources.model_requests.timing.calls == 0

    asyncio.run(run())
