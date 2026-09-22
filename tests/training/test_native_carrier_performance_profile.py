import asyncio
from dataclasses import replace
from pathlib import Path

from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.rollout_workflow import RolloutWorkflowResources


def test_carrier_latency_profile_only_changes_resource_admission():
    root = Path(__file__).resolve().parents[2]
    old = TrainingPerformanceConfig.load(root / "configs/training/protocol13_turn_latency.yaml")
    candidate = TrainingPerformanceConfig.load(
        root / "configs/training/protocol13_native_carrier_latency.yaml"
    )
    assert candidate.workflow().max_inflight_model_requests == candidate.max_running_requests
    assert (
        replace(candidate, actor_requests=old.actor_requests, process_graders=old.process_graders)
        == old
    )


def test_four_network_graders_do_not_block_a_ready_cpu_grader():
    root = Path(__file__).resolve().parents[2]
    profile = TrainingPerformanceConfig.load(
        root / "configs/training/protocol13_native_carrier_latency.yaml"
    )

    async def run():
        resources = RolloutWorkflowResources(profile.workflow())
        release = asyncio.Event()
        ready = [asyncio.Event() for _ in range(4)]

        async def network_grader(event):
            async with resources.process_graders.lease():
                event.set()
                await release.wait()

        async def code_grader():
            async with resources.process_graders.lease():
                return True

        waiting = [asyncio.create_task(network_grader(event)) for event in ready]
        try:
            await asyncio.gather(*(event.wait() for event in ready))
            assert await asyncio.wait_for(code_grader(), timeout=1)
            assert not release.is_set()
        finally:
            release.set()
            await asyncio.gather(*waiting)

    asyncio.run(run())
