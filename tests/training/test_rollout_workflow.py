from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from skillev.training import (
    AsyncResourceLimiter,
    ResourceTiming,
    RolloutBatchWorkflow,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
)


def _binding(*, resident: int = 4, model: int = 2) -> RolloutWorkflowBinding:
    return RolloutWorkflowBinding(
        max_resident_trajectories=resident,
        max_inflight_model_requests=model,
        max_inflight_environment_calls=2,
        max_inflight_terminal_evaluations=2,
        max_inflight_process_graders=1,
        transport_worker_threads=model,
    )


def test_workflow_binding_round_trips_without_changing_scientific_batch_size() -> None:
    binding = _binding(resident=8, model=4)

    assert RolloutWorkflowBinding.from_value(binding.to_value()) == binding
    assert "batch_size" not in binding.to_value()


def test_workflow_runs_concurrently_but_returns_planned_order() -> None:
    async def scenario() -> tuple[tuple[int, ...], int]:
        workflow = RolloutBatchWorkflow[int, int](_binding(resident=4, model=2))
        active = 0
        high_water = 0

        async def execute(item: int) -> int:
            nonlocal active, high_water
            active += 1
            high_water = max(high_water, active)
            await asyncio.sleep((4 - item) * 0.005)
            active -= 1
            return item

        return await workflow.run((1, 2, 3, 4), execute), high_water

    results, high_water = asyncio.run(scenario())
    assert results == (1, 2, 3, 4)
    assert high_water == 4


def test_resource_limiter_enforces_limit_and_records_queue_and_service() -> None:
    async def scenario() -> tuple[int, ResourceTiming, ResourceTiming]:
        limiter = AsyncResourceLimiter(2)
        active = 0
        high_water = 0

        async def execute() -> None:
            nonlocal active, high_water
            async with limiter.lease():
                active += 1
                high_water = max(high_water, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(execute() for _ in range(5)))
        timing = limiter.timing
        limiter.reset_timing()
        return high_water, timing, limiter.timing

    high_water, timing, reset = asyncio.run(scenario())
    assert high_water == 2
    assert timing.calls == 5
    assert timing.high_water_mark == 2
    assert timing.queue_seconds > 0
    assert timing.service_seconds > 0

    assert reset == ResourceTiming(0, 0, 0.0, 0.0)


@dataclass(slots=True)
class _InfrastructureError(RuntimeError):
    marker: str


def test_first_failure_stops_launching_and_drains_started_work() -> None:
    async def scenario() -> tuple[list[int], list[int], _InfrastructureError]:
        workflow = RolloutBatchWorkflow[int, int](_binding(resident=2))
        started: list[int] = []
        completed: list[int] = []
        both_started = asyncio.Event()
        release_second = asyncio.Event()

        async def execute(item: int) -> int:
            started.append(item)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            if item == 1:
                raise _InfrastructureError("failed")
            await release_second.wait()
            completed.append(item)
            return item

        async def release_after_failure() -> None:
            while len(started) < 2:
                await asyncio.sleep(0)
            await asyncio.sleep(0.01)
            release_second.set()

        release = asyncio.create_task(release_after_failure())
        with pytest.raises(_InfrastructureError) as captured:
            await workflow.run((1, 2, 3, 4), execute)
        await release
        return started, completed, captured.value

    started, completed, failure = asyncio.run(scenario())
    assert started == [1, 2]
    assert completed == [2]
    assert failure.marker == "failed"


def test_slow_terminal_evaluation_does_not_hold_the_model_lane() -> None:
    async def scenario() -> tuple[bool, int, int]:
        binding = _binding(resident=2, model=1)
        resources = RolloutWorkflowResources(binding)
        workflow = RolloutBatchWorkflow[int, int](binding)
        evaluator_started = asyncio.Event()
        model_finished_while_evaluator_waited = asyncio.Event()
        release_evaluator = asyncio.Event()

        async def execute(item: int) -> int:
            if item == 1:
                async with resources.terminal_evaluations.lease():
                    evaluator_started.set()
                    await release_evaluator.wait()
                return item
            await evaluator_started.wait()
            async with resources.model_requests.lease():
                await asyncio.sleep(0)
                model_finished_while_evaluator_waited.set()
            release_evaluator.set()
            return item

        result = await workflow.run((1, 2), execute)
        assert result == (1, 2)
        return (
            model_finished_while_evaluator_waited.is_set(),
            resources.model_requests.timing.high_water_mark,
            resources.terminal_evaluations.timing.high_water_mark,
        )

    overlapped, model_high_water, evaluator_high_water = asyncio.run(scenario())
    assert overlapped
    assert model_high_water == 1
    assert evaluator_high_water == 1
