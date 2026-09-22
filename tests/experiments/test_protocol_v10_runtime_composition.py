from __future__ import annotations

import pytest
from skillev_private.experiments.protocol_v10_attempt_builder import (
    ProtocolV10TaskProviderFactory,
)

from skillev.rollout import RolloutTask


def _tasks() -> tuple[RolloutTask, ...]:
    return tuple(
        RolloutTask(
            task_id=f"episode-{index}",
            environment_id="environment",
            task_family="hotpotqa",
            context_id=f"context-{index}",
            query=f"question {index}",
            available_tools=(),
            public_context={"benchmark_id": "hotpotqa"},
        )
        for index in range(3)
    )


def test_protocol_v10_task_provider_preserves_sealed_order_and_resume_cursor() -> None:
    factory = ProtocolV10TaskProviderFactory(_tasks(), "protocol-v10-selection")
    provider = factory.fresh()

    assert provider.next_task().task_id == "episode-0"
    resumed = factory.from_exact_state(provider.runtime_state)
    assert resumed.next_task().task_id == "episode-1"
    assert resumed.next_task().task_id == "episode-2"
    with pytest.raises(RuntimeError):
        resumed.next_task()


def test_protocol_v10_task_provider_rejects_a_cursor_from_another_selection() -> None:
    factory = ProtocolV10TaskProviderFactory(_tasks(), "protocol-v10-selection")
    another = ProtocolV10TaskProviderFactory(_tasks(), "another-selection").fresh()

    with pytest.raises(ValueError):
        factory.from_exact_state(another.runtime_state)
