import asyncio
from dataclasses import replace

import pytest


def test_valid_artifact_handoff_overlaps_cleanup_but_cleanup_failure_still_aborts(
    make_training_harness, monkeypatch
):
    harness = make_training_harness()
    collector = harness.loop._collector
    create = collector._session_factory.create
    received = []

    def session(task):
        bundle = create(task)

        async def cleanup():
            assert received  # The artifact can already be scored by its owner.
            raise RuntimeError("controlled environment cleanup failure")

        return replace(bundle, cleanup=cleanup)

    monkeypatch.setattr(
        type(collector._session_factory), "create", lambda self, task: session(task)
    )

    class Stream:
        provisional_edges = False

        async def begin(self, plan, snapshot):
            pass

        async def accept(self, position, artifact):
            received.append(artifact)

    with pytest.raises(RuntimeError):
        asyncio.run(collector.collect(optimizer_step=1, gradient_stream=Stream()))
    assert harness.loop.optimizer_step == 0
