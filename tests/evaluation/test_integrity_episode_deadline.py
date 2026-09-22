"""A cancelled owner request remains incomplete, with unknown usage recorded."""

import asyncio

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_broker_boundary import runtime


def test_episode_timeout_is_not_a_candidate_or_silent_zero_usage(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, [])
    instance.config["episode_timeout_seconds"] = 0.8
    scope = ("synthetic", "A2", "case")

    class WaitingGenerator:
        async def generate_evaluation(self, request, **kwargs):
            await asyncio.Event().wait()

    instance.generators = [WaitingGenerator()]
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
        failures = instance.journal.traces(scope, "model-transport-failure")
        assert len(failures) == 1
        assert failures[0]["failure_type"] == "CancelledError"
        assert "unknown" in failures[0]["usage"]
        with pytest.raises(KeyError):
            instance.journal.get(*scope)
        assert not instance.journal.model_outputs(scope)
        assert not any(instance.replicas.inflight)
        timing = instance.journal.traces(
            scope, "episode-timing", origin=EventOrigin.MODEL_TRANSPORT
        )
        assert len(timing) == 1
        assert timing[0]["wall_seconds"] >= 0.8
    finally:
        instance.journal.close()
