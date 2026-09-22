import asyncio
import json

import pytest
from skillev_private.experiments.zero_update_bridge import collect_training_condition

from skillev.rollout import UnskilledRolloutSessionBundle
from skillev.training.planning import CollectedTrainingBatch
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev.training.run_condition import EffectiveRunCondition
from tests.training.fakes import OrderedSessionFactory


def test_collect_only_reuses_real_collector_without_advancing_training(
    make_training_harness, tmp_path
):
    harness = make_training_harness()
    library_before = harness.library.state
    projection_before = harness.projections.runtime_state()
    fresh_sessions = OrderedSessionFactory((0.0, 1.0))

    class BaseSessions:
        def create(self, task):
            bundle = fresh_sessions.create(task)
            return UnskilledRolloutSessionBundle(
                environment=bundle.environment, evaluator=bundle.evaluator, cleanup=bundle.cleanup
            )

    condition = EffectiveRunCondition.create(
        condition_id="diagnostic-raw", scientific={"wire": "raw-json", "seed": 0}, execution={}
    )
    resources = RolloutWorkflowResources(RolloutWorkflowBinding())
    result = asyncio.run(
        collect_training_condition(
            root=tmp_path / "diagnostic",
            condition=condition,
            tasks=harness.task_provider.tasks[:2],
            generator=harness.generator,
            base_sessions=BaseSessions(),
            library_state=library_before,
            trainer=harness.config,
            maximum_h0_tokens=2048,
            workflow=RolloutWorkflowBinding(),
            workflow_resources=resources,
            sampling_schedule_id=harness.loop._collector._sampling_schedule_hash,
            ordered_task_sequence_id=harness.loop._collector._ordered_task_sequence_hash,
        )
    )
    assert len(result.diagnostic_artifacts) == 2
    assert result.consumed_budget.model_calls == 4
    assert resources.model_requests.timing.calls == 4
    assert harness.task_provider.cursor == 0
    assert harness.loop.optimizer_step == 0
    assert harness.projections.runtime_state() == projection_before
    assert harness.library.state == library_before
    assert fresh_sessions.cleanup_count == 2
    with pytest.raises(ValueError):
        CollectedTrainingBatch.from_value(result.to_value(), tokenizer=harness.generator.tokenizer)
    assert not (tmp_path / "diagnostic" / "checkpoints").exists()

    progress = json.loads((tmp_path / "diagnostic" / "rollout-progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["elapsed_seconds"] >= 0
    assert len(progress["trajectories"]) == 2
    for row in progress["trajectories"]:
        assert row["environment_seconds"] >= 0
        assert len(row["phases"]) == 2
        assert all(phase["elapsed_seconds"] >= 0 for phase in row["phases"])
    assert progress["performance"] is not None
