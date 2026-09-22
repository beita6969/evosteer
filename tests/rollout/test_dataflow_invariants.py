from types import SimpleNamespace

import pytest

from skillev.diagnostics.dataflow import (
    TrainingDataflowInvariantChecker,
    TrainingDataflowInvariantError,
)


def test_no_skill_condition_rejects_skill_bearing_artifact() -> None:
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            condition_id="structured-no-skill",
            task_id="task",
            trajectory_id="trajectory",
        ),
        initial_context=SimpleNamespace(
            contract=SimpleNamespace(
                meta={"task_id": "task"},
                active_skill_ids=("skill",),
                retrieved_skill_ids=(),
            )
        ),
        record=SimpleNamespace(trajectory_id="trajectory"),
    )
    checker = TrainingDataflowInvariantChecker(frozenset({"structured-no-skill"}))
    with pytest.raises(TrainingDataflowInvariantError):
        checker.check_rollout(artifact, condition_id="structured-no-skill")  # type: ignore[arg-type]
