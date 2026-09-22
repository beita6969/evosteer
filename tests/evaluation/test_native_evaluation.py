import pytest

from skillev.evaluation.native_evaluation import NativeEvaluation


def test_native_metrics_are_distinct_from_training_projection() -> None:
    outcome = NativeEvaluation(
        benchmark="hotpotqa",
        task_id="public-task",
        metrics={"em": 0.0, "f1": 0.75},
        training_value=0.75,
        success=False,
        candidate_valid=True,
        infrastructure_ok=True,
    )
    assert outcome.metrics["em"] == 0.0
    assert outcome.metrics["f1"] == outcome.training_value


def test_infrastructure_failure_cannot_be_admitted_as_candidate_zero() -> None:
    with pytest.raises(ValueError):
        NativeEvaluation(
            benchmark="webshop",
            task_id="public-task",
            metrics={},
            training_value=0.0,
            success=False,
            candidate_valid=True,
            infrastructure_ok=False,
        )
