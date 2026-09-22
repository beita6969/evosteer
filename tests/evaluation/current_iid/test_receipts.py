import pytest

from skillev.evaluation.current_iid.coverage import CurrentIIDResultStatus, project_status
from skillev.evaluation.current_iid.receipts import CurrentIIDRunReceipt
from skillev.evaluation.current_iid.targets import BenchmarkTarget, MetricTarget, TargetStatus
from skillev.experiments.protocol_v11 import BenchmarkV11


def _receipt(**changes: object) -> CurrentIIDRunReceipt:
    values: dict[str, object] = {
        "benchmark": BenchmarkV11.HUMAN_EVAL,
        "condition_id": "humaneval@1",
        "attempt_id": "attempt-1",
        "planned_count": 128,
        "final_record_count": 128,
        "definitive_verdict_count": 128,
        "candidate_failure_count": 0,
        "generation_infrastructure_failures": 0,
        "scorer_infrastructure_failures": 0,
        "environment_infrastructure_failures": 0,
        "metrics": {"pass_at_1": 90.0},
        "population_id": "panel",
        "prompt_profile": "prompt",
        "decoding_profile": "greedy",
        "actor_route": "base",
        "context_length": 4096,
        "adapter_active": False,
        "tool_surface": (),
        "grader_profile": None,
    }
    values.update(changes)
    return CurrentIIDRunReceipt(**values)  # type: ignore[arg-type]


def test_formal_receipt_rejects_adapter_and_incomplete_coverage() -> None:
    with pytest.raises(ValueError):
        _receipt(adapter_active=True)
    with pytest.raises(ValueError):
        _receipt(final_record_count=127)


def test_condition_mismatch_and_undefined_target_cannot_pass() -> None:
    receipt = _receipt()
    mismatch = BenchmarkTarget(
        benchmark=BenchmarkV11.HUMAN_EVAL,
        condition_id="other@1",
        status=TargetStatus.EXACT,
        metrics=(MetricTarget("pass_at_1", 92.0),),
    )
    assert project_status(receipt, mismatch)[0] is CurrentIIDResultStatus.PROTOCOL_MISMATCH
    undefined = BenchmarkTarget(BenchmarkV11.HUMAN_EVAL, None, TargetStatus.UNDEFINED)
    assert project_status(receipt, undefined)[0] is CurrentIIDResultStatus.TARGET_UNDEFINED
