import pytest

from skillev.evaluation.current_iid.conditions import (
    EvaluationCondition,
    EvaluationConditionKind,
)


def _direct(**changes: object) -> EvaluationCondition:
    values: dict[str, object] = {
        "condition_id": "direct@1",
        "kind": EvaluationConditionKind.DIRECT_BACKBONE,
        "actor_model": "Qwen3.5-9B",
        "actor_route": "base",
        "actor_service_profile": "sglang@1",
        "context_length": 1024,
        "adapter_policy": "forbidden",
        "prompt_profile": "prompt@1",
        "decoding_profile": "decode@1",
        "tool_surface": (),
        "environment_profile": None,
        "scorer_profile": "scorer@1",
        "grader_profile": None,
    }
    values.update(changes)
    return EvaluationCondition(**values)  # type: ignore[arg-type]


def test_direct_forbids_auxiliary_models_and_adapters() -> None:
    _direct()
    with pytest.raises(ValueError):
        _direct(auxiliary_models=("gpt",))
    with pytest.raises(ValueError):
        _direct(adapter_policy="allowed")
