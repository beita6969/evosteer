from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest
import skillev_private.benchmarks.mind2web as private_mind2web
from skillev_private.benchmarks import (
    PrivateMind2WebStepCase,
    PrivateMind2WebStepEvaluator,
    PrivateMind2WebStepSessionFactory,
    PrivateMind2WebStepTarget,
)

import skillev.benchmarks.mind2web as public_mind2web
from skillev.benchmarks import BenchmarkPublicItem, Mind2WebCompletionEnvironment
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    StructuredAction,
)

PRIVATE_VALUE_CANARY = "PRIVATE-MIND2WEB-VALUE-CANARY"
PRIVATE_ACTION_CANARY = "PRIVATE-MIND2WEB-ACTION-CANARY"
POSITIVE_NODE_ID = "node-positive"


def _case(*, task_id: str = "mind2web/annotation-1/action-1") -> PrivateMind2WebStepCase:
    public = BenchmarkPublicItem(
        benchmark_id="mind2web",
        dataset_revision="mind2web-fixture@1",
        split="test_domain",
        task_id=task_id,
        task_family="mind2web/test_domain/travel",
        query="Complete the next public browser action.",
        public_context={
            "action_count": 3,
            "answer_format": "browser-operation-and-element",
            "candidates": [
                {"backend_node_id": "node-negative", "tag": "button"},
                {"backend_node_id": POSITIVE_NODE_ID, "tag": "input"},
            ],
            "domain": "travel",
            "source_format": "mind2web-official-split@1",
            "step_index": 2,
            "subdomain": "booking",
            "website": "fixture.example",
        },
    )
    target = PrivateMind2WebStepTarget(
        task_id=task_id,
        annotation_id="annotation-1",
        action_uid="action-1",
        operation="TYPE",
        original_operation="TYPE",
        value=PRIVATE_VALUE_CANARY,
        positive_backend_node_ids=(POSITIVE_NODE_ID,),
        action_repr=PRIVATE_ACTION_CANARY,
    )
    return PrivateMind2WebStepCase(public=public, target=target)


def _request(
    case: PrivateMind2WebStepCase,
    *,
    operation: str = "TYPE",
    backend_node_id: str = POSITIVE_NODE_ID,
    value: str = PRIVATE_VALUE_CANARY,
) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-mind2web-1",
        task_id=case.public.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(
            {
                "backend_node_id": backend_node_id,
                "operation": operation,
                "value": value,
            }
        ),
        public_transcript_hash=stable_hash({"public": "mind2web-transcript"}),
    )


def _no_submission_request(case: PrivateMind2WebStepCase) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-mind2web-no-submission",
        task_id=case.public.task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )


def test_no_submission_is_zero_without_scoring_private_mind2web_target() -> None:
    case = _case()

    reward = asyncio.run(PrivateMind2WebStepEvaluator(case).evaluate(_no_submission_request(case)))

    assert reward.value == 0.0
    assert reward.success is False
    assert reward.native_payload["step_success"] == 0.0


def _skill_action(skill_id: str = "web-navigation") -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.SKILL,
        name="invoke-skill",
        arguments={},
        resource_id="skill-runtime",
        skill_id=skill_id,
    )


def _tool_action() -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name="click",
        arguments=normalize_json({"target": "node-negative"}),
        resource_id="mind2web",
    )


def test_completion_shape_is_exact_and_answer_free() -> None:
    case = _case()
    environment = Mind2WebCompletionEnvironment(case.public)
    valid = {
        "backend_node_id": POSITIVE_NODE_ID,
        "operation": "TYPE",
        "value": PRIVATE_VALUE_CANARY,
    }

    assert environment.validate_completion(valid)
    for invalid in (
        {"backend_node_id": POSITIVE_NODE_ID, "operation": "TYPE"},
        {**valid, "extra": True},
        {**valid, "operation": "type"},
        {**valid, "backend_node_id": "  "},
        {**valid, "value": 4},
        {**valid, "value": "bad\x00value"},
    ):
        assert not environment.validate_completion(cast(JsonValue, invalid))

    public_surface = {"task": case.public.to_rollout_task().to_value()}
    serialized = canonical_json(public_surface)
    assert PRIVATE_VALUE_CANARY not in serialized
    assert PRIVATE_ACTION_CANARY not in serialized
    assert not hasattr(environment, "checkpoint")
    assert not hasattr(environment, "restore")


def test_known_agent_tool_error_is_distinct_from_unknown_environment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    environment = Mind2WebCompletionEnvironment(case.public)
    with pytest.raises(EnvironmentMethodFailedError) as captured:
        asyncio.run(
            environment.execute(
                _tool_action(),
                step_index=1,
            )
        )
    assert captured.value.budget_usage == BudgetVector(tool_calls=1)

    def fail_observation(skill_id: str) -> object:
        del skill_id
        raise RuntimeError("unexpected Mind2Web environment defect")

    monkeypatch.setattr(public_mind2web, "_skill_observation", fail_observation)
    with pytest.raises(RuntimeError, match="unexpected Mind2Web environment defect"):
        asyncio.run(
            environment.execute(
                _skill_action(),
                step_index=1,
            )
        )


@pytest.mark.parametrize(
    ("operation", "node_id", "value", "expected"),
    [
        (
            "TYPE",
            POSITIVE_NODE_ID,
            PRIVATE_VALUE_CANARY,
            {
                "action_f1": 1.0,
                "element_accuracy": 1.0,
                "operation_accuracy": 1.0,
                "step_success": 1.0,
                "value_f1": 1.0,
            },
        ),
        (
            "TYPE",
            "node-negative",
            PRIVATE_VALUE_CANARY,
            {
                "action_f1": 1.0,
                "element_accuracy": 0.0,
                "operation_accuracy": 1.0,
                "step_success": 0.0,
                "value_f1": 1.0,
            },
        ),
        (
            "CLICK",
            POSITIVE_NODE_ID,
            "",
            {
                "action_f1": 0.0,
                "element_accuracy": 1.0,
                "operation_accuracy": 0.0,
                "step_success": 0.0,
                "value_f1": 0.0,
            },
        ),
    ],
)
def test_official_step_metrics_and_reward(
    operation: str,
    node_id: str,
    value: str,
    expected: dict[str, JsonValue],
) -> None:
    case = _case()
    evaluator = PrivateMind2WebStepEvaluator(case)

    reward = asyncio.run(
        evaluator.evaluate(
            _request(
                case,
                operation=operation,
                backend_node_id=node_id,
                value=value,
            )
        )
    )

    assert {
        key: value for key, value in reward.native_payload.items() if key != "public_metrics"
    } == expected
    assert reward.native_payload["public_metrics"] == {
        "mind2web-action-f1": expected["action_f1"],
        "mind2web-element-accuracy": expected["element_accuracy"],
        "mind2web-operation-accuracy": expected["operation_accuracy"],
        "mind2web-value-f1": expected["value_f1"],
    }
    assert reward.value == expected["step_success"]
    assert reward.success is (reward.value == 1.0)
    payload = canonical_json(reward.native_payload)
    assert PRIVATE_VALUE_CANARY not in payload
    assert PRIVATE_ACTION_CANARY not in payload
    assert POSITIVE_NODE_ID not in payload


def test_official_action_f1_uses_operation_and_value_token_sets() -> None:
    case = _case()
    target = replace(case.target, value="blue cotton")
    evaluator = PrivateMind2WebStepEvaluator(replace(case, target=target))

    reward = asyncio.run(evaluator.evaluate(_request(case, operation="TYPE", value="blue wool")))

    assert reward.native_payload["operation_accuracy"] == 1.0
    assert reward.native_payload["value_f1"] == pytest.approx(0.5)
    assert reward.native_payload["action_f1"] == pytest.approx(2.0 / 3.0)
    assert reward.value == 0.0


def test_missing_official_positive_candidate_is_an_unscorable_zero() -> None:
    case = _case()
    unavailable = replace(
        case,
        target=replace(case.target, positive_backend_node_ids=()),
    )
    evaluator = PrivateMind2WebStepEvaluator(unavailable)

    reward = asyncio.run(evaluator.evaluate(_request(unavailable)))

    assert reward.value == 0.0
    assert reward.success is False
    assert reward.native_payload["element_accuracy"] == 0.0
    assert POSITIVE_NODE_ID not in canonical_json(reward.native_payload)


def test_private_evaluator_rejects_bad_admission_and_unknown_faults_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    evaluator = PrivateMind2WebStepEvaluator(case)
    bad_shape = replace(
        _request(case),
        evaluation_input=SubmittedTerminalValue({"operation": "TYPE"}),
    )
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(evaluator.evaluate(bad_shape))
    wrong_task = replace(_request(case), task_id="mind2web/other/action")
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(evaluator.evaluate(wrong_task))

    def fail_score(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("unexpected private scorer defect")

    monkeypatch.setattr(private_mind2web, "_score_submission", fail_score)
    with pytest.raises(RuntimeError, match="unexpected private scorer defect"):
        asyncio.run(evaluator.evaluate(_request(case)))


def test_session_factory_requires_exact_public_identity() -> None:
    case = _case()
    factory = PrivateMind2WebStepSessionFactory((case,))

    bundle = factory.create(case.public.to_rollout_task())
    assert isinstance(bundle.environment, Mind2WebCompletionEnvironment)
    assert isinstance(bundle.evaluator, PrivateMind2WebStepEvaluator)
    assert bundle.retrieved_skills == ()

    altered = replace(case.public.to_rollout_task(), query="Changed public query")
    with pytest.raises(ValueError):
        factory.create(altered)
    with pytest.raises(ValueError):
        factory.create(replace(altered, task_id="mind2web/missing/action"))
