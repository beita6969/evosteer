"""Private Mind2Web offline action-prediction evaluation.

Only the answer-free candidate projection crosses into the rollout session.
Positive element identities and gold operations remain in this private wheel.
"""

from __future__ import annotations

from dataclasses import dataclass

from skillev.benchmarks.mind2web import (
    MIND2WEB_OPERATIONS,
    Mind2WebCompletionEnvironment,
)
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.training import RolloutSessionBundle

from .source_cases import PrivateMind2WebStepCase, PrivateMind2WebStepTarget
from .terminal_inputs import no_submission_reward, submitted_value


def _submission(value: JsonValue) -> tuple[str, str, str]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != {
        "backend_node_id",
        "operation",
        "value",
    }:
        raise TerminalEvaluatorError("admitted Mind2Web completion has incompatible fields")
    operation = normalized["operation"]
    backend_node_id = normalized["backend_node_id"]
    action_value = normalized["value"]
    if type(operation) is not str or operation not in MIND2WEB_OPERATIONS:
        raise TerminalEvaluatorError("admitted Mind2Web operation is unsupported")
    if type(backend_node_id) is not str or not backend_node_id.strip():
        raise TerminalEvaluatorError("admitted Mind2Web backend node identity is invalid")
    if type(action_value) is not str or "\x00" in action_value:
        raise TerminalEvaluatorError("admitted Mind2Web operation value is invalid")
    return operation, backend_node_id, action_value


def _token_set_f1(prediction: str, target: str) -> float:
    """Match the official offline evaluator's set-of-whitespace-tokens F1."""

    predicted_tokens = set(prediction.strip().split())
    target_tokens = set(target.strip().split())
    if not predicted_tokens and not target_tokens:
        return 1.0
    if not predicted_tokens or not target_tokens:
        return 0.0
    overlap = len(predicted_tokens & target_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class Mind2WebStepMetrics:
    """Aggregate components of the official offline step prediction score."""

    element_accuracy: float
    operation_accuracy: float
    value_f1: float
    action_f1: float
    step_success: float

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_f1": self.action_f1,
            "element_accuracy": self.element_accuracy,
            "operation_accuracy": self.operation_accuracy,
            "step_success": self.step_success,
            "value_f1": self.value_f1,
        }


def _score_submission(
    target: PrivateMind2WebStepTarget,
    *,
    operation: str,
    backend_node_id: str,
    value: str,
) -> Mind2WebStepMetrics:
    element_accuracy = float(backend_node_id in target.positive_backend_node_ids)
    operation_accuracy = float(operation == target.operation)
    value_f1 = _token_set_f1(value, target.value)
    predicted_action = f"{operation} {value}".strip()
    expected_action = f"{target.operation} {target.value}".strip()
    action_f1 = _token_set_f1(predicted_action, expected_action)
    step_success = float(element_accuracy == 1.0 and action_f1 == 1.0)
    return Mind2WebStepMetrics(
        element_accuracy=element_accuracy,
        operation_accuracy=operation_accuracy,
        value_f1=value_f1,
        action_f1=action_f1,
        step_success=step_success,
    )


def score_mind2web_submission(
    target: PrivateMind2WebStepTarget,
    *,
    operation: str,
    backend_node_id: str,
    value: str,
) -> Mind2WebStepMetrics:
    """Score one direct native action with the official offline metric."""

    if not isinstance(target, PrivateMind2WebStepTarget):
        raise TypeError("Mind2Web target has an incompatible type")
    return _score_submission(
        target,
        operation=operation,
        backend_node_id=backend_node_id,
        value=value,
    )


@dataclass(slots=True)
class PrivateMind2WebStepEvaluator:
    """Evaluate one static Mind2Web action while retaining all gold privately."""

    case: PrivateMind2WebStepCase

    def __post_init__(self) -> None:
        if not isinstance(self.case, PrivateMind2WebStepCase):
            raise TypeError("private Mind2Web evaluator requires a step case")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.case.public.task_id:
            raise TerminalEvaluatorError("terminal request reached a different Mind2Web step")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="mind2web-step-success",
                native_payload={
                    "action_f1": 0.0,
                    "element_accuracy": 0.0,
                    "operation_accuracy": 0.0,
                    "public_metrics": {
                        "mind2web-action-f1": 0.0,
                        "mind2web-element-accuracy": 0.0,
                        "mind2web-operation-accuracy": 0.0,
                        "mind2web-value-f1": 0.0,
                    },
                    "step_success": 0.0,
                    "value_f1": 0.0,
                },
                environment_id=self.case.public.environment_id,
                verifier_version="mind2web-offline-action-evaluator@1",
            )
        operation, backend_node_id, value = _submission(submitted_value(request))
        metrics = _score_submission(
            self.case.target,
            operation=operation,
            backend_node_id=backend_node_id,
            value=value,
        )
        success = metrics.step_success == 1.0
        native_payload: dict[str, JsonValue] = {
            **metrics.to_value(),
            "public_metrics": {
                "mind2web-action-f1": metrics.action_f1,
                "mind2web-element-accuracy": metrics.element_accuracy,
                "mind2web-operation-accuracy": metrics.operation_accuracy,
                "mind2web-value-f1": metrics.value_f1,
            },
        }
        return TerminalReward(
            value=metrics.step_success,
            success=success,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="mind2web-step-success",
            native_payload=native_payload,
            environment_id=self.case.public.environment_id,
            verifier_version="mind2web-offline-action-evaluator@1",
        )


@dataclass(slots=True)
class PrivateMind2WebStepSessionFactory:
    """Pair exact answer-free task projections with isolated private targets."""

    cases: tuple[PrivateMind2WebStepCase, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private Mind2Web session factory requires cases")
        if any(not isinstance(case, PrivateMind2WebStepCase) for case in self.cases):
            raise TypeError("private Mind2Web session factory cases are invalid")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private Mind2Web cases must have unique task identities")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("Mind2Web session creation requires RolloutTask")
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private Mind2Web case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public Mind2Web task projection differs from its private case")
        return RolloutSessionBundle(
            environment=Mind2WebCompletionEnvironment(case.public),
            evaluator=PrivateMind2WebStepEvaluator(case),
            retrieved_skills=(),
        )


__all__ = [
    "Mind2WebStepMetrics",
    "PrivateMind2WebStepEvaluator",
    "PrivateMind2WebStepSessionFactory",
    "score_mind2web_submission",
]
