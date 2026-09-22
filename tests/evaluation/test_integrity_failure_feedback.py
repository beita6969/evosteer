"""Public failures explain the actual operation, not a physical state or peer route."""

import asyncio
from dataclasses import replace

from skillev.evaluation.agent_communication import control_failure_feedback
from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
    public_repair_feedback,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import ServingFixture, runtime
from tests.evaluation.test_integrity_native_tool_calls import call


def test_unavailable_command_feedback_does_not_describe_a_physical_surface():
    surface = PublicSurface("alfworld", 3, ("look", "inventory", "go to cabinet 1"))
    result = normalize_decision(ExplicitDecision("Action: inspect marker 1", 3), surface)
    assert result.action is None
    feedback = public_repair_feedback(result, surface)
    assert result.error not in feedback
    assert "available-action list" in feedback
    assert "no state change" in feedback
    assert all(action in feedback for action in surface.native_actions)
    assert "surface" not in feedback
    assert "output-token limit" not in feedback


def test_a_generation_limit_is_reported_without_modifying_the_command():
    surface = PublicSurface("alfworld", 2, ("look",))
    result = normalize_decision(
        ExplicitDecision("I am still discussing.\nNo action yet.", 2), surface
    )
    assert result.action is None
    feedback = public_repair_feedback(result, surface, finish_reason="length")
    assert "output-token limit" in feedback
    assert "not sent to the environment" in feedback
    assert "look" in feedback  # Availability is retained, not automatically selected.


def test_plain_message_feedback_does_not_claim_a_native_envelope_failure():
    feedback = control_failure_feedback(
        "Message to researcher:\n", direct_submission="a native action", finish_reason="stop"
    )
    assert "not decoded or delivered" in feedback
    assert "submit a native action" in feedback
    assert "closing tags" not in feedback
    assert "final answer" not in feedback
    assert "solver" not in feedback


def test_truncated_owner_answer_is_not_redirected_to_a_peer_at_actual_broker(tmp_path):
    entry = PublicTaskView.from_record("case", "humaneval", {"prompt": "def f():\n    pass\n"})
    incomplete = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\ndef f():\n"
    source = "def f():\n    return 3"
    outputs = [incomplete, call("submit_answer", answer=source)]
    instance = runtime(tmp_path, entry, outputs, calls=2)

    class LimitedServing(ServingFixture):
        async def generate_evaluation(self, request, **kwargs):
            first = len(self.outputs) == 2
            result = await super().generate_evaluation(request, **kwargs)
            return replace(result, finish_reason="length" if first else "stop")

    instance.generators = [LimitedServing(outputs)]
    arm = InferenceArm("A2", tool_call_mode=ToolCallMode.QWEN_XML)
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    assert final.text == source
    assert final.submission["raw_response"] == outputs[-1]
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["communication_repairs"] == 1
    assert final.intervention_counts["peer_model_calls"] == 0
    scope = ("synthetic", "A2", "case")
    requests = instance.journal.traces(
        scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    feedback = "\n".join(m["content"] for m in requests[-1]["messages"] if m["role"] == "tool")
    assert "native tool call" in feedback
    assert "closing tags" in feedback
    assert "output-token limit" in feedback
    assert "submit your final answer" in feedback
    assert "solver" not in feedback
    for stage, origin in (
        ("model-response", EventOrigin.ACTOR_DIAGNOSTIC),
        ("model-transport-complete", EventOrigin.MODEL_TRANSPORT),
    ):
        events = instance.journal.traces(scope, stage, origin=origin)
        assert events[0]["finish_reason"] == "length"
        assert events[1]["finish_reason"] == "stop"
    instance.journal.close()
