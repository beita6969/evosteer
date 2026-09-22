"""A demonstrated literal wait carrier, not a policy for when to advance time."""

import asyncio
import json

import pytest

from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from skillev.evaluation.native_tool_calls import native_control_payload
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_native_tool_calls import call
from tests.evaluation.test_scienceworld_owner_finish import FINISH, science_runtime


@pytest.mark.parametrize(
    "carrier",
    [
        call("wait"),
        "wait()",
        json.dumps(
            {"kind": "tool", "resource_id": "scienceworld", "name": "wait", "arguments": {}}
        ),
    ],
)
def test_literal_wait_retains_native_command_and_public_state(carrier):
    surface = PublicSurface("scienceworld", 4, ("act",))
    decision = ExplicitDecision(carrier, 4)
    assert normalize_decision(decision, surface).action == "wait"
    assert normalize_decision(ExplicitDecision(carrier, 3), surface).action is None
    assert normalize_decision(decision, PublicSurface("scienceworld", 4, ())).action is None
    for mode in ("alfworld", "webshop", "completion"):
        assert normalize_decision(decision, PublicSurface(mode, 4, ("act",))).action is None


@pytest.mark.parametrize(
    "carrier",
    [
        call("wait", duration="10"),
        call("wait", command="focus on vessel"),
        "wait(10)",
        "I might wait, then inspect the vessel.",
        call("wait") + call("act", command="look around"),
        call("wait").replace("</function>", ""),
    ],
)
def test_wait_does_not_infer_duration_complete_truncation_or_choose_actions(carrier):
    surface = PublicSurface("scienceworld", 4, ("act",))
    assert normalize_decision(ExplicitDecision(carrier, 4), surface).action is None


def test_isolated_owner_wait_is_one_action_without_a_repair_or_extra_actor(monkeypatch, tmp_path):
    response = "I decide to wait.\n" + call("wait")
    assert native_control_payload(response) is None
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [response, FINISH])
    arm = InferenceArm("literal-wait", tool_call_mode=ToolCallMode.QWEN_XML)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert environment.actions == ["wait"]
        assert json.loads(final.text) == environment.actions
        assert final.intervention_counts["model_calls"] == 2
        assert final.intervention_counts["tool_calls"] == 2  # wait and explicit finish
        assert final.intervention_counts["communication_repairs"] == 0
        assert final.intervention_counts["peer_model_calls"] == 0
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
