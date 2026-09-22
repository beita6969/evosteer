"""Independent synthetic interfaces; no benchmark questions or solutions."""

import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.contracts.skill_invocation import SkillInvocationAdmissionError
from skillev.rollout.action_contract import ActionContract
from skillev.rollout.action_surface import (
    ActionSurface,
    CompletionSpec,
    TerminalMode,
    ToolActionSpec,
)
from skillev.rollout.codec import StructuredJsonActionCodec
from skillev.runtime.contracts import ActionKind, StructuredAction


def surface():
    return ActionSurface(
        TerminalMode.EXPLICIT_COMPLETION,
        tools=(ToolActionSpec("public", "read", {"key": "non-empty string"}, {"key": "EXAMPLE"}),),
        completion=CompletionSpec({"answer": "non-empty string"}, {"answer": "EXAMPLE"}),
    )


def test_rendered_action_examples_round_trip_through_same_contract():
    contract = ActionContract.freeze(surface())
    codec = StructuredJsonActionCodec()
    examples = [line for line in contract.render_public_instruction() if line.startswith("{")]
    assert len(examples) == 2
    for example in examples:
        parsed = codec.parse(example)
        action, error = contract.admit_surface(parsed.action)
        assert error is None
        assert json.loads(example)["arguments"] == action.arguments
    assert contract.to_scoring_metadata()["probability"] == "raw-full-vocabulary"


def test_frozen_contract_does_not_share_mutable_schema_or_examples():
    original = surface()
    contract = ActionContract.freeze(original)
    before = contract.render_public_instruction()
    original.tools[0].example_arguments["key"] = "MUTATED"
    original.tools[0].arguments_schema["key"] = "integer"
    assert contract.render_public_instruction() == before
    action = StructuredAction(ActionKind.TOOL, "read", {"key": 42}, resource_id="public")
    assert contract.admit_surface(action)[1] is not None


@pytest.mark.parametrize(
    "change", ["extra-argument", "wrong-tool", "wrong-resource", "empty-value"]
)
def test_public_tool_admission_rejects_without_repair(change):
    action = StructuredAction(ActionKind.TOOL, "read", {"key": "x"}, resource_id="public")
    if change == "extra-argument":
        action = replace(action, arguments={"key": "x", "extra": "y"})
    elif change == "wrong-tool":
        action = replace(action, name="write")
    elif change == "wrong-resource":
        action = replace(action, resource_id="private")
    else:
        action = replace(action, arguments={"key": ""})
    admitted, error = ActionContract.freeze(surface()).admit_surface(action)
    assert admitted == action
    assert error is not None


def test_environment_terminal_and_h0_skill_visibility_remain_distinct():
    contract = ActionContract.freeze(
        ActionSurface(TerminalMode.ENVIRONMENT),
        retrieved_skill_ids=("visible",),
        active_skill_ids=("hidden", "visible"),
    )
    complete = StructuredAction(ActionKind.COMPLETE, "complete", {"value": {"answer": "x"}})
    assert contract.admit_surface(complete)[1] is not None
    skill = StructuredAction(
        ActionKind.SKILL, "invoke", {}, resource_id="skills", skill_id="hidden"
    )
    with pytest.raises(SkillInvocationAdmissionError):
        contract.invoked_skill_ids(skill)
    assert contract.invoked_skill_ids(replace(skill, skill_id="visible")) == ("visible",)
    text = canonical_json(complete.to_value())
    assert (
        StructuredJsonActionCodec().parse("Here are choices: " + text + " " + text).action is None
    )


def test_completion_shape_validation_is_not_a_reward_or_answer_checker():
    contract = ActionContract.freeze(surface())
    action = StructuredAction(ActionKind.COMPLETE, "complete", {"value": {"answer": "WRONG"}})
    admission = contract.validate(
        action,
        validate_completion=lambda value: isinstance(value, dict) and bool(value.get("answer")),
    )
    assert admission.error is None
    assert admission.completion
    assert admission.submission == {"answer": "WRONG"}
    malformed = replace(action, arguments={"answer": "not-the-declared-carrier"})
    assert contract.validate(malformed, validate_completion=lambda _: True).error is not None
    empty = replace(action, arguments={"value": {"answer": ""}})
    assert contract.validate(empty, validate_completion=lambda value: bool(value["answer"])).error
