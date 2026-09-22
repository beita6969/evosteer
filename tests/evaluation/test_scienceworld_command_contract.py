"""Interface fixes explain, but never choose, execute, or score owner actions."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.scienceworld_native_diagnostics import ScienceWorldNativeDiagnostics
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldStepResult
from skillev_private.evaluation.ood_scienceworld import OODScienceWorldEnvironment

from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.native_tool_instructions import native_tool_reference
from skillev.evaluation.scienceworld_commands import (
    COMMAND_PROFILE,
    COMMANDS,
    SCOPED_TYPED_COMMAND_PROFILE,
    TYPED_COMMAND_PROFILE,
    command_profile,
    task_contract,
)
from skillev.evaluation.scienceworld_references import ScienceWorldReferences
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V6,
    PUBLIC_TASK_SEMANTICS_V7,
    PUBLIC_TASK_SEMANTICS_V8,
    evaluation_submission_instruction,
    public_task_semantics,
)
from tests.evaluation.test_scienceworld_owner_finish import FINISH, science_runtime

AMBIGUOUS = (
    "Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n"
    "0:\tfocus on sample (in vessel A)\n1:\tfocus on sample (in vessel B)\n"
)


def test_versioned_system_tool_and_task_semantics_share_the_same_catalog():
    for version in (
        PUBLIC_TASK_SEMANTICS_V6,
        PUBLIC_TASK_SEMANTICS_V7,
        PUBLIC_TASK_SEMANTICS_V8,
    ):
        assert command_profile(version) == COMMAND_PROFILE
    new = native_tool_reference("scienceworld", scienceworld_profile=COMMAND_PROFILE)
    assert all(command.help() in new for command in COMMANDS)
    tools = native_tool_definitions("scienceworld", scienceworld_profile=COMMAND_PROFILE)
    act = next(item["function"] for item in tools if item["function"]["name"] == "act")
    assert COMMANDS[0].help() in act["description"]
    assert COMMANDS[0].task_selection
    assert "not an inspection" in new
    assert "including any required initial" in new
    assert "not restricted to the final step" in new
    assert new != native_tool_reference("scienceworld")
    for version in (
        PUBLIC_TASK_SEMANTICS_V6,
        PUBLIC_TASK_SEMANTICS_V7,
        PUBLIC_TASK_SEMANTICS_V8,
    ):
        for profile in ("released-ood-source@1", "training-public-source-bridge@1"):
            assert (
                public_task_semantics("scienceworld", input_profile=profile, version=version)
                == task_contract()
            )


def test_gpqa_submission_wire_allows_reasoning_but_identifies_one_final_choice():
    instruction = evaluation_submission_instruction("gpqa-diamond-bioorganic")
    assert "After any explanation" in instruction
    assert "Final answer:" in instruction
    assert "A, B, C or D" in instruction


@pytest.mark.parametrize("reply", ["0", "1", "99", "look around", "annotated text"])
def test_choice_is_local_and_consumed_without_substitution(reply):
    references = ScienceWorldReferences()
    references.observe("focus on sample", AMBIGUOUS, terminal=False)
    assert references.pending.command == "focus on sample"
    assert references.pending.public_revision == 2
    assert references.pending.options[1] == ("1", "focus on sample (in vessel B)")
    references.observe(reply, "Confirmation or native cancellation", terminal=False)
    assert references.pending is None
    references.observe("1", "No known action matches that input.", terminal=False)
    assert references.pending is None
    references.observe("focus on another sample", AMBIGUOUS, terminal=False)
    assert references.pending.command == "focus on another sample"
    assert references.pending.public_revision == 5


@pytest.mark.parametrize(
    "profile", [COMMAND_PROFILE, TYPED_COMMAND_PROFILE, SCOPED_TYPED_COMMAND_PROFILE]
)
def test_literal_container_choice_and_native_negative_diagnostics_stay_private(profile):
    commands = []
    state = {
        "current_look": "a vessel (containing a sprout)\n\tan empty pot",
        "inventory": "a meter",
        "task_description": "A public goal",
    }
    results = iter(
        [
            OfficialScienceWorldStepResult(
                AMBIGUOUS, 10, False, public_state=state, native_moves=3
            ),
            OfficialScienceWorldStepResult(
                "You focus on the vessel.",
                -100,
                True,
                public_state=state,
                native_moves=4,
                private_diagnostics={"failure_branch": "hidden-goal-branch", "secret": "private"},
            ),
        ]
    )
    native = SimpleNamespace(step=lambda action: commands.append(action) or next(results))
    env = OODScienceWorldEnvironment(
        native,
        "synthetic",
        0,
        0,
        200,
        observation_profile="public-state@1",
        command_profile=profile,
    )
    env._raw_score = 10
    env._native_moves = 3
    first = asyncio.run(env.step("focus on vessel"))
    assert "Pending command: focus on vessel" in first.observation
    assert state["current_look"] in first.observation
    last = asyncio.run(env.step("1"))
    assert commands == ["focus on vessel", "1"]
    assert "hidden-goal" not in last.observation
    assert "-100" not in last.observation
    assert env.references.pending is None
    assert env.last_transition.native_score == -100
    assert env.last_transition.previous_simulator_moves == 3
    assert env.last_transition.native_diagnostics["failure_branch"] == "hidden-goal-branch"
    assert last.terminal
    assert not last.success


@pytest.mark.parametrize(
    ("raw_score", "moves", "steps", "horizon"),
    [
        (13, 211, 32, True),
        (40, 200, 32, False),
        (40, 100, 200, True),
        (-100, 211, 32, False),
        (100, 211, 32, False),
        (40, None, 32, False),
    ],
)
def test_native_clock_horizon_is_distinct_from_action_count_and_failure(
    raw_score, moves, steps, horizon
):
    env = OODScienceWorldEnvironment(SimpleNamespace(), "synthetic", 0, 0, 200)
    env._raw_score = raw_score
    env._score = max(0, raw_score) / 100
    env._native_moves = moves
    env._steps = steps
    env._terminal = True
    result = asyncio.run(env.outcome())
    assert result.terminated_by_horizon is horizon
    assert result.native_final_score == raw_score
    assert result.success is (raw_score == 100)


def test_diagnostic_failure_is_explicit_not_a_second_action_or_fake_branch():
    diagnostics = ScienceWorldNativeDiagnostics(SimpleNamespace())
    diagnostics.reset({"score": 10, "moves": 1})
    result = diagnostics.transition("focus on vessel", {"score": -100, "moves": 2})
    assert result["after"]["native_score"] == -100
    assert result["after"]["diagnostic_status"] == "unavailable"
    assert result["first_failure_this_step"] is None


def test_unknown_previous_goal_state_does_not_become_a_claimed_first_failure(monkeypatch):
    diagnostics = ScienceWorldNativeDiagnostics(SimpleNamespace())
    diagnostics.previous = {"diagnostic_status": "unavailable"}
    monkeypatch.setattr(diagnostics, "snapshot", lambda info: {"failed": True})
    result = diagnostics.transition("look at vessel", {"score": -100})
    assert result["first_failure_this_step"] is None
    assert result["first_failure"] is None


def test_private_diagnostic_latches_first_failure_not_the_latest_observation(monkeypatch):
    diagnostics = ScienceWorldNativeDiagnostics(SimpleNamespace())
    states = iter(
        [
            {"failed": False, "simulator_moves": 1},
            {"failed": True, "simulator_moves": 2, "failure_branch": "native predicate"},
            {"failed": True, "simulator_moves": 2, "failure_branch": "native predicate"},
        ]
    )
    monkeypatch.setattr(diagnostics, "snapshot", lambda info: next(states))
    diagnostics.reset({})
    first = diagnostics.transition("focus on vessel", {})
    later = diagnostics.transition("look around", {})
    assert first["first_failure_this_step"]
    assert not later["first_failure_this_step"]
    assert later["first_failure"]["command"] == "focus on vessel"
    assert later["first_failure"]["failure_branch"] == "native predicate"


def test_real_broker_renders_new_contract_without_rewriting_the_owner_command(
    monkeypatch, tmp_path
):
    call = (
        "<tool_call><function=act><parameter=command>focus on vessel"
        "</parameter></function></tool_call>"
    )
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [call, FINISH])
    arm = InferenceArm(
        "api-v2",
        tool_call_mode=ToolCallMode.QWEN_XML,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V6,
    )
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert environment.actions == ["focus on vessel"]
        assert final.intervention_counts["peer_model_calls"] == 0
        rendered = json.dumps(instance.tokenizer.messages)
        assert COMMAND_PROFILE in rendered
        assert task_contract() in rendered
        assert "not an inspection" in rendered
    finally:
        instance.journal.close()
