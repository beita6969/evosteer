"""One owner's literal API binding, without object inference or safer-action selection."""

import asyncio
import json

import pytest

from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from skillev.evaluation.direct_baseline.interactive_tasks import NativePublicState
from skillev.evaluation.native_tool_calls import native_control_payload, native_tool_call
from skillev.evaluation.scienceworld_commands import (
    COMMAND_PROFILE,
    MATERIAL_CONTACT_COMMAND_PROFILE,
    PERSISTENT_FOCUS_COMMAND_PROFILE,
    SCOPED_TYPED_COMMAND_PROFILE,
    TYPED_COMMAND_PROFILE,
    TYPED_COMMANDS,
    command_profile,
    native_functions,
)
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V9,
    PUBLIC_TASK_SEMANTICS_V10,
    PUBLIC_TASK_SEMANTICS_V12,
    PUBLIC_TASK_SEMANTICS_V13,
    PUBLIC_TASK_SEMANTICS_V14,
    public_task_semantics,
)
from tests.evaluation.test_integrity_native_tool_calls import call
from tests.evaluation.test_scienceworld_owner_finish import FINISH, science_runtime


@pytest.mark.parametrize("header", ["act(command", "act(command)"])
def test_observed_signature_header_keeps_its_one_literal_command(header):
    raw = (
        "I choose the vessel, not its contents.\n<tool_call>\n"
        f"<function={header}>\nfocus on glass jar\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    parsed = native_tool_call(raw)
    assert parsed.name == "act"
    assert parsed.arguments == {"command": "focus on glass jar"}
    surface = PublicSurface("scienceworld", 1, ("act",))
    assert normalize_decision(ExplicitDecision(raw, 1), surface).action == "focus on glass jar"
    with pytest.raises(ValueError):
        native_tool_call(raw + "\n" + call("inspect_object", target="plant"))
    with pytest.raises(ValueError):
        native_tool_call(raw.replace("focus on glass jar", "<parameter=other>plant"))
    assert (
        normalize_decision(
            ExplicitDecision(raw.replace("focus on glass jar", "focus on jar\nlook at plant"), 1),
            surface,
        ).action
        is None
    )


@pytest.mark.parametrize("name", list(TYPED_COMMANDS))
@pytest.mark.parametrize("carrier", ["native", "json", "function"])
@pytest.mark.parametrize("target", ["pot 4 (in room A)", "around", "task"])
@pytest.mark.parametrize(
    "profile",
    [
        TYPED_COMMAND_PROFILE,
        SCOPED_TYPED_COMMAND_PROFILE,
        PERSISTENT_FOCUS_COMMAND_PROFILE,
        MATERIAL_CONTACT_COMMAND_PROFILE,
    ],
)
def test_literal_binding_retains_the_owner_target(name, carrier, target, profile):
    text = (
        call(name, target=target)
        if carrier == "native"
        else json.dumps(
            {
                "kind": "tool",
                "resource_id": "scienceworld",
                "name": name,
                "arguments": {"target": target},
            }
        )
        if carrier == "json"
        else f"{name}({target!r})"
    )
    surface = PublicSurface("scienceworld", 1, native_functions(profile))
    result = normalize_decision(ExplicitDecision(text, 1), surface)
    assert result.action == TYPED_COMMANDS[name] + target
    assert (
        normalize_decision(
            ExplicitDecision(text, 1), PublicSurface("scienceworld", 1, ("act",))
        ).action
        is None
    )
    assert normalize_decision(ExplicitDecision(text, 0), surface).action is None
    if carrier == "native":
        assert native_control_payload(text) is None


@pytest.mark.parametrize("target", ["", "\n", "pot\nfocus on plant", "pot\x00"])
def test_typed_binding_never_executes_a_multiline_or_empty_target(target):
    result = normalize_decision(
        ExplicitDecision(call("select_task_object", target=target), 1),
        PublicSurface("scienceworld", 1, native_functions(TYPED_COMMAND_PROFILE)),
    )
    assert result.action is None


def test_typed_names_are_opt_in_and_other_semantics_are_inherited():
    assert command_profile(PUBLIC_TASK_SEMANTICS_V10) == TYPED_COMMAND_PROFILE
    assert command_profile(PUBLIC_TASK_SEMANTICS_V9) == COMMAND_PROFILE
    for version in (COMMAND_PROFILE, TYPED_COMMAND_PROFILE):
        tools = native_tool_definitions("scienceworld", scienceworld_profile=version)
        names = {row["function"]["name"] for row in tools}
        assert set(TYPED_COMMANDS).issubset(names) is (version == TYPED_COMMAND_PROFILE)
        assert "act" in names
    for domain in (
        "musique",
        "nq-open",
        "math-hard",
        "gpqa-diamond-bioorganic",
        "apps-introductory",
    ):
        assert public_task_semantics(
            domain, input_profile="released-ood-source@1", version=PUBLIC_TASK_SEMANTICS_V10
        ) == public_task_semantics(
            domain, input_profile="released-ood-source@1", version=PUBLIC_TASK_SEMANTICS_V9
        )


@pytest.mark.parametrize(
    ("version", "profile"),
    [
        (PUBLIC_TASK_SEMANTICS_V10, TYPED_COMMAND_PROFILE),
        (PUBLIC_TASK_SEMANTICS_V12, SCOPED_TYPED_COMMAND_PROFILE),
        (PUBLIC_TASK_SEMANTICS_V13, PERSISTENT_FOCUS_COMMAND_PROFILE),
        (PUBLIC_TASK_SEMANTICS_V14, MATERIAL_CONTACT_COMMAND_PROFILE),
    ],
)
def test_real_isolated_owner_binds_distinct_operations_and_keeps_free_act(
    monkeypatch, tmp_path, version, profile
):
    outputs = [
        call("inspect_object", target="vessel"),
        call("select_task_object", target="vessel"),
        call("act", command="1"),
        FINISH,
    ]
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, outputs)

    async def reset():
        return NativePublicState("A vessel containing a sprout.", native_functions(profile))

    original_step = environment.step

    async def step(action):
        from dataclasses import replace

        result = await original_step(action)
        return replace(result, available_actions=native_functions(profile))

    environment.reset, environment.step = reset, step
    arm = InferenceArm(
        "typed-sw",
        tool_call_mode=ToolCallMode.QWEN_XML,
        task_semantic_guidance=version,
    )
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert environment.actions == ["look at vessel", "focus on vessel", "1"]
        assert json.loads(final.text) == environment.actions
        assert final.intervention_counts["model_calls"] == 4
        # finish is a public tool call, but is not a simulator action.
        assert final.intervention_counts["tool_calls"] == 4
        assert final.intervention_counts["peer_model_calls"] == 0
        assert final.intervention_counts["communication_repairs"] == 0
        assert profile in json.dumps(instance.tokenizer.messages)
        requests = instance.journal.traces(
            ("synthetic", arm.arm_id, entry.task_id), "rendered-request"
        )
        for request in requests:
            current_tools = [
                message["content"]
                for message in request["messages"]
                if message["role"] == "tool"
                and "Available environment function" in message["content"]
            ]
            assert current_tools
            assert all(name in current_tools[-1] for name in TYPED_COMMANDS)
            assert all(name in request["messages"][0]["content"] for name in TYPED_COMMANDS)
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
