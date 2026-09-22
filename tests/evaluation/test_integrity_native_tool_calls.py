"""Model-native tools use literal payloads and the existing owner/peer boundaries."""

import asyncio
from types import SimpleNamespace

import pytest
from skillev_private.evaluation.integrity_runtime import PublicServiceTokenizer

from skillev.evaluation.agent_communication import peer_request
from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.native_tool_calls import native_control_payload, native_tool_call
from skillev.evaluation.native_tool_instructions import native_tool_instructions
from skillev.evaluation.owner_final import project_owner_final
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_completion import StepZeroTerminalMode
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import runtime


def call(name, **arguments):
    parameters = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def test_agreeing_final_field_and_single_submission_are_one_owner_answer():
    envelope = call("submit_answer", answer="C")
    response = "Discussion.\nFinal answer: C\n\n" + envelope
    assert native_tool_call(response).arguments == {"answer": "C"}
    assert native_control_payload(response) is None
    with pytest.raises(ValueError):
        native_tool_call("Final answer: D\n" + envelope)
    with pytest.raises(ValueError):
        native_tool_call("Final answer: C\n" + envelope + "\n" + envelope)


def test_repeated_final_can_accompany_explanation_without_rewriting_payload():
    payload = "My explanation.\nFinal answer: C\nAdditional explanation."
    response = "Final answer: C\n" + call("submit_answer", answer=payload)
    assert native_tool_call(response).arguments == {"answer": payload}
    for rejected in (
        "Explanation mentioning C without a final declaration.",
        "Final answer: C\nFinal answer: D",
        "```text\nFinal answer: C\n```\nExplanation without a declaration.",
    ):
        with pytest.raises(ValueError):
            native_tool_call("Final answer: C\n" + call("submit_answer", answer=rejected))


def test_native_peer_body_preserves_math_code_quotes_and_whitespace():
    body = '\nPlease inspect \\boxed{17}.\n\ndef f():\n    return "x < y & z"\n'
    response = "I would like another perspective.\n" + call("solver", body=body)
    assert peer_request(response) == ("solver", body)


def test_history_integer_parameters_are_transport_values_not_expressions():
    decoded = native_control_payload(call("history", archive="discussion", cursor=0, limit=4))
    assert decoded["cursor"] == 0
    assert decoded["limit"] == 4
    with pytest.raises(ValueError):
        native_control_payload(call("history", archive="discussion", cursor="1+1", limit=4))


@pytest.mark.parametrize("name", ["look", "inventory", "help"])
def test_native_observation_functions_preserve_explicit_command_and_state(name):
    surface = PublicSurface("alfworld", 3, (name, "go to desk 1"))
    response = call(name)
    assert native_control_payload(response) is None  # Not an attempted peer message.
    assert normalize_decision(ExplicitDecision(response, 3), surface).action == name
    assert normalize_decision(ExplicitDecision(f"{name}()", 3), surface).action == name
    assert normalize_decision(ExplicitDecision(response, 2), surface).action is None
    assert (
        normalize_decision(ExplicitDecision(response, 3), PublicSurface("alfworld", 3, ())).action
        is None
    )
    assert (
        normalize_decision(
            ExplicitDecision(response, 3), PublicSurface("webshop", 3, (name,))
        ).action
        is None
    )
    assert (
        normalize_decision(ExplicitDecision(call(name, command="look"), 3), surface).action is None
    )


def test_native_observation_functions_are_public_zero_argument_interfaces():
    functions = {
        item["function"]["name"]: item["function"] for item in native_tool_definitions("alfworld")
    }
    for name in ("look", "inventory", "help"):
        parameters = functions[name]["parameters"]
        assert not parameters["properties"]
        assert not parameters["required"]
    other = native_tool_definitions("webshop") + native_tool_definitions("completion")
    assert not any(item["function"]["name"] in ("look", "inventory", "help") for item in other)


@pytest.mark.parametrize(
    "command",
    [
        "go to shelf 2",
        "examine cabinet 3",
        "take marker 2 from desk 1",
        "move marker 2 to drawer 4",
        "cool item 1 with appliance 2",
        "use lamp 3",
    ],
)
@pytest.mark.parametrize("header_ending", [">", "", "\n</parameter>"])
def test_complete_literal_command_in_function_slot_keeps_owner_choice(command, header_ending):
    surface = PublicSurface("alfworld", 9, (command, "look"))
    envelope = call(command).replace(f"{command}>", command + header_ending)
    response = "I will perform this action.\n" + envelope
    assert native_control_payload(response) is None
    assert normalize_decision(ExplicitDecision(response, 9), surface).action == command
    assert normalize_decision(ExplicitDecision(response, 8), surface).action is None
    assert (
        normalize_decision(
            ExplicitDecision(response, 9), PublicSurface("alfworld", 9, ("look",))
        ).action
        is None
    )
    assert (
        normalize_decision(
            ExplicitDecision(response, 9), PublicSurface("webshop", 9, (command,))
        ).action
        is None
    )
    assert (
        normalize_decision(ExplicitDecision(call(command, command="look"), 9), surface).action
        is None
    )
    assert normalize_decision(ExplicitDecision("Example:\n" + envelope, 9), surface).action is None
    assert normalize_decision(ExplicitDecision(envelope + call("look"), 9), surface).action is None


@pytest.mark.parametrize(
    "body",
    [
        "<function=go to shelf 2\nlook\n</function>",
        "<function=go to shelf 2\n<parameter=command>look</parameter></function>",
        "<function=go to shelf 2\n</parameter></parameter></function>",
        "<function=go to shelf 2\n</parameter>",
        "<function=act\n<parameter=command>go to shelf 2</parameter></function>",
    ],
)
def test_literal_header_repair_does_not_invent_or_discard_argument_content(body):
    response = f"<tool_call>\n{body}\n</tool_call>"
    with pytest.raises(ValueError):
        native_tool_call(response)
    surface = PublicSurface("alfworld", 9, ("go to shelf 2", "look"))
    assert normalize_decision(ExplicitDecision(response, 9), surface).action is None


@pytest.mark.parametrize(
    "command",
    ["click[Red Mug]", "click[next >]", "click[b000000001]", "search[blue ceramic mug]"],
)
@pytest.mark.parametrize("header_ending", [">", "", "\n</parameter>"])
def test_bracket_command_in_native_function_slot_keeps_literal_owner_argument(
    command, header_ending
):
    envelope = call(command).replace(f"{command}>", command + header_ending)
    surface = PublicSurface("webshop", 9, ("search", "click[Red Mug]", command))
    assert native_control_payload(envelope) is None
    assert normalize_decision(ExplicitDecision(envelope, 9), surface).action == command
    assert normalize_decision(ExplicitDecision(envelope, 8), surface).action is None
    assert (
        normalize_decision(
            ExplicitDecision(envelope, 9), PublicSurface("webshop", 9, ("click[Other]",))
        ).action
        is None
    )
    assert (
        normalize_decision(
            ExplicitDecision(envelope, 9), PublicSurface("alfworld", 9, (command,))
        ).action
        is None
    )
    assert normalize_decision(ExplicitDecision("Example:\n" + envelope, 9), surface).action is None
    assert (
        normalize_decision(
            ExplicitDecision(envelope + call("click", target="Other"), 9), surface
        ).action
        is None
    )


@pytest.mark.parametrize(
    "body",
    [
        "<function=click[Red Mug]>\n<parameter=target>Other</parameter></function>",
        "<function=click[Red Mug]\n<parameter=target>Other</parameter></function>",
        "<function=click[Red Mug\n</parameter></function>",
        "<function=click[Red Mug]\n</parameter></parameter></function>",
    ],
)
def test_bracket_function_slot_does_not_replace_or_invent_target(body):
    response = f"<tool_call>\n{body}\n</tool_call>"
    surface = PublicSurface("webshop", 9, ("click[Red Mug]", "click[Other]"))
    assert normalize_decision(ExplicitDecision(response, 9), surface).action is None


@pytest.mark.parametrize(
    "response",
    [
        call("click", target="Red Mug") + call("click", target="Other"),
        call("solver", body="request") + "I am not submitting.",
        "Example:\n" + call("solver", body="request"),
        "Do not execute:\n" + call("solver", body="request"),
        "Action: click[Other]\n" + call("click", target="Red Mug"),
        "Final answer: 17\n" + call("submit_answer", answer="42"),
        "click[Other]\n" + call("click", target="Red Mug"),
        call("solver", body="request").replace(
            "</function>", "<parameter=body>x</parameter></function>"
        ),
        call("solver", body="request").removesuffix("</tool_call>"),
    ],
)
def test_ambiguous_or_unsubmitted_native_envelope_is_not_selected(response):
    with pytest.raises(ValueError):
        native_tool_call(response)


def test_tool_mentions_and_fenced_examples_remain_discussion():
    assert native_tool_call("An inline <tool_call> mention is not an envelope.") is None
    assert (
        native_tool_call("Example:\n```text\n" + call("solver", body="request") + "\n```") is None
    )


@pytest.mark.parametrize("target", ["Red Mug", '"Red Mug"', "red mug"])
def test_native_tool_call_does_not_rewrite_target_or_public_state(target):
    surface = PublicSurface("webshop", 2, ("click[Red Mug]",))
    decision = ExplicitDecision(call("click", target=target), 2)
    result = normalize_decision(decision, surface)
    assert result.action == ("click[Red Mug]" if target == "Red Mug" else None)
    assert normalize_decision(ExplicitDecision(decision.text, 1), surface).action is None


@pytest.mark.parametrize(
    ("mode", "answer", "expected"),
    [
        (StepZeroTerminalMode.AIME_INTEGER, "42", r"\boxed{42}"),
        (StepZeroTerminalMode.AIME_INTEGER, "017\n\\boxed{17}", r"\boxed{17}"),
        (
            StepZeroTerminalMode.PYTHON_SOURCE,
            'def f():\n    return "x < y"',
            'def f():\n    return "x < y"',
        ),
        (
            StepZeroTerminalMode.NATURAL_LANGUAGE,
            "Final answer: ordinary clinical prose.",
            "Final answer: ordinary clinical prose.",
        ),
    ],
)
def test_only_the_owner_designated_native_payload_is_projected(mode, answer, expected):
    text = "Here is my response.\n" + call("submit_answer", answer=answer)
    submitted = project_owner_final(mode, text, owner_id="owner", message_id="final")
    assert submitted.payload == expected
    assert submitted.raw_response == text
    assert (
        project_owner_final(mode, call("solver", body=answer), owner_id="owner", message_id="final")
        is None
    )


def test_public_tokenizer_passes_tools_into_the_actual_chat_template():
    captured = []
    tokenizer = object.__new__(PublicServiceTokenizer)
    tokenizer.inner = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: captured.append(kwargs) or [1, 2]
    )
    tools = native_tool_definitions("completion", peers=("solver", "researcher"))
    assert tokenizer.encode_integrity_messages(
        ({"role": "user", "content": "Task"},), enable_thinking=False, tools=tools
    ) == [1, 2]
    assert captured[0]["tools"] == list(tools)
    assert captured[0]["enable_thinking"] is False
    assert not any(tool["function"]["name"] == "search" for tool in tools)


@pytest.mark.parametrize("mode", ["webshop", "alfworld"])
def test_native_function_instructions_do_not_compete_with_the_tool_template(mode):
    instructions = native_tool_instructions(mode, native_functions=True)
    assert "Action:" not in instructions
    assert '"kind":"tool"' not in instructions
    definitions = native_tool_definitions(mode, peers=("solver",))
    functions = {tool["function"]["name"]: tool["function"] for tool in definitions}
    if mode == "webshop":
        assert "query" in functions["search"]["parameters"]["properties"]
        assert "target" in functions["click"]["parameters"]["properties"]
    else:
        assert "command" in functions["act"]["parameters"]["properties"]
    assert "body" in functions["solver"]["parameters"]["properties"]


def test_native_consultation_is_rejected_without_an_extra_model_or_tool(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    body = "Please check this calculation: \\boxed{17}."
    owner_final = call("submit_answer", answer="42")
    instance = runtime(tmp_path, entry, [call("solver", body=body), owner_final], calls=2)
    arm = InferenceArm("A2", tool_call_mode=ToolCallMode.QWEN_XML)
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    assert final.text == r"\boxed{42}"
    assert final.submission["raw_response"] == owner_final
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["peer_model_calls"] == 0
    scope = ("synthetic", "A2", "case")
    requests = instance.journal.traces(
        scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    transports = instance.journal.traces(
        scope, "model-transport-start", origin=EventOrigin.MODEL_TRANSPORT
    )
    assert requests[0]["tools"] == transports[0]["tools"]
    assert not any(
        tool["function"]["name"] in {"solver", "researcher"} for tool in transports[0]["tools"]
    )
    assert transports[1]["tools"] == transports[0]["tools"]
    assert all(row["participant"] == "owner" for row in requests)
    owner_system = requests[0]["messages"][0]["content"]
    assert '"kind":"history"' not in owner_system
    assert "Message to solver:" not in owner_system
    assert body in "\n".join(row["content"] for row in requests[1]["messages"])
    assert not instance.journal.traces(scope, "agent-reply", origin=EventOrigin.ACTOR_DIAGNOSTIC)
    instance.journal.close()


@pytest.mark.parametrize(
    ("benchmark", "public"),
    [
        ("musique", {"question": "Which tree?", "context": "A fictional public passage."}),
        ("nq-open", {"question": "Which tree?"}),
    ],
)
def test_qa_native_submission_crosses_the_actual_actor_broker_without_another_solver(
    tmp_path, benchmark, public
):
    entry = PublicTaskView.from_record("native-qa", benchmark, public)
    response = "I compared the available information.\n" + call("submit_answer", answer="Alder")
    instance = runtime(tmp_path, entry, [response])
    arm = InferenceArm("native-qa", tool_call_mode=ToolCallMode.QWEN_XML)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert final.text == "Alder"
        assert final.submission["raw_response"] == response
        assert final.intervention_counts["model_calls"] == 1
        assert final.intervention_counts["peer_model_calls"] == 0
        assert final.intervention_counts["extra_model_calls_for_serialization"] == 0
        scope = ("synthetic", arm.arm_id, entry.task_id)
        rendered = instance.journal.traces(scope, "rendered-request")
        transported = instance.journal.traces(scope, "model-transport-start")
        assert len(rendered) == len(transported) == 1
        assert rendered[0]["tools"] == transported[0]["tools"]
        functions = {tool["function"]["name"]: tool["function"] for tool in transported[0]["tools"]}
        assert functions["submit_answer"]["parameters"]["properties"]["answer"]["type"] == "string"
        assert not {"solver", "researcher", "search"}.intersection(functions)
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()


def test_malformed_native_call_spends_a_real_owner_repair_call(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(
        tmp_path,
        entry,
        [call("nonexistent", body="request"), call("submit_answer", answer="42")],
        calls=2,
    )
    arm = InferenceArm("A2", tool_call_mode=ToolCallMode.QWEN_XML)
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    assert final.text == r"\boxed{42}"
    assert final.intervention_counts["communication_repairs"] == 1
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["extra_model_calls_for_serialization"] == 0
    instance.journal.close()
