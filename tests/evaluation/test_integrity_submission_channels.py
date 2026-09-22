"""The actual actor advertises only the submission channel its broker accepts."""

import asyncio
import json

import pytest
from skillev_private.evaluation import integrity_runtime

from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.evaluation.thinking_policy import ThinkingPolicy
from tests.evaluation.test_aime_thinking_authority import policy as default_policy
from tests.evaluation.test_integrity_broker_boundary import runtime


def policy():
    """These channel tests deliberately exercise an explicit thinking-on condition."""
    baseline = default_policy()
    return ThinkingPolicy(
        "explicit-test-thinking-on", tuple((name, True) for name, _ in baseline.rows)
    )


@pytest.mark.parametrize(
    ("benchmark", "action"), [("webshop", "click[Proceed]"), ("alfworld", "look")]
)
def test_interactive_owner_and_control_feedback_do_not_offer_an_unsupported_chat_final(
    monkeypatch, tmp_path, benchmark, action
):
    entry = PublicTaskView.from_record("case", benchmark, {"task": "Observe the fictional scene."})
    instance = runtime(tmp_path, entry, ["Message to solver:\n", "Action: " + action], calls=2)
    instance.source.interactive["case"] = {"case": {"max_steps": 1}}
    executed = []

    class Environment:
        async def reset(self):
            return NativePublicState(
                "Your task is to: Observe the fictional scene.\nPublic scene.", (action,)
            )

        async def step(self, value):
            executed.append(value)
            return NativeEnvironmentStep("Scene observed.", True, 0.0, False, True)

        async def outcome(self):
            return NativeEnvironmentOutcome(0.0, False, True)

        async def close(self):
            pass

    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *args, **kwargs: Environment()
    )
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert json.loads(final.text) == executed == [action]
    requests = instance.journal.traces(
        ("synthetic", "A2", "case"), "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    for request in requests:
        system = request["messages"][0]["content"]
        assert "native action" in system
        assert "final answer" not in system
        assert "Message to solver:" not in system
        assert "Available peers:" not in system
    # No tool executed: the runtime's repair notice is addressed to the owner,
    # not presented as a return from an unexecuted tool.
    repair = "\n".join(m["content"] for m in requests[-1]["messages"] if m["role"] == "user")
    assert "submit a native action" in repair
    assert "final answer" not in repair
    assert final.intervention_counts["communication_repairs"] == 1
    assert final.intervention_counts["model_calls"] == 2
    instance.journal.close()


def test_noninteractive_owner_can_still_submit_its_final_after_control_feedback(tmp_path):
    entry = PublicTaskView.from_record(
        "case", "aime-2026", {"problem": "Compute a synthetic integer."}
    )
    instance = runtime(tmp_path, entry, ["Message to solver:\n", "Final answer: 42"], calls=2)
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert final.text == r"\boxed{42}"
    requests = instance.journal.traces(
        ("synthetic", "A2", "case"), "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    for request in requests:
        system = request["messages"][0]["content"]
        assert "submit your final answer" in system
        assert "native action" not in system
    repair = "\n".join(m["content"] for m in requests[-1]["messages"] if m["role"] == "tool")
    assert "submit your final answer" in repair
    assert final.intervention_counts["communication_repairs"] == 1
    assert final.intervention_counts["model_calls"] == 2
    instance.journal.close()


def test_bold_qa_final_crosses_actor_and_broker_with_its_original_provenance(tmp_path):
    entry = PublicTaskView.from_record(
        "synthetic-qa",
        "triviaqa",
        {"question": "Name the fictional tree.", "public_context": "An excerpt about trees."},
    )
    response = "The draft discusses Juniper.\n**Final answer: Alder**"
    instance = runtime(tmp_path, entry, ["Think carefully.</think>" + response])
    instance.thinking_policy = policy()
    arm = InferenceArm("A2", native_thinking=True)
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    assert final.text == "Alder"
    assert final.submission["raw_response"] == response
    assert final.intervention_counts["model_calls"] == 1
    assert final.intervention_counts["communication_repairs"] == 0
    instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    instance.journal.close()


@pytest.mark.parametrize("native_tools", [False, True])
def test_mbpp_public_call_contract_reaches_owner_without_changing_source(tmp_path, native_tools):
    prompt = '"""Return twice the supplied integer.\nassert double_value(3) == 6\n"""\n'
    entry = PublicTaskView.from_record("synthetic-code", "mbpp-plus", {"prompt": prompt})
    source = "def double_value(n):\n    return 2 * n"
    output = (
        "<tool_call>\n<function=submit_answer>\n<parameter=answer>\n"
        + source
        + "\n</parameter>\n</function>\n</tool_call>"
        if native_tools
        else "```py\n" + source + "\n```"
    )
    arm = InferenceArm(
        "A2",
        tool_call_mode=ToolCallMode.QWEN_XML if native_tools else ToolCallMode.PLAIN_TEXT,
    )
    instance = runtime(tmp_path, entry, [output])
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    requests = instance.journal.traces(
        ("synthetic", "A2", entry.task_id),
        "rendered-request",
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    system = "\n".join(m["content"] for m in requests[0]["messages"] if m["role"] == "system")
    assert "function name" in system
    assert "public examples" in system
    assert any(
        m["role"] == "user" and m["content"] == entry.render() for m in requests[0]["messages"]
    )
    assert final.text == source
    assert final.submission["raw_response"] == output
    assert final.intervention_counts["model_calls"] == 1
    assert not final.intervention_counts["peer_model_calls"]
    instance.journal.close()


def test_mbpp_wrong_function_name_is_not_rewritten_or_retried_by_framework(tmp_path):
    entry = PublicTaskView.from_record(
        "synthetic-code",
        "mbpp-plus",
        {"prompt": '"""Return twice the integer.\nassert double_value(3) == 6\n"""\n'},
    )
    source = "def a_different_name(n):\n    return 2 * n"
    instance = runtime(tmp_path, entry, [source])
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert final.text == source
    assert final.intervention_counts["model_calls"] == 1
    assert not final.intervention_counts["communication_repairs"]
    instance.journal.close()


def test_configured_mbpp_thinking_reaches_owner_and_final_source_is_not_reasoning(tmp_path):
    entry = PublicTaskView.from_record(
        "synthetic-code", "mbpp-plus", {"prompt": "Implement twice. Example: twice(3) == 6."}
    )
    source = "def twice(n):\n    return n * 2"
    output = (
        "Use the public call signature. </think>"
        "<tool_call>\n<function=submit_answer>\n<parameter=answer>\n"
        + source
        + "\n</parameter>\n</function>\n</tool_call>"
    )
    instance = runtime(tmp_path, entry, [output])
    instance.thinking_policy = policy()
    arm = InferenceArm("A2", tool_call_mode=ToolCallMode.QWEN_XML)
    final = asyncio.run(instance.generate(entry, arm, "synthetic"))
    scope = ("synthetic", "A2", entry.task_id)
    records = instance.journal.model_outputs(scope)
    assert len(records) == 1
    assert records[0].native_thinking
    assert instance.generators[0].profiles[0].enable_thinking
    assert final.text == source
    assert "Use the public call signature" not in final.text
    instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    instance.journal.close()
