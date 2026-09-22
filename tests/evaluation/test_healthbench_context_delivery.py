"""Complete public dialogue reaches the owner; private rubrics never do."""

import asyncio
import json

import pytest
from skillev_private.evaluation.integrity_sources import public_source_view

from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import ServingFixture, runtime
from tests.evaluation.test_integrity_native_tool_calls import call


@pytest.mark.parametrize("multiturn", [False, True])
@pytest.mark.parametrize("system", [False, True])
def test_health_conversation_roles_content_and_full_response_survive(tmp_path, multiturn, system):
    prompt = [{"role": "user", "content": "Please discuss the fictional situation above."}]
    if multiturn:
        prompt = [
            {"role": "user", "content": "Fictional history: café, 日常; no diagnosis supplied."},
            {"role": "assistant", "content": "Which part would you like to discuss?"},
            *prompt,
        ]
    if system:
        prompt.insert(0, {"role": "system", "content": "Reply in the user's language."})
    source = {
        "prompt": prompt,
        "rubrics": [{"criterion": "SCORER_ONLY_CRITERION", "points": 3}],
        "answer": "SCORER_ONLY_REFERENCE",
    }
    entry = public_source_view("synthetic-health", "healthbench", source)
    answer = "Here is the discussion.\n\nFinal answer: A complete response.\nFurther explanation."
    instance = runtime(tmp_path, entry, [answer])

    class RecordingServing(ServingFixture):
        async def generate_evaluation(self, request, **kwargs):
            self.request = request
            return await super().generate_evaluation(request, **kwargs)

    serving = RecordingServing([answer])
    instance.generators = [serving]
    scope = ("synthetic", "A2", entry.task_id)
    try:
        final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
        assert final.text == answer
        assert final.intervention_counts["model_calls"] == 1
        assert final.intervention_counts["peer_model_calls"] == 0
        assert json.loads(dict(entry.fields)["prompt"]) == prompt
        (trace,) = instance.journal.traces(
            scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
        )
        messages = trace["messages"]
        assert messages[0]["role"] == "system"
        assert not any(row["role"] in {"system", "tool"} for row in messages[1:])
        assert messages[1:] == [row for row in prompt if row["role"] != "system"]
        if system:
            assert prompt[0]["content"] in messages[0]["content"]
        assert "preceding dialogue" in messages[0]["content"]
        text = "\n".join(row["content"] for row in messages)
        assert "SCORER_ONLY" not in text
        assert trace["archived_message_count"] == 0
        assert (
            tuple(
                instance.tokenizer.encode_integrity_messages(
                    tuple(messages), enable_thinking=False, tools=tuple(trace["tools"])
                )
            )
            == serving.request.input_ids
        )
        instance.validate_candidate(instance.journal, entry, InferenceArm("A2"), "synthetic")
    finally:
        instance.journal.close()


@pytest.mark.parametrize(
    "answer",
    [
        "Could you clarify which part of the fictional situation you mean?",
        "I can discuss what you described.\n\nWhich part would you like to start with?",
    ],
)
@pytest.mark.parametrize("envelope", [False, True])
def test_owner_can_read_initial_dialogue_then_submit_a_conversational_reply(
    tmp_path, answer, envelope
):
    prompt = [
        {"role": "user", "content": "Fictional background: café, 日常."},
        {"role": "assistant", "content": "What would you like to discuss?"},
        {"role": "user", "content": "The earlier situation."},
    ]
    entry = public_source_view(
        "synthetic-health",
        "healthbench",
        {"prompt": prompt, "rubrics": [{"criterion": "SCORER_ONLY", "points": 3}]},
    )
    outputs = [
        call("history", archive="environment", cursor=0, limit=4),
        call("history", archive="conversation", cursor=0, limit=8),
        call("submit_answer", answer=answer) if envelope else answer,
    ]
    instance = runtime(tmp_path, entry, outputs)
    arm = InferenceArm("A2", tool_call_mode=ToolCallMode.QWEN_XML)
    scope = ("synthetic", "A2", entry.task_id)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert final.text == answer
        assert final.intervention_counts["model_calls"] == 3
        assert final.intervention_counts["tool_calls"] == 2
        assert final.intervention_counts["communication_repairs"] == 0
        assert final.intervention_counts["peer_model_calls"] == 0
        pages = instance.journal.traces(scope, "history-read", origin=EventOrigin.ACTOR_DIAGNOSTIC)
        assert pages[0]["page"]["entries"] == []
        assert "external records" in pages[0]["page"]["scope"]
        conversation = pages[1]["page"]["entries"]
        assert [
            {"role": row["role"], "content": row["content"]}
            for row in conversation
            if row["role"] != "system"
        ] == prompt
        traces = instance.journal.traces(
            scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
        )
        for trace in traces:
            assert trace["messages"][1 : len(prompt) + 1] == prompt
            assert "SCORER_ONLY" not in json.dumps(trace)
            functions = {tool["function"]["name"]: tool["function"] for tool in trace["tools"]}
            assert (
                "conversation"
                in functions["history"]["parameters"]["properties"]["archive"]["enum"]
            )
            assert "clarification" in functions["submit_answer"]["description"]
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
