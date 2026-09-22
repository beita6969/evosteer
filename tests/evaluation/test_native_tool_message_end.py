"""A complete native call can end with Qwen message framing at the length cap."""

import asyncio
import json
from dataclasses import replace

import pytest

from skillev.evaluation.native_tool_calls import native_corpus_queries, native_tool_call
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import ServingFixture
from tests.evaluation.test_integrity_native_tool_calls import call
from tests.evaluation.test_scienceworld_owner_finish import science_runtime

END = "<|im_end|>"


@pytest.mark.parametrize(
    "envelope",
    [
        call("act", command="look at marker"),
        call("inspect_object", target="marker"),
        call("wait"),
        call("act", command="look at marker").replace(
            "<function=act>\n<parameter=command>", "<function=act(command>"
        ),
        call("look at marker").replace("<function=look at marker>", "<function=look at marker"),
    ],
)
def test_message_framing_preserves_a_complete_literal_tool_call(envelope):
    original = native_tool_call(envelope)
    assert native_tool_call("My decision.\n" + envelope + END) == original
    assert native_tool_call(envelope + "\n" + END + "\n") == original


@pytest.mark.parametrize(
    "response",
    [
        call("act", command="look").replace("</function>", "") + END,
        call("act", command="look").replace("</tool_call>", "") + END,
        call("act", command="look") + END + "Further instructions.",
        call("act", command="look") + END + END,
        call("act", command="look") + END + call("act", command="inventory"),
        "For example:\n" + call("act", command="look") + END,
    ],
)
def test_end_framing_does_not_complete_or_select_an_action(response):
    with pytest.raises(ValueError):
        native_tool_call(response)


def test_literal_marker_inside_an_argument_is_not_deleted():
    answer = f'A literal string: "{END}".'
    response = call("submit_answer", answer=answer) + END
    assert native_tool_call(response).arguments == {"answer": answer}
    first, second = call("corpus_search", query="first"), call("corpus_search", query="second")
    queries = first + "\n" + second + END
    assert native_corpus_queries(queries) == ("first", "second")
    with pytest.raises(ValueError):
        native_corpus_queries(first + END + "\n" + second)


def test_isolated_length_capped_call_executes_once_without_more_generation(monkeypatch, tmp_path):
    response = "My decision.\n" + call("act", command="look at marker") + END
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [])

    class LengthStopped(ServingFixture):
        async def generate_evaluation(self, *args, **kwargs):
            return replace(
                await super().generate_evaluation(*args, **kwargs), finish_reason="length"
            )

    instance.generators = [LengthStopped([response])]
    instance.config["budgets"]["scienceworld"].update(
        total_model_calls=1, total_output_tokens=len(response)
    )
    arm = InferenceArm("framed", tool_call_mode=ToolCallMode.QWEN_XML)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        scope = ("synthetic", arm.arm_id, entry.task_id)
        assert environment.actions == json.loads(final.text) == ["look at marker"]
        assert final.completion_tokens == len(response)
        assert final.intervention_counts["model_calls"] == 1
        assert final.intervention_counts["tool_calls"] == 1
        assert final.intervention_counts["communication_repairs"] == 0
        assert final.intervention_counts["peer_model_calls"] == 0
        outputs = instance.journal.model_outputs(scope)
        assert len(outputs) == 1
        assert outputs[0].final_text == response
        assert outputs[0].result["finish_reason"] == "length"
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
