"""Identical integers cross chat, final-field and native envelopes without another call."""

import asyncio

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.owner_final import parse_explicit_integer_payload, project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode, project_terminal_candidate
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import runtime


def envelope(payload):
    return (
        "<tool_call>\n<function=submit_answer>\n<parameter=answer>\n"
        + payload
        + "\n</parameter>\n</function>\n</tool_call>"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "42",
        "42.",
        r"\boxed{42}",
        r"\boxed{42}.",
        " \t042.\r\n",
        "\\boxed{\n042\n}.",
        "42.\n\\boxed{042}.",
    ],
)
def test_equivalent_final_scalars_have_one_projection_in_every_entry(payload):
    assert parse_explicit_integer_payload(payload) == 42
    for text in (payload, "Final answer: " + payload, envelope(payload)):
        final = project_owner_final(
            StepZeroTerminalMode.AIME_INTEGER, text, owner_id="owner", message_id="case:owner-final"
        )
        assert final.payload == r"\boxed{42}"
        assert final.raw_response == text
    assert project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, payload) == r"\boxed{42}"


@pytest.mark.parametrize(
    "payload",
    [
        r"41 or \boxed{42}.",
        r"41. or \boxed{42}.",
        r"\boxed{41}. or 42.",
        r"\boxed{41}, 42.",
        "41 42",
        "41.\n42.",
        "41\n\\boxed{42}.",
        r"\boxed{41}. \boxed{42}.",
        "42.0",
        "1000.",
        "-1.",
    ],
)
def test_punctuation_compatibility_never_selects_between_values(payload):
    for text in (payload, "Final answer: " + payload, envelope(payload)):
        assert (
            project_owner_final(
                StepZeroTerminalMode.AIME_INTEGER,
                text,
                owner_id="owner",
                message_id="case:owner-final",
            )
            is None
        )


@pytest.mark.parametrize("wrap", [str, lambda text: "Final answer: " + text, envelope])
@pytest.mark.parametrize("payload", ["42.", r"\boxed{42}."])
def test_real_actor_does_not_spend_an_extra_call_removing_a_period(tmp_path, wrap, payload):
    raw = wrap(payload)
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic integer task."})
    instance = runtime(tmp_path, entry, [raw], calls=1)
    try:
        final = asyncio.run(
            instance.generate(
                entry,
                InferenceArm("punctuation", tool_call_mode=ToolCallMode.QWEN_XML),
                "synthetic",
            )
        )
        assert final.text == r"\boxed{42}"
        assert final.submission["raw_response"] == raw
        assert final.intervention_counts["model_calls"] == 1
        assert final.intervention_counts["communication_repairs"] == 0
        assert final.completion_tokens == len(raw)
        assert not instance.journal.traces(
            ("synthetic", "punctuation", "case"), "terminal-parse-failure"
        )
    finally:
        instance.journal.close()


@pytest.mark.parametrize(
    "raw",
    [
        "Final answer: 42\nThis follows from the calculation above.",
        "Intermediate: \\boxed{12}.\nFinal answer: \\boxed{42}\nThe intermediate was 12.",
        "Final answer:\n\\boxed{\n042\n}\nExplanation with intermediate \\boxed{12}.",
        "Final answer: 42\nExplanation.\nFinal answer: 042\nMore explanation.",
    ],
)
def test_explanation_after_declared_integer_is_not_another_final(raw):
    final = project_owner_final(
        StepZeroTerminalMode.AIME_INTEGER, raw, owner_id="owner", message_id="case:owner-final"
    )
    assert final.payload == r"\boxed{42}"
    assert final.raw_response == raw
