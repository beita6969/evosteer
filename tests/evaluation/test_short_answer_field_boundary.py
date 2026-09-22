"""An owner final field is not its following explanation or another answer."""

import asyncio

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.owner_final import project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_broker_boundary import runtime


@pytest.mark.parametrize("header", ["Final answer:", "Final:", "Terminal payload:"])
@pytest.mark.parametrize("separator", [" ", "\n"])
def test_declared_short_answer_ends_at_its_line(header, separator):
    response = f"Earlier discussion.\n{header}{separator}Public title\nThis explains the choice."
    final = project_owner_final(
        StepZeroTerminalMode.SHORT_ANSWER, response, owner_id="owner", message_id="synthetic"
    )
    assert final is not None
    assert final.payload == "Public title"
    assert final.raw_response == response


def test_conflicting_final_fields_are_not_resolved_by_taking_the_first():
    assert (
        project_owner_final(
            StepZeroTerminalMode.SHORT_ANSWER,
            "Final answer: Public title\nExplanation.\nFinal answer: Different title",
            owner_id="owner",
            message_id="synthetic",
        )
        is None
    )


@pytest.mark.parametrize("response", ["No, the examples differ.", "6", "First and second"])
def test_unmarked_answers_are_not_rewritten_to_match_a_reference(response):
    final = project_owner_final(
        StepZeroTerminalMode.SHORT_ANSWER, response, owner_id="owner", message_id="synthetic"
    )
    assert final is not None
    assert final.payload == response


@pytest.mark.parametrize("benchmark", ["hotpotqa", "triviaqa"])
def test_owner_and_persisted_source_validation_use_the_same_final_field(tmp_path, benchmark):
    fields = {"question": "Which synthetic title?"}
    if benchmark == "hotpotqa":
        fields["context"] = [f"Synthetic public passage {index}." for index in range(10)]
    else:
        fields["public_context"] = "Synthetic public passage."
    entry = PublicTaskView.from_record("synthetic-qa", benchmark, fields)
    response = "Final answer: Public title\nThis explains the choice."
    instance = runtime(tmp_path, entry, [response])
    arm = InferenceArm("A2")
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert final.text == "Public title"
        assert final.submission["raw_response"] == response
        assert final.intervention_counts["model_calls"] == 1
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
