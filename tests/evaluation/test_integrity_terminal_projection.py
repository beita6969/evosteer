"""The owner, not the framework or scorer, chooses an explicit final payload."""

import pytest

from skillev.evaluation.owner_final import parse_explicit_integer_payload, project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode, project_terminal_candidate


def project(mode, text):
    return project_owner_final(mode, text, owner_id="owner", message_id="task:owner-final")


def test_explicit_integer_final_is_distinct_from_a_discussed_draft():
    response = "Draft calculation: \\boxed{12}\nThis may be revised.\nFinal answer: 42"
    final = project(StepZeroTerminalMode.AIME_INTEGER, response)
    assert final.raw_response == response
    assert final.payload == r"\boxed{42}"
    assert final.owner_id == "owner"
    assert project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, response) is None


@pytest.mark.parametrize(
    "payload",
    ["42 or 12", "42\n12", "1000", "-1", "answer is 42", "42\n42\n12", "17.0", "1717"],
)
def test_integer_payload_is_not_a_search_for_the_last_or_best_number(payload):
    assert parse_explicit_integer_payload(payload) is None
    assert project(StepZeroTerminalMode.AIME_INTEGER, "Final answer: " + payload) is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("017\n\n\\boxed{17}", 17),
        ("\\boxed{\n000\n}\n0", 0),
        ("999\n\\boxed{999}", 999),
    ],
)
def test_equivalent_scalar_notations_in_one_final_field_are_not_conflicting(payload, expected):
    response = "Draft calculation: \\boxed{12}\nFinal answer: " + payload
    final = project(StepZeroTerminalMode.AIME_INTEGER, response)
    assert parse_explicit_integer_payload(payload) == expected
    assert final.payload == rf"\boxed{{{expected}}}"
    assert final.raw_response == response


def test_multiple_active_finals_remain_ambiguous():
    assert project(StepZeroTerminalMode.AIME_INTEGER, "Final answer: 12\nFinal answer: 42") is None


@pytest.mark.parametrize(
    "response",
    [
        "Final answer: 17\nFinal answer: 017",
        "Final answer:\n\\boxed{17}\nDiscussion continues.\nFinal answer: 17\n\\boxed{17}",
        "Final answer: 0\nI will check the reasoning.\nFinal answer:\n\\boxed{\n000\n}",
        "Final answer: 999\nFurther discussion.\nFinal response: 999\n\\boxed{999}",
    ],
)
def test_equivalent_repeated_final_declarations_preserve_legacy_compatibility(response):
    expected = project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, response)
    assert expected is not None
    final = project(StepZeroTerminalMode.AIME_INTEGER, response)
    assert final.payload == expected
    assert final.raw_response == response


@pytest.mark.parametrize(
    "response",
    [
        "Final answer: 17\nFinal answer: 17\nFinal answer: 18",
        "Final answer: 17\nFinal answer: 17 or 18\n\\boxed{17}",
        "Final answer: 17\nFinal answer: unsure\n\\boxed{17}",
        "Final answer: 17\nFinal answer: 1000\n\\boxed{17}",
        "Final answer: 17\nFinal answer: 17.5\n\\boxed{17}",
        "Final answer: 17\n18\nDiscussion.\nFinal answer: 17\n\\boxed{17}",
        "Final answer: 17\n\\boxed{18}\nFinal answer: 17",
    ],
)
def test_legacy_compatibility_never_hides_an_invalid_or_conflicting_final(response):
    assert project(StepZeroTerminalMode.AIME_INTEGER, response) is None


def test_owner_can_discuss_programs_then_explicitly_submit_one_verbatim():
    source = "def f():\n    return 17\n"
    response = (
        "Discussion:\n```python\ndef f():\n    return 1\n```\nFinal code:\n```python\n"
        + source
        + "```"
    )
    final = project(StepZeroTerminalMode.PYTHON_SOURCE, response)
    assert final.payload == source.rstrip("\n")
    assert final.raw_response == response


def test_explicit_python_payload_follows_the_fixed_final_block_convention():
    response = (
        "Final code:\n```python\ndef f():\n    return 1\n```\n"
        "```python\ndef f():\n    return 2\n```"
    )
    assert project(StepZeroTerminalMode.PYTHON_SOURCE, response).payload == "def f():\n    return 2"


def test_revised_code_declaration_submits_the_last_owner_declaration():
    response = (
        "Final code:\n```python\nprint(1)\n```\nAfter reconsidering:\n"
        "### **Final code:**\n```python\nprint(2)\n```"
    )
    final = project(StepZeroTerminalMode.PYTHON_SOURCE, response)
    assert final.payload == "print(2)"
    assert final.raw_response == response


def test_health_response_is_not_shortened_at_an_incidental_final_heading():
    text = "Discuss the symptoms.\nFinal answer: seek appropriate care.\nAdditional safety context."
    assert project(StepZeroTerminalMode.NATURAL_LANGUAGE, text).payload == text


@pytest.mark.parametrize(
    "field",
    [
        "**Final answer: Alder**",
        "__Final answer: Alder__",
        "*Final answer: Alder*",
        "_Final answer: Alder_",
        "**Final answer:** Alder",
        "**Final answer**: Alder",
        "### **Final answer: Alder**",
        "  Final answer: Alder",
        "**Final answer:**\nAlder",
    ],
)
def test_markdown_qa_field_preserves_the_owner_answer_not_its_explanation(field):
    response = "The draft mentions Juniper.\n" + field + "\nAn explanation follows."
    final = project(StepZeroTerminalMode.SHORT_ANSWER, response)
    assert final.payload == "Alder"
    assert final.raw_response == response


@pytest.mark.parametrize(
    "response",
    [
        "**Final answer: Alder**\nFinal answer: Birch",
        "**Final answer:** Alder\n__Final answer: Birch__",
        "**Final answer:**",
    ],
)
def test_markdown_qa_does_not_select_conflicting_or_empty_declarations(response):
    assert project(StepZeroTerminalMode.SHORT_ANSWER, response) is None


def test_markdown_in_discussion_or_natural_language_is_not_a_final_field():
    response = "The **Alder** suggestion remains uncertain.\nCompare it with Birch."
    assert project(StepZeroTerminalMode.SHORT_ANSWER, response).payload == response
    clinical = "Here is the context.\n**Final answer: discuss your symptoms.**\nFurther advice."
    assert project(StepZeroTerminalMode.NATURAL_LANGUAGE, clinical).payload == clinical


@pytest.mark.parametrize("label", ["Answer", "Short Answer"])
@pytest.mark.parametrize(
    "template", ["{}: Alder", "{} : Alder", "**{}: Alder**", "### **{}**:\nAlder"]
)
@pytest.mark.parametrize("native", [False, True])
def test_short_answer_aliases_are_field_transport_not_prose_search(label, template, native):
    response = "I considered Juniper.\n" + template.format(label) + "\nAn explanation follows."
    if native:
        from tests.evaluation.test_integrity_native_tool_calls import call

        response = call("submit_answer", answer=response)
    final = project(StepZeroTerminalMode.SHORT_ANSWER, response)
    assert final.payload == "Alder"
    assert final.raw_response == response


@pytest.mark.parametrize(
    "response",
    [
        "Answer: Birch\nShort answer: Alder",
        "Answer: Birch\nShort answer: Cedar\nFinal answer: Alder",
        "Final answer: Alder\nExplanation follows.\nAnswer: Birch",
        "Final answer: Alder\nA check follows.\n**Final answer: Alder**",
    ],
)
def test_explicit_final_priority_and_identical_repetition_are_answer_blind(response):
    assert project(StepZeroTerminalMode.SHORT_ANSWER, response).payload == "Alder"


@pytest.mark.parametrize("label", ["Answer", "Short answer", "Final answer"])
def test_conflicting_or_empty_short_answer_fields_remain_unsubmitted(label):
    assert project(StepZeroTerminalMode.SHORT_ANSWER, f"{label}: Alder\n{label}: Birch") is None
    assert project(StepZeroTerminalMode.SHORT_ANSWER, f"**{label}:**") is None


def test_short_answer_aliases_do_not_extract_from_discussion_or_other_payload_types():
    discussion = "A draft example:\n```text\nAnswer: Alder\n```\nStill considering Birch."
    assert project(StepZeroTerminalMode.SHORT_ANSWER, discussion).payload == discussion
    clinical = "Some context.\n**Answer: discuss symptoms.**\nFurther advice."
    assert project(StepZeroTerminalMode.NATURAL_LANGUAGE, clinical).payload == clinical
    code = 'print("Answer: Alder")'
    assert project(StepZeroTerminalMode.PYTHON_SOURCE, code).payload == code


@pytest.mark.parametrize(
    "response",
    [
        '```python\ndef f():\n    """Unfinished draft\n```python\ndef f():\n    return 2\n```',
        "```python\ndef f():\n    return 1\n```python\ndef f():\n    return 2",
        "```python\ndef f():\n    return 1\n```\n```python\ndef f():\n    return 2",
    ],
)
def test_unbalanced_program_fences_never_silently_submit_the_abandoned_draft(response):
    assert project(StepZeroTerminalMode.PYTHON_SOURCE, response) is None


@pytest.mark.parametrize("wrapper", ["```python\n{}\n```", "{}", "{}\n```", "```python\n{}"])
def test_python_transport_preserves_literal_backticks_and_source_indentation(wrapper):
    source = 'def f():\n    return "literal ``` marker"'
    final = project(StepZeroTerminalMode.PYTHON_SOURCE, wrapper.format(source))
    assert final.payload == source
