"""Transport wrappers must not turn unchanged Python into a syntax failure."""

import ast
import asyncio

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.owner_final import project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode, project_terminal_candidate
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_broker_boundary import runtime


@pytest.mark.parametrize("label", ["", "python", "py", "python3", "PY", "Python3"])
@pytest.mark.parametrize("marker", ["```", "````"])
def test_code_fence_aliases_share_the_owner_and_scorer_decoder(label, marker):
    source = 'import math\ndef add(a, b):\n    """A literal ``` marker."""\n    return a + b'
    raw = f"{marker}{label}\n{source}\n{marker}"
    final = project_owner_final(
        StepZeroTerminalMode.PYTHON_SOURCE, raw, owner_id="owner", message_id="case:owner-final"
    )
    assert final.payload == source
    assert final.raw_response == raw
    assert project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, raw) == source
    ast.parse(final.payload)


def test_fence_label_is_presentation_not_a_reason_to_reject_the_program():
    source = "def add(a,b):\n    return a+b"
    assert (
        project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, f"```ruby\n{source}\n```")
        == source
    )


def test_last_code_block_is_submitted_without_forcing_a_second_owner_call(tmp_path):
    entry = PublicTaskView.from_record("case", "mbpp-plus", {"prompt": "Implement add(a,b)."})
    module = "import math\ndef add(a,b):\n    return a+b"
    response = f"A sketch:\n```py\nimport math\n```\nComplete solution:\n```python3\n{module}\n```"
    instance = runtime(tmp_path, entry, [response], calls=2)
    try:
        final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
        assert final.text == module
        assert final.intervention_counts["model_calls"] == 1
        assert final.intervention_counts["communication_repairs"] == 0
        assert final.submission["raw_response"] == response
        instance.validate_candidate(instance.journal, entry, InferenceArm("A2"), "synthetic")
    finally:
        instance.journal.close()


@pytest.mark.parametrize("final_block", ["raise RuntimeError('owner program')", "def broken("])
def test_last_block_is_not_replaced_by_an_earlier_executable_program(final_block):
    response = f"```python\nprint(17)\n```\nRevised solution:\n```python\n{final_block}\n```"
    final = project_owner_final(
        StepZeroTerminalMode.PYTHON_SOURCE, response, owner_id="owner", message_id="case:final"
    )
    assert final.payload == final_block
    assert final.raw_response == response


def test_multiple_blocks_are_never_implicitly_joined():
    response = "```python\nimport math\n```\n```python\nprint(math.sqrt(4))\n```"
    assert project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, response) == (
        "print(math.sqrt(4))"
    )


def test_an_empty_final_block_does_not_fall_back_to_the_draft():
    response = "```python\nprint(17)\n```\n```python\n\n```"
    assert project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, response) is None
