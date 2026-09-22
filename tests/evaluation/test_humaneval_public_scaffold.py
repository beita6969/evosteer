"""Only public scaffold assembly changes; one candidate goes to native tests."""

import asyncio

import pytest
from skillev_private.benchmarks.code_math import CodeExecutionRequest, CodeExecutionStatus
from skillev_private.benchmarks.humaneval_official import IsolatedHumanEvalExecutionBackend
from skillev_private.evaluation.humaneval_source import humaneval_source_parts

PREFIX = "from typing import List\n\ndef public_helper(value):\n    return value + 1\n\n"
STUB = 'def transform(values: List[int]):\n    """Apply the public transform."""\n'
IMPLEMENTATION = (
    "def transform(values: List[int]):\n    return [public_helper(value) for value in values]\n"
)
TEST = (
    "def check(candidate):\n"
    "    assert public_helper(2) == 3\n"
    "    assert candidate([1, 2]) == [2, 3]\n"
)


def grade(prompt, candidate, test=TEST):
    prefix, completion = humaneval_source_parts(prompt, candidate)
    return asyncio.run(
        IsolatedHumanEvalExecutionBackend().run(
            CodeExecutionRequest("synthetic-scaffold", prefix, completion, test, "transform")
        )
    ).status


@pytest.mark.parametrize(
    "candidate",
    [
        IMPLEMENTATION,
        "    return [public_helper(value) for value in values]\n",
        PREFIX + IMPLEMENTATION,
        PREFIX + STUB + "    return [public_helper(value) for value in values]\n",
        '"""Owner module."""\nfrom __future__ import annotations\n\n' + IMPLEMENTATION,
    ],
)
def test_complete_function_and_native_body_both_retain_public_imports_and_helpers(candidate):
    assert grade(PREFIX + STUB, candidate) is CodeExecutionStatus.PASSED


def test_public_helper_used_by_native_check_is_not_dropped_for_a_self_contained_candidate():
    candidate = "def transform(values):\n    return [value + 1 for value in values]\n"
    assert grade(PREFIX + STUB, candidate) is CodeExecutionStatus.PASSED


def test_both_module_future_headers_remain_legal_and_owner_docstring_survives():
    prompt = '"""Public module."""\nfrom __future__ import annotations\n' + PREFIX + STUB
    candidate = '"""Owner module."""\nfrom __future__ import annotations\n' + IMPLEMENTATION
    test = TEST + '    assert __doc__ == "Owner module."\n'
    assert grade(prompt, candidate, test) is CodeExecutionStatus.PASSED


def test_owner_definition_is_not_replaced_by_a_more_successful_public_helper():
    candidate = "def public_helper(value):\n    return value - 1\n\n" + IMPLEMENTATION
    assert grade(PREFIX + STUB, candidate) is CodeExecutionStatus.TEST_FAILURE


def test_unfinished_signature_is_not_prepended_to_a_complete_function():
    candidate = "def transform(value):\n    return value + 1\n"
    assert (
        grade(
            "def transform(value):\n",
            candidate,
            "def check(candidate):\n    assert candidate(2) == 3\n",
        )
        is CodeExecutionStatus.PASSED
    )


def test_decorator_belongs_to_replaced_stub_not_to_public_prefix():
    prompt = PREFIX + "@public_helper\n" + STUB
    assert grade(prompt, IMPLEMENTATION) is CodeExecutionStatus.PASSED


def test_syntax_errors_are_not_repaired_by_harness_assembly():
    assert (
        grade(PREFIX + STUB, "def transform(:\n    return []\n") is CodeExecutionStatus.SYNTAX_ERROR
    )


def test_complete_candidate_content_is_preserved_around_public_scaffold_insertion():
    header = '"""Owner module."""\nfrom __future__ import annotations\n'
    candidate = header + IMPLEMENTATION
    prompt, body = humaneval_source_parts(PREFIX + STUB, candidate)
    assert prompt == header + PREFIX
    assert body == IMPLEMENTATION
    assert candidate == header + body
