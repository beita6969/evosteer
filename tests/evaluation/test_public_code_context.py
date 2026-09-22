"""Public synthetic syntax only; native child executes the same fixed candidates."""

import asyncio
from dataclasses import replace

import pytest
from skillev_private.benchmarks.code_math import CodeExecutionRequest
from skillev_private.benchmarks.humaneval_official import IsolatedHumanEvalExecutionBackend
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingOutput
from skillev_private.benchmarks.protocol_v13_training_sessions import _HumanEvalEvaluator
from skillev_private.benchmarks.public_code_context import (
    ASSEMBLY_VERSION,
    LEGACY_TRAINING_ASSEMBLY,
    CodeSubmission,
    PublicCodeContext,
    compose_code_candidate,
)

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from tests.benchmarks.test_protocol_v13_training_sessions import _record, _request

PUBLIC = (
    "from math import ceil\nSCALE = 2\n"
    "def helper(x):\n    return ceil(x) * SCALE\n\n"
    'def solve(x: float):\n    """Use the public helper."""\n'
)
BODY = "    return helper(x)\n"
DEF = "def solve(x: float):\n    return helper(x)\n"
TESTS = "def check(candidate):\n    assert candidate(1.2) == 4\n"


def native(assembled, tests=TESTS):
    return asyncio.run(
        IsolatedHumanEvalExecutionBackend().run(
            CodeExecutionRequest(
                "synthetic", assembled.prefix, assembled.completion, tests, "solve"
            )
        )
    )


@pytest.mark.parametrize(
    "candidate",
    [
        BODY,
        DEF,
        PUBLIC + BODY,
        "from __future__ import annotations\n" + DEF,
        "def helper(x):\n    return ceil(x)*SCALE\n" + DEF,
    ],
)
def test_target_and_body_retain_identical_public_namespace(candidate):
    result = compose_code_candidate(PublicCodeContext(PUBLIC, "solve"), CodeSubmission(candidate))
    assert result.original_payload == candidate
    assert result.assembly_version == ASSEMBLY_VERSION
    assert result.public_context_retained
    assert native(result).passed


def test_legacy_drop_reproduced_without_changing_candidate_or_test():
    context, submission = PublicCodeContext(PUBLIC, "solve"), CodeSubmission(DEF)
    old = compose_code_candidate(context, submission, assembly_version=LEGACY_TRAINING_ASSEMBLY)
    new = compose_code_candidate(context, submission)
    result = native(old)
    assert not result.passed
    assert result.exception_type == "NameError"
    assert result.stage == "run"
    assert not old.public_context_retained
    assert native(new).passed
    assert old.original_payload == new.original_payload


def test_standalone_module_requires_explicit_contract_and_does_not_get_missing_helpers():
    assembled = compose_code_candidate(
        PublicCodeContext(PUBLIC, "solve"), CodeSubmission(DEF, "module")
    )
    assert assembled.executable_source == DEF
    assert not native(assembled).passed
    complete_module = compose_code_candidate(
        PublicCodeContext(PUBLIC, "solve"), CodeSubmission(PUBLIC + BODY, "module")
    )
    assert native(complete_module).passed
    wrong = compose_code_candidate(
        PublicCodeContext(PUBLIC, "solve"),
        CodeSubmission(DEF.replace("helper(x)", "private_missing_helper(x)")),
    )
    result = native(wrong)
    assert not result.passed
    assert result.exception_type == "NameError"
    assert "private_missing_helper" in result.native_result


@pytest.mark.parametrize(
    "body",
    [
        '    message = "def solve(x): is just text"\n    return helper(x)\n',
        "    def solve(y):\n        return helper(y)\n    return solve(x)\n",
        (
            "    class Local:\n        def solve(self, y):\n            return helper(y)\n"
            "    return Local().solve(x)\n"
        ),
    ],
)
def test_strings_nested_functions_and_methods_remain_body_completions(body):
    assembled = compose_code_candidate(PublicCodeContext(PUBLIC, "solve"), CodeSubmission(body))
    assert assembled.form == "body"
    assert native(assembled).passed


def test_duplicate_submitted_definitions_keep_original_python_order_not_best_verdict():
    payload = DEF + "def solve(x):\n    return -1\n"
    assembled = compose_code_candidate(PublicCodeContext(PUBLIC, "solve"), CodeSubmission(payload))
    assert assembled.completion == payload
    result = native(assembled)
    assert not result.passed
    assert result.stage == "assertion"
    assert result.exception_type == "AssertionError"


def test_training_native_terminal_uses_shared_ast_assembly_and_private_diagnostics():
    original = _record()
    task = replace(
        original.input,
        query=PUBLIC,
        public_context={**original.input.public_context, "benchmark_id": "humaneval"},
    )
    record = replace(
        original,
        episode=replace(original.episode, benchmark=Protocol13Benchmark.HUMAN_EVAL),
        input=task,
        output=Protocol13TrainingOutput(
            evaluator_kind="humaneval-native", target={"entry_point": "solve", "test": TESTS}
        ),
    )
    evaluator = _HumanEvalEvaluator(record, task, IsolatedHumanEvalExecutionBackend())
    reward = asyncio.run(evaluator.evaluate(_request(task.task_id, DEF)))
    assert reward.success
    assert reward.value == 1
    assert reward.native_payload["native_diagnostics"]["assembly_version"] == ASSEMBLY_VERSION
    assert reward.native_payload["native_diagnostics"]["public_context_retained"]
    assert reward.native_payload["native_diagnostics"]["stage"] == "run"
    assert task.query == PUBLIC


def test_public_stub_alone_is_native_failure_not_an_empty_transport_request():
    assembled = compose_code_candidate(PublicCodeContext(PUBLIC, "solve"), CodeSubmission(PUBLIC))
    assert assembled.executable_source == PUBLIC
    assert not native(assembled).passed
