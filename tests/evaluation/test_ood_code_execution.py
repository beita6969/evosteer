"""Regression coverage for script versus callable execution in the APPS shim."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from skillev_private.evaluation.ood_scoring import grade_code

from skillev.evaluation.actor_sandbox import ActorSandbox

# A synthetic checker exercises the same module boundary as the released APPS
# checker. No benchmark problems, solutions or test cases are embedded here.
CHECKER = """
import contextlib
import io
import sys
import textwrap
from pyext import RuntimeModule

def run_test(problem, test, debug=False):
    data = problem["input_output"]
    name = data.get("fn_name")
    source = test if name else "def code():\\n" + textwrap.indent(test, "    ")
    module = RuntimeModule.from_string("candidate_module", "", source)
    results = []
    for value, expected in zip(data["inputs"], data["outputs"]):
        if name:
            owner = module.Solution() if hasattr(module, "Solution") else module
            result = getattr(owner, name)(*value)
        else:
            original = sys.stdin
            sys.stdin = io.StringIO(value)
            try:
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    module.code()
                result = captured.getvalue()
            finally:
                sys.stdin = original
        results.append(result == expected)
    return results
"""


@pytest.mark.parametrize(
    ("candidate", "function", "inputs", "outputs", "passed"),
    [
        ("print(sum(map(int, input().split())))", None, ["2 3\n", "7 4\n"], ["5\n", "11\n"], True),
        (
            "def solve():\n    print(sum(map(int, input().split())))\n"
            "if __name__ == '__main__':\n    solve()",
            None,
            ["2 3\n", "7 4\n"],
            ["5\n", "11\n"],
            True,
        ),
        ("if __name__ == '__main__':\n    print(999)", None, ["2 3\n"], ["5\n"], False),
        (
            "def add(a, b):\n    return a + b\n"
            "if __name__ == '__main__':\n    raise RuntimeError('not a stdin task')",
            "add",
            [[2, 3], [7, 4]],
            [5, 11],
            True,
        ),
        (
            "class Solution:\n    def add(self, a, b):\n        return a + b",
            "add",
            [[2, 3], [7, 4]],
            [5, 11],
            True,
        ),
        ("def add(a, b):\n    return 999", "add", [[2, 3]], [5], False),
    ],
)
def test_apps_preserves_script_and_callable_entry_semantics(
    tmp_path, candidate, function, inputs, outputs, passed
):
    if shutil.which("bwrap") is None:
        pytest.skip("the native scorer requires bubblewrap")
    checker = tmp_path / "checker.py"
    checker.write_text(CHECKER)
    sandbox = ActorSandbox.current(Path(__file__).resolve().parents[2] / "src")
    result = grade_code(
        "apps-introductory",
        candidate,
        {"input_output": {"inputs": inputs, "outputs": outputs, "fn_name": function}},
        {"checker_path": str(checker)},
        sandbox,
    )
    assert result["passed"] is passed
    assert len(result["results"]) == len(inputs)
