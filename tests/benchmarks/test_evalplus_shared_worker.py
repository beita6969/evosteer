"""Exercise the shared fixed EvalPlus source in real isolated child processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from skillev_private.benchmarks import protocol_v13_mbpp_worker
from skillev_private.benchmarks.mbpp_scoring import (
    MBPP_REQUEST_FORMAT,
    MBPPScorerProfile,
    decode_mbpp_verdict,
)


@pytest.mark.parametrize(
    ("submission", "base_passed", "plus_passed"),
    [
        ("def solve(value): return value + 1\n", True, True),
        ("def solve(value): return value - 1\n", False, False),
        ("def solve(value): return 2\n", True, False),
        ("def solve(value): return 3\n", False, True),
    ],
    ids=("correct", "both-fail", "plus-only-failure", "base-only-failure"),
)
def test_official_mbpp_worker_grades_in_fresh_process(
    tmp_path: Path, submission: str, base_passed: bool, plus_passed: bool
) -> None:
    # Entirely synthetic evidence: no dataset download or benchmark answer is
    # required. The plus-only failure ensures the extended suite really runs.
    request = {
        "format": MBPP_REQUEST_FORMAT,
        "scorer_profile": MBPPScorerProfile().to_value(),
        "operation": "evaluate-mbpp-plus",
        "private_target": {
            "assertion": [],
            "atol": 0,
            "base_input": [[1]],
            "canonical_solution": "def solve(value): return value + 1\n",
            "contract": [],
            "entry_point": "solve",
            "plus_input": [[2]],
        },
        "prompt": "",
        "source_task_id": "Mbpp/99999",
        "submission": submission,
        "task_id": "synthetic-mbpp-worker",
    }
    completed = subprocess.run(  # noqa: S603 -- fixed local worker, synthetic input on stdin
        [sys.executable, str(Path(protocol_v13_mbpp_worker.__file__).resolve())],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    result = decode_mbpp_verdict(json.loads(completed.stdout), MBPPScorerProfile())
    assert not result.get("infrastructure_error", False), result
    assert result["base_passed"] is base_passed
    assert result["plus_passed"] is plus_passed
    assert result["status"] == ("passed" if base_passed and plus_passed else "failed")
