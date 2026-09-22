from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

from skillev_private.direct_reference import load_humaneval_cases, load_nq_open_cases
from skillev_private.direct_reference.evaluators import score_static_case

from skillev.evaluation.direct_baseline.parsing import parse_python_source
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol


def test_nq_loader_freezes_128_before_generation_and_keeps_answers_private(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nq.jsonl"
    with source.open("w", encoding="utf-8") as stream:
        for index in range(130):
            stream.write(
                json.dumps({"question": f"public question {index}", "answer": [f"secret {index}"]})
                + "\n"
            )
    cases = load_nq_open_cases(
        source,
        protocol=load_direct_reference_protocol(
            Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
        ),
    )
    assert len(cases) == 128
    assert len({case.public_task.task_id for case in cases}) == 128
    rendered = "\n".join(
        message["content"] for case in cases for message in case.public_task.messages
    )
    assert "secret" not in rendered


def test_humaneval_direct_source_uses_official_isolated_executor(tmp_path: Path) -> None:
    source = tmp_path / "HumanEval.jsonl.gz"
    prompt = 'def increment(value):\n    """Return value plus one."""\n'
    row = {
        "task_id": "HumanEval/fixture",
        "prompt": prompt,
        "canonical_solution": "    return value + 1\n",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        "entry_point": "increment",
    }
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    case = load_humaneval_cases(
        source,
        protocol=load_direct_reference_protocol(
            Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
        ),
    )[0]
    parsed = parse_python_source(f"```python\n{prompt}    return value + 1\n```")
    score = asyncio.run(score_static_case(case, parsed))
    assert score.metrics == {"pass_at_1": 1.0}
    assert score.scorer_reached
