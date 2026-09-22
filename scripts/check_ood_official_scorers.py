"""CPU-only synthetic parity checks against explicitly supplied upstream sources.

No downloads, model calls, licensed benchmark items or hidden-test feedback.
APPS/LCB execute only through the production namespace sandbox. See
docs/ood-official-scorers.md for source filenames and provenance.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from skillev_private.benchmarks.qa_metrics import (
    normalize_nq_open_answer,
    score_musique_answers,
    score_qa_answers,
)
from skillev_private.evaluation.ood_scoring import grade_code

from skillev.evaluation.actor_sandbox import ActorSandbox


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sources(sources: Path) -> dict[str, Any]:
    started = time.monotonic()
    # Load MuSiQue's own abstract base, not its model/runtime dependencies.
    sys.modules["metrics.metric"] = load_module(sources / "musique-metric.py", "metrics.metric")
    musique = load_module(sources / "musique-answer.py", "upstream_musique_answer")
    fid = load_module(sources / "fid-evaluation.py", "upstream_fid_evaluation")
    lcb = load_module(sources / "lcb-passk.py", "upstream_lcb_passk")
    words = ["", "the", "a!", "blue", "blue bird", "BLUE-bird", "bird bird", "café", "thé bird"]
    comparisons = 0
    differences = []
    for text in words:
        if normalize_nq_open_answer(text) != fid.normalize_answer(text):
            differences.append({"benchmark": "nq-open-normalizer", "input": text})
        comparisons += 1
    for prediction, answer in itertools.product(words, repeat=2):
        accepted = [answer, "azure"]
        metric = musique.AnswerMetric()
        metric(prediction, accepted)
        official = metric.get_metric()
        ours = score_musique_answers(prediction, tuple(accepted))
        if (ours.em, ours.f1) != official:
            differences.append({"benchmark": "musique", "prediction": prediction, "answer": answer})
        em = score_qa_answers(prediction, tuple(accepted), normalizer=normalize_nq_open_answer).em
        if em != float(fid.ems(prediction, accepted)):
            differences.append({"benchmark": "nq-open", "prediction": prediction, "answer": answer})
        comparisons += 2

    sandbox = ActorSandbox.current(Path(__file__).resolve().parents[1] / "src")
    cases = [
        ("stdio", "print(sum(map(int, input().split())))", None, True),
        (
            "main-guard",
            "def solve():\n    print(sum(map(int, input().split())))\n"
            "if __name__ == '__main__':\n    solve()",
            None,
            True,
        ),
        ("function", "def add(a, b):\n    return a + b", "add", True),
        ("method", "class Solution:\n    def add(self, a, b):\n        return a + b", "add", True),
        ("wrong-answer", "print(999)", None, False),
        ("syntax-error", "if :", None, False),
        ("runtime-error", "raise ValueError('synthetic')", None, False),
        ("timeout", "while True:\n    pass", None, False),
    ]
    native_cases = []
    for benchmark, filename in [
        ("apps-introductory", "apps-testing.py"),
        ("livecodebench", "lcb-testing.py"),
    ]:
        native_results = {}
        for name, code, fn_name, expected in cases:
            if fn_name and benchmark == "livecodebench":
                inputs, outputs = ["2\n3", "7\n4"], ["5", "11"]
            elif fn_name:
                inputs, outputs = [[2, 3], [7, 4]], [5, 11]
            else:
                inputs, outputs = ["2 3\n", "7 4\n"], ["5\n", "11\n"]
            result = grade_code(
                benchmark,
                code,
                {"input_output": {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}},
                {"checker_path": str(sources / filename), "test_timeout_seconds": 1},
                sandbox,
            )
            native_cases.append({"benchmark": benchmark, "case": name, **result})
            native_results[name] = [result["results"]]
            if result["passed"] is not expected:
                differences.append({"benchmark": benchmark, "case": name})
        # One candidate per question: mean all-tests-pass equals official pass@1.
        upstream = lcb.compute_metrics_from_results(native_results, k_list=[1])["pass@1"]
        own = sum(row["passed"] for row in native_cases if row["benchmark"] == benchmark) / len(
            cases
        )
        assert own == upstream
    return {
        "qa_comparisons": comparisons,
        "native_code_cases": len(native_cases),
        "differences": differences,
        "code_results": native_cases,
        "model_calls": 0,
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = check_sources(args.sources.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "code_results"}))
    if result["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
