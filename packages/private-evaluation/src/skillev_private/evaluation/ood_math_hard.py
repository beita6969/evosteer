"""MATH-Hard final-answer grading with Hugging Face Math-Verify 0.9.0.

Only the final submitted carrier is parsed. Never search older mathematical
expressions using the reference. Untrusted symbolic parsing runs in the same
isolated, bounded worker boundary as the existing native code checkers.
https://github.com/huggingface/Math-Verify (Apache-2.0)
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev_private.benchmarks.converters import _last_boxed_answer

_WORKER = r"""
import contextlib, io, json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
from math_verify import parse, verify, LatexExtractionConfig
request = json.load(sys.stdin)
with contextlib.redirect_stdout(io.StringIO()):
    gold = parse("$" + request["reference"] + "$",
                 extraction_config=[LatexExtractionConfig()], extraction_mode="first_match")
    if not gold:
        raise ValueError("trusted reference could not be parsed")
    predicted = parse("$" + request["prediction"] + "$",
                      extraction_config=[LatexExtractionConfig()], extraction_mode="first_match")
    passed = verify(gold, predicted) if predicted else False
print(json.dumps({"passed": bool(passed), "prediction_parsed": bool(predicted)}))
"""


def final_math_answer(candidate: str) -> str | None:
    """Last explicit box, otherwise the complete submitted final field/text."""
    if "\\boxed" in candidate or "\\fbox" in candidate:
        start = max(candidate.rfind("\\boxed"), candidate.rfind("\\fbox"))
        final_box = candidate[start:].replace("\\fbox", "\\boxed", 1)
        try:
            return _last_boxed_answer(final_box)
        except ValueError:
            return None  # An unfinished final box cannot fall back to an earlier answer.
    return candidate.strip() or None


def grade_math_hard(candidate: str, reference: str, sandbox: ActorSandbox) -> dict[str, Any]:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("MATH-Hard requires a private reference answer")
    prediction = final_math_answer(candidate)
    if prediction is None:
        return {"passed": False, "prediction_parsed": False, "failure_kind": "invalid-final-box"}
    completed = subprocess.run(  # noqa: S603 -- isolated native scoring worker
        sandbox.command("-c", _WORKER),
        input=json.dumps({"prediction": prediction, "reference": reference}),
        text=True,
        capture_output=True,
        timeout=45,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    if completed.returncode:
        raise RuntimeError("MATH-Hard native scorer worker failed")
    result: dict[str, Any] = json.loads(completed.stdout)
    result["submitted_answer"] = prediction
    return result
