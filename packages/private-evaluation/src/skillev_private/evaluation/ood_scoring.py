"""OOD native scorers behind the existing private, post-submission boundary."""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.direct_baseline.parsing import parse_multiple_choice_label_v3
from skillev.evaluation.environment_scores import LearningRewardProjection, NativeEnvironmentScore
from skillev.evaluation.input_metric_contracts import CONTRACTS
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_results import EvaluationStatus, NativeScore
from skillev.evaluation.sealed_candidates import CandidateReader, EventOrigin
from skillev_private.benchmarks.qa_metrics import (
    normalize_nq_open_answer,
    score_musique_answers,
    score_qa_answers,
)

from .ood_luna_judge import judge_identity, read_judgement

# Both MIT-licensed official checkers execute candidate code. Run them only in
# the existing filesystem/process/network sandbox, with one target on stdin.
# APPS's old pyext RuntimeModule is replaced by the same standard-library
# ModuleType execution used by LiveCodeBench; test and comparison logic is intact.
_CODE_WORKER = r"""
import contextlib, importlib.util, io, json, resource, sys, types
request = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_AS, (request["memory_bytes"], request["memory_bytes"]))
resource.setrlimit(resource.RLIMIT_CPU, (request["cpu_seconds"], request["cpu_seconds"]))
if request["benchmark"] == "apps-introductory":
    pyext = types.ModuleType("pyext")
    class RuntimeModule:
        @staticmethod
        def from_string(name, doc, source):
            module = types.ModuleType(name, doc)
            # APPS wraps stdin programs in code(). Their original main guard
            # still reads this module's globals when the checker invokes code.
            # Preserve script semantics without rewriting the submitted source;
            # call-based tasks remain imported modules, not entry-point scripts.
            if request["input_output"].get("fn_name") is None:
                module.__name__ = "__main__"
            exec(source, module.__dict__)
            return module
    pyext.RuntimeModule = RuntimeModule
    sys.modules["pyext"] = pyext
spec = importlib.util.spec_from_file_location("official_checker", "/checker.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    if request["benchmark"] == "apps-introductory":
        checker.timeout = request["test_timeout"]
        results = checker.run_test(problem={"input_output": request["input_output"]},
                                   test=request["candidate"], debug=False)
        metadata = {}
    else:
        results, metadata = checker.run_test(
            {"input_output": json.dumps(request["input_output"])},
            test=request["candidate"], debug=False, timeout=request["test_timeout"])
results = [value.item() if hasattr(value, "item") else value for value in results]
print(json.dumps({"results": results, "metadata": metadata,
                  "diagnostic": capture.getvalue()[-4000:]}, default=str))
"""


def grade_code(
    benchmark: str,
    candidate: str,
    target: dict[str, Any],
    settings: dict[str, Any],
    sandbox: ActorSandbox,
) -> dict[str, Any]:
    checker = Path(settings["checker_path"]).resolve()
    if not checker.is_file():
        raise FileNotFoundError("official code checker is unavailable")
    in_out = target["input_output"]
    if not in_out["inputs"] or len(in_out["inputs"]) != len(in_out["outputs"]):
        raise ValueError("official code task has no complete test population")
    timeout = settings.get("test_timeout_seconds", 4 if benchmark == "apps-introductory" else 6)
    if type(timeout) is not int or timeout < 1:
        raise ValueError("official per-test timeout must be a positive integer")
    outer = (timeout + 1) * len(in_out["inputs"]) + 30
    wall_limit = float(settings.get("wall_timeout_seconds", outer + 10))
    if wall_limit <= 0:
        raise ValueError("code scorer wall timeout must be positive")
    command = sandbox.command("-c", _CODE_WORKER)
    command = (command[0], "--ro-bind", str(checker), "/checker.py", *command[1:])
    completed = subprocess.run(  # noqa: S603 -- existing namespace sandbox, official checker
        command,
        input=json.dumps(
            {
                "benchmark": benchmark,
                "candidate": candidate,
                "input_output": in_out,
                "test_timeout": timeout,
                "memory_bytes": 4 * 1024**3,
                "cpu_seconds": outer,
            }
        ),
        text=True,
        capture_output=True,
        timeout=min(outer + 10, wall_limit),
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    if completed.returncode:
        raise RuntimeError("official scorer worker failed: " + completed.stderr[-2000:])
    result: dict[str, Any] = json.loads(completed.stdout)
    if completed.stderr:
        result["worker_stderr"] = completed.stderr[-4000:]
    if not isinstance(result.get("results"), list) or not result["results"]:
        raise RuntimeError("official checker produced no test verdicts")
    # Official APPS/LCB aggregation tests > 0, never truthiness (-1/-2 are true).
    result["passed"] = all(value > 0 for value in result["results"])
    if result["passed"] and len(result["results"]) != len(in_out["inputs"]):
        raise RuntimeError("all-passing native verdicts do not cover the complete test population")
    try:
        ast.parse(candidate)
    except SyntaxError as error:
        result["python_syntax"] = {"status": "invalid", "line": error.lineno, "reason": error.msg}
    else:
        result["python_syntax"] = {"status": "valid"}
    result["test_timeout_seconds"] = timeout
    result["wall_timeout_seconds"] = min(outer + 10, wall_limit)
    result["test_count"] = len(in_out["inputs"])
    result["returned_test_verdicts"] = len(result["results"])
    result["public_test_count"] = target.get("public_test_count")
    result["private_test_count"] = target.get("private_test_count")
    result["failure_kind"] = code_failure_kind(benchmark, result)
    return result


def code_failure_kind(benchmark: str, result: dict[str, Any]) -> str | None:
    """Keep benchmark-specific error codes separate; never change a verdict."""
    if result["passed"]:
        return None
    if benchmark == "livecodebench":
        # LCB -2 is wrong answer, unlike APPS -2 (load/compile failure).
        code = result.get("metadata", {}).get("error_code")
        return {
            -2: "wrong-answer",
            -3: "time-limit-exceeded",
            -4: "runtime-or-load-error",
        }.get(code, "unclassified-native-failure")
    verdicts = result["results"]
    if -2 in verdicts:
        return "compile-or-load-error"
    if -1 in verdicts:
        # APPS does not reliably distinguish these; do not invent precision.
        return "runtime-error-or-timeout"
    return "wrong-answer"


async def score_ood(
    reader: CandidateReader,
    scope: tuple[str, str, str],
    benchmark: str,
    target: dict[str, Any],
    *,
    settings: dict[str, Any],
    sandbox: ActorSandbox,
    diagnostics: Callable[[dict[str, Any]], None],
) -> NativeScore:
    started = time.monotonic()
    candidate = reader.get(*scope)
    metric = CONTRACTS[benchmark].metric
    verifier = NATIVE_VERIFIER_VERSIONS[benchmark]
    if benchmark == "omni-math":
        _, _, metric, verifier = judge_identity(settings[benchmark])
    secondary: dict[str, float] = {}
    status = EvaluationStatus.SCORED
    failure = None
    grader_used = None
    grader_cost: dict[str, float] = {}
    if benchmark == "scienceworld":
        outcomes = reader.traces(scope, "native-outcome", origin=EventOrigin.ENVIRONMENT)
        if len(outcomes) != 1 or not isinstance(outcomes[0], dict):
            raise RuntimeError("ScienceWorld native outcome is unavailable")
        native = NativeEnvironmentScore.from_outcome(outcomes[0])
        learning = LearningRewardProjection.from_native(native)
        value = native.value
        secondary = {
            "success": float(learning.success),
            "zero-clipped-learning-reward": learning.value,
        }
        diagnostics(
            {
                "native_outcome": outcomes[0],
                "primary_projection": "native_final_score/100",
                "learning_reward_projection": learning.projection,
            }
        )
    elif not candidate.text.strip():
        value, status, failure = 0.0, EvaluationStatus.CANDIDATE_FAILURE, "empty-submission"
        if benchmark in {"musique", "nq-open"}:
            secondary = {"answer-exact-match": 0.0, "answer-f1": 0.0}
        elif benchmark == "livemedbench":
            secondary, grader_used = {"triggered-negative-rubric-count": 0.0}, False
    elif benchmark == "livemedbench":
        from .ood_livemedbench import grade_case

        result = await asyncio.to_thread(grade_case, candidate, target, settings[benchmark])
        diagnostics(result)
        value = float(result["raw_score"])
        secondary = {"triggered-negative-rubric-count": float(result["triggered_negative_count"])}
        grader_used, grader_cost = True, result["cost"]
    elif benchmark in {"musique", "nq-open"}:
        metrics = (
            score_musique_answers(candidate.text, tuple(target["accepted_answers"]))
            if benchmark == "musique"
            else score_qa_answers(
                candidate.text,
                tuple(target["accepted_answers"]),
                normalizer=normalize_nq_open_answer,
            )
        )
        value = metrics.f1 if benchmark == "musique" else metrics.em
        secondary = {"answer-exact-match": metrics.em, "answer-f1": metrics.f1}
    elif benchmark in {"livecodebench", "apps-introductory"}:
        result = await asyncio.to_thread(
            grade_code, benchmark, candidate.text, target, settings[benchmark], sandbox
        )
        diagnostics(result)
        value = float(result["passed"])
    elif benchmark == "omni-math":
        if settings[benchmark].get("backend") == "openai-chat-completions":
            from .ood_api_judge import grade_candidate

            result = await asyncio.to_thread(
                grade_candidate, candidate, target, settings[benchmark]
            )
            grader_cost = result["cost"]
        else:
            result = await asyncio.to_thread(read_judgement, candidate, settings[benchmark])
        diagnostics(result)
        value = float(result["passed"])
    elif benchmark == "gpqa-diamond-bioorganic":
        expected = target["correct_label"]
        if expected not in {"A", "B", "C", "D"}:
            raise ValueError("GPQA private target must be a frozen A-D choice label")
        parsed = parse_multiple_choice_label_v3(candidate.text)
        value = float(parsed.value == expected)
        if parsed.value is None:
            status, failure = EvaluationStatus.CANDIDATE_FAILURE, "invalid-choice-submission"
        diagnostics({"predicted_label": parsed.value, "parse_reason": parsed.reason.value})
    elif benchmark == "math-hard":
        from .ood_math_hard import grade_math_hard

        result = await asyncio.to_thread(grade_math_hard, candidate.text, target["answer"], sandbox)
        diagnostics(result)
        value = float(result["passed"])
        if not result["prediction_parsed"]:
            status, failure = EvaluationStatus.CANDIDATE_FAILURE, "invalid-math-submission"
    else:
        raise ValueError("OOD scorer is not enabled")
    return NativeScore(
        candidate.episode_id,
        benchmark,
        metric,
        value,
        status,
        secondary,
        verifier_version=verifier,
        grader_used=grader_used,
        scorer_cost={
            "judgement_import_wall_seconds"
            if benchmark == "omni-math" and not grader_cost
            else "wall_seconds": time.monotonic() - started,
            **grader_cost,
        },
        failure_kind=failure,
    )
