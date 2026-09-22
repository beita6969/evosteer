"""Real QA/math/MBPP training-curve bindings for EvoSteer.

The builder writes public prompts and verifier-only targets to a local data
directory.  This module never places answers or tests in ``EvoTask.prompt``.
Use ``scripts/build_curve_benchmarks.py`` before launching the CLI and pass
``--task-factory skillev.experiments.curve_benchmarks:task_factory``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from skillev.contracts.evosteer import EvoTask
from skillev.evosteer_application import ResetReceipt, SessionRequest, TaskBinding, TaskSession
from skillev.orchestration.model_executor import FrozenTextExecutor
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy

_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def _hotpot_norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, answer: str) -> float:
    normalized_prediction, normalized_answer = _hotpot_norm(prediction), _hotpot_norm(answer)
    # Official HotpotQA rule: yes/no/noanswer only match exactly.
    special = {"yes", "no", "noanswer"}
    if (normalized_prediction in special or normalized_answer in special) and (
        normalized_prediction != normalized_answer
    ):
        return 0.0
    p, g = normalized_prediction.split(), normalized_answer.split()
    if not p or not g:
        return float(p == g)
    overlap = sum((__import__("collections").Counter(p) & __import__("collections").Counter(g)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _math_answer(solution: str) -> str:
    # GSM8K's official train answers end with ``#### final``.  Keep only the
    # final answer and normalize commas/whitespace, never the worked solution.
    match = re.findall(r"####\s*([^\n]+)", solution)
    if not match:
        return ""
    return re.sub(r"[,$\s]", "", match[-1])


_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def _last_number(text: str) -> Decimal | None:
    numbers = _NUMBER.findall(text.replace("\\!", "").replace("{,}", ","))
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


_MATH_CUE = re.compile(r"(?:final answer|answer)[*_\s]*(?:is\b[*_\s]*[:：]?|[:：=])", re.IGNORECASE)


def _math_prediction(output: str) -> Decimal | None:
    """The final numeric answer.

    Priority: last ``\\boxed{}``, last ``####`` line, the first number after the
    last "answer" cue (so "Answer: Tom spends $45 in 10 days" reads 45), and
    only then the last number in the text.

    ``\\boxed{}`` is read first because in model output ``####`` is a markdown
    heading far more often than a final-answer line ("#### Case 3: $n$ is a
    3-digit number"). The ``####`` convention holds for GSM8K's gold solutions,
    which are parsed by ``_math_answer``, not here; it stays second so a
    genuinely GSM8K-formatted output is still read.
    """
    for pattern in (r"\\boxed\{([^{}]+)\}", r"####\s*([^\n]+)"):
        spans = re.findall(pattern, output)
        if spans:
            value = _last_number(spans[-1])
            if value is not None:
                return value
    cues = list(_MATH_CUE.finditer(output))
    if cues:
        match = _NUMBER.search(output[cues[-1].end() :][:200].replace("{,}", ","))
        if match:
            try:
                return Decimal(match.group(0).replace(",", ""))
            except InvalidOperation:
                pass
    return _last_number(output)


def _math_score(output: str, answer: str) -> float:
    try:
        gold = Decimal(answer)
    except InvalidOperation:
        return 0.0
    prediction = _math_prediction(output)
    return float(prediction is not None and prediction == gold)


_ANSWER_CUE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE)


def _short_answer(output: str) -> str:
    """The span a reader would take as the answer; formatting and lead-ins removed."""
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    cues = _ANSWER_CUE.findall(output)
    span = cues[-1] if cues else lines[-1]
    span = re.sub(r"[*_`]+", "", span.splitlines()[0])
    return span.strip().strip("\"'").rstrip(".").strip()


def _hotpot_score(output: str, answer: str) -> float:
    # Token F1 is the HotpotQA metric; it is taken on the extracted answer span
    # so a correct answer phrased as a sentence is not scored as a near miss.
    short = _short_answer(output)
    gold = _hotpot_norm(answer)
    if gold in {"yes", "no"}:
        # "Yes, both are American." answers yes; the span's first word decides.
        return float((_hotpot_norm(short).split() or [""])[0] == gold)
    return max(_f1(short, answer), _f1(output, answer))


def _open_qa_score(output: str, answers: list[str]) -> float:
    """NQ-Open: best token F1 of the answer span over all gold aliases."""
    return max((_hotpot_score(output, answer) for answer in answers), default=0.0)


# Cue words are case-insensitive; the option letter must be a capital so that
# "answer, a patient ..." is never read as option A.
_CHOICE_CUES = (
    re.compile(r"(?i:answer)[*_\s]*(?i:is)?[*_\s]*[:：]?[*_\s(\[]*([A-J])(?![A-Za-z])"),
    re.compile(r"(?i:option|choice)[*_\s]*(?i:is)?[*_\s]*[:：]?[*_\s(\[]*([A-J])(?![A-Za-z])"),
)


def _choice_letter(output: str, letters: list[str]) -> str | None:
    """Option letter: last 'Answer: X' cue, else 'option X', else a reply that is only a letter."""
    for cue in _CHOICE_CUES:
        found = [letter for letter in cue.findall(output) if letter in letters]
        if found:
            return found[-1]
    bare = re.fullmatch(r"\s*[*_(\[]*\s*([A-J])\s*[)\].:*_]*\s*", output)
    if bare and bare.group(1) in letters:
        return bare.group(1)
    return None


def _integer_score(output: str, answer: str) -> float:
    prediction = _math_prediction(output)
    return float(prediction is not None and prediction == Decimal(answer))


def _mbpp_tests(row: dict[str, Any]) -> list[str]:
    tests = row.get("test_list", []) or []
    if isinstance(tests, str):
        # The manifest stores MBPP's test list as a Python literal.
        try:
            tests = ast.literal_eval(tests)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(tests, list) or any(not isinstance(t, str) for t in tests):
        return []
    return tests


def _task_prompt(row: dict[str, Any]) -> str:
    if row["kind"] != "mbpp":
        return row["prompt"]
    # Standard MBPP protocol: the task names the function through a test. Only
    # the first assertion is shown; every assertion is run by the verifier.
    tests = _mbpp_tests(row)
    if not tests:
        return row["prompt"]
    return f"{row['prompt']}\nYour code should pass this test:\n{tests[0]}"


def _extract_code(output: str) -> str:
    blocks = [match.group(1).strip() for match in _FENCE.finditer(output)]
    if blocks:
        return max(blocks, key=len)
    return output.strip()


def _code_score(output: str, row: dict[str, Any]) -> float:
    source = _extract_code(output)
    if not source:
        return 0.0
    try:
        ast.parse(source)
    except SyntaxError:
        return 0.0
    if row["kind"] == "mbpp_plus":
        # EvalPlus: base and plus inputs are checked by the dataset's own script.
        imports = "\n".join(row.get("test_imports", []) or [])
        return float(_passes(source + "\n\n" + imports + "\n" + row["plus_test"], cpu_seconds=20))
    setup = row.get("test_setup_code", "") or ""
    tests = _mbpp_tests(row)
    if not isinstance(setup, str) or not tests:
        return 0.0
    return float(_passes(source + "\n\n" + setup + "\n" + "\n".join(tests), cpu_seconds=3))


def _passes(program: str, *, cpu_seconds: int) -> bool:
    """Run a candidate plus its assertions in a child process.

    Assertions stay verifier-side. A per-run sentinel printed after the last
    assertion rejects programs that exit early with status 0. Limits are set by
    the child itself: preexec_fn is unsafe in this multithreaded trainer.
    """
    sentinel = f"evosteer-tests-passed-{uuid.uuid4().hex}"
    limits = (
        "import resource as _evosteer_resource\n"
        f"_evosteer_resource.setrlimit(_evosteer_resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "_evosteer_resource.setrlimit(_evosteer_resource.RLIMIT_AS, (4 << 30, 4 << 30))\n"
        "_evosteer_resource.setrlimit(_evosteer_resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))\n"
        if sys.platform.startswith("linux")
        else ""
    )
    with tempfile.TemporaryDirectory(prefix="evosteer-mbpp-") as temp:
        path = Path(temp) / "candidate.py"
        path.write_text(limits + program + f"\nprint('\\n' + {sentinel!r})\n", encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"}
        result = subprocess.run(
            [sys.executable, "-I", str(path)],
            cwd=temp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=cpu_seconds + 10,
            check=False,
        )
    return result.returncode == 0 and result.stdout.decode("utf-8", "replace").splitlines()[-1:] == [sentinel]


def _score(kind: str, output: str, row: dict[str, Any]) -> float:
    if kind == "hotpotqa":
        return _hotpot_score(output, row["answer"])
    if kind == "gsm8k":
        return _math_score(output, _math_answer(row["answer"]))
    if kind == "nq_open":
        return _open_qa_score(output, row["answers"])
    if kind == "medqa":
        return float(_choice_letter(output, row["options"]) == row["answer"])
    if kind == "aime_2026":
        return _integer_score(output, row["answer"])
    if kind in {"mbpp", "mbpp_plus"}:
        try:
            return _code_score(output, row)
        except (OSError, subprocess.SubprocessError, ValueError, MemoryError):
            return 0.0
    raise ValueError(f"unsupported curve benchmark: {kind}")


def _load_rows(root: Path) -> list[dict[str, Any]]:
    path = Path(os.environ.get("EVOSTEER_CURVE_DATA", root / "curve_benchmarks.jsonl"))
    if not path.is_file():
        raise FileNotFoundError(f"curve data not found: {path}; run build_curve_benchmarks.py")
    requested_split = os.environ.get("EVOSTEER_CURVE_SPLIT", "train")
    if requested_split not in {"train", "validation", "all"}:
        raise ValueError("EVOSTEER_CURVE_SPLIT must be train, validation, or all")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if requested_split != "all":
        rows = [row for row in rows if row.get("split") == requested_split]
    if not rows:
        raise ValueError("curve benchmark manifest is empty")
    return rows


def _executor_model(policy: Any) -> Any:
    """The frozen base model that runs role nodes: in-process, or the same checkpoint on SGLang."""
    url = os.environ.get("EVOSTEER_EXECUTOR_URL")
    if not url:
        return policy
    from skillev.policy.sglang_text import SGLangFrozenText

    return SGLangFrozenText(
        url,
        policy.tokenizer,
        reference_id=f"{policy.reference_id}/sglang-executor",
        context_window=int(os.environ.get("EVOSTEER_EXECUTOR_CONTEXT", "32768")),
    )


def task_factory(*, policy: Any, config: Any) -> tuple[TaskBinding, ...]:
    """Return resettable bindings from a frozen, answer-private JSONL manifest."""
    root = Path(os.environ.get("EVOSTEER_CURVE_DATA_ROOT", "data/curve_benchmarks")).resolve()
    rows = _load_rows(root)
    executor_model = _executor_model(policy)
    executor_id = FrozenTextExecutor(executor_model).frozen_identity
    bindings: list[TaskBinding] = []
    for row in rows:
        kind, task_id = row["kind"], row["task_id"]
        prompt = _task_prompt(row)
        task = EvoTask(task_id, kind, prompt, reset_id="initial", environment_config_id=f"curve-{kind}@2")

        def session(request: SessionRequest, *, _row=row, _kind=kind, _task=task) -> TaskSession:
            def evaluate(output: str) -> float:
                return _score(_kind, output, _row)

            return TaskSession(
                FrozenTextExecutor(executor_model),
                evaluate,
                # Paired arms share a seed and an initial state but are distinct
                # sessions, so the session ID carries a per-session suffix.
                reset_receipt=ResetReceipt(_task.identity, _task.reset_id, _task.environment_config_id, _task.identity, f"{_task.task_id}:{request.seed}:{uuid.uuid4().hex}", request.seed),
                risk_assessor=ExecutionRiskPolicy(executor_id, _task.environment_config_id, scope="text_only", capability_id="curve-static-verifier@1"),
            )

        bindings.append(TaskBinding(task, executor_id, session, replay_safe=True))
    return tuple(bindings)


__all__ = ["task_factory"]
