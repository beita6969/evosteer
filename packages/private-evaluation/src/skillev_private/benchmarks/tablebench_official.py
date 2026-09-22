"""Pinned official TableBench scoring behind the private evaluator boundary.

The public rollout task contains the table and question, but never the answer.
This module loads the answer-bearing materialized manifest, applies the metric
dispatch from the pinned TableBench evaluator, and returns only content-free
per-item metrics.  Visualization programs execute in a separate interpreter;
the trusted reference values are compared in the parent and are never placed
in the candidate process.
"""

from __future__ import annotations

import ast
import json
import math
import re
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import ClassVar, Protocol, cast

from skillev.contracts import JsonValue, artifact_hash, normalize_json, validate_sha256
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

from .external_completion import (
    ExternalCompletionInfrastructureError,
    ExternalCompletionResult,
    ExternalCompletionSessionFactory,
)

TABLEBENCH_SOURCE_REVISION = "6c61a20340409dea8f96edb8675e66bb4abd7b28"
TABLEBENCH_VERIFIER_VERSION = f"tablebench-official@{TABLEBENCH_SOURCE_REVISION}"

_FINAL_ANSWER = re.compile(r"Final Answer: (.+)")
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", flags=re.DOTALL)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_ALPHANUMERIC = re.compile(r"^[a-z0-9]+$")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?%?$")
_TOLERANT_DATA_ANALYSIS = frozenset(
    {"CorrelationAnalysis", "StatisticalAnalysis", "TrendForecasting"}
)


class TableBenchEvaluationInfrastructureError(ExternalCompletionInfrastructureError):
    """The private manifest or isolated official evaluator is unusable."""


@dataclass(frozen=True, slots=True)
class TableBenchEvaluationCase:
    task_id: str
    qtype: str
    qsubtype: str
    answer: str
    table: dict[str, JsonValue]
    chart_type: str | None

    def __post_init__(self) -> None:
        for name in ("task_id", "qtype", "qsubtype", "answer"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or "\x00" in value:
                raise ValueError(f"TableBench {name} must be non-empty text without NUL")
        if self.qtype == "Visualization":
            if type(self.chart_type) is not str or not self.chart_type.strip():
                raise ValueError("TableBench visualization requires chart_type")
        elif self.chart_type is not None:
            raise ValueError("non-visual TableBench case cannot carry chart_type")


@dataclass(frozen=True, slots=True)
class TableBenchChartRequest:
    """Answer-free request to the isolated candidate-code interpreter."""

    task_id: str
    prediction: str
    table: dict[str, JsonValue]
    chart_type: str


@dataclass(frozen=True, slots=True)
class TableBenchChartResult:
    executed: bool
    y_predictions: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if type(self.executed) is not bool:
            raise TypeError("TableBench chart executed must be boolean")
        if any(not math.isfinite(value) for row in self.y_predictions for value in row):
            raise ValueError("TableBench chart predictions must be finite")


class TableBenchChartBackend(Protocol):
    async def run(self, request: TableBenchChartRequest) -> TableBenchChartResult: ...


def _manifest_record(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != {"answer", "official_row", "task_id"}:
        raise ValueError("TableBench private manifest record has incompatible fields")
    if any(type(key) is not str for key in value):
        raise TypeError("TableBench private manifest keys must be text")
    return cast(dict[str, JsonValue], value)


def _text(value: JsonValue, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"TableBench private {field} must be non-empty text without NUL")
    return value


def load_tablebench_cases(private_manifest: Path) -> tuple[TableBenchEvaluationCase, ...]:
    if not private_manifest.is_file():
        raise FileNotFoundError(private_manifest)
    cases: list[TableBenchEvaluationCase] = []
    with private_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("TableBench private manifest contains an empty record")
            record = _manifest_record(json.loads(line))
            official = record["official_row"]
            if not isinstance(official, dict):
                raise ValueError("TableBench official_row must be an object")
            answer = _text(record["answer"], field="answer")
            if normalize_json(official.get("answer")) != answer:
                raise ValueError("TableBench materialized answer differs from official row")
            table = normalize_json(official.get("table"))
            if not isinstance(table, dict):
                raise ValueError("TableBench official table must be an object")
            qtype = _text(official.get("qtype"), field="qtype")
            chart_value = official.get("chart_type")
            chart_type = None if chart_value is None else _text(chart_value, field="chart_type")
            cases.append(
                TableBenchEvaluationCase(
                    task_id=_text(record["task_id"], field="task_id"),
                    qtype=qtype,
                    qsubtype=_text(official.get("qsubtype"), field="qsubtype"),
                    answer=answer,
                    table=cast(dict[str, JsonValue], table),
                    chart_type=chart_type,
                )
            )
    if not cases or len({case.task_id for case in cases}) != len(cases):
        raise ValueError("TableBench private cases must be non-empty and task-unique")
    return tuple(cases)


def load_tablebench_session_factory(
    *,
    tasks: tuple[RolloutTask, ...],
    private_manifest: Path,
    expected_sha256: str,
    chart_backend: TableBenchChartBackend | None = None,
) -> ExternalCompletionSessionFactory:
    """Load a task-exact factory from a raw-byte-pinned private manifest."""

    validate_sha256(expected_sha256)
    if artifact_hash(private_manifest.read_bytes()) != expected_sha256:
        raise ValueError("TableBench private manifest differs from its frozen identity")
    worker = TableBenchOfficialWorker(
        cases=load_tablebench_cases(private_manifest),
        chart_backend=chart_backend or IsolatedTableBenchChartBackend(),
    )
    return ExternalCompletionSessionFactory(tasks=tasks, worker=worker)


def _normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _parsed_prediction(prediction: str) -> str:
    match = _FINAL_ANSWER.search(prediction)
    return "" if match is None else match.group(1)


def _is_number(value: str) -> bool:
    return _NUMBER.fullmatch(value.strip()) is not None


def _decimal(value: str) -> Decimal:
    if value.endswith("%"):
        return (Decimal(value.removesuffix("%")) / Decimal("100")).quantize(
            Decimal("1.0000"), rounding=ROUND_HALF_UP
        )
    return Decimal(value)


def _reference_precision(values: list[str]) -> int:
    precisions = [
        len(value.rsplit(".", 1)[1]) if "." in value else 0
        for value in values
        if not value.endswith("%")
    ]
    return min(precisions) if precisions else 0


def _rounded(value: Decimal, precision: int) -> str:
    return str(value.quantize(Decimal(f"1.{'0' * precision}"), rounding=ROUND_HALF_UP))


def _exact_match(reference: str, prediction: str) -> float:
    references = [item.strip() for item in reference.split(",")]
    predictions = [item.strip() for item in prediction.split(",")]
    weight = 1.0 / len(references)
    score = 0.0
    for index, expected in enumerate(references):
        if index >= len(predictions):
            continue
        actual = predictions[index]
        if _is_number(expected):
            try:
                if expected.endswith("%"):
                    matched = _decimal(expected) == _decimal(actual)
                else:
                    numeric_references = [
                        item for item in references if _is_number(item) and not item.endswith("%")
                    ]
                    precision = _reference_precision(numeric_references)
                    matched = _rounded(_decimal(expected), precision) == _rounded(
                        _decimal(actual), precision
                    )
            except InvalidOperation:
                matched = False
        else:
            matched = expected == actual
        if matched:
            score += weight
    return score


def _tolerant_exact_match(reference: str, prediction: str) -> float:
    references = [item.strip() for item in reference.split(",")]
    predictions = [item.strip() for item in prediction.split(",")]
    weight = 1.0 / len(references)
    score = 0.0
    for index, expected in enumerate(references):
        if index >= len(predictions):
            continue
        actual = predictions[index]
        if _is_number(expected):
            try:
                expected_number = _decimal(expected)
                actual_number = _decimal(actual)
                matched = (
                    actual_number == expected_number
                    if expected_number == 0
                    else abs(expected_number - actual_number) / abs(expected_number)
                    <= Decimal("0.10")
                )
            except InvalidOperation:
                matched = False
        else:
            matched = expected == actual
        if matched:
            score += weight
    return score


def _rouge_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _NON_ALPHANUMERIC.sub(" ", value.lower()).split()
        if _ALPHANUMERIC.fullmatch(token)
    )


def _rouge_l(reference: str, prediction: str) -> float:
    expected = _rouge_tokens(reference)
    actual = _rouge_tokens(prediction)
    if not expected or not actual:
        return 0.0
    previous = [0] * (len(actual) + 1)
    for expected_token in expected:
        current = [0]
        for index, actual_token in enumerate(actual, start=1):
            current.append(
                previous[index - 1] + 1
                if expected_token == actual_token
                else max(previous[index], current[-1])
            )
        previous = current
    overlap = previous[-1]
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _qa_score(case: TableBenchEvaluationCase, submission: str) -> tuple[str, float, bool]:
    parsed = _parsed_prediction(submission)
    reference = _normalize_answer(case.answer)
    prediction = _normalize_answer(parsed)
    parsed_ok = bool(parsed)
    if case.qtype in {"FactChecking", "NumericalReasoning"} or (
        case.qtype == "DataAnalysis" and case.qsubtype == "ImpactAnalysis"
    ):
        return "EM", _exact_match(reference, prediction), parsed_ok
    if case.qtype == "DataAnalysis" and case.qsubtype in _TOLERANT_DATA_ANALYSIS:
        return "EM_with_error_10", _tolerant_exact_match(reference, prediction), parsed_ok
    if case.qtype == "DataAnalysis":
        return "ROUGE-L", _rouge_l(reference, prediction), parsed_ok
    raise TableBenchEvaluationInfrastructureError("TableBench qtype has no official QA metric")


def _chart_reference(answer: str) -> tuple[tuple[float, ...], ...]:
    try:
        module = ast.parse(answer, mode="exec")
    except SyntaxError as error:
        raise TableBenchEvaluationInfrastructureError(
            "TableBench trusted chart reference is invalid"
        ) from error
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        raise TableBenchEvaluationInfrastructureError(
            "TableBench trusted chart reference has an incompatible shape"
        )
    assignment = module.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "y_references"
    ):
        raise TableBenchEvaluationInfrastructureError(
            "TableBench trusted chart reference has an incompatible target"
        )
    try:
        value = ast.literal_eval(assignment.value)
    except (ValueError, TypeError) as error:
        raise TableBenchEvaluationInfrastructureError(
            "TableBench trusted chart reference is not literal data"
        ) from error
    rows = value if isinstance(value, list) else [value]
    normalized: list[tuple[float, ...]] = []
    for row in rows:
        values = row if isinstance(row, list) else [row]
        if any(type(item) not in {int, float} for item in values):
            raise TableBenchEvaluationInfrastructureError(
                "TableBench trusted chart reference is not numeric"
            )
        numeric = tuple(float(item) for item in values)
        if any(not math.isfinite(item) for item in numeric):
            raise TableBenchEvaluationInfrastructureError(
                "TableBench trusted chart reference is not finite"
            )
        normalized.append(numeric)
    return tuple(normalized)


def _flatten(values: tuple[tuple[float, ...], ...]) -> list[float]:
    return [round(value, 2) for row in values for value in row]


def _chart_passed(
    *,
    chart_type: str,
    reference: tuple[tuple[float, ...], ...],
    prediction: tuple[tuple[float, ...], ...],
) -> bool:
    expected = [value for row in reference for value in row]
    if chart_type == "pie":
        total = sum(expected)
        if total == 0.0:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench trusted pie reference has zero total"
            )
        expected = [round(value / total, 2) for value in expected]
    else:
        expected = [round(value, 2) for value in expected]
    actual = _flatten(prediction)
    expected.sort()
    actual.sort()
    return expected == actual


_CHART_CHILD = r"""
import contextlib
import csv
import json
import os
import re
import sys

os.environ["MPLBACKEND"] = "Agg"
payload = json.loads(sys.stdin.buffer.read())
match = re.findall(r"```python\n(.*?)```", payload["prediction"], flags=re.S)
if not match:
    print(json.dumps({"executed": False, "y_predictions": []}))
    raise SystemExit(0)

table = payload["table"]
with open("table.csv", "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(table["columns"])
    writer.writerows(table["data"])

try:
    import matplotlib.pyplot as plt
    namespace = {"__name__": "__main__"}
    with open(os.devnull, "w", encoding="utf-8") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            exec(compile(match[-1], "<tablebench-candidate>", "exec"), namespace, namespace)
    chart_type = payload["chart_type"]
    if chart_type in {"bar", "waterfall"}:
        values = [[patch.get_height() for patch in plt.gca().patches]]
    elif chart_type == "hbar":
        values = [[patch.get_width() for patch in plt.gca().patches]]
    elif chart_type == "pie":
        values = [[round((patch.theta2 - patch.theta1) / 360.0, 2) for patch in plt.gca().patches]]
    elif chart_type == "line":
        values = [list(line.get_ydata()) for line in plt.gca().get_lines()]
    elif chart_type == "area":
        values = [
            [item for item in collection.get_paths()[0].vertices[:, 1] if item != 0]
            for collection in plt.gca().collections
        ]
    elif chart_type == "radar":
        values = [list(line.get_ydata())[:-1] for line in plt.gca().get_lines()]
    elif chart_type == "scatter":
        values = [
            [item[1] for item in collection.get_offsets()]
            for collection in plt.gca().collections
        ]
    else:
        raise ValueError("unknown chart type")
    values = [[float(item) for item in row] for row in values]
    if any(not __import__("math").isfinite(item) for row in values for item in row):
        raise ValueError("non-finite chart value")
except BaseException:
    print(json.dumps({"executed": False, "y_predictions": []}))
    raise SystemExit(0)
print(json.dumps({"executed": True, "y_predictions": values}, separators=(",", ":")))
"""


@dataclass(frozen=True, slots=True)
class IsolatedTableBenchChartBackend:
    wall_timeout_seconds: float = 20.0

    async def run(self, request: TableBenchChartRequest) -> TableBenchChartResult:
        if not isinstance(request, TableBenchChartRequest):
            raise TypeError("TableBench chart backend requires TableBenchChartRequest")
        payload = json.dumps(
            {
                "chart_type": request.chart_type,
                "prediction": request.prediction,
                "table": request.table,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        with tempfile.TemporaryDirectory(prefix="skillev-tablebench-") as directory:
            try:
                completed = subprocess.run(  # noqa: S603 - fixed interpreter and child source
                    (sys.executable, "-I", "-c", _CHART_CHILD),
                    input=payload,
                    cwd=directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=self.wall_timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise TableBenchEvaluationInfrastructureError(
                    "TableBench chart evaluator process failed"
                ) from error
        if completed.returncode != 0 or len(completed.stdout) > 2 * 1024 * 1024:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench chart evaluator returned an invalid result"
            )
        try:
            result = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench chart evaluator returned an invalid result"
            ) from error
        if not isinstance(result, dict) or set(result) != {"executed", "y_predictions"}:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench chart evaluator returned an invalid result"
            )
        rows = result["y_predictions"]
        if type(rows) is not list or any(type(row) is not list for row in rows):
            raise TableBenchEvaluationInfrastructureError(
                "TableBench chart evaluator returned an invalid result"
            )
        try:
            predictions = tuple(tuple(float(value) for value in row) for row in rows)
        except (TypeError, ValueError) as error:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench chart evaluator returned an invalid result"
            ) from error
        return TableBenchChartResult(executed=result["executed"], y_predictions=predictions)


@dataclass(frozen=True, slots=True)
class TableBenchOfficialWorker:
    benchmark_id: ClassVar[str] = Benchmark.TABLEBENCH.value

    cases: tuple[TableBenchEvaluationCase, ...]
    chart_backend: TableBenchChartBackend
    verifier_version: str = TABLEBENCH_VERIFIER_VERSION

    def __post_init__(self) -> None:
        if not self.cases or len({case.task_id for case in self.cases}) != len(self.cases):
            raise ValueError("TableBench worker cases must be non-empty and task-unique")
        if not callable(getattr(self.chart_backend, "run", None)):
            raise TypeError("TableBench chart backend must implement run")

    async def evaluate(self, *, task_id: str, submission: str) -> ExternalCompletionResult:
        matches = tuple(case for case in self.cases if case.task_id == task_id)
        if len(matches) != 1:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench task is absent from the frozen private manifest"
            )
        case = matches[0]
        if case.qtype == "Visualization":
            assert case.chart_type is not None
            result = await self.chart_backend.run(
                TableBenchChartRequest(
                    task_id=case.task_id,
                    prediction=submission,
                    table=case.table,
                    chart_type=case.chart_type,
                )
            )
            score = float(
                result.executed
                and _chart_passed(
                    chart_type=case.chart_type,
                    reference=_chart_reference(case.answer),
                    prediction=result.y_predictions,
                )
            )
            metric_name = "Pass@1"
            public_metrics: dict[str, JsonValue] = {
                "ECR@1": float(result.executed),
                "Pass@1": score,
                "Parse@1": float(bool(_PYTHON_BLOCK.findall(submission))),
            }
        else:
            metric_name, score, parsed = _qa_score(case, submission)
            public_metrics = {metric_name: score, "Parse@1": float(parsed)}
        return ExternalCompletionResult(
            reward_value=score,
            success=score == 1.0,
            native_metric_name=metric_name,
            public_metrics=public_metrics,
            verifier_version=self.verifier_version,
        )

    def no_submission_result(self, *, task_id: str) -> ExternalCompletionResult:
        matches = tuple(case for case in self.cases if case.task_id == task_id)
        if len(matches) != 1:
            raise TableBenchEvaluationInfrastructureError(
                "TableBench task is absent from the frozen private manifest"
            )
        case = matches[0]
        if case.qtype == "Visualization":
            metric_name = "Pass@1"
            public_metrics: dict[str, JsonValue] = {
                "ECR@1": 0.0,
                "Parse@1": 0.0,
                "Pass@1": 0.0,
            }
        else:
            metric_name, _, _ = _qa_score(case, "")
            public_metrics = {metric_name: 0.0, "Parse@1": 0.0}
        public_metrics["no_submission_reason"] = "horizon-exhausted"
        return ExternalCompletionResult(
            reward_value=0.0,
            success=False,
            native_metric_name=metric_name,
            public_metrics=public_metrics,
            verifier_version=self.verifier_version,
        )


__all__ = [
    "TABLEBENCH_SOURCE_REVISION",
    "TABLEBENCH_VERIFIER_VERSION",
    "IsolatedTableBenchChartBackend",
    "TableBenchChartBackend",
    "TableBenchChartRequest",
    "TableBenchChartResult",
    "TableBenchEvaluationCase",
    "TableBenchEvaluationInfrastructureError",
    "TableBenchOfficialWorker",
    "load_tablebench_cases",
    "load_tablebench_session_factory",
]
