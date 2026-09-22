from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from skillev_private.benchmarks.external_materialization import materialize_tablebench
from skillev_private.benchmarks.tablebench_official import (
    TABLEBENCH_VERIFIER_VERSION,
    TableBenchChartRequest,
    TableBenchChartResult,
    TableBenchOfficialWorker,
    load_tablebench_session_factory,
)

from skillev.contracts import artifact_hash, canonical_json
from skillev.rollout import RolloutTask


@dataclass(slots=True)
class _ChartBackend:
    result: TableBenchChartResult
    requests: list[TableBenchChartRequest] = field(default_factory=list)

    async def run(self, request: TableBenchChartRequest) -> TableBenchChartResult:
        self.requests.append(request)
        return self.result


def _materialized(
    root: Path,
) -> tuple[tuple[RolloutTask, ...], Path, str, str]:
    answer_canary = "private tablebench answer canary"
    rows = (
        {
            "answer": "Supported",
            "chart_type": None,
            "id": "fact-1",
            "qsubtype": "MatchBased",
            "qtype": "FactChecking",
            "question": "Is the statement supported?",
            "table": {"columns": ["claim"], "data": [["Supported"]]},
        },
        {
            "answer": "100",
            "chart_type": None,
            "id": "analysis-1",
            "qsubtype": "CorrelationAnalysis",
            "qtype": "DataAnalysis",
            "question": "Estimate the correlation statistic.",
            "table": {"columns": ["x", "y"], "data": [[1, 100]]},
        },
        {
            "answer": answer_canary,
            "chart_type": None,
            "id": "analysis-2",
            "qsubtype": "DescriptiveAnalysis",
            "qtype": "DataAnalysis",
            "question": "Summarize the table.",
            "table": {"columns": ["label"], "data": [["public row"]]},
        },
        {
            "answer": "y_references = [[2, 4]]",
            "chart_type": "line",
            "id": "chart-1",
            "qsubtype": "ChartGeneration",
            "qtype": "Visualization",
            "question": "Plot the values as a line chart.",
            "table": {"columns": ["x", "y"], "data": [[1, 2], [2, 4]]},
        },
    )
    source = root / "source.jsonl"
    source.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    public = root / "derived" / "tasks.jsonl"
    private = root / "private" / "cases.jsonl"
    materialize_tablebench(
        source_jsonl=source,
        dataset_revision="fixture-revision",
        public_manifest=public,
        private_manifest=private,
    )
    tasks = tuple(
        RolloutTask.from_value(json.loads(line)["task"])
        for line in public.read_text(encoding="utf-8").splitlines()
    )
    return tasks, private, artifact_hash(private.read_bytes()), answer_canary


def test_tablebench_loader_pins_manifest_and_keeps_answers_behind_worker(
    tmp_path: Path,
) -> None:
    tasks, manifest, manifest_hash, canary = _materialized(tmp_path)
    backend = _ChartBackend(TableBenchChartResult(True, ((2.0, 4.0),)))

    factory = load_tablebench_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_sha256=manifest_hash,
        chart_backend=backend,
    )

    assert isinstance(factory.worker, TableBenchOfficialWorker)
    assert all(canary not in canonical_json(task.to_value()) for task in tasks)
    assert "Final Answer: <answer>" in tasks[0].query
    assert "Python code block" in tasks[3].query
    assert tasks[3].public_context["chart_type"] == "line"
    with pytest.raises(ValueError):
        load_tablebench_session_factory(
            tasks=tasks,
            private_manifest=manifest,
            expected_sha256=artifact_hash(b"different private manifest"),
            chart_backend=backend,
        )


def test_tablebench_no_submission_is_zero_without_chart_backend(tmp_path: Path) -> None:
    tasks, manifest, manifest_hash, _ = _materialized(tmp_path)
    backend = _ChartBackend(TableBenchChartResult(True, ((2.0, 4.0),)))
    worker = load_tablebench_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_sha256=manifest_hash,
        chart_backend=backend,
    ).worker

    results = tuple(worker.no_submission_result(task_id=task.task_id) for task in tasks)

    assert all(result.reward_value == 0.0 for result in results)
    assert all(result.success is False for result in results)
    assert backend.requests == []


def test_tablebench_worker_matches_official_qa_metric_dispatch(tmp_path: Path) -> None:
    tasks, manifest, manifest_hash, _ = _materialized(tmp_path)
    factory = load_tablebench_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_sha256=manifest_hash,
        chart_backend=_ChartBackend(TableBenchChartResult(False, ())),
    )
    worker = factory.worker

    fact = asyncio.run(
        worker.evaluate(task_id=tasks[0].task_id, submission="Reasoning\nFinal Answer: Supported")
    )
    missing_official_parse_marker = asyncio.run(
        worker.evaluate(task_id=tasks[0].task_id, submission="Supported")
    )
    within_ten_percent = asyncio.run(
        worker.evaluate(task_id=tasks[1].task_id, submission="Final Answer: 109")
    )
    outside_ten_percent = asyncio.run(
        worker.evaluate(task_id=tasks[1].task_id, submission="Final Answer: 111")
    )
    partial_rouge = asyncio.run(
        worker.evaluate(
            task_id=tasks[2].task_id,
            submission="Final Answer: private answer canary",
        )
    )

    assert fact.reward_value == 1.0
    assert fact.native_metric_name == "EM"
    assert fact.public_metrics == {"EM": 1.0, "Parse@1": 1.0}
    assert missing_official_parse_marker.reward_value == 0.0
    assert missing_official_parse_marker.public_metrics["Parse@1"] == 0.0
    assert within_ten_percent.reward_value == 1.0
    assert within_ten_percent.native_metric_name == "EM_with_error_10"
    assert outside_ten_percent.reward_value == 0.0
    assert partial_rouge.native_metric_name == "ROUGE-L"
    assert partial_rouge.reward_value == pytest.approx(6 / 7)
    assert fact.verifier_version == TABLEBENCH_VERIFIER_VERSION


def test_tablebench_visualization_keeps_reference_out_of_chart_backend(
    tmp_path: Path,
) -> None:
    tasks, manifest, manifest_hash, _ = _materialized(tmp_path)
    backend = _ChartBackend(TableBenchChartResult(True, ((4.0, 2.0),)))
    factory = load_tablebench_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_sha256=manifest_hash,
        chart_backend=backend,
    )

    result = asyncio.run(
        factory.worker.evaluate(
            task_id=tasks[3].task_id,
            submission="```python\nimport matplotlib.pyplot as plt\nplt.plot([2, 4])\n```",
        )
    )

    assert result.reward_value == 1.0
    assert result.native_metric_name == "Pass@1"
    assert result.public_metrics == {"ECR@1": 1.0, "Pass@1": 1.0, "Parse@1": 1.0}
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.task_id == tasks[3].task_id
    assert request.chart_type == "line"
    assert "y_references" not in canonical_json(
        {
            "chart_type": request.chart_type,
            "prediction": request.prediction,
            "table": request.table,
            "task_id": request.task_id,
        }
    )


def test_tablebench_factory_requires_exact_public_task_projection(tmp_path: Path) -> None:
    tasks, manifest, manifest_hash, _ = _materialized(tmp_path)
    factory = load_tablebench_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_sha256=manifest_hash,
        chart_backend=_ChartBackend(TableBenchChartResult(False, ())),
    )
    changed = RolloutTask(
        task_id=tasks[0].task_id,
        environment_id=tasks[0].environment_id,
        task_family=tasks[0].task_family,
        context_id=tasks[0].context_id,
        query=tasks[0].query + " changed",
        available_tools=tasks[0].available_tools,
        public_context=tasks[0].public_context,
    )

    with pytest.raises(ValueError):
        factory.create(changed)
