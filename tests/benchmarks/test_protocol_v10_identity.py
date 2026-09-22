from __future__ import annotations

import json
from pathlib import Path

import pytest
from skillev_private.benchmarks import (
    PopulationOverlapKind,
    ProtocolV10TaskIdentityMaterializer,
    load_protocol_v10_population_file,
    normalize_public_content,
)

from skillev.experiments import BenchmarkV10, PopulationRole, load_active_protocol_v10
from skillev.rollout import RolloutTask

ROOT = Path(__file__).parents[2]


def _spec(benchmark: BenchmarkV10):
    protocol = load_active_protocol_v10(ROOT / "configs" / "evaluation" / "protocol_v10.yaml")
    benchmark_spec = next(item for item in protocol.benchmarks if item.benchmark is benchmark)
    return benchmark_spec.population(PopulationRole.TRAINING)[0]


def _task(benchmark: BenchmarkV10, source_id: str) -> RolloutTask:
    context = {"benchmark_id": benchmark.value}
    if benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
        context["code_signature"] = "def solve(value: int) -> int"
    if benchmark is BenchmarkV10.SPREADSHEETBENCH:
        context["workbook_structure_id"] = "Workbook / Sheet 1 / A1:C8"
    if benchmark in {
        BenchmarkV10.WEBSHOP,
        BenchmarkV10.ALFWORLD,
        BenchmarkV10.APPWORLD,
    }:
        context["scenario_id"] = "Scenario Group 7"
    return RolloutTask(
        task_id=source_id,
        environment_id=f"private:{benchmark.value}",
        task_family=benchmark.value,
        context_id=f"context:{source_id}",
        query="  Solve\tTHE   public\n task.  ",
        available_tools=(),
        public_context=context,
    )


def test_public_content_normalization_is_unicode_and_whitespace_stable() -> None:
    assert normalize_public_content("  \uff28ello\tWORLD\n") == "hello world"


@pytest.mark.parametrize("benchmark", tuple(BenchmarkV10))
def test_materializer_rebuilds_required_identities(benchmark: BenchmarkV10) -> None:
    source_id = f"{benchmark.value}/record/7"
    keys = ProtocolV10TaskIdentityMaterializer().materialize(
        spec=_spec(benchmark),
        source_id=source_id,
        task=_task(benchmark, source_id),
    )
    kinds = {item.kind for item in keys}
    assert PopulationOverlapKind.SOURCE_RECORD in kinds
    assert PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT in kinds
    if benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
        assert PopulationOverlapKind.CODE_SIGNATURE in kinds
    if benchmark is BenchmarkV10.SPREADSHEETBENCH:
        assert PopulationOverlapKind.WORKBOOK_STRUCTURE in kinds
    if benchmark in {
        BenchmarkV10.WEBSHOP,
        BenchmarkV10.ALFWORLD,
        BenchmarkV10.APPWORLD,
    }:
        assert PopulationOverlapKind.INTERACTIVE_SCENARIO in kinds


def test_population_loader_rejects_declared_identity_drift(tmp_path: Path) -> None:
    materializer = ProtocolV10TaskIdentityMaterializer()
    spec = _spec(BenchmarkV10.HOTPOT_QA)
    task = _task(BenchmarkV10.HOTPOT_QA, "hotpot/record/1")
    keys = materializer.materialize(spec=spec, source_id=task.task_id, task=task)
    value = {
        "source_id": task.task_id,
        "task": task.to_value(),
        "overlap_keys": [item.to_value() for item in keys],
    }
    value["overlap_keys"][1]["value"] = "declared-content-that-is-not-the-task"
    path = tmp_path / "population.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_protocol_v10_population_file(spec, path, identity_materializer=materializer)


def test_group_identity_requires_source_metadata() -> None:
    benchmark = BenchmarkV10.APPWORLD
    task = _task(benchmark, "appworld/record/1")
    task = RolloutTask(
        task_id=task.task_id,
        environment_id=task.environment_id,
        task_family=task.task_family,
        context_id=task.context_id,
        query=task.query,
        available_tools=task.available_tools,
        public_context={"benchmark_id": benchmark.value},
    )
    with pytest.raises(ValueError):
        ProtocolV10TaskIdentityMaterializer().materialize(
            spec=_spec(benchmark),
            source_id=task.task_id,
            task=task,
        )


@pytest.mark.parametrize(
    ("benchmark", "source_prefix"),
    [
        (BenchmarkV10.ALFWORLD, "alfworld"),
        (BenchmarkV10.APPWORLD, "appworld"),
    ],
)
def test_interactive_variations_keep_distinct_item_identities(
    benchmark: BenchmarkV10, source_prefix: str
) -> None:
    materializer = ProtocolV10TaskIdentityMaterializer()
    first = _task(benchmark, f"{source_prefix}/scenario_1")
    second = _task(benchmark, f"{source_prefix}/scenario_2")

    first_keys = materializer.materialize(
        spec=_spec(benchmark), source_id=first.task_id, task=first
    )
    second_keys = materializer.materialize(
        spec=_spec(benchmark), source_id=second.task_id, task=second
    )

    def public_content(keys):
        return next(
            key.value for key in keys if key.kind is PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT
        )

    assert public_content(first_keys) != public_content(second_keys)
    assert {
        key.value for key in first_keys if key.kind is PopulationOverlapKind.INTERACTIVE_SCENARIO
    } == {
        key.value for key in second_keys if key.kind is PopulationOverlapKind.INTERACTIVE_SCENARIO
    }


def test_code_tasks_include_visible_signature_in_content_identity() -> None:
    benchmark = BenchmarkV10.MBPP_PLUS_FIXED_100
    materializer = ProtocolV10TaskIdentityMaterializer()
    first = _task(benchmark, "code/task/1")
    second_base = _task(benchmark, "code/task/2")
    second_context = dict(second_base.public_context)
    second_context["code_signature"] = "def another(value: int) -> int"
    second = RolloutTask(
        task_id=second_base.task_id,
        environment_id=second_base.environment_id,
        task_family=second_base.task_family,
        context_id=second_base.context_id,
        query=first.query,
        available_tools=second_base.available_tools,
        public_context=second_context,
    )

    first_keys = materializer.materialize(
        spec=_spec(benchmark), source_id=first.task_id, task=first
    )
    second_keys = materializer.materialize(
        spec=_spec(benchmark), source_id=second.task_id, task=second
    )

    first_content = next(
        key.value
        for key in first_keys
        if key.kind is PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT
    )
    second_content = next(
        key.value
        for key in second_keys
        if key.kind is PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT
    )
    assert first_content != second_content
