from __future__ import annotations

import json
import random
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from skillev_private.benchmarks.protocol_v10_materialization import (
    ProtocolV10SourceRecord,
    load_protocol_v10_materialized_catalog,
    materialize_protocol_v10_population,
    publish_protocol_v10_materialized_catalog,
)
from skillev_private.benchmarks.protocol_v10_sources import (
    _exclude_alfworld_evaluation_scenarios,
    _humaneval_records,
    _load_webshop_runtime_goals,
    _spreadsheet_training_records,
    _unique_alfworld_scenarios,
    _unique_code_rows,
    _unique_webshop_goal_indices,
)

from skillev.experiments import BenchmarkV10
from skillev.experiments.protocol_v10 import load_active_protocol_v10
from skillev.rollout import RolloutTask


def _protocol():
    return load_active_protocol_v10(Path("configs/evaluation/protocol_v10.yaml").resolve())


def _record(spec, index: int) -> ProtocolV10SourceRecord:
    context: dict[str, object] = {
        "benchmark_id": spec.benchmark.value,
        "population_id": spec.population_id,
        "source_version": spec.source_version,
    }
    if spec.benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
        context["code_signature"] = f"function_{spec.population_id}_{index}()"
    if spec.benchmark is BenchmarkV10.SPREADSHEETBENCH:
        context["workbook_structure_id"] = f"workbook/{spec.population_id}/{index}"
    if spec.benchmark in {
        BenchmarkV10.WEBSHOP,
        BenchmarkV10.ALFWORLD,
        BenchmarkV10.APPWORLD,
    }:
        context["scenario_id"] = f"scenario/{spec.population_id}/{index}"
    source_id = f"{spec.benchmark.value}/{spec.population_id}/{index}"
    task = RolloutTask(
        task_id=source_id,
        environment_id=f"benchmark:{spec.benchmark.value}",
        task_family=f"{spec.benchmark.value}/test",
        context_id=spec.population_id,
        query=f"public query {spec.population_id} {index}",
        available_tools=(),
        public_context=context,
    )
    return ProtocolV10SourceRecord(
        source_id=source_id,
        task=task,
        private_payload={"answer": f"private-{spec.population_id}-{index}"},
    )


def test_materialization_separates_private_payload(tmp_path: Path) -> None:
    protocol = _protocol()
    spec = protocol.benchmarks[0].populations[0]
    materialized = materialize_protocol_v10_population(spec, (_record(spec, 0),))

    public_value = materialized.population.items[0].to_value()
    assert "private_payload" not in json.dumps(public_value)
    assert materialized.private_records[0].private_payload == {
        "answer": f"private-{spec.population_id}-0"
    }


def test_publish_builds_complete_skillflow_mix(tmp_path: Path) -> None:
    protocol = _protocol()
    materialized = tuple(
        materialize_protocol_v10_population(
            spec,
            tuple(_record(spec, index) for index in range(2)),
        )
        for benchmark in protocol.benchmarks
        for spec in benchmark.populations
    )
    output = tmp_path / "protocol-v10-private"
    catalog = publish_protocol_v10_materialized_catalog(
        protocol=protocol,
        populations=materialized,
        output_root=output,
    )

    selection = json.loads((output / "training-selection.json").read_text())
    assert len(selection["episodes"]) == 4_608
    counts = {
        benchmark.value: sum(
            episode["benchmark"] == benchmark.value for episode in selection["episodes"]
        )
        for benchmark in BenchmarkV10
    }
    assert set(counts.values()) == {512}
    assert len(catalog.populations) == sum(
        len(benchmark.populations) for benchmark in protocol.benchmarks
    )
    private_path = (
        output / "private-records" / f"{protocol.benchmarks[0].populations[0].population_id}.jsonl"
    )
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert "private_payload" in private_path.read_text()
    assert (
        "private_payload"
        not in (
            output / "populations" / f"{protocol.benchmarks[0].populations[0].population_id}.jsonl"
        ).read_text()
    )

    loaded = load_protocol_v10_materialized_catalog(protocol, output.resolve())
    assert loaded.catalog == catalog
    assert len(loaded.selection.episodes) == 4_608
    assert set(loaded.private_record_files) == {
        population.spec.population_id for population in catalog.populations
    }


def test_spreadsheet_training_accepts_chat_prompt_column(tmp_path: Path) -> None:
    protocol = _protocol()
    spec = next(
        population
        for benchmark in protocol.benchmarks
        for population in benchmark.populations
        if population.population_id == "independent-spreadsheet-train-v1"
    )
    source = tmp_path / "spreadsheet.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "Update the workbook."}],
                    "reward_model": {"ground_truth": "private", "style": "rule"},
                    "extra_info": {"id": "sheet-1", "answer_position": "A1"},
                }
            ]
        ),
        source,
    )

    records = _spreadsheet_training_records(spec, source)

    assert records[0].task.query == "user: Update the workbook."
    assert "private" not in records[0].task.query
    assert records[0].private_payload == {
        "answer_position": "A1",
        "spreadsheet_task_route": "private",
    }


def test_webshop_split_uses_one_index_per_visible_goal() -> None:
    goals = ["Find a blue mug", " find  A   BLUE mug ", "Find a red mug"]

    assert _unique_webshop_goal_indices(goals) == [0, 2]


def test_webshop_source_order_matches_official_runtime_sessions(tmp_path: Path) -> None:
    source = tmp_path / "goals.jsonl"
    rows = [{"instruction_text": f"public instruction {index}"} for index in range(12_087)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    expected = list(rows)
    random.Random(233).shuffle(expected)  # noqa: S311 - official runtime fixture

    goals = _load_webshop_runtime_goals(source)

    assert goals == tuple(row["instruction_text"] for row in expected)


def test_alfworld_keeps_one_trial_per_scenario_and_protects_evaluation() -> None:
    protocol = _protocol()
    spec = next(
        population
        for benchmark in protocol.benchmarks
        for population in benchmark.populations
        if population.population_id == "alfworld-train-main"
    )

    def record(source: str, scenario: str) -> ProtocolV10SourceRecord:
        task = RolloutTask(
            task_id=source,
            environment_id="benchmark:alfworld",
            task_family="alfworld/test",
            context_id="alfworld:training",
            query="Complete the interactive task.",
            available_tools=("act",),
            public_context={
                "benchmark_id": spec.benchmark.value,
                "scenario_id": scenario,
            },
        )
        return ProtocolV10SourceRecord(source, task, {"route": source})

    training = (
        record("alfworld/train/a/1", "scenario/a"),
        record("alfworld/train/a/2", "scenario/a"),
        record("alfworld/train/b/1", "scenario/b"),
    )
    evaluation = (record("alfworld/final/b/1", "scenario/b"),)

    unique = _unique_alfworld_scenarios(training)
    isolated = _exclude_alfworld_evaluation_scenarios(unique, evaluation)

    assert [item.source_id for item in unique] == [
        "alfworld/train/a/1",
        "alfworld/train/b/1",
    ]
    assert [item.source_id for item in isolated] == ["alfworld/train/a/1"]


def test_independent_code_record_keeps_tests_private() -> None:
    protocol = _protocol()
    spec = next(
        population
        for benchmark in protocol.benchmarks
        for population in benchmark.populations
        if population.population_id == "independent-code-train-excluding-all-mbpp-plus"
    )
    records = _humaneval_records(
        spec,
        (
            {
                "task_id": "HumanEval/0",
                "prompt": 'def add(a, b):\n    """Return the sum."""\n',
                "canonical_solution": "    return a + b\n",
                "test": "def check(candidate): assert candidate(1, 2) == 3",
                "entry_point": "add",
            },
        ),
    )

    assert records[0].task.query.endswith('Return the sum."""\n')
    assert "candidate(1, 2)" not in records[0].task.query
    assert records[0].private_payload["evaluator_kind"] == "humaneval-sandbox"


def test_independent_code_rows_remove_prompt_or_signature_duplicates() -> None:
    rows = (
        {"task_id": "1", "prompt": "def first(x):\n    pass"},
        {"task_id": "2", "prompt": "def first(x):\n    return x"},
        {"task_id": "3", "prompt": "def third(x):\n    pass"},
    )

    unique = _unique_code_rows(rows)

    assert [row["task_id"] for row in unique] == ["1", "3"]
