from __future__ import annotations

import json
from pathlib import Path

from skillev_private.benchmarks.external_process_sources import (
    materialize_appworld_source,
    materialize_bfcl_v3_source,
    materialize_skillflow_bench_source,
    materialize_tua_bench_source,
)

from skillev.contracts import canonical_json
from skillev.rollout import RolloutTask


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_skillflow_materialization_reads_every_public_instruction_only(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    repository = root / "skillflow-bench" / "repository"
    rows = []
    for ordinal in range(2):
        family = f"family-{ordinal}"
        name = f"task-{ordinal}"
        task_path = f"{family}/{name}"
        rows.append(
            {
                "category": "spreadsheets",
                "difficulty": "medium",
                "family": family,
                "tags": ["xlsx", "analysis"],
                "task_name": name,
                "task_path": task_path,
            }
        )
        _write(repository / task_path / "instruction.md", f"Complete public task {ordinal}.\n")
        _write(repository / task_path / "environment" / "input.txt", "public input\n")
        _write(repository / task_path / "solution" / "answer.txt", "private-answer-canary\n")
        _write(repository / task_path / "tests" / "test.py", "private-test-canary\n")
    _write(
        repository / "benchmark_manifest.json",
        canonical_json({"task_count": len(rows)}) + "\n",
    )
    _write(
        repository / "viewer" / "task_manifest.jsonl",
        "".join(canonical_json(row) + "\n" for row in rows),
    )

    result = materialize_skillflow_bench_source(
        repository_root=repository,
        target_root=root,
        dataset_revision="a" * 40,
    )

    output = root / result.task_manifest_relative_path
    tasks = tuple(
        RolloutTask.from_value(json.loads(line))
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert result.task_count == len(tasks) == 2
    assert all(task.available_tools == ("skillflow-bench.execute", "submit") for task in tasks)
    assert all(
        set(task.public_context["tools"]) == {"skillflow-bench.execute", "submit"} for task in tasks
    )
    assert {task.query for task in tasks} == {
        "Complete public task 0.\n",
        "Complete public task 1.\n",
    }
    wire = output.read_text(encoding="utf-8")
    assert "private-answer-canary" not in wire
    assert "private-test-canary" not in wire
    assert any(path.endswith("solution/answer.txt") for path in result.asset_relative_files)
    assert any(path.endswith("tests/test.py") for path in result.asset_relative_files)


def test_tua_materialization_reads_public_task_files_without_tests(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    repository = root / "tua-bench" / "repository"
    task = repository / "tasks" / "001-public-task"
    _write(task / "instruction.md", "Prepare the public spreadsheet.\n")
    _write(
        task / "task.toml",
        """
[metadata]
difficulty = "hard"
category = "office"

[task]
name = "local/001-public-task"
description = "Prepare a spreadsheet using public inputs."
keywords = ["spreadsheet", "analysis"]
""".strip()
        + "\n",
    )
    _write(task / "environment" / "input.csv", "public,input\n")
    _write(task / "tests" / "test.py", "private-tua-test-canary\n")

    result = materialize_tua_bench_source(
        repository_root=repository,
        target_root=root,
        dataset_revision="b" * 40,
    )

    output = root / result.task_manifest_relative_path
    loaded = RolloutTask.from_value(json.loads(output.read_text(encoding="utf-8")))
    assert result.task_count == 1
    assert loaded.query == "Prepare the public spreadsheet.\n"
    assert loaded.public_context["public_files"] == ["environment/input.csv"]
    assert set(loaded.public_context["tools"]) == {"submit"}
    assert "private-tua-test-canary" not in output.read_text(encoding="utf-8")


def test_appworld_materialization_projects_only_public_instruction(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    appworld = root / "appworld"
    _write(appworld / "data" / "datasets" / "train.txt", "scenario_1\n")
    specs = {
        "canary_string": "private-appworld-canary",
        "datetime": "2026-01-01T00:00:00",
        "db_version": "0.2.0",
        "instruction": "Schedule the public appointment.",
        "supervisor": {"email": "private@example.invalid"},
    }
    _write(
        appworld / "data" / "tasks" / "scenario_1" / "specs.json",
        canonical_json(specs) + "\n",
    )
    _write(
        appworld / "data" / "tasks" / "scenario_1" / "ground_truth" / "answer.json",
        '{"answer":"private-answer-canary"}\n',
    )
    _write(appworld / "repository" / "README.md", "official source\n")

    result = materialize_appworld_source(
        benchmark_root=appworld,
        target_root=root,
        dataset_revision="c" * 40,
    )

    output = root / result.task_manifest_relative_path
    loaded = RolloutTask.from_value(json.loads(output.read_text(encoding="utf-8")))
    assert result.task_count == 1
    assert loaded.query == "Schedule the public appointment."
    assert loaded.available_tools == ("appworld.execute", "submit")
    assert set(loaded.public_context["tools"]) == set(loaded.available_tools)
    wire = output.read_text(encoding="utf-8")
    assert "private-appworld-canary" not in wire
    assert "private-answer-canary" not in wire
    assert "private@example.invalid" not in wire


def test_bfcl_v3_materialization_uses_official_all_collection_without_answers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    repository = root / "bfcl-v3" / "repository"
    data = repository / "berkeley-function-call-leaderboard" / "data"
    categories = (
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
        "multi_turn_composite",
    )
    for category in categories:
        _write(
            data / f"BFCL_v3_{category}.json",
            canonical_json(
                {
                    "function": [
                        {
                            "description": "A public function.",
                            "name": f"public_{category}",
                            "parameters": {"properties": {}, "type": "dict"},
                        }
                    ],
                    "ground_truth": ["private-bfcl-answer-canary"],
                    "id": f"{category}_0",
                    "question": [[{"content": f"Public {category} query.", "role": "user"}]],
                }
            )
            + "\n",
        )
    _write(data / "possible_answer" / "answer.json", "private-possible-answer-canary\n")

    result = materialize_bfcl_v3_source(
        repository_root=repository,
        target_root=root,
        dataset_revision="d" * 40,
    )

    output = root / result.task_manifest_relative_path
    tasks = tuple(
        RolloutTask.from_value(json.loads(line))
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert result.task_count == len(tasks) == len(categories)
    assert {task.task_family for task in tasks} == {
        f"bfcl-v3/{category}" for category in categories
    }
    assert all(len(task.available_tools) == 1 for task in tasks)
    assert all(set(task.public_context["tools"]) == set(task.available_tools) for task in tasks)
    wire = output.read_text(encoding="utf-8")
    assert "private-bfcl-answer-canary" not in wire
    assert "private-possible-answer-canary" not in wire
    assert any(path.endswith("possible_answer/answer.json") for path in result.asset_relative_files)
