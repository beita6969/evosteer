from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest
from skillev_private.benchmarks import (
    EvalPlusEvaluationInfrastructureError,
    load_evalplus_session_factory,
)

from skillev.contracts import canonical_json
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask


def _task(benchmark: Benchmark, task_id: str) -> RolloutTask:
    return RolloutTask(
        task_id=task_id,
        environment_id=f"{benchmark.value}:fixture:test",
        task_family=f"{benchmark.value}/python/code-generation",
        context_id=f"{benchmark.value}:test:one",
        query="def add_one(value):\n",
        available_tools=(),
        public_context={
            "benchmark_id": benchmark.value,
            "dataset_revision": "fixture",
            "language": "python",
            "source_id": "one",
            "split": "test",
        },
    )


def _humaneval_row(
    *,
    prompt: str = "def add_one(value):\n",
    canonical_solution: str = "    return value + 1\n",
    entry_point: str = "add_one",
    test: str = (
        "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(10) == 11\n"
    ),
) -> dict[str, object]:
    return {
        "canonical_solution": canonical_solution,
        "entry_point": entry_point,
        "prompt": prompt,
        "task_id": "official/one",
        "test": test,
    }


def _manifest(path: Path, *, task_id: str, row: dict[str, object]) -> str:
    wire = canonical_json({"official_row": row, "task_id": task_id}) + "\n"
    path.write_text(wire, encoding="utf-8")
    return f"sha256:{hashlib.sha256(wire.encode()).hexdigest()}"


def test_loader_binds_frozen_manifest_to_exact_public_tasks(tmp_path: Path) -> None:
    benchmark = Benchmark.HUMANEVAL_PLUS
    task = _task(benchmark, "humaneval-plus/task-one")
    private_manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(private_manifest, task_id=task.task_id, row=_humaneval_row())

    factory = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=private_manifest,
        expected_sha256=manifest_hash,
    )

    bundle = factory.create(task)
    assert bundle.environment.environment_id == task.environment_id
    assert factory.worker.benchmark_id == benchmark.value

    private_manifest.write_text(private_manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_evalplus_session_factory(
            benchmark=benchmark,
            tasks=(task,),
            private_manifest_path=private_manifest,
            expected_sha256=manifest_hash,
        )


def test_evalplus_no_submission_is_zero_without_code_execution(tmp_path: Path) -> None:
    benchmark = Benchmark.HUMANEVAL_PLUS
    task = _task(benchmark, "humaneval-plus/task-one")
    manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(manifest, task_id=task.task_id, row=_humaneval_row())
    worker = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=manifest,
        expected_sha256=manifest_hash,
    ).worker

    result = worker.no_submission_result(task_id=task.task_id)

    assert result.reward_value == 0.0
    assert result.success is False
    assert result.public_metrics["status"] == "no-submission"


def test_humaneval_plus_requires_both_base_and_plus_tests(tmp_path: Path) -> None:
    benchmark = Benchmark.HUMANEVAL_PLUS
    task = _task(benchmark, "humaneval-plus/task-one")
    manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(manifest, task_id=task.task_id, row=_humaneval_row())
    factory = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=manifest,
        expected_sha256=manifest_hash,
    )

    passed = asyncio.run(
        factory.worker.evaluate(task_id=task.task_id, submission="    return value + 1\n")
    )
    failed_plus = asyncio.run(
        factory.worker.evaluate(
            task_id=task.task_id,
            submission="    return value + 1 if value == 1 else 0\n",
        )
    )

    assert passed.success
    assert passed.reward_value == 1.0
    assert passed.public_metrics == {
        "base_passed": True,
        "plus_passed": True,
        "status": "passed",
    }
    assert not failed_plus.success
    assert failed_plus.public_metrics["base_passed"] is False
    assert failed_plus.public_metrics["plus_passed"] is False


def test_mbpp_plus_requires_original_and_augmented_assertions(tmp_path: Path) -> None:
    benchmark = Benchmark.MBPP_PLUS
    task = _task(benchmark, "mbpp-plus/task-one")
    row = {
        "code": "def add_one(value):\n    return value + 1\n",
        "prompt": "Write a function add_one that increments an integer.",
        "source_file": "fixture.jsonl",
        "task_id": 1,
        "test": ["assert add_one(10) == 11"],
        "test_imports": [],
        "test_list": ["assert add_one(1) == 2"],
    }
    manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(manifest, task_id=task.task_id, row=row)
    factory = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=manifest,
        expected_sha256=manifest_hash,
    )

    result = asyncio.run(
        factory.worker.evaluate(
            task_id=task.task_id,
            submission="def add_one(value):\n    return value + 1\n",
        )
    )

    assert result.success
    assert result.native_metric_name == "pass@1"


def test_mbpp_plus_augmented_tests_can_import_pinned_dependencies(tmp_path: Path) -> None:
    benchmark = Benchmark.MBPP_PLUS
    task = _task(benchmark, "mbpp-plus/task-one")
    row = {
        "code": "def add_one(value):\n    return value + 1\n",
        "prompt": "Write a function add_one that increments a number.",
        "source_file": "fixture.jsonl",
        "task_id": 1,
        "test": "import numpy as np\nassert np.array([add_one(1)]).tolist() == [2]",
        "test_imports": [],
        "test_list": ["assert add_one(1) == 2"],
    }
    manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(manifest, task_id=task.task_id, row=row)
    worker = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=manifest,
        expected_sha256=manifest_hash,
    ).worker

    result = asyncio.run(
        worker.evaluate(
            task_id=task.task_id,
            submission="def add_one(value):\n    return value + 1\n",
        )
    )

    assert result.success


def test_candidate_really_executes_in_an_isolated_child(tmp_path: Path) -> None:
    benchmark = Benchmark.HUMANEVAL_PLUS
    task = _task(benchmark, "humaneval-plus/task-one")
    parent_pid = os.getpid()
    row = _humaneval_row(
        prompt="def is_child(parent_pid):\n",
        canonical_solution="    import os\n    return os.getpid() != parent_pid\n",
        entry_point="is_child",
        test=(f"def check(candidate):\n    assert candidate({parent_pid})\n"),
    )
    manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(manifest, task_id=task.task_id, row=row)
    factory = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=manifest,
        expected_sha256=manifest_hash,
    )

    result = asyncio.run(
        factory.worker.evaluate(
            task_id=task.task_id,
            submission="    import os\n    return os.getpid() != parent_pid\n",
        )
    )

    assert result.success


def test_broken_private_reference_is_infrastructure_failure_without_canary(
    tmp_path: Path,
) -> None:
    benchmark = Benchmark.HUMANEVAL_PLUS
    task = _task(benchmark, "humaneval-plus/task-one")
    canary = "PRIVATE-REFERENCE-CANARY"
    manifest = tmp_path / "cases.jsonl"
    manifest_hash = _manifest(
        manifest,
        task_id=task.task_id,
        row=_humaneval_row(canonical_solution=f"    this is invalid syntax  # {canary}\n"),
    )
    factory = load_evalplus_session_factory(
        benchmark=benchmark,
        tasks=(task,),
        private_manifest_path=manifest,
        expected_sha256=manifest_hash,
    )

    with pytest.raises(EvalPlusEvaluationInfrastructureError) as caught:
        asyncio.run(
            factory.worker.evaluate(
                task_id=task.task_id,
                submission="    return value + 1\n",
            )
        )

    assert canary not in str(caught.value)
    assert canary not in json.dumps(task.to_value())
