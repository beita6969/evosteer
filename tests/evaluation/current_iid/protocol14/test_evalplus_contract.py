import asyncio
import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_qwen35_protocol14_evalplus as evalplus_runner
from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.code_eval import (
    CodeExtractionStatus,
    CodeOutcomeKind,
    CodeTerminalOutcome,
    evalplus_composite_percent,
    load_evalplus_protocol_profile,
    project_evalplus_outcomes,
)
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def _success() -> CodeTerminalOutcome:
    return CodeTerminalOutcome(
        CodeOutcomeKind.SCORED_SUCCESS,
        CodeExtractionStatus.VALID,
        True,
        True,
    )


def test_mbpp_plus_contract_is_nonthinking_greedy_v020_projection() -> None:
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    execution = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )[Protocol14Benchmark.MBPP_PLUS]
    assert execution.thinking_mode.value == "disabled"
    assert execution.decoding.sampling_mode == "greedy"
    assert execution.decoding.temperature == 0.0
    assert execution.dataset_revision == "evalplus-mbppplus-v0.2.0"
    assert execution.code_test_suite == "mbpp-base-plus-v0.2.0-378"
    assert "projected-from-full378" in execution.selection_rule
    profile = load_evalplus_protocol_profile(ROOT / "configs/evaluation/protocol_v14_evalplus.yaml")
    assert profile.mbpp_plus_task_count == 378
    assert profile.humaneval_plus_task_count == 164
    assert profile.candidate_count_per_task == 1
    assert profile.maximum_memory_bytes == 4 * 1024**3
    humaneval = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )[Protocol14Benchmark.HUMAN_EVAL]
    assert humaneval.code_test_suite == "humaneval-original-v1"
    assert humaneval.scorer_profile == "humaneval-original-pass1@3"
    assert "plus" not in humaneval.code_test_suite


def test_evalplus_projection_uses_same_full_outcomes_in_panel_order() -> None:
    outcomes = {task_id: _success() for task_id in ("a", "b", "c", "d")}
    projection = project_evalplus_outcomes(
        full_task_ids=("a", "b", "c", "d"),
        panel_task_ids=("d", "b"),
        outcomes=outcomes,
        expected_full_count=4,
        expected_panel_count=2,
    )
    assert projection == (outcomes["d"], outcomes["b"])


def test_evalplus_aggregate_is_cross_check_not_mbpp_target() -> None:
    assert evalplus_composite_percent(80.0, 60.0) == 70.0
    with pytest.raises(ValueError):
        evalplus_composite_percent(101.0, 60.0)


def test_evalplus_scoring_resume_reuses_complete_generation_population(
    tmp_path: Path,
) -> None:
    task_ids = ("task-a", "task-b")
    marker = {"attempt_id": "reference-attempt", "lane": "mbpp-plus"}
    (tmp_path / "attempt.json").write_text(json.dumps(marker), encoding="utf-8")
    with (tmp_path / "generations.jsonl").open("w", encoding="utf-8") as stream:
        for task_id in task_ids:
            stream.write(json.dumps({"task_id": task_id}) + "\n")
    with (tmp_path / "official-samples.jsonl").open("w", encoding="utf-8") as stream:
        for task_id in task_ids:
            stream.write(json.dumps({"task_id": task_id, "solution": "pass"}) + "\n")

    rows, carrier, generation_seconds = evalplus_runner._resume_generation_state(
        tmp_path, marker=marker, full_task_ids=task_ids
    )

    assert tuple(row["task_id"] for row in rows) == task_ids
    assert carrier == tmp_path / "official-samples.jsonl"
    assert generation_seconds >= 0


def _assert_mbpp_source_rejected(source: Path | None) -> None:
    arguments = Namespace(
        concurrency=1,
        lane="mbpp-plus",
        manifest=Path("panel.json"),
        mbpp_source=source,
    )
    with pytest.raises(ValueError, match="absolute official v0.2.0 source asset"):
        asyncio.run(evalplus_runner.run(arguments))


def test_mbpp_runtime_requires_source_asset() -> None:
    _assert_mbpp_source_rejected(None)


def test_mbpp_runtime_rejects_relative_source_asset() -> None:
    _assert_mbpp_source_rejected(Path("relative.jsonl.gz"))


def _assert_humaneval_source_rejected(source: Path | None) -> None:
    arguments = Namespace(
        concurrency=1,
        lane="humaneval-plus-crosscheck",
        attempt_role="reference",
        mbpp_summary=Path("mbpp-summary.json"),
        humaneval_source=source,
    )
    with pytest.raises(ValueError, match="absolute official v0.1.9 source asset"):
        asyncio.run(evalplus_runner.run(arguments))


def test_humaneval_plus_runtime_requires_source_asset() -> None:
    _assert_humaneval_source_rejected(None)


def test_humaneval_plus_runtime_rejects_relative_source_asset() -> None:
    _assert_humaneval_source_rejected(Path("relative.jsonl"))
