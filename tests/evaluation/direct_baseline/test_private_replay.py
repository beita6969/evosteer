import json
from pathlib import Path

import pytest
from skillev_private.direct_reference.replay import (
    collapse_generation_attempts,
    load_generation_records,
    validate_replay_contract,
)

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol


def _row(task_id: str = "task-1") -> dict[str, object]:
    return {
        "task_id": task_id,
        "benchmark": "hotpotqa",
        "raw_text": "Final answer: Paris",
        "response_model": "qwen35-direct-base",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "prompt_profile_id": "hotpotqa-full-context@1",
        "parser_profile_id": "short-answer-final@2",
        "decoding_profile_id": "qwen35-nonthinking-general@1",
        "population_id": "skillflow-released-iid-v3",
        "run_seed": 42,
        "request_attempt": 1,
        "infrastructure_error": None,
    }


def _complete(directory: Path, count: int = 1) -> None:
    (directory / "attempt-complete.json").write_text(
        json.dumps(
            {
                "format": "skillev-direct-attempt@1",
                "status": "complete",
                "final_record_count": count,
            }
        ),
        encoding="utf-8",
    )


def test_replay_loads_raw_without_endpoint_and_validates_contract(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    _complete(tmp_path)
    journal.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    record = load_generation_records(journal)[0]
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    validate_replay_contract(
        record,
        protocol.benchmark(DirectBenchmark.HOTPOT_QA),
        served_model_name=protocol.model.served_model_name,
    )


def test_replay_rejects_duplicate_generation_and_contract_drift(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    _complete(tmp_path, 2)
    journal.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        collapse_generation_attempts(load_generation_records(journal))
    drift = _row("task-2")
    drift["population_id"] = "another-panel"
    _complete(tmp_path)
    journal.write_text(json.dumps(drift) + "\n", encoding="utf-8")
    record = load_generation_records(journal)[0]
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    with pytest.raises(ValueError):
        validate_replay_contract(
            record,
            protocol.benchmark(DirectBenchmark.HOTPOT_QA),
            served_model_name=protocol.model.served_model_name,
        )


def test_replay_allows_only_infrastructure_then_candidate_retry(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    first = _row()
    first.update(
        raw_text=None,
        response_model=None,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        infrastructure_error="DirectGenerationError",
    )
    second = _row()
    second["request_attempt"] = 2
    _complete(tmp_path, 2)
    journal.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    assert collapse_generation_attempts(load_generation_records(journal))[0].raw_text is not None

    first = _row()
    second = _row()
    second["request_attempt"] = 2
    journal.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate response"):
        collapse_generation_attempts(load_generation_records(journal))


def test_replay_rejects_incomplete_or_mistyped_journal(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    row = _row()
    row["raw_text"] = {"not": "text"}
    _complete(tmp_path)
    journal.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text or null"):
        load_generation_records(journal)

    (tmp_path / "attempt-complete.json").unlink()
    with pytest.raises(ValueError, match="complete formal attempt"):
        load_generation_records(journal)
