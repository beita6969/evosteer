from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from skillev_private.benchmarks import (
    fixed_result_blind_subsample,
    load_private_static_jsonl,
)

from skillev.contracts import canonical_json
from skillev.experiments import fixed_subsample_rank


def _row(task_id: str, answer: str) -> dict[str, object]:
    return {
        "accepted_answers": [answer],
        "benchmark_id": "synthetic-private",
        "dataset_revision": "fixture@1",
        "public_context": {"answer_format": "short-text"},
        "query": f"Public synthetic query {task_id}",
        "scoring_rule": "exact-text",
        "split": "dev",
        "task_family": "synthetic-private/private-loader",
        "task_id": task_id,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def test_private_jsonl_splits_answers_and_exposes_only_snapshot_hash(tmp_path: Path) -> None:
    source = tmp_path / "private.jsonl"
    canary = "PRIVATE-LOADER-CANARY"
    _write(source, [_row("task-1", canary), _row("task-2", "another-private-answer")])

    snapshot = load_private_static_jsonl(
        source,
        snapshot_name="synthetic-private",
        snapshot_version="fixture@1",
    )

    assert snapshot.identity.snapshot_hash.startswith("sha256:")
    assert canary not in canonical_json(snapshot.identity.to_value())
    assert canary not in canonical_json(
        [item.to_rollout_task().to_value() for item in snapshot.public_items()]
    )
    assert snapshot.cases[0].target.accepted_answers == (canary,)


def test_result_blind_subsample_is_order_independent_and_uses_fixed_seed(tmp_path: Path) -> None:
    rows = [_row(f"task-{index}", f"private-{index}") for index in range(10)]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write(first_path, rows)
    _write(second_path, list(reversed(rows)))
    first = load_private_static_jsonl(first_path, snapshot_name="suite", snapshot_version="v1")
    second = load_private_static_jsonl(second_path, snapshot_name="suite", snapshot_version="v1")

    selected_first = fixed_result_blind_subsample(first, 4)
    selected_second = fixed_result_blind_subsample(second, 4)

    assert tuple(case.public.task_id for case in selected_first.cases) == tuple(
        case.public.task_id for case in selected_second.cases
    )
    expected = tuple(
        sorted(
            (case.public.task_id for case in first.cases),
            key=lambda task_id: (
                fixed_subsample_rank("synthetic-private", task_id),
                task_id,
            ),
        )[:4]
    )
    assert tuple(case.public.task_id for case in selected_first.cases) == expected
    assert "seed" not in inspect.signature(fixed_result_blind_subsample).parameters


def test_private_jsonl_rejects_duplicate_identity_and_noncanonical_record(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    _write(duplicate, [_row("task-1", "a"), _row("task-1", "b")])
    with pytest.raises(ValueError):
        load_private_static_jsonl(duplicate, snapshot_name="suite", snapshot_version="v1")

    noncanonical = tmp_path / "noncanonical.jsonl"
    noncanonical.write_text('{"task_id": "task-1"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_private_static_jsonl(noncanonical, snapshot_name="suite", snapshot_version="v1")
