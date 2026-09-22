from __future__ import annotations

from pathlib import Path

import pytest
from skillev_private.benchmarks import create_private_dataset_snapshot

from skillev.experiments import DatasetSnapshotIdentity


def test_private_snapshot_is_content_addressed_without_file_contents(tmp_path: Path) -> None:
    root = tmp_path / "private-data"
    root.mkdir()
    private_canary = "PRIVATE-DATASET-ANSWER-CANARY"
    (root / "part-a.jsonl").write_text(private_canary, encoding="utf-8")
    (root / "part-b.jsonl").write_text("public metadata", encoding="utf-8")

    first = create_private_dataset_snapshot(
        name="fixture",
        version="source-revision@1",
        root=root,
        relative_files=("part-a.jsonl", "part-b.jsonl"),
    )
    second = create_private_dataset_snapshot(
        name="fixture",
        version="source-revision@1",
        root=root,
        relative_files=("part-a.jsonl", "part-b.jsonl"),
    )

    assert isinstance(first, DatasetSnapshotIdentity)
    assert first == second
    assert private_canary not in str(first.to_value())

    (root / "part-a.jsonl").write_text(private_canary + " changed", encoding="utf-8")
    changed = create_private_dataset_snapshot(
        name="fixture",
        version="source-revision@1",
        root=root,
        relative_files=("part-a.jsonl", "part-b.jsonl"),
    )
    assert changed.snapshot_hash != first.snapshot_hash


def test_private_snapshot_requires_exact_sorted_file_set(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("a", encoding="utf-8")
    (tmp_path / "b").write_text("b", encoding="utf-8")

    with pytest.raises(ValueError):
        create_private_dataset_snapshot(
            name="fixture",
            version="v1",
            root=tmp_path,
            relative_files=("b", "a"),
        )
    with pytest.raises(FileNotFoundError):
        create_private_dataset_snapshot(
            name="fixture",
            version="v1",
            root=tmp_path,
            relative_files=("missing",),
        )
    with pytest.raises(ValueError):
        create_private_dataset_snapshot(
            name="fixture",
            version="v1",
            root=tmp_path,
            relative_files=("../a",),
        )
