from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from skillev.runtime.formal_storage import (
    FormalStorageBinding,
    FormalStorageBudget,
    validate_formal_storage,
)


def _budget() -> FormalStorageBudget:
    return FormalStorageBudget(
        estimated_attempt_peak_bytes=400,
        rollback_checkpoint_bytes=200,
        atomic_publication_bytes=200,
        minimum_free_inodes=1,
    )


def test_storage_budget_adds_rollback_publication_and_25_percent_margin() -> None:
    budget = _budget()

    assert budget.minimum_free_bytes == 1_000
    assert FormalStorageBudget.from_value(budget.to_value()) == budget


def test_storage_preflight_proves_capacity_fsync_and_atomic_rename(tmp_path: Path) -> None:
    binding = FormalStorageBinding(
        output_root=tmp_path,
        temporary_root=tmp_path,
        budget=_budget(),
    )

    observation = validate_formal_storage(binding)

    assert observation.available_bytes >= binding.budget.minimum_free_bytes
    assert not tuple(tmp_path.glob(".skillev-storage-preflight-*"))


def test_storage_preflight_fails_closed_below_budget(tmp_path: Path) -> None:
    available = validate_formal_storage(
        FormalStorageBinding(tmp_path, tmp_path, _budget())
    ).available_bytes
    binding = FormalStorageBinding(
        output_root=tmp_path,
        temporary_root=tmp_path,
        budget=replace(_budget(), estimated_attempt_peak_bytes=available + 1),
    )

    with pytest.raises(RuntimeError):
        validate_formal_storage(binding)
