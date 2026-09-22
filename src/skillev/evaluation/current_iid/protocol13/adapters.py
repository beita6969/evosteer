"""Common adapter primitives for private runner artifacts.

Benchmark-specific adapters live with private data loaders.  This public module only
defines the answer-free boundary and exact manifest join shared by every adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contracts import ExecutionContractV3
from .receipts import Protocol13RunReceipt, RunProvenanceV3


class Protocol13ReceiptAdapter(Protocol):
    def build(
        self,
        *,
        artifact_dir: Path,
        execution: ExecutionContractV3,
        provenance: RunProvenanceV3,
    ) -> Protocol13RunReceipt: ...


@dataclass(frozen=True, slots=True)
class PrivatePanelOutcome:
    task_id: str
    outcome: str
    metrics: dict[str, str]
    infrastructure_error: str | None

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.outcome.strip():
            raise ValueError("private outcome identity is incomplete")
        if any(not name.strip() for name in self.metrics):
            raise ValueError("private metric names must be non-empty")


def require_exact_private_panel(
    *,
    expected_task_ids: tuple[str, ...],
    outcomes: tuple[PrivatePanelOutcome, ...],
) -> tuple[PrivatePanelOutcome, ...]:
    if len(expected_task_ids) != len(set(expected_task_ids)):
        raise ValueError("manifest task IDs are not unique")
    by_id: dict[str, PrivatePanelOutcome] = {}
    for outcome in outcomes:
        if outcome.task_id in by_id:
            raise ValueError(f"duplicate task outcome: {outcome.task_id}")
        by_id[outcome.task_id] = outcome
    expected = set(expected_task_ids)
    missing = [task_id for task_id in expected_task_ids if task_id not in by_id]
    extra = sorted(set(by_id).difference(expected))
    if missing or extra:
        raise ValueError(f"private panel differs: missing={missing}, extra={extra}")
    return tuple(by_id[task_id] for task_id in expected_task_ids)


__all__ = [
    "PrivatePanelOutcome",
    "Protocol13ReceiptAdapter",
    "require_exact_private_panel",
]
