from __future__ import annotations

import pytest

from skillev.training.distributed_ttb import (
    DistributedTTBError,
    DistributedTTBOOMKind,
    classify_distributed_ttb_oom,
)


def _oom(*positions: int) -> dict[str, object]:
    return {
        "artifact_count": len(positions),
        "artifact_positions": list(positions),
        "artifact_token_counts": [100 + position for position in positions],
        "rank": 1,
        "status": "oom",
    }


def test_multiple_artifacts_are_partitionable_after_worker_oom() -> None:
    assert classify_distributed_ttb_oom((_oom(0, 2, 4),)) is (
        DistributedTTBOOMKind.PARTITIONABLE_WORKLOAD
    )


def test_any_single_artifact_oom_is_not_recoverable_by_another_worker() -> None:
    assert classify_distributed_ttb_oom((_oom(0, 2), _oom(3))) is (
        DistributedTTBOOMKind.SINGLE_ARTIFACT
    )


def test_oom_classification_rejects_missing_partition_evidence() -> None:
    with pytest.raises(DistributedTTBError):
        classify_distributed_ttb_oom(({"rank": 1, "status": "oom"},))
