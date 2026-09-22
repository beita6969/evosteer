"""Coverage gate for ten-domain Protocol 11 final evaluation receipts."""

from __future__ import annotations

from skillev.evaluation.current_iid.receipts import CurrentIIDRunReceipt
from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11


def require_protocol_v11_final_coverage(
    receipts: tuple[CurrentIIDRunReceipt, ...],
) -> None:
    if tuple(item.benchmark for item in receipts) != ACTIVE_BENCHMARKS_V11:
        raise ValueError("Protocol 11 final receipts differ from the current ten catalog")
    if any(item.infrastructure_failure_count for item in receipts):
        raise RuntimeError("Protocol 11 final evaluation contains infrastructure failures")


__all__ = ["require_protocol_v11_final_coverage"]
