"""Fixed child entrypoint for one formal private benchmark attempt.

This module intentionally has no builder argument or environment-selection
switch.  Formal inputs always use :func:`build_private_benchmark_attempt`; the
public completion-smoke worker remains a distinct executable.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from skillev.experiments import FormalAttemptLaunch, FormalRunLedger
from skillev.runtime.attempt_worker import execute_attempt_request

from .benchmark_attempt_builders import build_private_benchmark_attempt


def _configure_formal_determinism() -> None:
    """Apply the backend flags frozen into every formal build identity."""

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


async def run_private_benchmark_attempt_worker(launch: FormalAttemptLaunch) -> None:
    """Run the sole formal private catalog builder after consuming its claim."""

    if not isinstance(launch, FormalAttemptLaunch):
        raise TypeError("formal private worker requires FormalAttemptLaunch")
    # PyTorch backend flags are process-local.  The supervisor's settings do
    # not cross this child-process boundary, so establish the frozen state
    # before build identity is measured by the private attempt builder.
    _configure_formal_determinism()
    ledger = FormalRunLedger.open(directory=launch.ledger_directory)
    claim = ledger.consume_private_launch(launch)
    if claim.run_group_id != launch.run_group_id:
        raise ValueError("formal worker launch differs from consumed claim")
    await execute_attempt_request(
        launch.request,
        build=build_private_benchmark_attempt,
        formal_run_group_id=claim.run_group_id,
    )


def main(request_path: str) -> None:
    value = json.loads(Path(request_path).read_text(encoding="utf-8"))
    launch = FormalAttemptLaunch.from_value(value)
    asyncio.run(run_private_benchmark_attempt_worker(launch))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m skillev_private.experiments.attempt_worker REQUEST.json")
    main(sys.argv[1])


__all__ = ["main", "run_private_benchmark_attempt_worker"]
