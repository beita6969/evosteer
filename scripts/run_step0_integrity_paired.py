#!/usr/bin/env python3
"""Run clean base/trained single-arm or one-axis paired evaluation, never a selector."""

from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path

from skillev.evaluation.integrity_pipeline import run_paired
from skillev.evaluation.integrity_resume import EvaluationRunMode, require_execution_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-factory",
        default="skillev_private.evaluation.integrity_runtime:create_runtime",
        help="Trusted private module:function returning runtime, panel, selected arms",
    )
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--scoring-concurrency", type=int, default=8)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--reuse-completed-from", type=Path, help="Withdrawn: always rejected")
    parser.add_argument(
        "--run-mode",
        choices=[EvaluationRunMode.FORMAL_FRESH.value, EvaluationRunMode.SAME_RUN_RESUME.value],
        default=EvaluationRunMode.FORMAL_FRESH.value,
    )
    parser.add_argument("--minimum-records-per-hour", type=float, default=0.0)
    args = parser.parse_args()
    run_mode = EvaluationRunMode(args.run_mode)
    require_execution_mode(
        run_mode,
        directory=args.private_output,
        importing_other_run=args.reuse_completed_from is not None,
    )
    module, name = args.runtime_factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    runtime, panel, arms = factory(args.runtime_config, private_output=args.private_output)
    asyncio.run(
        run_paired(
            runtime,
            panel,
            arms,
            run_id=args.run_id,
            directory=args.private_output,
            concurrency=args.concurrency,
            scoring_concurrency=args.scoring_concurrency,
            canary=args.canary,
            reuse_completed_from=args.reuse_completed_from,
            run_mode=run_mode,
            minimum_records_per_hour=args.minimum_records_per_hour,
        )
    )


if __name__ == "__main__":
    main()
