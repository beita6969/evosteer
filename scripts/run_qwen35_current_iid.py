#!/usr/bin/env python3
"""Compose completed benchmark receipts without owning benchmark execution.

Each benchmark family retains its official/native runner.  This command reads
their answer-free receipts, recomputes coverage and parity, and writes the
single public aggregate consumed by the renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.evaluation.current_iid.aggregation import aggregate_current_iid
from skillev.evaluation.current_iid.receipts import CurrentIIDRunReceipt
from skillev.evaluation.current_iid.targets import load_target_registry
from skillev.experiments.protocol_v11 import BenchmarkV11


def _receipt(path: Path) -> CurrentIIDRunReceipt:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("receipt must be a JSON object")
    return CurrentIIDRunReceipt(
        benchmark=BenchmarkV11(str(raw["benchmark"])),
        condition_id=str(raw["condition_id"]),
        attempt_id=str(raw["attempt_id"]),
        planned_count=int(raw["planned_count"]),
        final_record_count=int(raw["final_record_count"]),
        definitive_verdict_count=int(raw["definitive_verdict_count"]),
        candidate_failure_count=int(raw["candidate_failure_count"]),
        generation_infrastructure_failures=int(raw["generation_infrastructure_failures"]),
        scorer_infrastructure_failures=int(raw["scorer_infrastructure_failures"]),
        environment_infrastructure_failures=int(raw["environment_infrastructure_failures"]),
        metrics={str(k): float(v) for k, v in dict(raw["metrics"]).items()},
        population_id=str(raw["population_id"]),
        prompt_profile=str(raw["prompt_profile"]),
        decoding_profile=str(raw["decoding_profile"]),
        actor_route=str(raw["actor_route"]),
        context_length=int(raw["context_length"]),
        adapter_active=bool(raw["adapter_active"]),
        tool_surface=tuple(str(item) for item in raw.get("tool_surface", [])),
        grader_profile=(None if raw.get("grader_profile") is None else str(raw["grader_profile"])),
        auxiliary_models=tuple(str(item) for item in raw.get("auxiliary_models", [])),
        scaffold_id=(None if raw.get("scaffold_id") is None else str(raw["scaffold_id"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument(
        "--targets", type=Path, default=Path("configs/evaluation/qwen35_current_iid_targets.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    receipts = {
        benchmark: _receipt(args.receipt_dir / f"{benchmark.value}.json")
        for benchmark in BenchmarkV11
    }
    aggregate = aggregate_current_iid(
        receipts, load_target_registry(args.targets), source_commit=args.source_commit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
