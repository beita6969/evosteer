#!/usr/bin/env python3
"""Compose exactly nine typed Protocol 12 receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.evaluation.current_iid.aggregation import aggregate_protocol_v12
from skillev.evaluation.current_iid.catalog import ACTIVE_CURRENT_IID_BENCHMARKS
from skillev.evaluation.current_iid.config import load_execution_contracts
from skillev.evaluation.current_iid.receipt_io import load_current_iid_receipt
from skillev.evaluation.current_iid.targets import load_target_registry_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("configs/evaluation/protocol_v12_conditions.yaml"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("configs/evaluation/qwen35_current_iid_v12_targets.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executions = load_execution_contracts(args.conditions)
    targets = load_target_registry_v2(args.targets, executions=executions)
    receipts = {
        benchmark: load_current_iid_receipt(args.receipt_dir / f"{benchmark.value}.json")
        for benchmark in ACTIVE_CURRENT_IID_BENCHMARKS
    }
    aggregate = aggregate_protocol_v12(receipts=receipts, targets=targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
