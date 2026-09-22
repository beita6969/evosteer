#!/usr/bin/env python3
"""Compose public Protocol 13 results from eight answer-free receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.evaluation.current_iid.protocol13.aggregation import aggregate_protocol_v13
from skillev.evaluation.current_iid.protocol13.config import (
    load_execution_contracts_v3,
    load_receipts_v3,
    load_targets_v3,
)
from skillev.evaluation.current_iid.protocol13.rendering import (
    render_protocol_v13_markdown,
)
from skillev.experiments.protocol_v13 import load_protocol_v13


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    protocol = load_protocol_v13(args.protocol, args.sources)
    executions = load_execution_contracts_v3(args.conditions, protocol=protocol)
    targets = load_targets_v3(args.targets, executions=executions, protocol=protocol)
    receipts = load_receipts_v3(
        args.receipt_dir,
        executions=executions,
        protocol=protocol,
    )
    aggregate = aggregate_protocol_v13(receipts, targets)
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_protocol_v13_markdown(aggregate),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
