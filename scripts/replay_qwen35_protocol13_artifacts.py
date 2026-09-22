#!/usr/bin/env python3
"""Replay Protocol 13 artifacts offline with exact task-order conservation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.execution_receipts import (
    load_execution_receipt,
)


@dataclass(frozen=True, slots=True)
class ReplayPanel:
    benchmark: Protocol13Benchmark
    expected_task_ids: tuple[str, ...]
    records_by_task_id: dict[str, dict[str, object]]

    def ordered(self) -> tuple[dict[str, object], ...]:
        observed = set(self.records_by_task_id)
        expected = set(self.expected_task_ids)
        missing = [item for item in self.expected_task_ids if item not in observed]
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(f"replay panel differs: missing={missing}, extra={extra}")
        return tuple(self.records_by_task_id[item] for item in self.expected_task_ids)


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _summary(
    benchmark: Protocol13Benchmark, rows: tuple[dict[str, object], ...]
) -> dict[str, int | str]:
    counts: Counter[str] = Counter(planned=len(rows))
    for row in rows:
        if benchmark in {
            Protocol13Benchmark.HOTPOT_QA,
            Protocol13Benchmark.TRIVIA_QA,
            Protocol13Benchmark.AIME_2026,
            Protocol13Benchmark.HUMAN_EVAL,
        }:
            counts["parse_invalid"] += row.get("parse_status") != "extracted"
            counts["finish_length"] += row.get("finish_reason") == "length"
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                counts["exact_success"] += all(value == 1 for value in metrics.values())
                counts["partial"] += any(
                    isinstance(value, int | float) and 0 < value < 1 for value in metrics.values()
                )
        elif benchmark in {Protocol13Benchmark.WEB_SHOP, Protocol13Benchmark.ALF_WORLD}:
            counts["success"] += row.get("success") is True
            counts["candidate_invalid"] += row.get("termination_reason") == "candidate-invalid"
            counts["horizon"] += row.get("terminated_by_horizon") is True
            counts["official_terminal"] += row.get("terminal_reached") is True
            counts["steps"] += int(cast(int, row.get("steps", 0)))
            for step in cast(list[dict[str, object]], row.get("trace") or []):
                counts["unlisted_action"] += step.get("action_listed_before") is False
        elif benchmark is Protocol13Benchmark.MBPP_PLUS:
            outcome = str(row.get("outcome"))
            counts[outcome] += 1
            counts["base_failure"] += row.get("base_passed") is False
            counts["plus_only_failure"] += (
                row.get("base_passed") is True and row.get("plus_passed") is False
            )
        else:
            metrics = row.get("metrics")
            counts["scored"] += isinstance(metrics, dict) and "overall_score" in metrics
    return {"benchmark": benchmark.value, **dict(sorted(counts.items()))}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=[item.value for item in Protocol13Benchmark], required=True
    )
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--grader-profile")
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    benchmark = Protocol13Benchmark(arguments.benchmark)
    runtime = load_execution_receipt(arguments.execution_receipt)
    if benchmark is Protocol13Benchmark.MBPP_PLUS:
        rows = _jsonl(arguments.artifact_dir / "verdicts.jsonl")
    elif benchmark is Protocol13Benchmark.HEALTHBENCH:
        if arguments.grader_profile is None:
            raise ValueError("HealthBench replay requires --grader-profile")
        rows = _jsonl(arguments.artifact_dir / "grades" / arguments.grader_profile / "scores.jsonl")
    else:
        rows = _jsonl(arguments.artifact_dir / "per-task-results.jsonl")
        rows = tuple(row for row in rows if row.get("benchmark") == benchmark.value)
    id_key = "prompt_id" if benchmark is Protocol13Benchmark.HEALTHBENCH else "task_id"
    by_id = {str(row[id_key]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("replay records contain duplicate task IDs")
    panel = ReplayPanel(benchmark, runtime.planned_task_ids, by_id)
    ordered = panel.ordered()
    arguments.private_output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.private_output.open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    public = _summary(benchmark, ordered)
    arguments.public_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.public_output.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
