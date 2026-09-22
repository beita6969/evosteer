#!/usr/bin/env python3
"""Emit answer-free QA failure counts from private direct-run artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from skillev.evaluation.direct_baseline.parsing import ParseStatus


def _rows(path: Path) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(line)) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--scorer-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generations = {str(row["task_id"]): row for row in _rows(arguments.journal)}
    scorer_rows = _rows(arguments.scorer_rows)
    scorer_ids = tuple(str(row["task_id"]) for row in scorer_rows)
    if len(scorer_ids) != len(set(scorer_ids)):
        raise ValueError("QA scorer rows contain duplicate task IDs")
    if set(generations) != set(scorer_ids):
        raise ValueError("QA generation and scorer task IDs differ")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scorer_rows:
        benchmark = str(row.get("benchmark", "unknown"))
        metrics = cast(dict[str, float], row.get("metrics") or {})
        em = float(metrics.get("em", 0.0))
        f1 = float(metrics.get("f1", em))
        bucket = counts[benchmark]
        bucket["records"] += 1
        bucket["parse_invalid"] += row.get("parse_status") != ParseStatus.EXTRACTED.value
        bucket["exact_match"] += em == 1.0
        bucket["partial_f1"] += em == 0.0 and f1 > 0.0
        bucket["zero_overlap"] += f1 == 0.0
        generation = generations.get(str(row["task_id"]), {})
        bucket["finish_reason_length"] += generation.get("finish_reason") == "length"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({key: dict(value) for key, value in sorted(counts.items())}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
