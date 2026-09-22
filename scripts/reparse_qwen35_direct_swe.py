#!/usr/bin/env python3
"""Replay a frozen SWE generation journal through a versioned patch parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from skillev.evaluation.direct_baseline.parsing import PARSER_REGISTRY, ParseStatus


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--prior-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parser-profile", default="unified-diff@2")
    return parser.parse_args()


def _read_rows(path: Path, *, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{label} rows must be objects")
            rows.append(cast(dict[str, object], value))
    if len(rows) != 128:
        raise ValueError(f"{label} must contain exactly 128 rows")
    task_ids = tuple(str(row.get("task_id") or "") for row in rows)
    if not all(task_ids) or len(set(task_ids)) != 128:
        raise ValueError(f"{label} task IDs must be non-empty and unique")
    return rows


def reparse(
    *,
    generations: Path,
    prior_predictions: Path,
    output: Path,
    parser_profile: str,
) -> dict[str, int | str]:
    try:
        patch_parser = PARSER_REGISTRY[parser_profile]
    except KeyError as exc:
        raise ValueError(f"unknown parser profile {parser_profile}") from exc
    generation_rows = _read_rows(generations, label="generation journal")
    prediction_rows = _read_rows(prior_predictions, label="prior predictions")
    generation_by_task = {str(row["task_id"]): row for row in generation_rows}
    if set(generation_by_task) != {str(row["task_id"]) for row in prediction_rows}:
        raise ValueError("generation and prediction panels differ")
    if output.exists():
        raise FileExistsError("replay output already exists")

    replayed: list[dict[str, object]] = []
    parse_counts = {status.value: 0 for status in ParseStatus}
    changed = 0
    for prior in prediction_rows:
        task_id = str(prior["task_id"])
        generation = generation_by_task[task_id]
        generation_error = generation.get("infrastructure_error")
        if generation_error is not None:
            raise ValueError("generation journal contains infrastructure failures")
        raw_text = generation.get("raw_text")
        if type(raw_text) is not str:
            raise ValueError("successful generation row is missing raw text")
        parsed = patch_parser(raw_text)
        parse_counts[parsed.status.value] += 1
        old_patch = prior.get("model_patch")
        old_value = old_patch if type(old_patch) is str else None
        if old_value != parsed.value:
            changed += 1
        row = dict(prior)
        row.update(
            {
                "model_patch": parsed.value,
                "parse_reason": parsed.reason.value,
                "parse_status": parsed.status.value,
                "parser_profile_id": parser_profile,
            }
        )
        replayed.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in replayed:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "planned_count": 128,
        "changed_prediction_count": changed,
        "submission_count": parse_counts[ParseStatus.EXTRACTED.value],
        "ambiguous_count": parse_counts[ParseStatus.AMBIGUOUS.value],
        "invalid_count": parse_counts[ParseStatus.EMPTY.value],
        "parser_profile": parser_profile,
    }


def main() -> None:
    arguments = _arguments()
    print(
        json.dumps(
            reparse(
                generations=arguments.generations,
                prior_predictions=arguments.prior_predictions,
                output=arguments.output,
                parser_profile=arguments.parser_profile,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
