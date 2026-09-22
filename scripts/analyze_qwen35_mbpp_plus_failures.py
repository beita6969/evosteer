#!/usr/bin/env python3
"""Classify MBPP+ failures without modifying candidates or evaluator verdicts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import cast

from skillev_private.benchmarks.evalplus_adapter import EvalPlusCandidateVerdict


class MBPPFailureKind(StrEnum):
    EMPTY = "empty"
    MULTIPLE_CODE_BLOCKS = "multiple-code-blocks"
    PYTHON_SYNTAX = "python-syntax"
    BASE_TIMEOUT = "base-timeout"
    BASE_FAILURE = "base-failure"
    PLUS_TIMEOUT = "plus-timeout"
    PLUS_ONLY_FAILURE = "plus-only-failure"
    BASE_PLUS_PASS = "base-plus-pass"  # noqa: S105 -- outcome label, not a credential
    EVALUATOR_INFRASTRUCTURE = "evaluator-infrastructure"


def classify_mbpp(
    generation: dict[str, object], verdict: EvalPlusCandidateVerdict | None
) -> MBPPFailureKind:
    submission = generation.get("submission")
    if not isinstance(submission, str) or not submission.strip():
        if generation.get("parse_reason") == "conflicting-finals":
            return MBPPFailureKind.MULTIPLE_CODE_BLOCKS
        return MBPPFailureKind.EMPTY
    try:
        compile(submission, "<mbpp-candidate>", "exec")
    except SyntaxError:
        return MBPPFailureKind.PYTHON_SYNTAX
    if verdict is None:
        return MBPPFailureKind.EVALUATOR_INFRASTRUCTURE
    if verdict.base_plus_passed:
        return MBPPFailureKind.BASE_PLUS_PASS
    if "timed" in verdict.base_status.casefold():
        return MBPPFailureKind.BASE_TIMEOUT
    if not verdict.base_passed:
        return MBPPFailureKind.BASE_FAILURE
    if "timed" in verdict.plus_status.casefold():
        return MBPPFailureKind.PLUS_TIMEOUT
    return MBPPFailureKind.PLUS_ONLY_FAILURE


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    generations = {
        str(row["task_id"]): row
        for row in (
            cast(dict[str, object], json.loads(line))
            for line in arguments.generations.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    verdict_rows = {
        str(row["task_id"]): row
        for row in (
            cast(dict[str, object], json.loads(line))
            for line in arguments.verdicts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if set(generations) != set(verdict_rows):
        raise ValueError("MBPP+ generation and verdict IDs differ")
    counts: Counter[str] = Counter()
    for task_id, generation in generations.items():
        row = verdict_rows[task_id]
        verdict = None
        if isinstance(row.get("base_passed"), bool) and isinstance(row.get("plus_passed"), bool):
            verdict = EvalPlusCandidateVerdict(
                task_id,
                "success" if row["base_passed"] else "failed",
                "success" if row["plus_passed"] else "failed",
            )
        counts[classify_mbpp(generation, verdict).value] += 1
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
