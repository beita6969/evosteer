#!/usr/bin/env python3
"""Classify AIME visible outputs without selecting a parser by final score."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from skillev.evaluation.current_iid.protocol13.aime_extraction import (
    extract_aime_boxed,
    extract_aime_explicit_final,
    extract_aime_final_line,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    arguments = parser.parse_args()
    counts: Counter[str] = Counter()
    completion_tokens: list[int] = []
    with arguments.generations.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("AIME generation row must be an object")
            text = row.get("raw_text")
            if not isinstance(text, str):
                counts["infrastructure"] += 1
                continue
            result = (
                extract_aime_boxed(text)
                if r"\boxed{" in text
                else extract_aime_explicit_final(text)
            )
            if result.value is None and result.reason.value == "empty":
                result = extract_aime_final_line(text)
            counts[result.reason.value] += 1
            counts["finish_length"] += row.get("finish_reason") == "length"
            tokens = row.get("completion_tokens")
            if isinstance(tokens, int):
                completion_tokens.append(tokens)
    ordered = sorted(completion_tokens)
    output: dict[str, object] = {"counts": dict(sorted(counts.items()))}
    if ordered:
        output["completion_tokens"] = {
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "max": ordered[-1],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
