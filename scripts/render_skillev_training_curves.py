#!/usr/bin/env python3
"""Render answer-free training curves from private step-point JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.diagnostics.curves import TrainingCurvePoint, ewma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    points: list[TrainingCurvePoint] = []
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("curve row must be an object")
            points.append(TrainingCurvePoint(**raw))
    smooth = ewma(tuple(point.mean_reward for point in points), alpha=args.alpha)
    lines = [
        "# SKILLEV private training curve",
        "",
        "| Step | Reward | Reward EWMA | Success | Completion | TTB loss | Infra |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point, mean in zip(points, smooth, strict=True):
        lines.append(
            f"| {point.optimizer_step} | {point.mean_reward:.4f} | {mean:.4f} | "
            f"{point.success_rate:.4f} | {point.completion_rate:.4f} | "
            f"{point.ttb_loss:.4f} | {point.infrastructure_failure_count} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
