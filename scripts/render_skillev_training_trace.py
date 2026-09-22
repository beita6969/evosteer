#!/usr/bin/env python3
"""Render one private trajectory trace after rejecting private-evaluator keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.diagnostics.rollout_trace import reject_private_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("trace row must be an object")
            if value.get("trajectory_id") == args.trajectory_id:
                reject_private_keys(value, location="trace")  # type: ignore[arg-type]
                rows.append(value)
    if not rows:
        raise ValueError("trajectory is absent from trace")
    lines = [f"# Private rollout trace: {args.trajectory_id}", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('stage')} / step {row.get('step_index')}",
                "",
                "```json",
                json.dumps(row.get("public_payload"), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
