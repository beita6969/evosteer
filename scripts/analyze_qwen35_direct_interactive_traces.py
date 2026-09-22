#!/usr/bin/env python3
"""Emit content-free action and termination telemetry from private traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for line in arguments.per_task_results.read_text(encoding="utf-8").splitlines():
        row = cast(dict[str, object], json.loads(line))
        bucket = counts[str(row["benchmark"])]
        reward = row.get("reward")
        bucket["records"] += 1
        bucket["official_terminal"] += row.get("terminal_reached") is True
        bucket["horizon_terminated"] += row.get("terminated_by_horizon") is True
        bucket["candidate_invalid"] += row.get("termination_reason") == "candidate-invalid"
        bucket["reward_zero"] += reward == 0.0
        bucket["reward_partial"] += isinstance(reward, int | float) and 0.0 < reward < 1.0
        bucket["reward_full"] += reward == 1.0
        actions: list[str] = []
        for step in cast(list[dict[str, object]], row.get("trace") or []):
            action = step.get("parsed_action")
            if isinstance(action, str):
                actions.append(action)
                bucket["search_actions"] += action.startswith("search[")
                bucket["click_actions"] += action.startswith("click[")
        bucket["repeated_actions"] += len(actions) - len(set(actions))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({key: dict(value) for key, value in sorted(counts.items())}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
