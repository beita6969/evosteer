#!/usr/bin/env python3
"""Offline structural audit of a private rollout trace lifecycle."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from skillev.diagnostics.rollout_trace import RolloutTraceStage, reject_private_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    events: dict[str, list[dict[str, object]]] = defaultdict(list)
    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("trace row must be an object")
            reject_private_keys(row)  # type: ignore[arg-type]
            events[str(row["trajectory_id"])].append(row)
    for trajectory_id, rows in events.items():
        stages = Counter(str(row["stage"]) for row in rows)
        if stages[RolloutTraceStage.EPISODE_STARTED.value] != 1:
            raise ValueError(f"{trajectory_id} lacks one episode-started event")
        if stages[RolloutTraceStage.TERMINAL_REQUEST.value] != 1:
            raise ValueError(f"{trajectory_id} lacks one terminal-request event")
        if stages[RolloutTraceStage.TERMINAL_RESULT.value] != 1:
            raise ValueError(f"{trajectory_id} lacks one terminal-result event")
        for stage in (
            RolloutTraceStage.REASONING_REQUEST,
            RolloutTraceStage.REASONING_RESULT,
            RolloutTraceStage.ACTION_REQUEST,
            RolloutTraceStage.ACTION_RESULT,
            RolloutTraceStage.ACTION_PARSED,
            RolloutTraceStage.ENVIRONMENT_RESULT,
        ):
            if stages[stage.value] == 0:
                raise ValueError(f"{trajectory_id} has no {stage.value} event")
    print(json.dumps({"trajectories": len(events), "status": "pass"}))


if __name__ == "__main__":
    main()
