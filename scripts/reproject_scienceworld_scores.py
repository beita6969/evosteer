"""Write a new native-score report from immutable environment journals, without generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from skillev.evaluation.environment_scores import LearningRewardProjection, NativeEnvironmentScore
from skillev.evaluation.input_metric_contracts import CONTRACTS
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_results import (
    NativeScore,
    exact_panel_join,
    native_mean,
    scienceworld_terminal_metrics,
)
from skillev.evaluation.sealed_candidates import CandidateReader, EventOrigin


def reproject(
    reader: CandidateReader, task_ids: tuple[str, ...], run_id: str, arm_id: str
) -> dict[str, object]:
    scores, outcomes = [], []
    for task_id in task_ids:
        scope = run_id, arm_id, task_id
        reader.get(*scope)  # Require the original committed candidate; never create a new attempt.
        events = reader.traces(scope, "native-outcome", origin=EventOrigin.ENVIRONMENT)
        if len(events) != 1 or not isinstance(events[0], dict):
            raise RuntimeError("native terminal journal is incomplete")
        native = NativeEnvironmentScore.from_outcome(events[0])
        learning = LearningRewardProjection.from_native(native)
        outcomes.append(events[0])
        scores.append(
            NativeScore(
                task_id,
                "scienceworld",
                CONTRACTS["scienceworld"].metric,
                native.value,
                secondary_metrics={
                    "success": float(learning.success),
                    "zero-clipped-learning-reward": learning.value,
                },
                verifier_version=NATIVE_VERIFIER_VERSIONS["scienceworld"],
            )
        )
    joined = exact_panel_join(
        task_ids, tuple(scores), expected_verifier=NATIVE_VERIFIER_VERSIONS["scienceworld"]
    )
    return {
        "schema": "scienceworld-readonly-native-reprojection@1",
        "source_run_id": run_id,
        "source_arm_id": arm_id,
        "new_model_calls": 0,
        "new_environment_steps": 0,
        "source_records_modified": False,
        "planned": len(task_ids),
        "scored": len(joined),
        "metric": CONTRACTS["scienceworld"].metric,
        "value": native_mean(joined),
        "unit": "native-score/100",
        "terminal_metrics": scienceworld_terminal_metrics(outcomes),
        "scores": [asdict(score) for score in joined],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.frozen_plan.read_text())
    task_ids = tuple(
        entry["task_id"]
        for entry in plan["panel"]["entries"]
        if entry["benchmark"] == "scienceworld"
    )
    reader = CandidateReader(args.database)
    try:
        result = reproject(reader, task_ids, args.run_id, args.arm_id)
    finally:
        reader.close()
    # New report only. No in-place regrading, old-score replacement or implicit overwrite.
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "scores"}))


if __name__ == "__main__":
    main()
