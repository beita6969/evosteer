"""Report development gates per run, without pooling or selecting trajectories."""

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

METRICS = {
    "musique": "answer-f1",
    "nq-open": "answer-exact-match",
    "math-hard": "accuracy",
    "gpqa-diamond-bioorganic": "accuracy",
    "scienceworld": "native-final-score",
    "apps-introductory": "pass-at-1",
}


def assess(summary: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """All native values are fractions; ScienceWorld's fraction may be negative."""
    rows = {}
    for benchmark, target in policy["targets"].items():
        entry = summary.get("benchmarks", {}).get(benchmark, {})
        threshold = Decimal(str(target)) * Decimal(str(policy["gate_ratio"]))
        value = entry.get("value")
        score = None if value is None else Decimal(str(value)) * 100
        count = entry.get("count", 0)
        diagnostics = entry.get("native_diagnostics", {})
        statuses = [item.get("status_counts", {}) for item in diagnostics.values()]
        complete = (
            count in (policy["initial_samples"], policy["confirmation_samples"])
            and entry.get("metric") == METRICS[benchmark]
            and entry.get("execution_status") == "complete"
            and entry.get("native_metric_contract_status") == "pass"
            and len(statuses) == 1
            and statuses[0].get("infrastructure-failure", 0) == 0
            and statuses[0].get("scored", 0) + statuses[0].get("candidate-failure", 0) == count
            and score is not None
            and score.is_finite()
        )
        passed = complete and score is not None and score > threshold
        rows[benchmark] = {
            "count": count,
            "score": None if score is None else str(score),
            "strict_threshold": str(threshold),
            "status": "incomplete" if not complete else "passed" if passed else "below-gate",
            "promote_to_64": passed and count == policy["initial_samples"],
            "accepted_64": passed and count == policy["confirmation_samples"],
        }
    return {"run_id": summary.get("run_id"), "condition": policy["condition"], "benchmarks": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/evaluation/current_ood_targets.json",
    )
    args = parser.parse_args()
    policy = json.loads(args.targets.read_text())
    for path in args.summaries:
        print(json.dumps(assess(json.loads(path.read_text()), policy)))


if __name__ == "__main__":
    main()
