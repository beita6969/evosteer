"""Declare the owner-selected external judging without changing old rewards."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.evaluation.healthbench_luna_profile import PROFILE_ID


def declare(source: Path, destination: Path) -> BayesianFormalConfig:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "config" in raw:
        raw = raw["config"]
    before = BayesianFormalConfig(**raw)
    after = replace(before, healthbench_judge=PROFILE_ID)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(after.to_value(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    declare(args.source, args.destination)
    print(
        "Declared external HealthBench Judge; "
        "use --allow-healthbench-judge at a complete checkpoint."
    )
    print("No process restarted and no historical reward or posterior changed.")


if __name__ == "__main__":
    main()
