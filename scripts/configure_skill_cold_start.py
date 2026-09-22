"""Write an explicit new method condition, preserving every unrelated control.

No training process is touched. Use the output with --allow-skill-cold-start at
a complete checkpoint; the normal condition journal records the boundary.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.evolution.cold_start_config import ColdStartConfig


def declare(source: Path, destination: Path) -> BayesianFormalConfig:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "config" in raw:
        raw = raw["config"]
    if not isinstance(raw, dict):
        raise ValueError("source must be a formal configuration or condition-current document")
    before = BayesianFormalConfig(**raw)
    if before.cold_start is not None:
        raise ValueError("source already declares a cold-start extension")
    after = replace(before, skill_exposure="catalog-then-read@1", cold_start=ColdStartConfig())
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(after.to_value(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    config = declare(args.source, args.destination)
    print("Declared zero-coverage-generate@1 with optional catalog reads; no process restarted.")
    print(
        f"B={config.batch_size}, steps={config.steps}, "
        f"horizons={config.static_max_turns}/{config.max_turns}; other controls preserved."
    )


if __name__ == "__main__":
    main()
