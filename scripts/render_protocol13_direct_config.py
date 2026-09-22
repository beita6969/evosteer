#!/usr/bin/env python3
"""Render the Protocol 13 direct-runner compatibility config from pinned inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import yaml


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    runner = yaml.safe_load(args.runner_config.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or base.get("format") != "skillev-qwen35-direct-reference@2":
        raise ValueError("base direct-runner config is incompatible")
    if not isinstance(runner, dict) or runner.get("format") != (
        "skillev-qwen35-protocol13-runner@1"
    ):
        raise ValueError("Protocol 13 runner config is incompatible")
    profiles = base.get("profiles")
    runner_profiles = runner.get("profiles")
    if not isinstance(profiles, dict) or not isinstance(runner_profiles, dict):
        raise ValueError("runner profiles are absent")
    math_profile = runner_profiles.get("qwen35-thinking-math-reference@2")
    if not isinstance(math_profile, dict):
        raise ValueError("Protocol 13 AIME profile is absent")
    profiles["qwen35-thinking-math-reference@2"] = dict(math_profile)
    rows = base.get("benchmarks")
    if not isinstance(rows, list):
        raise ValueError("direct benchmark rows are absent")
    changes: dict[str, dict[str, object]] = {
        "hotpotqa": {
            "population": "hotpotqa-final-128-v13",
            "dataset_revision": "skillflow-released",
            "selection_rule": "frozen-manifest-order",
            "scorer": "hotpotqa-official-em-f1@2",
        },
        "triviaqa": {
            "population": "triviaqa-final-128-v13",
            "dataset_revision": "skillflow-released",
            "selection_rule": "frozen-manifest-order",
            "scorer": "triviaqa-official-alias-em-f1@2",
        },
        "aime-2026": {
            "population": "aime-2026-all-30-v13",
            "dataset_revision": "aime-2026",
            "selection_rule": "all-30",
            "prompt": "integer-cot-boxed@1",
            "decoding": "qwen35-thinking-math-reference@2",
            "parser": "aime-boxed-integer@1",
            "scorer": "aime-official-integer@2",
            "seed_aggregation": {"mode": "single-run", "seeds": [42]},
        },
        "webshop": {
            "population": "webshop-final-128-v13",
            "dataset_revision": "official-text-environment",
            "selection_rule": "frozen-manifest-order",
        },
        "alfworld": {
            "population": "alfworld-final-seen97-unseen31-v13",
            "dataset_revision": "official-alfworld",
            "selection_rule": "frozen-seen97-unseen31",
        },
        "humaneval": {
            "population": "humaneval-native-128-v13",
            "dataset_revision": "openai-humaneval-v1",
            "selection_rule": "frozen-native-ids-128",
            "scorer": "humaneval-official-pass1@2",
        },
    }
    seen: set[str] = set()
    for value in rows:
        if not isinstance(value, dict) or type(value.get("id")) is not str:
            raise ValueError("direct benchmark row is incompatible")
        benchmark = cast(str, value["id"])
        if benchmark in changes:
            value.update(changes[benchmark])
            seen.add(benchmark)
    if seen != set(changes):
        raise ValueError("base config lacks a Protocol 13 direct benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
