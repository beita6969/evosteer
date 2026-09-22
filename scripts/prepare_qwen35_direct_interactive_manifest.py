#!/usr/bin/env python3
"""Freeze one benchmark-specific Protocol 13 interactive manifest v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from skillev_private.benchmarks.alfworld_taxonomy import classify_alfworld_game_id

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.experiments.protocol_v13 import load_protocol_v13


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("webshop", "alfworld"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _alf_split(raw: dict[str, object], case: dict[str, object]) -> str:
    payload = _mapping(case["payload"], "ALFWorld payload")
    if payload.get("split") in {"valid_seen", "valid_unseen"}:
        return cast(str, payload["split"])
    deployments = _mapping(raw["deployments"], "deployments")
    deployment = _mapping(deployments[str(case["deployment"])], "ALF deployment")
    games = _mapping(deployment["games"], "ALF games")
    game = _mapping(games[str(payload["game_id"])], "ALF game")
    source = str(game["train_eval"])
    aliases = {
        "valid_seen": "valid_seen",
        "eval_in_distribution": "valid_seen",
        "valid_unseen": "valid_unseen",
        "eval_out_of_distribution": "valid_unseen",
    }
    try:
        return aliases[source]
    except KeyError as exc:
        raise ValueError(f"unknown ALFWorld final split: {source}") from exc


def main() -> None:
    arguments = _arguments()
    protocol = load_protocol_v13(arguments.protocol, arguments.sources)
    benchmark = Protocol13Benchmark(arguments.benchmark)
    execution = load_execution_contracts_v3(arguments.conditions, protocol=protocol)[benchmark]
    raw = _mapping(json.loads(arguments.input.read_text(encoding="utf-8")), "manifest")
    if raw.get("format") not in {
        "skillev-qwen35-direct-interactive@1",
        "skillev-qwen35-direct-interactive@2",
    }:
        raise ValueError("only released interactive manifests can be upgraded")
    source_cases = raw.get("cases")
    if not isinstance(source_cases, list):
        raise ValueError("interactive manifest cases must be an array")
    cases: list[dict[str, object]] = []
    for value in source_cases:
        case = _mapping(value, "interactive case")
        if case.get("benchmark") != benchmark.value:
            continue
        payload = _mapping(case["payload"], "case payload")
        upgraded = dict(case)
        if benchmark is Protocol13Benchmark.WEB_SHOP:
            upgraded["source_identity"] = f"WebShop/{payload['goal_id']}"
        else:
            game_id = str(payload["game_id"])
            split = _alf_split(raw, case)
            upgraded["source_identity"] = f"ALFWorld/{split}/{game_id}"
            upgraded["payload"] = {
                **payload,
                "split": split,
                "task_type": classify_alfworld_game_id(game_id).value,
            }
        cases.append(upgraded)
    if len(cases) != execution.expected_count:
        raise ValueError("interactive final panel count differs from Protocol 13")
    if execution.interactive is None:
        raise ValueError("interactive execution lacks its extension")
    contract = {
        "expected_count": execution.expected_count,
        "prompt_profile": execution.prompt_profile,
        "decoding_profile": execution.decoding_profile,
        "parser_profile": execution.parser_profile,
        "environment_contract": execution.environment_profile,
        **execution.interactive.to_mapping(),
    }
    output = {
        **{key: raw[key] for key in ("runtimes", "deployments")},
        "format": "skillev-qwen35-direct-interactive@3",
        "benchmark": benchmark.value,
        "population_id": execution.population_id,
        "dataset_revision": execution.dataset_revision,
        "selection_rule": execution.selection_rule,
        "panel_manifest_id": execution.panel_manifest_id,
        "benchmarks": {benchmark.value: contract},
        "cases": cases,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        raise FileExistsError("refusing to replace an existing private manifest")
    arguments.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
