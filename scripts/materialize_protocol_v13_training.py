#!/usr/bin/env python3
"""Materialize the exact-eight, 2,000-question private Protocol 13 training set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev_private.benchmarks.protocol_v13_training import (
    build_protocol13_training_records,
    validate_protocol13_training_dataset,
    write_protocol13_training_dataset,
)
from skillev_private.benchmarks.protocol_v13_training_sources import (
    Protocol13TrainingSourcePaths,
    load_protocol13_training_sources,
    source_diagnostics,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-qa-train", type=Path, required=True)
    parser.add_argument("--released-final-source", type=Path, required=True)
    parser.add_argument("--aime-train", type=Path, required=True)
    parser.add_argument("--healthbench", type=Path, required=True)
    parser.add_argument("--webshop-goals", type=Path, required=True)
    parser.add_argument("--webshop-final-manifest", type=Path, required=True)
    parser.add_argument("--alfworld-train", type=Path, required=True)
    parser.add_argument("--alfworld-final-manifest", type=Path, required=True)
    parser.add_argument("--mbpp-plus", type=Path, required=True)
    parser.add_argument("--mbpp-plus-final-manifest", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--humaneval-final-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    paths = Protocol13TrainingSourcePaths(
        joint_qa_train=arguments.joint_qa_train,
        released_final_source=arguments.released_final_source,
        aime_train=arguments.aime_train,
        healthbench=arguments.healthbench,
        webshop_goals=arguments.webshop_goals,
        webshop_final_manifest=arguments.webshop_final_manifest,
        alfworld_train=arguments.alfworld_train,
        alfworld_final_manifest=arguments.alfworld_final_manifest,
        mbpp_plus=arguments.mbpp_plus,
        mbpp_plus_final_manifest=arguments.mbpp_plus_final_manifest,
        humaneval=arguments.humaneval,
        humaneval_final_manifest=arguments.humaneval_final_manifest,
    )
    bundle = load_protocol13_training_sources(paths)
    records = build_protocol13_training_records(bundle.sources)
    summary = write_protocol13_training_dataset(records, arguments.output_dir)
    validated = validate_protocol13_training_dataset(arguments.output_dir)
    if validated != summary:
        raise RuntimeError("freshly materialized Protocol 13 dataset failed validation")
    print(
        json.dumps(
            {"dataset": summary, "sources": source_diagnostics(bundle)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
