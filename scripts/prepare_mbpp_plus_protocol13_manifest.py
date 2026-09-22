#!/usr/bin/env python3
"""Freeze the result-blind Random(0) MBPP+ Protocol 13 panel once."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from skillev_private.direct_reference.manifests import (
    PopulationEntry,
    PopulationManifest,
    write_population_manifest,
)

from skillev.evaluation.direct_baseline.config import DirectBenchmark


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mbpp-source",
        type=Path,
        required=True,
        help="absolute path to the official EvalPlus MBPP+ v0.2.0 JSONL asset",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if not arguments.mbpp_source.is_absolute() or not arguments.mbpp_source.is_file():
        raise ValueError("MBPP+ manifest preparation requires an absolute source asset")
    os.environ["MBPP_OVERRIDE_PATH"] = str(arguments.mbpp_source.resolve())
    from evalplus.data import get_mbpp_plus  # type: ignore[import-not-found]

    task_ids = sorted(get_mbpp_plus())
    if len(task_ids) != 378 or len(set(task_ids)) != 378:
        raise ValueError("MBPP+ source asset must be the official v0.2.0 population")
    positions = sorted(random.Random(0).sample(range(len(task_ids)), 128))  # noqa: S311
    entries = tuple(
        PopulationEntry(
            task_id=task_ids[position],
            source_identity=task_ids[position],
            source_split="test",
            source_position=position,
        )
        for position in positions
    )
    write_population_manifest(
        arguments.output,
        PopulationManifest(
            format_version="skillev-direct-population-manifest@1",
            population_id="mbpp-plus-v0.2.0-random0-128-v13",
            benchmark=DirectBenchmark.MBPP_PLUS,
            dataset_revision="evalplus-mbppplus-v0.2.0",
            selection_rule="frozen-manifest-random0-128",
            entries=entries,
        ),
    )


if __name__ == "__main__":
    main()
