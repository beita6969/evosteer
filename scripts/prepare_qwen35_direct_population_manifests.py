#!/usr/bin/env python3
"""Freeze private IID task identity/order manifests before direct generation."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import cast

from skillev_private.direct_reference.manifests import (
    PopulationEntry,
    PopulationManifest,
    write_population_manifest,
)
from skillev_private.direct_reference.populations import (
    PrivateDirectCase,
    load_gpqa_cases,
    load_humaneval_cases,
    load_math_hard_cases,
    load_mind2web_cases,
    load_musique_cases,
    load_nq_open_cases,
    load_skillflow_iid_cases,
)

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iid-population", type=Path)
    parser.add_argument("--musique-archive", type=Path)
    parser.add_argument("--nq-open-jsonl", type=Path)
    parser.add_argument("--math-hard-parquet", type=Path)
    parser.add_argument("--gpqa-csv", type=Path)
    parser.add_argument("--humaneval-jsonl-gz", type=Path)
    parser.add_argument("--mind2web-archive", type=Path)
    parser.add_argument("--mind2web-scores", type=Path)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    protocol = load_direct_reference_protocol(arguments.config)
    iid = frozenset(
        {
            DirectBenchmark.HOTPOT_QA,
            DirectBenchmark.TRIVIA_QA,
            DirectBenchmark.AIME_2026,
            DirectBenchmark.MED_QA,
            DirectBenchmark.SWE_BENCH,
        }
    )
    cases: tuple[PrivateDirectCase, ...] = ()
    if arguments.iid_population is not None:
        cases += load_skillflow_iid_cases(
            arguments.iid_population,
            protocol=protocol,
            include=iid,
        )
    loaders = (
        (arguments.musique_archive, load_musique_cases),
        (arguments.nq_open_jsonl, load_nq_open_cases),
        (arguments.math_hard_parquet, load_math_hard_cases),
        (arguments.gpqa_csv, load_gpqa_cases),
        (arguments.humaneval_jsonl_gz, load_humaneval_cases),
    )
    for path, loader in loaders:
        if path is not None:
            cases += loader(path, protocol=protocol)
    if arguments.mind2web_archive is not None or arguments.mind2web_scores is not None:
        if arguments.mind2web_archive is None or arguments.mind2web_scores is None:
            raise ValueError("Mind2Web requires archive and score artifact together")
        password = os.environ.get("SKILLEV_MIND2WEB_ZIP_PASSWORD")
        if not password:
            raise ValueError("SKILLEV_MIND2WEB_ZIP_PASSWORD is required")
        cases += load_mind2web_cases(
            arguments.mind2web_archive,
            arguments.mind2web_scores,
            protocol=protocol,
            archive_password=password.encode("utf-8"),
        )
    if not cases:
        raise ValueError("at least one private population source is required")
    grouped: dict[DirectBenchmark, list[PrivateDirectCase]] = defaultdict(list)
    for case in cases:
        grouped[case.public_task.benchmark].append(case)
    for benchmark in sorted(grouped, key=lambda item: item.value):
        spec = protocol.benchmark(benchmark)
        entries = tuple(
            PopulationEntry(
                task_id=case.public_task.task_id,
                source_identity=cast(str, case.private_metadata["source_identity"]),
                source_split=cast(str, case.private_metadata["source_split"]),
                source_position=cast(int, case.private_metadata["source_position"]),
                option_order=(
                    cast(tuple[str, ...], case.private_metadata["option_order"])
                    if "option_order" in case.private_metadata
                    else None
                ),
            )
            for case in grouped[benchmark]
        )
        manifest = PopulationManifest(
            format_version="skillev-direct-population-manifest@1",
            population_id=spec.population,
            benchmark=benchmark,
            dataset_revision=spec.dataset_revision,
            selection_rule=spec.selection_rule,
            entries=entries,
        )
        write_population_manifest(
            arguments.private_output_dir / f"{benchmark.value}.json",
            manifest,
        )


if __name__ == "__main__":
    main()
