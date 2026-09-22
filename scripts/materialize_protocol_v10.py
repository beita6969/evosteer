#!/usr/bin/env python3
"""Materialize all Protocol 10 populations in server-private storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev_private.benchmarks.protocol_v10_materialization import (
    materialize_protocol_v10_population,
    publish_protocol_v10_materialized_catalog,
)
from skillev_private.benchmarks.protocol_v10_sources import (
    load_protocol_v10_source_records,
)

from skillev.experiments.protocol_v10 import load_active_protocol_v10


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    protocol = load_active_protocol_v10(args.protocol.resolve())
    source_populations = load_protocol_v10_source_records(protocol, args.data_root.resolve())
    populations = tuple(
        materialize_protocol_v10_population(spec, records) for spec, records in source_populations
    )
    catalog = publish_protocol_v10_materialized_catalog(
        protocol=protocol,
        populations=populations,
        output_root=args.output_root.resolve(),
    )
    counts = {
        population.spec.population_id: len(population.items) for population in catalog.populations
    }
    print(
        json.dumps(
            {
                "population_counts": counts,
                "population_count": len(counts),
                "status": "materialized",
                "training_episode_count": 4_608,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
