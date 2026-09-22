from pathlib import Path

import pytest
from skillev_private.direct_reference.manifests import (
    PopulationEntry,
    PopulationManifest,
    load_population_manifest,
    validate_manifest_task_ids,
    write_population_manifest,
)

from skillev.evaluation.direct_baseline.config import DirectBenchmark


def _manifest() -> PopulationManifest:
    return PopulationManifest(
        "skillev-direct-population-manifest@1",
        "panel@1",
        DirectBenchmark.AIME_2026,
        "dataset@1",
        "native-full-population",
        tuple(
            PopulationEntry(f"aime:{index}", f"problem:{index}", "test", index)
            for index in range(30)
        ),
    )


def test_manifest_round_trip_and_runtime_order_check(tmp_path: Path) -> None:
    path = tmp_path / "aime.json"
    manifest = _manifest()
    write_population_manifest(path, manifest)
    assert load_population_manifest(path) == manifest
    validate_manifest_task_ids(
        manifest,
        population_id="panel@1",
        task_ids=tuple(f"aime:{index}" for index in range(30)),
        source_identities=tuple(f"problem:{index}" for index in range(30)),
    )
    with pytest.raises(ValueError):
        validate_manifest_task_ids(
            manifest,
            population_id="panel@1",
            task_ids=tuple(reversed(tuple(f"aime:{index}" for index in range(30)))),
        )
