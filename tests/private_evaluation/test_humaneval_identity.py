from dataclasses import replace

import pytest
from skillev_private.benchmarks.humaneval_identity import validate_humaneval_manifest
from skillev_private.direct_reference.manifests import PopulationEntry, PopulationManifest

from skillev.evaluation.direct_baseline.config import DirectBenchmark


def _manifest() -> PopulationManifest:
    return PopulationManifest(
        format_version="skillev-direct-population-manifest@1",
        population_id="humaneval-native-128-v13",
        benchmark=DirectBenchmark.HUMAN_EVAL,
        dataset_revision="openai-humaneval-v1",
        selection_rule="frozen-native-ids-128",
        entries=tuple(
            PopulationEntry(
                task_id=f"humaneval:{index}",
                source_identity=f"HumanEval/{index}",
                source_split="test",
                source_position=index,
            )
            for index in range(128)
        ),
    )


def test_humaneval_manifest_is_limited_to_original_164_task_namespace() -> None:
    validate_humaneval_manifest(_manifest())
    manifest = _manifest()
    invalid = replace(
        manifest,
        entries=(
            *manifest.entries[:-1],
            replace(
                manifest.entries[-1],
                source_identity="HumanEval/164",
            ),
        ),
    )
    with pytest.raises(ValueError):
        validate_humaneval_manifest(invalid)
