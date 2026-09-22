from pathlib import Path

from skillev_private.direct_reference.manifests import (
    PopulationEntry,
    PopulationManifest,
    write_population_manifest,
)
from skillev_private.direct_reference.protocol14_runner import (
    load_protocol14_panel_manifest,
)

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def test_logical_panel_id_can_wrap_a_legacy_answer_free_manifest(tmp_path: Path) -> None:
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    execution = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )[Protocol14Benchmark.HOTPOT_QA]
    manifest_path = tmp_path / "hotpotqa.json"
    write_population_manifest(
        manifest_path,
        PopulationManifest(
            format_version="skillev-direct-population-manifest@1",
            population_id="skillflow-released-iid-v3",
            benchmark=DirectBenchmark.HOTPOT_QA,
            dataset_revision=execution.dataset_revision,
            selection_rule="released-panel-order",
            entries=tuple(
                PopulationEntry(f"task-{index}", f"source-{index}", "test", index)
                for index in range(128)
            ),
        ),
    )
    panel = load_protocol14_panel_manifest(manifest_path, execution=execution)
    assert panel.logical_population_id == execution.population_id
    assert panel.adapter_population_id == "skillflow-released-iid-v3"
    assert panel.panel_manifest_id == "hotpotqa-final-128-v13"
    assert len(panel.task_ids) == 128
