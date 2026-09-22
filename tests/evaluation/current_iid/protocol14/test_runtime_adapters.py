from pathlib import Path

from skillev_private.direct_reference.manifests import (
    PopulationEntry,
    PopulationManifest,
)
from skillev_private.direct_reference.protocol14_runner import (
    LoadedProtocol14Panel,
    _single_benchmark_protocol,
    build_protocol14_interactive_adapter,
    profile_from_execution,
)

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.direct_baseline.config import (
    BenchmarkComparability,
    DirectBenchmark,
)
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def _executions():
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    return load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )


def test_static_runtime_adapter_uses_exact_external_comparability() -> None:
    execution = _executions()[Protocol14Benchmark.HOTPOT_QA]
    manifest = PopulationManifest(
        format_version="skillev-direct-population-manifest@1",
        population_id="legacy-answer-free-panel",
        benchmark=DirectBenchmark.HOTPOT_QA,
        dataset_revision=execution.dataset_revision,
        selection_rule=execution.selection_rule,
        entries=tuple(
            PopulationEntry(f"task-{index}", f"source-{index}", "test", index)
            for index in range(execution.expected_count)
        ),
    )
    adapter = _single_benchmark_protocol(execution, profile_from_execution(execution), manifest)
    assert adapter.benchmarks[0].comparability is BenchmarkComparability.EXACT_EXTERNAL


def test_interactive_runtime_adapter_uses_exact_external_comparability() -> None:
    execution = _executions()[Protocol14Benchmark.WEB_SHOP]
    panel = LoadedProtocol14Panel(
        benchmark=execution.benchmark,
        logical_population_id=execution.population_id,
        adapter_population_id="legacy-answer-free-panel",
        dataset_revision=execution.dataset_revision,
        selection_rule=execution.selection_rule,
        panel_manifest_id=execution.panel_manifest_id,
        task_ids=tuple(f"task-{index}" for index in range(execution.expected_count)),
        source_identities=tuple(f"source-{index}" for index in range(execution.expected_count)),
    )
    adapter = build_protocol14_interactive_adapter(
        execution, profile_from_execution(execution), panel
    )
    assert adapter.benchmarks[0].comparability is BenchmarkComparability.EXACT_EXTERNAL
