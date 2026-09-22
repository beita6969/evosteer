from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.protocol_v10_population import (
    PopulationOverlapIdentityMaterializer,
    PopulationOverlapKey,
    PopulationOverlapKind,
    PrivateBenchmarkPopulation,
    PrivatePopulationItem,
    ProtocolV10PopulationCatalog,
    ProtocolV10PopulationSessionRegistry,
    ProtocolV10TrainingSelection,
    build_protocol_v10_training_selection,
    load_protocol_v10_population_catalog,
    load_protocol_v10_population_file,
    materialize_protocol_v10_training_mix,
    write_protocol_v10_population_file,
)
from skillev_private.experiments.protocol_v10_attempt_builder import (
    ProtocolV10AttemptBuilder,
    ProtocolV10MethodBuilders,
)

from skillev.application import FormalRuntimeDependencies
from skillev.contracts import SuccessRule, TerminalReward
from skillev.experiments.protocol_v10 import (
    ACTIVE_BENCHMARKS_V10,
    BenchmarkPopulation,
    BenchmarkV10,
    PopulationRole,
    ProtocolV10Error,
    load_active_protocol_v10,
)
from skillev.experiments.protocol_v10_formal import load_protocol_v10_formal_experiment
from skillev.rollout import (
    RolloutSessionBundle,
    RolloutTask,
    TerminalEvaluationRequest,
)
from skillev.training import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
)

ROOT = Path(__file__).parents[2]
PROTOCOL_PATH = ROOT / "configs" / "evaluation" / "protocol_v10.yaml"
FORMAL_EXPERIMENT_PATH = ROOT / "configs" / "experiments" / "protocol_v10_formal.yaml"


@dataclass(frozen=True, slots=True)
class _Evaluator:
    source_id: str

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.source_id:
            raise AssertionError("training episode was not routed to its source evaluator")
        return TerminalReward(
            value=0.0,
            success=False,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="protocol-v10-population-fixture",
            native_payload={},
            environment_id="protocol-v10-population-fixture",
            verifier_version="protocol-v10-population-fixture@1",
        )


@dataclass(frozen=True, slots=True)
class _Factory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        return RolloutSessionBundle(
            environment=object(),  # type: ignore[arg-type]
            evaluator=_Evaluator(task.task_id),
            retrieved_skills=(),
        )


def _required_keys(benchmark: BenchmarkV10, identity: str) -> tuple[PopulationOverlapKey, ...]:
    keys = [
        PopulationOverlapKey(PopulationOverlapKind.SOURCE_RECORD, identity),
        PopulationOverlapKey(
            PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
            f"normalized:{identity}",
        ),
    ]
    if benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
        keys.append(PopulationOverlapKey(PopulationOverlapKind.CODE_SIGNATURE, identity))
    if benchmark is BenchmarkV10.SPREADSHEETBENCH:
        keys.append(PopulationOverlapKey(PopulationOverlapKind.WORKBOOK_STRUCTURE, identity))
    if benchmark in {
        BenchmarkV10.WEBSHOP,
        BenchmarkV10.ALFWORLD,
        BenchmarkV10.APPWORLD,
    }:
        keys.append(PopulationOverlapKey(PopulationOverlapKind.INTERACTIVE_SCENARIO, identity))
    return tuple(keys)


def _item(benchmark: BenchmarkV10, identity: str) -> PrivatePopulationItem:
    return PrivatePopulationItem(
        source_id=identity,
        task=RolloutTask(
            task_id=identity,
            environment_id=f"fixture:{benchmark.value}",
            task_family=f"{benchmark.value}/fixture",
            context_id=f"fixture:{identity}",
            query=f"Private synthetic population fixture {identity}.",
            available_tools=(),
            public_context={"benchmark_id": benchmark.value},
        ),
        overlap_keys=_required_keys(benchmark, identity),
    )


@dataclass(frozen=True, slots=True)
class _IdentityMaterializer(PopulationOverlapIdentityMaterializer):
    version: str = "fixture-overlap-materializer@1"

    def materialize(
        self,
        *,
        spec: BenchmarkPopulation,
        source_id: str,
        task: RolloutTask,
    ) -> tuple[PopulationOverlapKey, ...]:
        assert source_id in task.query
        return _required_keys(spec.benchmark, source_id)


MATERIALIZER = _IdentityMaterializer()


def _population(spec: BenchmarkPopulation, count: int) -> PrivateBenchmarkPopulation:
    return PrivateBenchmarkPopulation(
        spec=spec,
        items=tuple(
            _item(spec.benchmark, f"{spec.population_id}/source/{index:04d}")
            for index in range(count)
        ),
        identity_materializer_version=MATERIALIZER.version,
    )


def _catalog() -> ProtocolV10PopulationCatalog:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    populations: list[PrivateBenchmarkPopulation] = []
    for benchmark_index, benchmark in enumerate(protocol.benchmarks):
        for population in benchmark.populations:
            if population.role is PopulationRole.TRAINING:
                count = 3 if benchmark_index == 0 else 513 if benchmark_index == 1 else 17
            else:
                count = 2
            populations.append(_population(population, count))
    return ProtocolV10PopulationCatalog(protocol, tuple(populations))


def test_training_mix_has_nine_skillflow_domains_of_exactly_512_then_global_shuffle() -> None:
    selection = build_protocol_v10_training_selection(_catalog())

    assert len(selection.episodes) == 9 * 512
    assert len({item.episode_id for item in selection.episodes}) == 9 * 512
    assert len({item.benchmark for item in selection.episodes[:32]}) > 1
    for benchmark in ACTIVE_BENCHMARKS_V10:
        domain = tuple(item for item in selection.episodes if item.benchmark is benchmark)
        assert len(domain) == 512
        assert tuple(sorted(item.block_position for item in domain)) == tuple(range(512))


def test_small_training_population_repeats_shuffled_cycles_with_distinct_episode_ids() -> None:
    selection = build_protocol_v10_training_selection(_catalog())
    first = tuple(item for item in selection.episodes if item.benchmark is ACTIVE_BENCHMARKS_V10[0])
    source_counts = Counter(item.source_id for item in first)

    assert len(source_counts) == 3
    assert set(source_counts.values()) <= {170, 171}
    assert max(item.repeat_ordinal for item in first) == 170
    assert len({item.episode_id for item in first}) == 512
    by_domain_position = tuple(sorted(first, key=lambda item: item.block_position))
    for cycle_start in range(0, 510, 3):
        cycle = by_domain_position[cycle_start : cycle_start + 3]
        assert len({item.source_id for item in cycle}) == 3

    second = tuple(
        item for item in selection.episodes if item.benchmark is ACTIVE_BENCHMARKS_V10[1]
    )
    assert len({item.source_id for item in second}) == 512
    assert {item.repeat_ordinal for item in second} == {0}


def test_training_selection_is_method_independent_deterministic_and_private(tmp_path: Path) -> None:
    catalog = _catalog()
    first = build_protocol_v10_training_selection(catalog)
    second = build_protocol_v10_training_selection(catalog)
    assert first == second

    output = tmp_path / "training-selection.json"
    first.write_once(output)
    assert ProtocolV10TrainingSelection.read(output) == first
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert len(stored["episodes"]) == 4608
    assert "Private synthetic population fixture" not in output.read_text(encoding="utf-8")


def test_population_catalog_loads_each_role_from_a_distinct_private_file(tmp_path: Path) -> None:
    catalog = _catalog()
    paths: dict[str, Path] = {}
    for population in catalog.populations:
        path = tmp_path / f"{population.spec.population_id}.jsonl"
        path.write_text(
            "".join(json.dumps(item.to_value()) + "\n" for item in population.items),
            encoding="utf-8",
        )
        paths[population.spec.population_id] = path

    materializers = dict.fromkeys(paths, MATERIALIZER)
    loaded = load_protocol_v10_population_catalog(catalog.protocol, paths, materializers)
    assert loaded == catalog

    first, second = tuple(paths)[:2]
    paths[second] = paths[first]
    with pytest.raises(ValueError):
        load_protocol_v10_population_catalog(catalog.protocol, paths, materializers)


def test_population_writer_is_private_atomic_and_non_overwriting(tmp_path: Path) -> None:
    population = _catalog().populations[0]
    output = (tmp_path / "private" / "population.jsonl").resolve()

    write_protocol_v10_population_file(population, output)

    assert (
        load_protocol_v10_population_file(
            population.spec,
            output,
            identity_materializer=MATERIALIZER,
        )
        == population
    )
    assert output.stat().st_mode & 0o777 == 0o600
    assert not output.with_name(f".{output.name}.staging").exists()
    with pytest.raises(FileExistsError):
        write_protocol_v10_population_file(population, output)


@pytest.mark.parametrize(
    "overlap_field",
    ["source_id", "normalized_content"],
)
def test_population_catalog_rejects_training_final_overlap(overlap_field: str) -> None:
    catalog = _catalog()
    populations = list(catalog.populations)
    training_index = next(
        index
        for index, population in enumerate(populations)
        if population.spec.benchmark is BenchmarkV10.HOTPOT_QA
        and population.spec.role is PopulationRole.TRAINING
    )
    final_index = next(
        index
        for index, population in enumerate(populations)
        if population.spec.benchmark is BenchmarkV10.HOTPOT_QA
        and population.spec.role is PopulationRole.FINAL_EVALUATION
    )
    training_item = populations[training_index].items[0]
    final = populations[final_index]
    final_item = final.items[0]
    if overlap_field == "source_id":
        replacement = PrivatePopulationItem(
            source_id=training_item.source_id,
            task=final_item.task,
            overlap_keys=final_item.overlap_keys,
        )
    else:
        training_content = next(
            key
            for key in training_item.overlap_keys
            if key.kind is PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT
        )
        final_source = next(
            key
            for key in final_item.overlap_keys
            if key.kind is PopulationOverlapKind.SOURCE_RECORD
        )
        replacement = PrivatePopulationItem(
            source_id=final_item.source_id,
            task=final_item.task,
            overlap_keys=(final_source, training_content),
        )
    populations[final_index] = PrivateBenchmarkPopulation(
        final.spec,
        (replacement, *final.items[1:]),
        MATERIALIZER.version,
    )

    with pytest.raises(ValueError):
        ProtocolV10PopulationCatalog(catalog.protocol, tuple(populations))


def test_special_domains_require_structural_overlap_keys() -> None:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    spreadsheet = protocol.benchmark(BenchmarkV10.SPREADSHEETBENCH).population(
        PopulationRole.TRAINING
    )[0]
    incomplete = PrivatePopulationItem(
        source_id="spreadsheet/source/1",
        task=RolloutTask(
            task_id="spreadsheet/source/1",
            environment_id="fixture:spreadsheetbench",
            task_family="spreadsheetbench/fixture",
            context_id="spreadsheet/source/1",
            query="Private synthetic population fixture.",
            available_tools=(),
            public_context={"benchmark_id": "spreadsheetbench"},
        ),
        overlap_keys=(
            PopulationOverlapKey(
                PopulationOverlapKind.SOURCE_RECORD,
                "spreadsheet/source/1",
            ),
            PopulationOverlapKey(
                PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
                "normalized:spreadsheet/source/1",
            ),
        ),
    )
    with pytest.raises(ValueError):
        PrivateBenchmarkPopulation(spreadsheet, (incomplete,), MATERIALIZER.version)


def test_population_accepts_shared_group_identity_within_one_population() -> None:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    spreadsheet = protocol.benchmark(BenchmarkV10.SPREADSHEETBENCH).population(
        PopulationRole.TRAINING
    )[0]
    first = _item(BenchmarkV10.SPREADSHEETBENCH, "spreadsheet/source/1")
    second = _item(BenchmarkV10.SPREADSHEETBENCH, "spreadsheet/source/2")
    shared_group = PopulationOverlapKey(
        PopulationOverlapKind.WORKBOOK_STRUCTURE,
        "spreadsheet/shared-workbook",
    )
    first = PrivatePopulationItem(
        source_id=first.source_id,
        task=first.task,
        overlap_keys=(*first.overlap_keys[:-1], shared_group),
    )
    second = PrivatePopulationItem(
        source_id=second.source_id,
        task=second.task,
        overlap_keys=(*second.overlap_keys[:-1], shared_group),
    )

    population = PrivateBenchmarkPopulation(
        spreadsheet,
        (first, second),
        MATERIALIZER.version,
    )
    assert population.items == (first, second)


def test_source_population_rejects_duplicate_public_content_before_mixing() -> None:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    hotpot = protocol.benchmark(BenchmarkV10.HOTPOT_QA).population(PopulationRole.TRAINING)[0]
    first = _item(BenchmarkV10.HOTPOT_QA, "hotpot/source/1")
    second = _item(BenchmarkV10.HOTPOT_QA, "hotpot/source/2")
    second = PrivatePopulationItem(
        source_id=second.source_id,
        task=second.task,
        overlap_keys=first.overlap_keys,
    )

    with pytest.raises(ValueError):
        PrivateBenchmarkPopulation(hotpot, (first, second), MATERIALIZER.version)


def test_catalog_rejects_overlap_across_benchmarks_and_same_role_populations() -> None:
    catalog = _catalog()
    populations = list(catalog.populations)
    hotpot_training = next(
        population
        for population in populations
        if population.spec.benchmark is BenchmarkV10.HOTPOT_QA
        and population.spec.role is PopulationRole.TRAINING
    )
    trivia_final_index = next(
        index
        for index, population in enumerate(populations)
        if population.spec.benchmark is BenchmarkV10.TRIVIA_QA
        and population.spec.role is PopulationRole.FINAL_EVALUATION
    )
    trivia_final = populations[trivia_final_index]
    original = trivia_final.items[0]
    hotpot_content = next(
        key
        for key in hotpot_training.items[0].overlap_keys
        if key.kind is PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT
    )
    populations[trivia_final_index] = PrivateBenchmarkPopulation(
        trivia_final.spec,
        (
            PrivatePopulationItem(
                source_id=original.source_id,
                task=original.task,
                overlap_keys=(original.overlap_keys[0], hotpot_content),
            ),
            *trivia_final.items[1:],
        ),
        MATERIALIZER.version,
    )
    with pytest.raises(ValueError):
        ProtocolV10PopulationCatalog(catalog.protocol, tuple(populations))

    populations = list(catalog.populations)
    alfworld_finals = [
        index
        for index, population in enumerate(populations)
        if population.spec.benchmark is BenchmarkV10.ALFWORLD
        and population.spec.role is PopulationRole.FINAL_EVALUATION
    ]
    first_final = populations[alfworld_finals[0]]
    second_index = alfworld_finals[1]
    second_final = populations[second_index]
    first_group = next(
        key
        for key in first_final.items[0].overlap_keys
        if key.kind is PopulationOverlapKind.INTERACTIVE_SCENARIO
    )
    second_item = second_final.items[0]
    populations[second_index] = PrivateBenchmarkPopulation(
        second_final.spec,
        (
            PrivatePopulationItem(
                source_id=second_item.source_id,
                task=second_item.task,
                overlap_keys=(*second_item.overlap_keys[:-1], first_group),
            ),
            *second_final.items[1:],
        ),
        MATERIALIZER.version,
    )
    with pytest.raises(ValueError):
        ProtocolV10PopulationCatalog(catalog.protocol, tuple(populations))


def test_loader_recomputes_and_rejects_declared_overlap_identity(tmp_path: Path) -> None:
    population = _catalog().populations[0]
    item = population.items[0]
    wrong = PrivatePopulationItem(
        source_id=item.source_id,
        task=item.task,
        overlap_keys=(
            item.overlap_keys[0],
            PopulationOverlapKey(
                PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
                "declared-but-not-recomputed",
            ),
        ),
    )
    path = tmp_path / "population.jsonl"
    path.write_text(json.dumps(wrong.to_value()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_protocol_v10_population_file(
            population.spec,
            path,
            identity_materializer=MATERIALIZER,
        )


def test_materialized_mix_preserves_order_and_routes_repeated_episode_to_source() -> None:
    catalog = _catalog()
    selection = build_protocol_v10_training_selection(catalog)
    sessions = ProtocolV10PopulationSessionRegistry(
        tuple(
            (catalog.training_population(benchmark).spec.population_id, _Factory())
            for benchmark in ACTIVE_BENCHMARKS_V10
        )
    )
    mix = materialize_protocol_v10_training_mix(catalog, selection, sessions)

    assert len(mix.tasks) == 4608
    assert tuple(task.task_id for task in mix.tasks) == tuple(
        episode.episode_id for episode in selection.episodes
    )
    repeated_episode = next(episode for episode in selection.episodes if episode.repeat_ordinal > 0)
    task = mix.tasks[repeated_episode.global_position]
    bundle = mix.session_factory.create(task)
    assert bundle.evaluator.source_evaluator.source_id == repeated_episode.source_id  # type: ignore[attr-defined]


def test_materialization_rebuilds_and_compares_supplied_selection() -> None:
    catalog = _catalog()
    selection = build_protocol_v10_training_selection(catalog)
    first = selection.episodes[0]
    alternative = next(
        episode
        for episode in selection.episodes
        if episode.benchmark is first.benchmark and episode.source_id != first.source_id
    )
    tampered = ProtocolV10TrainingSelection(
        episodes=(replace(first, source_id=alternative.source_id), *selection.episodes[1:]),
        population_manifests=selection.population_manifests,
    )
    sessions = ProtocolV10PopulationSessionRegistry(
        tuple(
            (catalog.training_population(benchmark).spec.population_id, _Factory())
            for benchmark in ACTIVE_BENCHMARKS_V10
        )
    )

    with pytest.raises(ValueError):
        materialize_protocol_v10_training_mix(catalog, tampered, sessions)


def _formal_runtime(binding: RolloutWorkflowBinding) -> FormalRuntimeDependencies:
    return FormalRuntimeDependencies(
        rollout_generator_factory=lambda backbone, resources: object(),  # type: ignore[return-value]
        gradient_preparer=DistributedTTBGradientCoordinator(
            DistributedTTBTopology(rank=0, world_size=2, local_rank=0, backend="nccl")
        ),
        workflow_resources=RolloutWorkflowResources(binding),
        step_adapter_publisher_factory=lambda backbone: object(),  # type: ignore[return-value]
        skill_author_factory=lambda *args: object(),  # type: ignore[return-value]
    )


def _protocol_v10_attempt_builder(*, executable: bool) -> ProtocolV10AttemptBuilder[str]:
    original = _catalog()
    protocol = replace(original.protocol, executable=executable)
    catalog = ProtocolV10PopulationCatalog(protocol, original.populations)
    selection = build_protocol_v10_training_selection(catalog)
    sessions = ProtocolV10PopulationSessionRegistry(
        tuple(
            (catalog.training_population(benchmark).spec.population_id, _Factory())
            for benchmark in ACTIVE_BENCHMARKS_V10
        )
    )
    experiment = replace(
        load_protocol_v10_formal_experiment(FORMAL_EXPERIMENT_PATH),
        executable=executable,
    )
    binding = RolloutWorkflowBinding()

    def baseline(inputs) -> str:
        return f"baseline:{inputs.training_mix.tasks[0].task_id}"

    def full(inputs) -> str:
        return f"full:{inputs.training_mix.tasks[0].task_id}"

    def no_calibration(inputs) -> str:
        return f"no-calibration:{inputs.training_mix.tasks[0].task_id}"

    return ProtocolV10AttemptBuilder(
        experiment=experiment,
        catalog=catalog,
        selection=selection,
        sessions=sessions,
        runtime=_formal_runtime(binding),
        workflow_binding=binding,
        methods=ProtocolV10MethodBuilders(
            skillflow_baseline=baseline,
            bayesian_improve_full=full,
            bayesian_improve_no_calibration=no_calibration,
        ),
    )


def test_protocol_v10_composition_dispatches_distinct_methods_over_one_mix() -> None:
    builder = _protocol_v10_attempt_builder(executable=True)
    built = tuple(builder.build(binding.method) for binding in builder.experiment.methods)

    assert [item.application.split(":", 1)[0] for item in built] == [
        "baseline",
        "full",
        "no-calibration",
    ]
    ordered_ids = tuple(task.task_id for task in built[0].training_mix.tasks)
    assert len(ordered_ids) == 4608
    assert all(
        tuple(task.task_id for task in item.training_mix.tasks) == ordered_ids for item in built
    )


def test_protocol_v10_composition_stays_closed_until_both_gates_open() -> None:
    builder = _protocol_v10_attempt_builder(executable=False)
    with pytest.raises(ProtocolV10Error):
        builder.build(builder.experiment.methods[0].method)
