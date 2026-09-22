from __future__ import annotations

from pathlib import Path

import yaml

from skillev.experiments.protocol_v10 import load_active_protocol_v10
from skillev.experiments.protocol_v10_sources import load_protocol_v10_source_plan

ROOT = Path(__file__).parents[2]
PROTOCOL_PATH = ROOT / "configs/evaluation/protocol_v10.yaml"
SOURCE_PATH = ROOT / "configs/evaluation/protocol_v10_sources.yaml"


def test_source_plan_covers_each_population_once_and_preserves_512() -> None:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    plan = load_protocol_v10_source_plan(SOURCE_PATH, protocol=protocol)

    expected = tuple(
        population.population_id
        for benchmark in protocol.benchmarks
        for population in benchmark.populations
    )
    actual = tuple(
        route.population_id for benchmark in plan.benchmarks for route in benchmark.routes
    )
    assert set(actual) == set(expected)
    assert len(actual) == len(set(actual))
    assert plan.training_episodes_per_benchmark == 512


def test_spreadsheet_training_source_excludes_evaluation_workbooks() -> None:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    plan = load_protocol_v10_source_plan(SOURCE_PATH, protocol=protocol)

    training = plan.route("independent-spreadsheet-train-v1")
    final = plan.route("spreadsheetbench-v1-verified-400")
    assert training.source == "Spreadsheet-RL/Spreadsheet-RL"
    assert training.subset == "excelforum"
    assert final.subset == "v1-verified"
    assert training.source != final.source


def test_trivia_source_uses_the_official_no_context_projection() -> None:
    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    plan = load_protocol_v10_source_plan(SOURCE_PATH, protocol=protocol)

    assert plan.route("triviaqa-v1.0-unfiltered-train-main").subset == "unfiltered.nocontext"
    assert plan.route("triviaqa-v1.0-unfiltered-dev").subset == "unfiltered.nocontext"


def test_source_plan_reader_rejects_missing_population(tmp_path: Path) -> None:
    value = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    del value["benchmarks"]["healthbench"]["validation"]
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    protocol = load_active_protocol_v10(PROTOCOL_PATH)
    try:
        load_protocol_v10_source_plan(path, protocol=protocol)
    except ValueError:
        pass
    else:
        raise AssertionError("source plan accepted a missing population")
