from pathlib import Path

from skillev.evaluation.current_iid.protocol13.catalog import ACTIVE_PROTOCOL13_BENCHMARKS
from skillev.evaluation.current_iid.protocol13.config import (
    load_execution_contracts_v3,
    load_targets_v3,
)
from skillev.experiments.protocol_v13 import load_protocol_v13

ROOT = Path(__file__).parents[2]


def test_protocol13_configuration_closes_exact_eight() -> None:
    protocol = load_protocol_v13(
        ROOT / "configs/evaluation/protocol_v13.yaml",
        ROOT / "configs/evaluation/protocol_v13_sources.yaml",
    )
    executions = load_execution_contracts_v3(
        ROOT / "configs/evaluation/protocol_v13_conditions.yaml",
        protocol=protocol,
    )
    targets = load_targets_v3(
        ROOT / "configs/evaluation/qwen35_protocol13_iid_targets.yaml",
        executions=executions,
        protocol=protocol,
    )
    assert protocol.benchmarks == ACTIVE_PROTOCOL13_BENCHMARKS
    assert len(protocol.populations) == 24
    assert tuple(executions) == ACTIVE_PROTOCOL13_BENCHMARKS
    assert tuple(targets) == ACTIVE_PROTOCOL13_BENCHMARKS
    assert not protocol.executable
