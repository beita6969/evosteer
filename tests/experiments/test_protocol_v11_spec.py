from pathlib import Path

import pytest

from skillev.experiments.protocol_v10 import ACTIVE_BENCHMARKS_V10
from skillev.experiments.protocol_v11 import (
    ACTIVE_BENCHMARKS_V11,
    BenchmarkV11,
    ProtocolV11Error,
    ProtocolV11TrainingShape,
    load_protocol_v11,
)

ROOT = Path(__file__).parents[2]


def test_current_catalog_and_historical_catalog_are_independent() -> None:
    assert ACTIVE_BENCHMARKS_V11 == (
        BenchmarkV11.HOTPOT_QA,
        BenchmarkV11.TRIVIA_QA,
        BenchmarkV11.AIME_2026,
        BenchmarkV11.HEALTHBENCH,
        BenchmarkV11.WEBSHOP,
        BenchmarkV11.ALFWORLD,
        BenchmarkV11.SPREADSHEETBENCH,
        BenchmarkV11.APPWORLD,
        BenchmarkV11.MBPP_PLUS_HARD,
        BenchmarkV11.HUMAN_EVAL,
    )
    assert "appworld" in ACTIVE_BENCHMARKS_V10
    assert "mbpp-plus-fixed-100" in ACTIVE_BENCHMARKS_V10
    assert "appworld" in ACTIVE_BENCHMARKS_V11
    assert "mbpp-plus-hard" in ACTIVE_BENCHMARKS_V11
    assert "swe-bench-verified" not in ACTIVE_BENCHMARKS_V11


def test_training_shape_closes() -> None:
    shape = ProtocolV11TrainingShape()
    shape.validate()
    assert shape.total_episodes == 5_120 == shape.batch_size * shape.optimizer_steps


def test_repository_protocol_is_closed_and_has_all_roles() -> None:
    protocol = load_protocol_v11(
        ROOT / "configs/evaluation/protocol_v11.yaml",
        ROOT / "configs/evaluation/protocol_v11_sources.yaml",
    )
    assert protocol.executable is False
    assert len(protocol.populations) == 30
    with pytest.raises(ProtocolV11Error):
        protocol.require_execution_ready()
