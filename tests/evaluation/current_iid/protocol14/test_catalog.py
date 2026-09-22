from pathlib import Path

from skillev.evaluation.current_iid.protocol14.catalog import (
    ACTIVE_PROTOCOL14_BENCHMARKS,
    FINAL_RECORD_COUNT,
    Protocol14Benchmark,
    expected_final_count,
)
from skillev.evaluation.current_iid.protocol14.config import (
    load_execution_contracts_v4,
    load_targets_v5,
)
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def test_protocol14_is_exactly_the_authoritative_eight() -> None:
    assert tuple(Protocol14Benchmark) == ACTIVE_PROTOCOL14_BENCHMARKS
    assert [item.value for item in ACTIVE_PROTOCOL14_BENCHMARKS] == [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "mbpp-plus",
        "humaneval",
    ]
    assert FINAL_RECORD_COUNT == 926
    assert sum(expected_final_count(item) for item in ACTIVE_PROTOCOL14_BENCHMARKS) == 926


def test_protocol14_active_identity_has_no_retired_benchmark_or_nonofficial_hard_split() -> None:
    paths = (
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml",
        ROOT / "configs/evaluation/qwen35_protocol14_iid.yaml",
        ROOT / "configs/evaluation/qwen35_protocol14_iid_targets.yaml",
    )
    active_text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    assert "spreadsheetbench" not in active_text
    assert "mbpp+ hard" not in active_text
    assert "mbpp-plus-hard" not in active_text
    assert "mbpp_plus_hard" not in active_text


def test_protocol14_configs_close_and_only_diagnostic_target_remains_nonformal() -> None:
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    executions = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )
    targets = load_targets_v5(
        ROOT / "configs/evaluation/qwen35_protocol14_iid_targets.yaml",
        executions=executions,
    )
    assert tuple(executions) == ACTIVE_PROTOCOL14_BENCHMARKS
    assert tuple(targets) == ACTIVE_PROTOCOL14_BENCHMARKS
    assert protocol.executable
    assert {
        benchmark for benchmark, target in targets.items() if target.can_enter_formal_gate
    } == set(ACTIVE_PROTOCOL14_BENCHMARKS) - {Protocol14Benchmark.HEALTHBENCH}
    health = targets[Protocol14Benchmark.HEALTHBENCH]
    assert health.metrics
    assert not any(metric.required_for_formal_gate for metric in health.metrics)
    assert not health.can_enter_formal_gate
    mbpp = targets[Protocol14Benchmark.MBPP_PLUS]
    assert [metric.metric_id for metric in mbpp.metrics] == ["pass_at_1"]
    assert len(mbpp.published_anchors) == 1
    assert mbpp.published_anchors[0].metric_id == "evalplus_aggregate"
    assert not mbpp.published_anchors[0].usable_as_formal_target
