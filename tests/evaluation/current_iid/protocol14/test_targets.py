from decimal import Decimal
from pathlib import Path

import pytest

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.targets import (
    EvidenceLocator,
    MetricProjection,
    PublishedAnchor,
    ReferenceMetricV4,
    ReferenceTargetV5,
    TargetScope,
    TargetStatus,
)
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def test_composite_anchor_cannot_be_promoted_to_formal() -> None:
    evidence = EvidenceLocator("paper", "source", "version", "table", ("metric",))
    with pytest.raises(ValueError, match="composite anchor"):
        PublishedAnchor(
            "EvalPlus",
            TargetScope.COMPOSITE,
            "evalplus_aggregate",
            Decimal("71.8"),
            evidence,
            usable_as_formal_target=True,
        )


def test_mbpp_target_rejects_evalplus_aggregate_metric() -> None:
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    execution = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )[Protocol14Benchmark.MBPP_PLUS]
    metric = ReferenceMetricV4(
        "evalplus_aggregate",
        "percent",
        "composite",
        MetricProjection.IDENTITY_PERCENT,
        "mean(humaneval_plus, mbpp_plus)",
        "external",
        Decimal("71.8"),
    )
    with pytest.raises(ValueError, match="aggregate"):
        ReferenceTargetV5(
            execution.benchmark,
            TargetStatus.PENDING,
            TargetScope.STANDALONE_BENCHMARK,
            execution,
            None,
            (metric,),
            None,
            None,
            None,
        )


def test_interactive_conditions_are_action_only_and_source_pinned() -> None:
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    executions = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )
    for benchmark in (Protocol14Benchmark.WEB_SHOP, Protocol14Benchmark.ALF_WORLD):
        execution = executions[benchmark]
        assert execution.thinking_mode.value == "disabled"
        assert execution.interactive is not None
        assert not execution.interactive.include_reasoning_in_history
        assert execution.interactive.prompt_source_revision == (
            "74be52bb6bd9f0e9e68dacb72636b75649197983"
        )
