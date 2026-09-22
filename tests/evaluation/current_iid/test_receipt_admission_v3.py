from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from skillev.evaluation.current_iid.protocol13.admission import (
    MetricComparisonMode,
    Protocol13MismatchError,
    admit_against_target_v3,
)
from skillev.evaluation.current_iid.protocol13.aggregation import (
    compare_against_numeric_target_v3,
)
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import (
    load_execution_contracts_v3,
    load_targets_v3,
)
from skillev.evaluation.current_iid.protocol13.receipts import (
    MetricObservationV3,
    OutcomeCountsV4,
    Protocol13RunReceipt,
    RunProvenanceV3,
)
from skillev.evaluation.current_iid.protocol13.targets import (
    MetricProjection,
    ReferenceAggregationV4,
    ReferencePopulationV4,
    TargetStatus,
)
from skillev.experiments.protocol_v13 import load_protocol_v13

ROOT = Path(__file__).parents[3]


def _contracts():
    protocol = load_protocol_v13(
        ROOT / "configs/evaluation/protocol_v13.yaml",
        ROOT / "configs/evaluation/protocol_v13_sources.yaml",
    )
    executions = load_execution_contracts_v3(
        ROOT / "configs/evaluation/protocol_v13_conditions.yaml", protocol=protocol
    )
    targets = load_targets_v3(
        ROOT / "configs/evaluation/qwen35_protocol13_iid_targets.yaml",
        executions=executions,
        protocol=protocol,
    )
    return executions, targets


def _receipt(denominator_id: str = "planned-panel-128") -> Protocol13RunReceipt:
    executions, _targets = _contracts()
    execution = executions[Protocol13Benchmark.HUMAN_EVAL]
    return Protocol13RunReceipt(
        benchmark=Protocol13Benchmark.HUMAN_EVAL,
        execution=execution,
        planned_count=128,
        outcomes=OutcomeCountsV4(128, 0, 0, 0, 0, 114, 14),
        metrics=(
            MetricObservationV3(
                "pass_at_1",
                "percent",
                Decimal(114),
                128,
                denominator_id,
                MetricProjection.IDENTITY_PERCENT,
                Decimal("89.0625"),
                "count(official_humaneval_pass) / planned_count * 100",
                "mean-over-records",
            ),
        ),
        provenance=RunProvenanceV3(
            "attempt",
            "generation",
            "scoring",
            "skillev-benchmark-protocol@13",
            "revision",
            "revision",
            "revision",
            "humaneval@1",
            "2026-08-28T00:00:00Z",
            "2026-08-28T01:00:00Z",
        ),
        diagnostics={},
        runtime_execution_attempt_id="attempt",
        runtime_contract_matched=True,
        final_panel_used_for_selection=False,
    )


def _formal_target():
    executions, targets = _contracts()
    execution = executions[Protocol13Benchmark.HUMAN_EVAL]
    target = targets[Protocol13Benchmark.HUMAN_EVAL]
    return replace(
        target,
        status=TargetStatus.EXACT_MATCHED,
        reference_execution=execution,
        reference_population=ReferencePopulationV4(
            execution.population_id,
            execution.dataset_revision,
            execution.selection_rule,
            execution.expected_count,
            execution.panel_manifest_id,
        ),
        reference_aggregation=ReferenceAggregationV4("single-run", (42,), 1, "identity", None),
    )


def test_admission_checks_denominator_identity_and_strict_gap() -> None:
    admitted = admit_against_target_v3(_receipt(), _formal_target())
    assert admitted[0].passed
    with pytest.raises(Protocol13MismatchError, match="metric contract"):
        admit_against_target_v3(
            _receipt("valid-candidates-only"),
            _formal_target(),
        )


def test_infrastructure_receipt_is_never_formal() -> None:
    original = _receipt()
    broken = replace(original, outcomes=OutcomeCountsV4(127, 0, 1, 0, 0, 113, 14))
    with pytest.raises(Protocol13MismatchError, match="incomplete"):
        admit_against_target_v3(broken, _formal_target())


def test_seven_point_gap_is_excluded_but_smaller_gap_passes() -> None:
    original = _receipt()
    metric = original.metrics[0]
    at_boundary = replace(
        original,
        metrics=(
            replace(
                metric,
                numerator=Decimal("105.0368"),
                observed_percent=Decimal("82.06"),
            ),
        ),
    )
    inside = replace(
        original,
        metrics=(
            replace(
                metric,
                numerator=Decimal("105.03680128"),
                observed_percent=Decimal("82.060001"),
            ),
        ),
    )
    target = _formal_target()
    assert not admit_against_target_v3(at_boundary, target)[0].passed
    assert admit_against_target_v3(inside, target)[0].passed


def test_owner_defined_score_goal_is_a_one_sided_minimum() -> None:
    original = _receipt()
    _executions, targets = _contracts()
    target = targets[Protocol13Benchmark.HUMAN_EVAL]
    owner_goal = replace(
        target,
        status=TargetStatus.OWNER_DEFINED_GOAL,
        metrics=(replace(target.metrics[0], reference_percent=Decimal("80.00")),),
    )
    above = compare_against_numeric_target_v3(original, owner_goal)[0]
    assert above.passed
    assert above.gap_pp == 0
    assert above.comparison_mode is MetricComparisonMode.MINIMUM_GOAL

    below = replace(
        original,
        metrics=(
            replace(
                original.metrics[0],
                numerator=Decimal("92.16"),
                observed_percent=Decimal("72.00"),
            ),
        ),
    )
    shortfall = compare_against_numeric_target_v3(below, owner_goal)[0]
    assert not shortfall.passed
    assert shortfall.gap_pp == Decimal("8.00")
