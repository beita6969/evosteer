from decimal import Decimal
from pathlib import Path

from skillev_private.current_iid.base import (
    MetricAggregationSpec,
    PerTaskFinalOutcome,
    build_protocol12_receipt,
)

from skillev.evaluation.current_iid.admission import admit_against_target
from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CURRENT_IID_FINAL_RECORD_COUNT,
    CURRENT_IID_OPTIMIZER_STEPS,
    CURRENT_IID_TRAINING_RECORD_COUNT,
    CurrentIIDBenchmark,
)
from skillev.evaluation.current_iid.config import load_execution_contracts
from skillev.evaluation.current_iid.receipts import (
    FinalOutcomeKind,
    MetricObservation,
    OutcomeCounts,
    Protocol12RunReceipt,
    RunProvenance,
)
from skillev.evaluation.current_iid.targets import load_target_registry_v2
from skillev.experiments.protocol_v12 import load_protocol_v12

ROOT = Path(__file__).parents[3]


def test_protocol_v12_catalog_is_exact_nine() -> None:
    assert tuple(item.value for item in ACTIVE_CURRENT_IID_BENCHMARKS) == (
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "spreadsheetbench",
        "mbpp-plus",
        "humaneval",
    )
    assert CURRENT_IID_TRAINING_RECORD_COUNT == 4_608
    assert CURRENT_IID_OPTIMIZER_STEPS == 288
    assert CURRENT_IID_FINAL_RECORD_COUNT == 1_054
    protocol = load_protocol_v12(
        ROOT / "configs/evaluation/protocol_v12.yaml",
        ROOT / "configs/evaluation/protocol_v12_sources.yaml",
    )
    assert protocol.benchmarks == ACTIVE_CURRENT_IID_BENCHMARKS


def test_protocol_v12_contracts_and_targets_fail_closed() -> None:
    executions = load_execution_contracts(ROOT / "configs/evaluation/protocol_v12_conditions.yaml")
    targets = load_target_registry_v2(
        ROOT / "configs/evaluation/qwen35_current_iid_v12_targets.yaml",
        executions=executions,
    )
    assert set(executions) == set(ACTIVE_CURRENT_IID_BENCHMARKS)
    assert not targets[CurrentIIDBenchmark.HEALTHBENCH].can_enter_formal_gate
    assert not targets[CurrentIIDBenchmark.SPREADSHEETBENCH].can_enter_formal_gate
    assert not targets[CurrentIIDBenchmark.MBPP_PLUS].can_enter_formal_gate


def test_receipt_outcomes_conserve_and_strict_seven_pp() -> None:
    executions = load_execution_contracts(ROOT / "configs/evaluation/protocol_v12_conditions.yaml")
    execution = executions[CurrentIIDBenchmark.HOTPOT_QA]
    provenance = RunProvenance(
        "attempt",
        "generation",
        "scoring",
        "protocol@12",
        "generation-revision",
        "scoring-revision",
        "renderer-revision",
        "evaluator@1",
        "2026-08-28T00:00:00Z",
        "2026-08-28T00:01:00Z",
    )
    targets = load_target_registry_v2(
        ROOT / "configs/evaluation/qwen35_current_iid_v12_targets.yaml",
        executions=executions,
    )
    receipt = Protocol12RunReceipt(
        benchmark=CurrentIIDBenchmark.HOTPOT_QA,
        execution=execution,
        planned_count=128,
        outcomes=OutcomeCounts(64, 64, 0, 0, 0, 0),
        metrics=(
            MetricObservation(
                "em",
                Decimal("69.0432"),
                128,
                Decimal("53.94"),
                "official_hotpotqa_exact_match",
                "mean-over-records",
            ),
            MetricObservation(
                "f1",
                Decimal("87.936"),
                128,
                Decimal("68.70"),
                "official_hotpotqa_token_f1",
                "mean-over-records",
            ),
        ),
        provenance=provenance,
        diagnostics={},
    )
    admitted = admit_against_target(receipt, targets[CurrentIIDBenchmark.HOTPOT_QA])
    assert all(not item.passed for item in admitted)


def test_private_receipt_builder_conserves_tasks() -> None:
    executions = load_execution_contracts(ROOT / "configs/evaluation/protocol_v12_conditions.yaml")
    execution = executions[CurrentIIDBenchmark.AIME_2026]
    outcomes = tuple(
        PerTaskFinalOutcome(
            f"task-{index}",
            FinalOutcomeKind.SCORED_SUCCESS if index == 0 else FinalOutcomeKind.SCORED_FAILURE,
            {"accuracy": Decimal(index == 0)},
            "passed" if index == 0 else "failed",
        )
        for index in range(30)
    )
    provenance = RunProvenance(
        "attempt",
        "generation",
        "scoring",
        "protocol-12",
        "generation-revision",
        "scoring-revision",
        "renderer-revision",
        "aime-official-integer@1",
        "2026-08-28T00:00:00Z",
        "2026-08-28T01:00:00Z",
    )
    receipt = build_protocol12_receipt(
        execution=execution,
        outcomes=outcomes,
        metric_specs=(MetricAggregationSpec("accuracy", "accuracy", "integer equality", "mean"),),
        provenance=provenance,
    )
    assert receipt.outcomes.total == 30
    assert receipt.metrics[0].observed_percent == Decimal(100) / Decimal(30)
