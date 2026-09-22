from decimal import Decimal
from pathlib import Path

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.receipts import (
    AttemptRole,
    MetricObservationV4,
    OutcomeCountsV5,
    Protocol14RunReceipt,
    load_run_receipt,
    write_run_receipt,
)
from skillev.evaluation.current_iid.protocol14.targets import MetricProjection
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def _execution():
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    return load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )[Protocol14Benchmark.AIME_2026]


def test_model_invalid_is_definitive_and_receipt_round_trips(tmp_path: Path) -> None:
    execution = _execution()
    receipt = Protocol14RunReceipt(
        benchmark=execution.benchmark,
        role=AttemptRole.CANDIDATE,
        execution=execution,
        attempt_id="candidate-1",
        planned_count=30,
        outcomes=OutcomeCountsV5(
            scored_valid=29,
            model_output_invalid=1,
            generation_infrastructure=0,
            environment_infrastructure=0,
            scorer_infrastructure=0,
            binary_success=20,
            binary_failure=10,
            definitive_detail={"scored": 29, "model-invalid": 1},
        ),
        metrics=(
            MetricObservationV4(
                "accuracy",
                "percent",
                Decimal(20),
                30,
                "planned-panel-30",
                MetricProjection.IDENTITY_PERCENT,
                Decimal(20) / Decimal(30) * Decimal(100),
                "official_aime_integer_exact_match",
                "mean-over-planned-records",
            ),
        ),
        generation_code_revision="revision",
        scoring_code_revision="revision",
        evaluator_version="aime-official-integer@2",
        started_at="2026-08-29T00:00:00Z",
        completed_at="2026-08-29T00:10:00Z",
        runtime_contract_matched=True,
        final_panel_used_for_selection=False,
    )
    assert receipt.is_formally_complete
    path = tmp_path / "receipt.json"
    write_run_receipt(path, receipt)
    loaded = load_run_receipt(path, execution=execution)
    assert loaded == receipt


def test_unresolved_infrastructure_blocks_formal_completion() -> None:
    execution = _execution()
    outcomes = OutcomeCountsV5(
        scored_valid=29,
        model_output_invalid=0,
        generation_infrastructure=1,
        environment_infrastructure=0,
        scorer_infrastructure=0,
        binary_success=20,
        binary_failure=9,
    )
    assert outcomes.total == 30
    assert outcomes.infrastructure == 1
    receipt = Protocol14RunReceipt(
        benchmark=execution.benchmark,
        role=AttemptRole.CANDIDATE,
        execution=execution,
        attempt_id="candidate-infra",
        planned_count=30,
        outcomes=outcomes,
        metrics=(),
        generation_code_revision="revision",
        scoring_code_revision="revision",
        evaluator_version="aime-official-integer@2",
        started_at="2026-08-29T00:00:00Z",
        completed_at="2026-08-29T00:10:00Z",
        runtime_contract_matched=True,
        final_panel_used_for_selection=False,
    )
    assert not receipt.is_formally_complete
