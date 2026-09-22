from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from skillev.evaluation.current_iid.protocol14.admission import (
    CapabilityFloorStatus,
    ParityStatus,
    Protocol14MismatchError,
    admit_against_target_v4,
)
from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.contracts import (
    AggregationIdentity,
    DecodingParameters,
    ExecutionContractV4,
    ExecutionLane,
    FormalEligibility,
    ThinkingMode,
)
from skillev.evaluation.current_iid.protocol14.receipts import (
    AttemptRole,
    MetricObservationV4,
    OutcomeCountsV5,
    Protocol14RunReceipt,
)
from skillev.evaluation.current_iid.protocol14.targets import (
    EvidenceLocator,
    MetricProjection,
    ReferenceMetricV4,
    ReferenceTargetV5,
    TargetScope,
    TargetStatus,
)


def _execution() -> ExecutionContractV4:
    return ExecutionContractV4(
        benchmark=Protocol14Benchmark.HOTPOT_QA,
        condition_id="hotpotqa-source-parity-128@1",
        lane=ExecutionLane.REFERENCE_BACKBONE,
        formal_eligibility=FormalEligibility.FORMAL,
        actor_model="Qwen3.5-9B",
        actor_model_revision="revision",
        actor_route="qwen35-direct-base",
        actor_service_profile="sglang",
        context_length=98_304,
        adapter_policy="forbidden",
        population_id="panel",
        dataset_revision="dataset",
        selection_rule="frozen",
        expected_count=128,
        panel_manifest_id="manifest",
        prompt_profile="prompt",
        thinking_mode=ThinkingMode.DISABLED,
        decoding_profile="decoding",
        decoding=DecodingParameters("sampling", 0.7, 0.8, 20, 0.0, 1.5, 1.0, 8192, 42),
        parser_profile="parser",
        completion_profile="completion",
        tool_surface=(),
        environment_profile=None,
        scorer_profile="scorer",
        grader_profile=None,
        metric_profile="metric",
        aggregation=AggregationIdentity((42,), 1, "identity", None),
    )


def _target(execution: ExecutionContractV4, frozen_at: datetime) -> ReferenceTargetV5:
    return ReferenceTargetV5(
        benchmark=execution.benchmark,
        status=TargetStatus.FROZEN_MATCHED_REFERENCE,
        scope=TargetScope.STANDALONE_BENCHMARK,
        observed_execution=execution,
        reference_execution=execution,
        metrics=(
            ReferenceMetricV4(
                "em",
                "percent",
                "planned-panel-128",
                MetricProjection.IDENTITY_PERCENT,
                "em",
                "mean",
                Decimal("60.94"),
            ),
        ),
        evidence=EvidenceLocator("reference-run", "project", "revision", "receipt", ("all",)),
        reference_receipt_id="reference-attempt",
        frozen_at=frozen_at.isoformat(),
    )


def _receipt(
    execution: ExecutionContractV4,
    observed: Decimal,
    started_at: datetime,
    *,
    outcomes: OutcomeCountsV5 | None = None,
) -> Protocol14RunReceipt:
    return Protocol14RunReceipt(
        benchmark=execution.benchmark,
        role=AttemptRole.CANDIDATE,
        execution=execution,
        attempt_id="candidate-attempt",
        planned_count=128,
        outcomes=outcomes or OutcomeCountsV5(128, 0, 0, 0, 0),
        metrics=(
            MetricObservationV4(
                "em",
                "percent",
                observed * Decimal(128) / Decimal(100),
                128,
                "planned-panel-128",
                MetricProjection.IDENTITY_PERCENT,
                observed,
                "em",
                "mean",
            ),
        ),
        generation_code_revision="revision",
        scoring_code_revision="revision",
        evaluator_version="scorer",
        started_at=started_at.isoformat(),
        completed_at=(started_at + timedelta(minutes=1)).isoformat(),
        runtime_contract_matched=True,
        final_panel_used_for_selection=False,
    )


@pytest.mark.parametrize(
    ("observed", "parity", "floor"),
    [
        (Decimal("67.939"), ParityStatus.PASS, CapabilityFloorStatus.PASS),
        (Decimal("67.94"), ParityStatus.ABOVE_REFERENCE, CapabilityFloorStatus.PASS),
        (Decimal("53.941"), ParityStatus.PASS, CapabilityFloorStatus.PASS),
        (Decimal("53.94"), ParityStatus.BELOW_REFERENCE, CapabilityFloorStatus.FAIL),
    ],
)
def test_strict_7pp_boundary_and_capability_floor(
    observed: Decimal,
    parity: ParityStatus,
    floor: CapabilityFloorStatus,
) -> None:
    now = datetime.now(UTC)
    admitted = admit_against_target_v4(
        _receipt(_execution(), observed, now), _target(_execution(), now - timedelta(minutes=1))
    )
    assert admitted[0].parity_status is parity
    assert admitted[0].capability_floor_status is floor
    assert admitted[0].absolute_gap_pp == abs(observed - Decimal("60.94"))
    assert admitted[0].relative_error_percent is not None


def test_model_output_invalid_is_definitive_zero_not_infrastructure() -> None:
    now = datetime.now(UTC)
    outcomes = OutcomeCountsV5(
        scored_valid=127,
        model_output_invalid=1,
        generation_infrastructure=0,
        environment_infrastructure=0,
        scorer_infrastructure=0,
        binary_success=100,
        binary_failure=28,
    )
    receipt = _receipt(_execution(), Decimal("60.94"), now, outcomes=outcomes)
    assert receipt.is_formally_complete
    assert receipt.outcomes.definitive == 128
    assert receipt.outcomes.infrastructure == 0


def test_unresolved_infrastructure_blocks_admission() -> None:
    now = datetime.now(UTC)
    outcomes = OutcomeCountsV5(127, 0, 1, 0, 0)
    with pytest.raises(Protocol14MismatchError, match="unresolved infrastructure"):
        admit_against_target_v4(
            _receipt(_execution(), Decimal("60.94"), now, outcomes=outcomes),
            _target(_execution(), now - timedelta(minutes=1)),
        )


def test_population_mismatch_and_late_target_freeze_are_rejected() -> None:
    now = datetime.now(UTC)
    execution = _execution()
    with pytest.raises(Protocol14MismatchError, match="execution differs"):
        admit_against_target_v4(
            _receipt(replace(execution, population_id="other"), Decimal("60.94"), now),
            _target(execution, now - timedelta(minutes=1)),
        )
    with pytest.raises(Protocol14MismatchError, match="not frozen before"):
        admit_against_target_v4(
            _receipt(execution, Decimal("60.94"), now),
            _target(execution, now),
        )
