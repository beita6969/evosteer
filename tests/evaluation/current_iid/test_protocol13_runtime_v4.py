from dataclasses import replace
from pathlib import Path

import pytest

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import (
    load_execution_contracts_v3,
    load_targets_v3,
)
from skillev.evaluation.current_iid.protocol13.execution_receipts import (
    COMPLETE_FORMAT,
    Protocol13ExecutionReceipt,
    RuntimeGenerationProfile,
)
from skillev.evaluation.current_iid.protocol13.runner_profiles import (
    load_protocol13_runner_profiles,
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
    return protocol, executions


def test_runtime_receipt_rejects_one_field_mutation() -> None:
    _protocol, executions = _contracts()
    execution = executions[Protocol13Benchmark.HUMAN_EVAL]
    profile = load_protocol13_runner_profiles(
        ROOT / "configs/evaluation/qwen35_protocol13_iid.yaml"
    ).require(execution.decoding_profile)
    ids = tuple(f"HumanEval/{index}" for index in range(128))
    evaluator_contract = {
        "profile_id": execution.evaluator_profile,
        "python_version": "3.12.0",
    }
    receipt = Protocol13ExecutionReceipt(
        format_version=COMPLETE_FORMAT,
        status="complete",
        attempt_id="attempt",
        execution=execution.to_mapping(),
        generation_profile=RuntimeGenerationProfile.from_direct_profile(profile),
        environment=None,
        model_route=execution.actor_route,
        service_instance_ids=("service-0",),
        response_model_ids=(execution.actor_route,),
        context_length=execution.context_length,
        adapter_active=False,
        panel_manifest_id=execution.panel_manifest_id,
        planned_task_ids=ids,
        generated_task_ids=ids,
        scored_task_ids=ids,
        generation_code_revision="revision",
        scoring_code_revision="revision",
        evaluator_version=str(execution.evaluator_profile),
        evaluator_contract=evaluator_contract,
        started_at="2026-08-29T00:00:00Z",
        completed_at="2026-08-29T01:00:00Z",
    )
    receipt.validate_against(
        execution=execution,
        expected_task_ids=ids,
        expected_generation_profile=profile,
        expected_evaluator_contract=evaluator_contract,
    )
    with pytest.raises(ValueError, match="context length"):
        replace(receipt, context_length=execution.context_length - 1).validate_against(
            execution=execution,
            expected_task_ids=ids,
            expected_generation_profile=profile,
            expected_evaluator_contract=evaluator_contract,
        )


def test_external_anchors_and_incomplete_paper_targets_are_nonformal() -> None:
    protocol, executions = _contracts()
    targets = load_targets_v3(
        ROOT / "configs/evaluation/qwen35_protocol13_iid_targets.yaml",
        executions=executions,
        protocol=protocol,
    )
    assert not any(target.can_enter_formal_gate for target in targets.values())
    assert targets[Protocol13Benchmark.HEALTHBENCH].evidence_status == ("external-aggregate-anchor")
    assert targets[Protocol13Benchmark.HOTPOT_QA].evidence_status == (
        "paper-reported-protocol-incomplete"
    )
    aime = targets[Protocol13Benchmark.AIME_2026]
    assert aime.evidence_status == "owner-defined-goal"
    assert str(aime.metrics[0].reference_percent) == "80.00"
    assert aime.metrics[0].required_for_formal_gate


def test_webshop_condition_freezes_the_validation_winner() -> None:
    _protocol, executions = _contracts()
    execution = executions[Protocol13Benchmark.WEB_SHOP]
    assert execution.condition_id == "webshop-benchmark-native-stateact-memory-v17@1"
    assert execution.prompt_profile == "webshop-native-stateact-memory-v17@1"
    assert execution.interactive is not None
    assert execution.interactive.prompt_asset_id == ("webshop-official-train-efficient-memory-v7")
    assert execution.interactive.max_steps_cap == 10
    assert execution.interactive.required_max_steps == 10
