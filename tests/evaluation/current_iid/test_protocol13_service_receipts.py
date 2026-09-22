from dataclasses import replace
from pathlib import Path

import pytest

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.current_iid.protocol13.service_receipts import (
    ModelServiceContract,
    ModelServiceReceipt,
    validate_replica_services,
)
from skillev.experiments.protocol_v13 import load_protocol_v13

ROOT = Path(__file__).parents[3]


def _execution():
    protocol = load_protocol_v13(
        ROOT / "configs/evaluation/protocol_v13.yaml",
        ROOT / "configs/evaluation/protocol_v13_sources.yaml",
    )
    return load_execution_contracts_v3(
        ROOT / "configs/evaluation/protocol_v13_conditions.yaml", protocol=protocol
    )[Protocol13Benchmark.HOTPOT_QA]


def _contract() -> ModelServiceContract:
    return ModelServiceContract(
        profile_id="sglang-qwen35-base@3",
        served_model_name="qwen35-direct-base",
        model_repo_id="Qwen/Qwen3.5-9B",
        model_revision="release",
        tokenizer_repo_id="Qwen/Qwen3.5-9B",
        tokenizer_revision="release",
        chat_template_profile="qwen3.5-native-thinking-switch@1",
        context_length=98_304,
        dtype="bfloat16",
        reasoning_parser="qwen3",
        tool_call_parser=None,
        adapter_policy="forbidden",
    )


def _receipt(instance: str) -> ModelServiceReceipt:
    contract = _contract()
    return ModelServiceReceipt(
        service_instance_id=instance,
        service_profile_id=contract.profile_id,
        served_model_name=contract.served_model_name,
        model_repo_id=contract.model_repo_id,
        model_revision=contract.model_revision,
        tokenizer_repo_id=contract.tokenizer_repo_id,
        tokenizer_revision=contract.tokenizer_revision,
        chat_template_profile=contract.chat_template_profile,
        context_length=contract.context_length,
        dtype=contract.dtype,
        tensor_parallel_size=1,
        reasoning_parser=contract.reasoning_parser,
        tool_call_parser=contract.tool_call_parser,
        adapter_paths=(),
        lora_modules=(),
        launch_code_revision="sglang-release",
        started_at="2026-08-30T00:00:00Z",
    )


def test_equivalent_service_replicas_validate_with_distinct_instance_ids() -> None:
    validate_replica_services(
        (_receipt("replica-0"), _receipt("replica-1")),
        contract=_contract(),
        execution=_execution(),
    )


def test_service_receipt_rejects_any_active_lora() -> None:
    with pytest.raises(ValueError):
        replace(_receipt("replica-0"), lora_modules=("forbidden",))
