"""Runtime-owned model-service identities for Protocol 13.

The receipt records resolved launch semantics.  It deliberately does not hash
model files and does not infer backbone-only status from the public route name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .contracts import ExecutionContractV3


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = tuple(_text(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    return result


@dataclass(frozen=True, slots=True)
class ModelServiceContract:
    profile_id: str
    served_model_name: str
    model_repo_id: str
    model_revision: str
    tokenizer_repo_id: str
    tokenizer_revision: str
    chat_template_profile: str
    context_length: int
    dtype: str
    reasoning_parser: str | None
    tool_call_parser: str | None
    adapter_policy: str

    def __post_init__(self) -> None:
        required = (
            self.profile_id,
            self.served_model_name,
            self.model_repo_id,
            self.model_revision,
            self.tokenizer_repo_id,
            self.tokenizer_revision,
            self.chat_template_profile,
            self.dtype,
        )
        if any(not item.strip() for item in required):
            raise ValueError("model service contract is incomplete")
        if self.context_length <= 0 or self.adapter_policy != "forbidden":
            raise ValueError("model service contract must be positive and adapter-free")


@dataclass(frozen=True, slots=True)
class ModelServiceReceipt:
    service_instance_id: str
    service_profile_id: str
    served_model_name: str
    model_repo_id: str
    model_revision: str
    tokenizer_repo_id: str
    tokenizer_revision: str
    chat_template_profile: str
    context_length: int
    dtype: str
    tensor_parallel_size: int
    reasoning_parser: str | None
    tool_call_parser: str | None
    adapter_paths: tuple[str, ...]
    lora_modules: tuple[str, ...]
    launch_code_revision: str
    started_at: str
    format_version: str = "skillev-model-service-receipt@1"

    def __post_init__(self) -> None:
        required = (
            self.service_instance_id,
            self.service_profile_id,
            self.served_model_name,
            self.model_repo_id,
            self.model_revision,
            self.tokenizer_repo_id,
            self.tokenizer_revision,
            self.chat_template_profile,
            self.dtype,
            self.launch_code_revision,
            self.started_at,
        )
        if any(not item.strip() for item in required):
            raise ValueError("model service receipt is incomplete")
        if self.format_version != "skillev-model-service-receipt@1":
            raise ValueError("unsupported model service receipt")
        if self.context_length <= 0 or self.tensor_parallel_size <= 0:
            raise ValueError("model service capacity is invalid")
        if self.adapter_paths or self.lora_modules:
            raise ValueError("backbone-only service exposes an adapter")

    def validate_against(
        self,
        contract: ModelServiceContract,
        *,
        execution: ExecutionContractV3,
    ) -> None:
        observed = (
            self.service_profile_id,
            self.served_model_name,
            self.model_repo_id,
            self.model_revision,
            self.tokenizer_repo_id,
            self.tokenizer_revision,
            self.chat_template_profile,
            self.context_length,
            self.dtype,
            self.reasoning_parser,
            self.tool_call_parser,
        )
        expected = (
            contract.profile_id,
            contract.served_model_name,
            contract.model_repo_id,
            contract.model_revision,
            contract.tokenizer_repo_id,
            contract.tokenizer_revision,
            contract.chat_template_profile,
            contract.context_length,
            contract.dtype,
            contract.reasoning_parser,
            contract.tool_call_parser,
        )
        if observed != expected:
            raise ValueError("runtime service differs from the frozen service contract")
        if (
            contract.adapter_policy != "forbidden"
            or execution.adapter_policy != "forbidden"
            or self.served_model_name != execution.actor_route
            or self.service_profile_id != execution.actor_service_profile
            or self.context_length != execution.context_length
        ):
            raise ValueError("runtime service differs from the Protocol 13 execution")

    @property
    def semantic_identity(self) -> tuple[object, ...]:
        return (
            self.service_profile_id,
            self.served_model_name,
            self.model_repo_id,
            self.model_revision,
            self.tokenizer_repo_id,
            self.tokenizer_revision,
            self.chat_template_profile,
            self.context_length,
            self.dtype,
            self.reasoning_parser,
            self.tool_call_parser,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": self.format_version,
            "service_instance_id": self.service_instance_id,
            "service_profile_id": self.service_profile_id,
            "served_model_name": self.served_model_name,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "tokenizer_repo_id": self.tokenizer_repo_id,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_profile": self.chat_template_profile,
            "context_length": self.context_length,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "reasoning_parser": self.reasoning_parser,
            "tool_call_parser": self.tool_call_parser,
            "adapter_paths": list(self.adapter_paths),
            "lora_modules": list(self.lora_modules),
            "launch_code_revision": self.launch_code_revision,
            "started_at": self.started_at,
        }


def load_model_service_contract(path: Path) -> ModelServiceContract:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-model-service-contract@1":
        raise ValueError("unsupported model service contract")
    expected = {
        "format",
        "profile_id",
        "served_model_name",
        "model_repo_id",
        "model_revision",
        "tokenizer_repo_id",
        "tokenizer_revision",
        "chat_template_profile",
        "context_length",
        "dtype",
        "reasoning_parser",
        "tool_call_parser",
        "adapter_policy",
    }
    if set(root) != expected:
        raise ValueError("model service contract fields differ")
    return ModelServiceContract(
        profile_id=_text(root["profile_id"], "profile ID"),
        served_model_name=_text(root["served_model_name"], "served model name"),
        model_repo_id=_text(root["model_repo_id"], "model repository"),
        model_revision=_text(root["model_revision"], "model revision"),
        tokenizer_repo_id=_text(root["tokenizer_repo_id"], "tokenizer repository"),
        tokenizer_revision=_text(root["tokenizer_revision"], "tokenizer revision"),
        chat_template_profile=_text(root["chat_template_profile"], "chat template profile"),
        context_length=_positive_integer(root["context_length"], "context length"),
        dtype=_text(root["dtype"], "dtype"),
        reasoning_parser=_optional_text(root["reasoning_parser"], "reasoning parser"),
        tool_call_parser=_optional_text(root["tool_call_parser"], "tool call parser"),
        adapter_policy=_text(root["adapter_policy"], "adapter policy"),
    )


def load_model_service_receipt(path: Path) -> ModelServiceReceipt:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("model service receipt must be an object")
    expected = {
        "format",
        "service_instance_id",
        "service_profile_id",
        "served_model_name",
        "model_repo_id",
        "model_revision",
        "tokenizer_repo_id",
        "tokenizer_revision",
        "chat_template_profile",
        "context_length",
        "dtype",
        "tensor_parallel_size",
        "reasoning_parser",
        "tool_call_parser",
        "adapter_paths",
        "lora_modules",
        "launch_code_revision",
        "started_at",
    }
    if set(root) != expected:
        raise ValueError("model service receipt fields differ")
    return ModelServiceReceipt(
        format_version=_text(root["format"], "format"),
        service_instance_id=_text(root["service_instance_id"], "service instance ID"),
        service_profile_id=_text(root["service_profile_id"], "service profile ID"),
        served_model_name=_text(root["served_model_name"], "served model name"),
        model_repo_id=_text(root["model_repo_id"], "model repository"),
        model_revision=_text(root["model_revision"], "model revision"),
        tokenizer_repo_id=_text(root["tokenizer_repo_id"], "tokenizer repository"),
        tokenizer_revision=_text(root["tokenizer_revision"], "tokenizer revision"),
        chat_template_profile=_text(root["chat_template_profile"], "chat template profile"),
        context_length=_positive_integer(root["context_length"], "context length"),
        dtype=_text(root["dtype"], "dtype"),
        tensor_parallel_size=_positive_integer(
            root["tensor_parallel_size"], "tensor parallel size"
        ),
        reasoning_parser=_optional_text(root["reasoning_parser"], "reasoning parser"),
        tool_call_parser=_optional_text(root["tool_call_parser"], "tool call parser"),
        adapter_paths=_strings(root["adapter_paths"], "adapter paths"),
        lora_modules=_strings(root["lora_modules"], "LoRA modules"),
        launch_code_revision=_text(root["launch_code_revision"], "launch code revision"),
        started_at=_text(root["started_at"], "start time"),
    )


def write_model_service_receipt(path: Path, receipt: ModelServiceReceipt) -> None:
    if path.exists():
        raise FileExistsError(f"service receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.to_mapping(), indent=2) + "\n", encoding="utf-8")


def validate_replica_services(
    receipts: tuple[ModelServiceReceipt, ...],
    *,
    contract: ModelServiceContract,
    execution: ExecutionContractV3,
) -> None:
    if not receipts:
        raise ValueError("at least one service receipt is required")
    instance_ids = tuple(item.service_instance_id for item in receipts)
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("replica service instance IDs must be unique")
    for receipt in receipts:
        receipt.validate_against(contract, execution=execution)
    if len({item.semantic_identity for item in receipts}) != 1:
        raise ValueError("replica services are not semantically equivalent")


__all__ = [
    "ModelServiceContract",
    "ModelServiceReceipt",
    "load_model_service_contract",
    "load_model_service_receipt",
    "validate_replica_services",
    "write_model_service_receipt",
]
