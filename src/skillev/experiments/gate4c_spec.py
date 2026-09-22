"""Immutable identity for the one-shot production-shape Gate 4c."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256
from skillev.experiments.build_identity import sha256_file
from skillev.policy import (
    BaseModelArtifactIdentity,
    TokenizerArtifactIdentity,
    TrainableStateIdentity,
)

from .execution_hardware import ExecutionHardwareIdentity
from .gate4c_execution import (
    GATE_4C_CHILD_TERMINAL_FORMAT,
    GATE_4C_OPERATOR_TERMINAL_FORMAT,
    GATE_4C_STAGE_EVENT_FORMAT,
)

GATE_4C_SPEC_FORMAT: Final = "skillev-gate-4c-spec@6"
DETERMINISTIC_BACKEND_IDENTITY_FORMAT: Final = "skillev-deterministic-backend@2"
GATE_4C_PROCEDURE_VERSION: Final = "skillev-real-9b-gate-4c@6"
GATE_4C_OPERATOR_VERSION: Final = "skillev-gate-4c-one-shot-operator@1"


def checkpoint_tree_hash(root: Path) -> str:
    """Commit every regular file in one immutable trainable checkpoint."""

    if not root.is_dir():
        raise NotADirectoryError(root)
    files: list[dict[str, JsonValue]] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        names[:] = sorted(names)
        directory_path = Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            if path.is_symlink():
                raise ValueError("Gate 4c checkpoint cannot contain symbolic links")
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
    if not files:
        raise ValueError("Gate 4c checkpoint tree cannot be empty")
    return stable_hash({"files": files, "format": "skillev-gate-4c-checkpoint-tree@1"})


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _source_commit(value: object) -> str:
    text = _text(value, field="source_commit")
    if len(text) not in {40, 64} or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("source_commit must be a lowercase Git object ID")
    return text


@dataclass(frozen=True, slots=True)
class DeterministicBackendIdentity:
    """Runtime switches that affect deterministic CUDA execution."""

    torch_version: str
    transformers_version: str
    peft_version: str
    flash_linear_attention_version: str
    tokenizers_version: str
    safetensors_version: str
    attention_implementation: str
    cublas_workspace_config: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    format: str = DETERMINISTIC_BACKEND_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        for field in (
            "torch_version",
            "transformers_version",
            "peft_version",
            "flash_linear_attention_version",
            "tokenizers_version",
            "safetensors_version",
            "attention_implementation",
            "cublas_workspace_config",
        ):
            _text(getattr(self, field), field=field)
        for field in (
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be boolean")
        if self.format != DETERMINISTIC_BACKEND_IDENTITY_FORMAT:
            raise ValueError("unsupported deterministic backend identity format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attention_implementation": self.attention_implementation,
            "cublas_workspace_config": self.cublas_workspace_config,
            "cuda_matmul_allow_tf32": self.cuda_matmul_allow_tf32,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cudnn_deterministic": self.cudnn_deterministic,
            "deterministic_algorithms": self.deterministic_algorithms,
            "format": self.format,
            "peft_version": self.peft_version,
            "flash_linear_attention_version": self.flash_linear_attention_version,
            "safetensors_version": self.safetensors_version,
            "tokenizers_version": self.tokenizers_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
        }

    @classmethod
    def from_value(cls, value: object) -> DeterministicBackendIdentity:
        normalized = normalize_json(value)
        fields = {
            "attention_implementation",
            "cublas_workspace_config",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
            "format",
            "peft_version",
            "flash_linear_attention_version",
            "safetensors_version",
            "tokenizers_version",
            "torch_version",
            "transformers_version",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("deterministic backend identity has incompatible fields")
        bool_fields = {
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        }
        if any(type(normalized[field]) is not bool for field in bool_fields):
            raise TypeError("deterministic backend flags must be boolean")
        if any(type(normalized[field]) is not str for field in fields - bool_fields):
            raise TypeError("deterministic backend text fields must be text")
        return cls(
            torch_version=normalized["torch_version"],
            transformers_version=normalized["transformers_version"],
            peft_version=normalized["peft_version"],
            flash_linear_attention_version=normalized["flash_linear_attention_version"],
            tokenizers_version=normalized["tokenizers_version"],
            safetensors_version=normalized["safetensors_version"],
            attention_implementation=normalized["attention_implementation"],
            cublas_workspace_config=normalized["cublas_workspace_config"],
            deterministic_algorithms=normalized["deterministic_algorithms"],
            cudnn_deterministic=normalized["cudnn_deterministic"],
            cudnn_benchmark=normalized["cudnn_benchmark"],
            cuda_matmul_allow_tf32=normalized["cuda_matmul_allow_tf32"],
            cudnn_allow_tf32=normalized["cudnn_allow_tf32"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class Gate4cSpec:
    """Path-free frozen inputs for one Gate 4c @6 execution."""

    source_commit: str
    source_tree_hash: str
    source_archive_sha256: str
    public_wheel_sha256: str
    private_wheel_sha256: str
    lockfile_sha256: str
    base_model_artifact: BaseModelArtifactIdentity
    tokenizer_artifact: TokenizerArtifactIdentity
    qwen_deployment_hash: str
    initial_trainable_state: TrainableStateIdentity
    initial_checkpoint_tree_hash: str
    initial_checkpoint_inspection_hash: str
    expected_hardware_identity: ExecutionHardwareIdentity
    deterministic_backend_identity: DeterministicBackendIdentity
    gate_procedure_version: str
    operator_version: str
    stage_journal_version: str
    child_terminal_version: str
    operator_terminal_version: str
    gpu_sampling_interval_ms: int
    expected_physical_gpu_uuid: str
    production_config_hash: str
    seed: int
    initialization_seed: int
    horizon: int
    maximum_h0_tokens: int
    max_reasoning_tokens: int
    max_action_tokens: int
    model_input_cap: int
    peak_reserved_cap_bytes: int
    eos_token_id: int
    format: str = GATE_4C_SPEC_FORMAT

    def __post_init__(self) -> None:
        _source_commit(self.source_commit)
        for field in (
            "source_tree_hash",
            "source_archive_sha256",
            "public_wheel_sha256",
            "private_wheel_sha256",
            "lockfile_sha256",
            "qwen_deployment_hash",
            "initial_checkpoint_tree_hash",
            "initial_checkpoint_inspection_hash",
            "production_config_hash",
        ):
            validate_sha256(getattr(self, field))
        if not isinstance(self.base_model_artifact, BaseModelArtifactIdentity):
            raise TypeError("Gate 4c spec requires a base-model artifact identity")
        if not isinstance(self.tokenizer_artifact, TokenizerArtifactIdentity):
            raise TypeError("Gate 4c spec requires a tokenizer artifact identity")
        if not isinstance(self.initial_trainable_state, TrainableStateIdentity):
            raise TypeError("Gate 4c spec requires an initial trainable-state identity")
        if not isinstance(self.expected_hardware_identity, ExecutionHardwareIdentity):
            raise TypeError("Gate 4c spec requires an execution hardware identity")
        if not isinstance(self.deterministic_backend_identity, DeterministicBackendIdentity):
            raise TypeError("Gate 4c spec requires a deterministic backend identity")
        expected_versions = {
            "gate_procedure_version": GATE_4C_PROCEDURE_VERSION,
            "operator_version": GATE_4C_OPERATOR_VERSION,
            "stage_journal_version": GATE_4C_STAGE_EVENT_FORMAT,
            "child_terminal_version": GATE_4C_CHILD_TERMINAL_FORMAT,
            "operator_terminal_version": GATE_4C_OPERATOR_TERMINAL_FORMAT,
        }
        if any(getattr(self, field) != expected for field, expected in expected_versions.items()):
            raise ValueError("Gate 4c execution contract versions differ from procedure @6")
        if type(self.gpu_sampling_interval_ms) is not int or self.gpu_sampling_interval_ms < 1:
            raise ValueError("Gate 4c GPU sampling interval must be positive")
        if not self.expected_physical_gpu_uuid.startswith("GPU-"):
            raise ValueError("Gate 4c expected physical GPU UUID is invalid")
        for field in (
            "seed",
            "initialization_seed",
            "horizon",
            "maximum_h0_tokens",
            "max_reasoning_tokens",
            "max_action_tokens",
            "model_input_cap",
            "peak_reserved_cap_bytes",
        ):
            _positive_int(getattr(self, field), field=field)
        if type(self.eos_token_id) is not int or self.eos_token_id < 0:
            raise ValueError("eos_token_id must be a non-negative integer")
        if self.initial_trainable_state.backbone_deployment_hash != self.qwen_deployment_hash:
            raise ValueError("initial trainable state targets another Qwen deployment")
        if self.tokenizer_artifact.revision != self.base_model_artifact.upstream_revision:
            raise ValueError("model and tokenizer revisions differ")
        if self.format != GATE_4C_SPEC_FORMAT:
            raise ValueError("unsupported Gate 4c spec format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "base_model_artifact": self.base_model_artifact.to_value(),
            "deterministic_backend_identity": self.deterministic_backend_identity.to_value(),
            "eos_token_id": self.eos_token_id,
            "expected_hardware_identity": self.expected_hardware_identity.to_value(),
            "format": self.format,
            "gate_procedure_version": self.gate_procedure_version,
            "gpu_sampling_interval_ms": self.gpu_sampling_interval_ms,
            "horizon": self.horizon,
            "initial_checkpoint_inspection_hash": self.initial_checkpoint_inspection_hash,
            "initial_checkpoint_tree_hash": self.initial_checkpoint_tree_hash,
            "initial_trainable_state": self.initial_trainable_state.to_value(),
            "initialization_seed": self.initialization_seed,
            "lockfile_sha256": self.lockfile_sha256,
            "max_action_tokens": self.max_action_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "model_input_cap": self.model_input_cap,
            "peak_reserved_cap_bytes": self.peak_reserved_cap_bytes,
            "operator_terminal_version": self.operator_terminal_version,
            "operator_version": self.operator_version,
            "private_wheel_sha256": self.private_wheel_sha256,
            "production_config_hash": self.production_config_hash,
            "public_wheel_sha256": self.public_wheel_sha256,
            "qwen_deployment_hash": self.qwen_deployment_hash,
            "seed": self.seed,
            "source_archive_sha256": self.source_archive_sha256,
            "source_commit": self.source_commit,
            "source_tree_hash": self.source_tree_hash,
            "stage_journal_version": self.stage_journal_version,
            "tokenizer_artifact": self.tokenizer_artifact.to_value(),
            "child_terminal_version": self.child_terminal_version,
            "expected_physical_gpu_uuid": self.expected_physical_gpu_uuid,
        }

    @classmethod
    def from_value(cls, value: object) -> Gate4cSpec:
        normalized = normalize_json(value)
        fields = {
            "base_model_artifact",
            "deterministic_backend_identity",
            "eos_token_id",
            "expected_hardware_identity",
            "format",
            "gate_procedure_version",
            "gpu_sampling_interval_ms",
            "horizon",
            "initial_checkpoint_inspection_hash",
            "initial_checkpoint_tree_hash",
            "initial_trainable_state",
            "initialization_seed",
            "lockfile_sha256",
            "max_action_tokens",
            "max_reasoning_tokens",
            "maximum_h0_tokens",
            "model_input_cap",
            "peak_reserved_cap_bytes",
            "operator_terminal_version",
            "operator_version",
            "private_wheel_sha256",
            "production_config_hash",
            "public_wheel_sha256",
            "qwen_deployment_hash",
            "seed",
            "source_archive_sha256",
            "source_commit",
            "source_tree_hash",
            "stage_journal_version",
            "tokenizer_artifact",
            "child_terminal_version",
            "expected_physical_gpu_uuid",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Gate 4c spec has incompatible fields")
        integer_fields = {
            "eos_token_id",
            "gpu_sampling_interval_ms",
            "horizon",
            "initialization_seed",
            "max_action_tokens",
            "max_reasoning_tokens",
            "maximum_h0_tokens",
            "model_input_cap",
            "peak_reserved_cap_bytes",
            "seed",
        }
        if any(type(normalized[field]) is not int for field in integer_fields):
            raise TypeError("Gate 4c numeric fields must be integers")
        text_fields = (
            fields
            - integer_fields
            - {
                "base_model_artifact",
                "deterministic_backend_identity",
                "expected_hardware_identity",
                "initial_trainable_state",
                "tokenizer_artifact",
            }
        )
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("Gate 4c identity fields must be text")
        return cls(
            source_commit=normalized["source_commit"],
            source_tree_hash=normalized["source_tree_hash"],
            source_archive_sha256=normalized["source_archive_sha256"],
            public_wheel_sha256=normalized["public_wheel_sha256"],
            private_wheel_sha256=normalized["private_wheel_sha256"],
            lockfile_sha256=normalized["lockfile_sha256"],
            base_model_artifact=BaseModelArtifactIdentity.from_value(
                normalized["base_model_artifact"]
            ),
            tokenizer_artifact=TokenizerArtifactIdentity.from_value(
                normalized["tokenizer_artifact"]
            ),
            qwen_deployment_hash=normalized["qwen_deployment_hash"],
            initial_trainable_state=TrainableStateIdentity.from_value(
                normalized["initial_trainable_state"]
            ),
            initial_checkpoint_inspection_hash=normalized["initial_checkpoint_inspection_hash"],
            initial_checkpoint_tree_hash=normalized["initial_checkpoint_tree_hash"],
            expected_hardware_identity=ExecutionHardwareIdentity.from_value(
                normalized["expected_hardware_identity"]
            ),
            deterministic_backend_identity=DeterministicBackendIdentity.from_value(
                normalized["deterministic_backend_identity"]
            ),
            gate_procedure_version=normalized["gate_procedure_version"],
            operator_version=normalized["operator_version"],
            stage_journal_version=normalized["stage_journal_version"],
            child_terminal_version=normalized["child_terminal_version"],
            operator_terminal_version=normalized["operator_terminal_version"],
            gpu_sampling_interval_ms=normalized["gpu_sampling_interval_ms"],
            expected_physical_gpu_uuid=normalized["expected_physical_gpu_uuid"],
            production_config_hash=normalized["production_config_hash"],
            seed=normalized["seed"],
            initialization_seed=normalized["initialization_seed"],
            horizon=normalized["horizon"],
            maximum_h0_tokens=normalized["maximum_h0_tokens"],
            max_reasoning_tokens=normalized["max_reasoning_tokens"],
            max_action_tokens=normalized["max_action_tokens"],
            model_input_cap=normalized["model_input_cap"],
            peak_reserved_cap_bytes=normalized["peak_reserved_cap_bytes"],
            eos_token_id=normalized["eos_token_id"],
            format=normalized["format"],
        )


__all__ = [
    "DETERMINISTIC_BACKEND_IDENTITY_FORMAT",
    "GATE_4C_OPERATOR_VERSION",
    "GATE_4C_PROCEDURE_VERSION",
    "GATE_4C_SPEC_FORMAT",
    "DeterministicBackendIdentity",
    "Gate4cSpec",
    "checkpoint_tree_hash",
]
