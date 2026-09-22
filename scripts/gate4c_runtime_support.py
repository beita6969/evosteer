"""Identity and deployment support for the fixed Gate 4c @6 procedure."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Final

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import skillev
from skillev.calibration import CalibrationConfig
from skillev.contracts import stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.experiments import (
    DeterministicBackendIdentity,
    ExecutionHardwareIdentity,
    Gate4cSpec,
    checkpoint_tree_hash,
)
from skillev.experiments.build_identity import (
    require_source_archive_matches_execution,
    sha256_file,
)
from skillev.policy import (
    QWEN35_GATED_DELTA_KERNEL_PACKAGE,
    QWEN35_GATED_DELTA_KERNEL_VERSION,
    TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS,
    QwenMultimodalBackboneConfig,
    QwenMultimodalPolicyBackbone,
    qwen_tokenizer_artifact_identity,
)
from skillev.policy.checkpoint import PolicyCheckpointInspection, inspect_policy_checkpoint
from skillev.policy.config import DEFAULT_LORA_TARGET_MODULES
from skillev.training.config import OptimizerConfig

GATE_4C_FORMAT: Final = "skillev-real-9b-gate-4c@6"
GATE_4C_FIXTURE_FORMAT: Final = "skillev-gate-4c-public-synthetic@5"
GATE_4C_ENVIRONMENT_ID: Final = "gate-4c-environment"
GATE_4C_TASK_FAMILY: Final = "gate-4c-public-synthetic"
GATE_4C_CONTEXT_ID: Final = "gate-4c-public-synthetic"
GATE_SEED: Final = 20_260_810
INITIALIZATION_SEED: Final = 20_260_811
HORIZON: Final = 15
MAX_REASONING_TOKENS: Final = 768
MAX_ACTION_TOKENS: Final = 768
MAXIMUM_H0_TOKENS: Final = 32_768
MODEL_INPUT_CAP: Final = 65_536
PEAK_RESERVED_LIMIT_BYTES: Final = 64 * 1024**3
CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"
SOURCE_ARCHIVE_NAME: Final = "gate-4c-source.tar.gz"
PUBLIC_WHEEL_NAME: Final = "gate-4c-public.whl"
PRIVATE_WHEEL_NAME: Final = "gate-4c-private.whl"
INITIAL_POLICY_DIRECTORY_NAME: Final = "gate-4c-initial-policy"


def production_config_hash() -> str:
    return stable_hash(
        {
            "attention_implementation": "sdpa",
            "calibration": CalibrationConfig().to_value(),
            "diagnostics": DiagnosticsConfig().to_value(),
            "format": GATE_4C_FORMAT,
            "gradient_checkpointing": {"teacher_forced": True, "use_reentrant": False},
            "h0": MAXIMUM_H0_TOKENS,
            "horizon": HORIZON,
            "model_input_cap": MODEL_INPUT_CAP,
            "optimizer": OptimizerConfig(
                adapter_learning_rate=1e-5, z_learning_rate=1e-4
            ).to_value(),
            "rollout": {
                "action": MAX_ACTION_TOKENS,
                "reasoning": MAX_REASONING_TOKENS,
                "tail_reuse_tokens": 64,
            },
            "scoring": {"temperature_beta": 1.0},
            "teacher_forced_activation_offload_min_tokens": (
                TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS
            ),
            "qwen35_gated_delta_kernel": {
                "package": QWEN35_GATED_DELTA_KERNEL_PACKAGE,
                "version": QWEN35_GATED_DELTA_KERNEL_VERSION,
            },
        }
    )


def _driver_version() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("nvidia-smi")
    completed = subprocess.run(  # noqa: S603 -- fixed arguments to resolved binary
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise RuntimeError("Gate 4c requires one visible driver version")
    return versions.pop()


def hardware_identity() -> ExecutionHardwareIdentity:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Gate 4c requires exactly one explicitly visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    nccl = getattr(torch.cuda, "nccl", None)
    nccl_version = nccl.version() if nccl is not None else None
    if not cuda_version or not cudnn_version or not nccl_version:
        raise RuntimeError("Gate 4c requires complete CUDA runtime identity")
    return ExecutionHardwareIdentity(
        accelerator_name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        visible_device_count=1,
        nvidia_driver_version=_driver_version(),
        cuda_runtime_version=cuda_version,
        cudnn_version=str(cudnn_version),
        nccl_version=".".join(str(item) for item in nccl_version),
        kernel_release=platform.release(),
        safetensors_version=importlib.metadata.version("safetensors"),
    )


def backend_identity() -> DeterministicBackendIdentity:
    return DeterministicBackendIdentity(
        torch_version=str(torch.__version__),
        transformers_version=importlib.metadata.version("transformers"),
        peft_version=importlib.metadata.version("peft"),
        flash_linear_attention_version=importlib.metadata.version(
            QWEN35_GATED_DELTA_KERNEL_PACKAGE
        ),
        tokenizers_version=importlib.metadata.version("tokenizers"),
        safetensors_version=importlib.metadata.version("safetensors"),
        attention_implementation="sdpa",
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )


def configure_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("Gate 4c requires the frozen CUBLAS_WORKSPACE_CONFIG")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def read_spec(path: Path) -> Gate4cSpec:
    spec = Gate4cSpec.from_value(json.loads(path.read_text(encoding="utf-8")))
    fixed = (
        (spec.seed, GATE_SEED, "seed"),
        (spec.initialization_seed, INITIALIZATION_SEED, "initialization seed"),
        (spec.horizon, HORIZON, "horizon"),
        (spec.maximum_h0_tokens, MAXIMUM_H0_TOKENS, "H0 cap"),
        (spec.max_reasoning_tokens, MAX_REASONING_TOKENS, "reasoning cap"),
        (spec.max_action_tokens, MAX_ACTION_TOKENS, "action cap"),
        (spec.model_input_cap, MODEL_INPUT_CAP, "model input cap"),
        (spec.peak_reserved_cap_bytes, PEAK_RESERVED_LIMIT_BYTES, "peak cap"),
    )
    for actual, expected, label in fixed:
        if actual != expected:
            raise ValueError(f"Gate 4c spec {label} differs from procedure @6")
    if spec.production_config_hash != production_config_hash():
        raise ValueError("Gate 4c production config differs from procedure @6")
    return spec


def verify_bundle(spec: Gate4cSpec, spec_path: Path, work: Path) -> Path:
    bundle = spec_path.resolve().parent
    source_archive = bundle / SOURCE_ARCHIVE_NAME
    public_wheel = bundle / PUBLIC_WHEEL_NAME
    private_wheel = bundle / PRIVATE_WHEEL_NAME
    initial_policy = bundle / INITIAL_POLICY_DIRECTORY_NAME
    executing_root = Path(__file__).resolve().parents[1]
    imported_root = Path(skillev.__file__).resolve().parent
    if imported_root != (executing_root / "src" / "skillev").resolve():
        raise RuntimeError("Gate 4c imported SKILLEV from another source tree")
    provenance = require_source_archive_matches_execution(
        source_archive=source_archive,
        executing_root=executing_root,
        temporary_parent=work,
    )
    if (provenance.source_commit, provenance.source_tree_hash) != (
        spec.source_commit,
        spec.source_tree_hash,
    ):
        raise ValueError("Gate 4c source provenance differs from spec")
    measured = {
        "source archive": sha256_file(source_archive),
        "public wheel": sha256_file(public_wheel),
        "private wheel": sha256_file(private_wheel),
        "lockfile": sha256_file(executing_root / "uv.lock"),
        "initial checkpoint": checkpoint_tree_hash(initial_policy),
    }
    expected = {
        "source archive": spec.source_archive_sha256,
        "public wheel": spec.public_wheel_sha256,
        "private wheel": spec.private_wheel_sha256,
        "lockfile": spec.lockfile_sha256,
        "initial checkpoint": spec.initial_checkpoint_tree_hash,
    }
    if measured != expected:
        raise ValueError("Gate 4c execution bundle bytes differ from spec")
    return initial_policy


def backbone_config(spec: Gate4cSpec, model_path: Path) -> QwenMultimodalBackboneConfig:
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = raw.get("text_config")
    if not isinstance(text_config, dict) or type(text_config.get("hidden_size")) is not int:
        raise ValueError("pinned Qwen config lacks text hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=spec.base_model_artifact.upstream_revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Gate 4c requires the pinned fast tokenizer")
    tokenizer_identity = qwen_tokenizer_artifact_identity(
        tokenizer=tokenizer,
        tokenizer_id=spec.tokenizer_artifact.tokenizer_id,
        revision=spec.tokenizer_artifact.revision,
    )
    if tokenizer_identity != spec.tokenizer_artifact:
        raise ValueError("local tokenizer bytes differ from Gate 4c spec")
    return QwenMultimodalBackboneConfig(
        base_model_path=str(model_path),
        revision=spec.base_model_artifact.upstream_revision,
        tokenizer_id=tokenizer_identity.tokenizer_id,
        tokenizer_content_hash=tokenizer_identity.content_hash,
        hidden_size=text_config["hidden_size"],
        device="cuda",
        torch_dtype="bfloat16",
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        z_hidden_width=32,
        eos_token_ids=(spec.eos_token_id,),
        teacher_forced_gradient_checkpointing=True,
        attention_implementation="sdpa",
        base_model_artifact=spec.base_model_artifact,
    )


def verify_initial_checkpoint(
    *, spec: Gate4cSpec, initial_policy: Path
) -> PolicyCheckpointInspection:
    inspection = inspect_policy_checkpoint(initial_policy)
    state = inspection.state
    if state.optimizer_step != 0:
        raise ValueError("Gate 4c initial checkpoint must be optimizer step zero")
    if state.backbone_id != spec.qwen_deployment_hash:
        raise ValueError("Gate 4c checkpoint targets another Qwen deployment")
    if state.trainable_state != spec.initial_trainable_state:
        raise ValueError("Gate 4c checkpoint metadata differs from frozen trainable state")
    if inspection.content_hash != spec.initial_checkpoint_inspection_hash:
        raise ValueError("Gate 4c checkpoint inspection differs from spec")
    return inspection


def construct_backbone(
    config: QwenMultimodalBackboneConfig,
    *,
    initialization_seed: int,
) -> QwenMultimodalPolicyBackbone:
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    return QwenMultimodalPolicyBackbone(config)


def load_initial_checkpoint(
    *,
    backbone: QwenMultimodalPolicyBackbone,
    initial_policy: Path,
    spec: Gate4cSpec,
) -> None:
    if backbone.backbone_id != spec.qwen_deployment_hash:
        raise ValueError("Qwen deployment differs from Gate 4c spec")
    backbone.load_checkpoint(str(initial_policy))


def bind_initial_state(
    *,
    backbone: QwenMultimodalPolicyBackbone,
    spec: Gate4cSpec,
) -> None:
    backbone.bind_initial_trainable_state(spec.initial_trainable_state)
    if backbone.trainable_state_identity != spec.initial_trainable_state:
        raise ValueError("loaded Gate 4c trainable state differs after binding")


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "GATE_4C_CONTEXT_ID",
    "GATE_4C_ENVIRONMENT_ID",
    "GATE_4C_FIXTURE_FORMAT",
    "GATE_4C_FORMAT",
    "GATE_4C_TASK_FAMILY",
    "GATE_SEED",
    "HORIZON",
    "INITIALIZATION_SEED",
    "INITIAL_POLICY_DIRECTORY_NAME",
    "MAXIMUM_H0_TOKENS",
    "MAX_ACTION_TOKENS",
    "MAX_REASONING_TOKENS",
    "MODEL_INPUT_CAP",
    "PEAK_RESERVED_LIMIT_BYTES",
    "PRIVATE_WHEEL_NAME",
    "PUBLIC_WHEEL_NAME",
    "SOURCE_ARCHIVE_NAME",
    "backbone_config",
    "backend_identity",
    "bind_initial_state",
    "configure_determinism",
    "construct_backbone",
    "hardware_identity",
    "load_initial_checkpoint",
    "production_config_hash",
    "read_spec",
    "verify_bundle",
    "verify_initial_checkpoint",
]
