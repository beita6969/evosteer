#!/usr/bin/env python3
"""Create the deterministic CPU-initialized LoRA/Z checkpoint for Gate 4c @6."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scripts.gate4c_runtime_support import INITIALIZATION_SEED
from skillev.contracts import canonical_json
from skillev.policy import (
    BaseModelArtifactIdentity,
    QwenMultimodalBackboneConfig,
    QwenMultimodalPolicyBackbone,
    qwen_tokenizer_artifact_identity,
)
from skillev.policy.config import (
    DEFAULT_LORA_TARGET_MODULES,
    qwen_backend_class,
    qwen_dtype_conversion_policy,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity-directory", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    model_path = Path(args.local_model_path).resolve()
    output = Path(args.output).resolve()
    identity_directory = Path(args.identity_directory).resolve()
    if output.exists() or identity_directory.exists():
        raise FileExistsError(output if output.exists() else identity_directory)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = raw.get("text_config")
    if not isinstance(text_config, dict) or type(text_config.get("hidden_size")) is not int:
        raise ValueError("pinned Qwen config lacks text hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=args.revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Gate 4c requires the pinned fast tokenizer")
    tokenizer_identity = qwen_tokenizer_artifact_identity(
        tokenizer=tokenizer,
        tokenizer_id="Qwen/Qwen3.5-9B",
        revision=args.revision,
    )
    provisional = QwenMultimodalBackboneConfig(
        base_model_path=str(model_path),
        revision=args.revision,
        tokenizer_id=tokenizer_identity.tokenizer_id,
        tokenizer_content_hash=tokenizer_identity.content_hash,
        hidden_size=text_config["hidden_size"],
        device="cpu",
        torch_dtype="bfloat16",
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        z_hidden_width=32,
        eos_token_ids=(args.eos_token_id,),
        teacher_forced_gradient_checkpointing=True,
        attention_implementation="sdpa",
    )
    base_identity = BaseModelArtifactIdentity.from_directory(
        directory=model_path,
        backend_class=qwen_backend_class(provisional),
        upstream_revision=args.revision,
        dtype_conversion_policy=qwen_dtype_conversion_policy(provisional),
    )
    config = replace(provisional, base_model_artifact=base_identity)
    torch.manual_seed(INITIALIZATION_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    backbone = QwenMultimodalPolicyBackbone(config)
    backbone.bind_initial_trainable_state(backbone.trainable_state_identity)
    backbone.save_checkpoint(str(output))
    identity_directory.mkdir(parents=True)
    values = {
        "base-model-artifact.json": base_identity.to_value(),
        "initial-trainable-state.json": backbone.trainable_state_identity.to_value(),
        "tokenizer-artifact.json": tokenizer_identity.to_value(),
    }
    for name, value in values.items():
        (identity_directory / name).write_text(canonical_json(value) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
