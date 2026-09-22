"""Create a new seed-zero preparation, never resume or repair an old adapter.

The input is the existing backbone declaration JSON (not preparation.json).
The output retains the original preparation@1 reader contract. An independent
CPU mapping beside preparation.json is for the actual fresh-start comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from skillev.contracts import canonical_json
from skillev.policy import (
    PrivateInitialCheckpointBinding,
    QwenMultimodalBackboneConfig,
    build_qwen_policy_backbone,
)

from .protocol_v13_training_debug import _PREPARATION_FORMAT

INITIAL_NAMED_PARAMETERS_FILE = "initial_named_parameters.pt"


def prepare_fresh_restart(
    *, backbone_config: QwenMultimodalBackboneConfig, output_root: Path, seed: int = 0
) -> Path:
    """Build BF16 base + new F/B LoRA + declared Z; return preparation.json.

    No old adapter/checkpoint input exists. The same production builder and
    checkpoint binding used by the original prepare own all initialization.
    Failed preparation leaves its directory intact and cannot be retried over it.
    """
    if type(seed) is not int or seed != 0 or backbone_config.z_initialization.initial_seed != 0:
        raise ValueError("fresh restart requires the declared single seed zero")
    if type(backbone_config) is not QwenMultimodalBackboneConfig:
        raise TypeError("fresh restart requires the original Qwen multimodal declaration")
    if backbone_config.torch_dtype != "bfloat16":
        raise ValueError("fresh restart requires original BF16 base precision")
    base = Path(backbone_config.base_model_path)
    # Prevent the observed old-adapter-as-preparation failure, not arbitrary
    # provenance certification. The builder has no load_checkpoint call here.
    if any((base / name).exists() for name in ("adapter_config.json", "policy_state.json")):
        raise ValueError("base_model_path is an adapter/checkpoint, not the original base")
    if backbone_config.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CUDA preparation requires explicit selected device visibility")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    checkpoint_directory = output_root / "initial-policy"
    torch.manual_seed(seed)
    backbone = build_qwen_policy_backbone(backbone_config)
    initial_state = backbone.trainable_state_identity
    backbone.bind_initial_trainable_state(initial_state)
    backbone.save_checkpoint(str(checkpoint_directory))
    named = {
        name: parameter.detach().to(device="cpu", copy=True)
        for name, parameter in backbone.named_trainable_parameters().items()
    }
    if not named:
        raise ValueError("fresh backbone has no named trainable parameters")
    mapping_path = output_root / INITIAL_NAMED_PARAMETERS_FILE
    with mapping_path.open("xb") as stream:
        torch.save(named, stream)
        stream.flush()
        os.fsync(stream.fileno())
    mapping_path.chmod(0o600)
    binding = PrivateInitialCheckpointBinding(
        directory=str(checkpoint_directory), trainable_state=initial_state
    )
    preparation = output_root / "preparation.json"
    with preparation.open("x", encoding="utf-8") as stream:
        stream.write(
            canonical_json(
                {
                    "backbone": backbone_config.to_value(),
                    "format": _PREPARATION_FORMAT,
                    "initial_checkpoint": binding.to_value(),
                }
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    preparation.chmod(0o600)
    return preparation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = QwenMultimodalBackboneConfig.from_value(
        json.loads(arguments.backbone_config.read_text(encoding="utf-8"))
    )
    path = prepare_fresh_restart(backbone_config=config, output_root=arguments.output_root)
    print(json.dumps({"preparation": str(path), "status": "prepared"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
