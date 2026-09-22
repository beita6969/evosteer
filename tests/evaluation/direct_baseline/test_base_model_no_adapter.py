from pathlib import Path

import pytest

from scripts.launch_direct_sglang import build_command, load_config


def _environment() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "5",
        "SKILLEV_MODEL_PATH": "/private/base-model",
        "SKILLEV_TOKENIZER_PATH": "/private/base-model",
    }


def test_direct_server_command_has_no_adapter_or_lora_flags() -> None:
    config = load_config(Path("configs/serving/qwen35_9b_direct_reference.yaml"))
    command = build_command(config, _environment())
    rendered = " ".join(command).lower()
    assert "adapter" not in rendered
    assert "lora" not in rendered
    assert "--enable-deterministic-inference" in command
    assert "--reasoning-parser" in command


def test_direct_server_requires_one_explicit_physical_gpu() -> None:
    config = load_config(Path("configs/serving/qwen35_9b_direct_reference.yaml"))
    with pytest.raises(RuntimeError):
        build_command(config, {**_environment(), "CUDA_VISIBLE_DEVICES": "4,5"})


def test_direct_server_accepts_one_explicit_physical_gpu_uuid() -> None:
    config = load_config(Path("configs/serving/qwen35_9b_direct_reference.yaml"))
    environment = {
        **_environment(),
        "CUDA_VISIBLE_DEVICES": "GPU-12345678-abcd-4321-abcd-1234567890ab",
    }
    assert build_command(config, environment)
