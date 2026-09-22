from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.launch_sglang import _load, build_command, build_environment


def test_formal_service_requests_native_timing_without_changing_sampling(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b_bayesian_250.yaml")
    command = build_command(config)
    option = "--enable-request-time-stats-logging"
    assert option in command
    assert tuple(value for value in command if value != option) == build_command(
        {**config, "enable_request_time_stats_logging": False}
    )
    with pytest.raises(TypeError):
        build_command({**config, "enable_request_time_stats_logging": "true"})


def test_fresh_service_admits_largest_reasoning_without_reducing_input(monkeypatch):
    from skillev_private.experiments.fresh_restart import load_fresh_config

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b_fresh_restart.yaml")
    formal = load_fresh_config(Path("configs/training/bayesianimprove_fresh_restart.yaml"))
    command = build_command(config)
    capacity = int(command[command.index("--context-length") + 1])
    assert capacity >= formal.max_input_tokens + max(
        formal.maximum_reasoning_tokens, formal.max_action_tokens
    )
    assert "--skillev-fp32-mamba-checkpoints" in command
    assert "--disable-radix-cache" not in command
    assert command[command.index("--dtype") + 1] == formal.base_dtype
    assert int(command[command.index("--random-seed") + 1]) == formal.seed


@pytest.mark.parametrize(
    ("field", "value"),
    [("dtype", "auto"), ("dtype", "float16"), ("random_seed", True), ("random_seed", -1)],
)
def test_explicit_fresh_precision_and_seed_cannot_be_unresolved(monkeypatch, field, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b_fresh_restart.yaml")
    with pytest.raises(ValueError):
        build_command({**config, field: value})


def test_project_service_config_enables_qwen35_multimodal_processing(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/tokenizers/qwen35")

    command = build_command(_load("configs/serving/qwen35_9b.yaml"))

    assert "--enable-multimodal" in command
    assert "--enable-deterministic-inference" in command
    assert command[command.index("--sampling-backend") + 1] == "pytorch"
    assert command[command.index("--tokenizer-path") + 1] == "/tokenizers/qwen35"


def test_build_environment_exposes_environment_tools_and_matching_cuda(tmp_path: Path) -> None:
    environment_bin = tmp_path / "environment" / "bin"
    environment_bin.mkdir(parents=True)
    cuda_root = tmp_path / "toolkits"
    nvcc = cuda_root / "cuda-12.9" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()

    environment = build_environment(
        {"PATH": "/usr/bin"},
        executable=str(environment_bin / "python"),
        cuda_version="12.9",
        cuda_root=cuda_root,
    )

    assert environment["PATH"].split(os.pathsep) == [str(environment_bin), "/usr/bin"]
    assert environment["CUDA_HOME"] == str(cuda_root / "cuda-12.9")
    assert environment["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"


def test_build_environment_preserves_explicit_runtime_choices(tmp_path: Path) -> None:
    environment_bin = tmp_path / "environment" / "bin"
    environment = build_environment(
        {
            "CUDA_HOME": "/custom/cuda",
            "PATH": f"{environment_bin}{os.pathsep}/usr/bin",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
        },
        executable=str(environment_bin / "python"),
        cuda_version="12.9",
        cuda_root=tmp_path,
    )

    assert environment["PATH"].split(os.pathsep) == [str(environment_bin), "/usr/bin"]
    assert environment["CUDA_HOME"] == "/custom/cuda"
    assert environment["SGLANG_ENABLE_JIT_DEEPGEMM"] == "1"


def test_mixed_serving_capacity_is_explicit_and_v2_remains_readable(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b.yaml")
    command = build_command(config)
    assert command[2] == "skillev.runtime.sglang_server"
    assert "--disable-radix-cache" in command
    assert command[command.index("--max-loras-per-batch") + 1] == "2"
    assert command[command.index("--max-running-requests") + 1] == "12"
    with pytest.raises(ValueError):
        build_command({**config, "max_loras_per_batch": 1})
    with pytest.raises(ValueError):
        build_command({**config, "max_running_requests": 0})
    old = {
        key: value
        for key, value in config.items()
        if key not in {"max_running_requests", "max_loras_per_batch", "disable_radix_cache"}
    }
    old["format"] = "skillev-sglang-service@2"
    assert "--max-loras-per-batch" not in build_command(old)
    assert build_command(old)[2] == "sglang.launch_server"


def test_native_evaluation_service_enables_decode_graph_and_prefix_reuse(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b_native_evaluation.yaml")
    command = build_command(config)
    assert command[2] == "skillev.runtime.sglang_server"
    assert command[command.index("--cuda-graph-backend-decode") + 1] == "full"
    assert command[command.index("--cuda-graph-max-bs-decode") + 1] == "48"
    assert "--disable-prefill-cuda-graph" in command
    assert "--disable-cuda-graph" not in command
    assert "--disable-radix-cache" not in command
    assert "--enable-deterministic-inference" in command
    assert command[command.index("--context-length") + 1] == "98304"
    assert command[command.index("--max-running-requests") + 1] == "48"
    assert command[command.index("--chunked-prefill-size") + 1] == "8192"
    assert command[command.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer"
    assert command[command.index("--page-size") + 1] == "64"
    assert "--skillev-fp32-mamba-checkpoints" in command
    assert "--disable-overlap-schedule" not in command


@pytest.mark.parametrize(
    ("field", "value"),
    [("mem_fraction_static", 1.0), ("cuda_graph_max_bs_decode", 0), ("max_prefill_tokens", True)],
)
def test_native_evaluation_service_rejects_invalid_capacity(monkeypatch, field, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b_native_evaluation.yaml")
    with pytest.raises(ValueError):
        build_command({**config, field: value})


def test_training_cache_candidate_has_explicit_compatible_hybrid_state_strategy(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000004")
    monkeypatch.setenv("SKILLEV_MODEL_PATH", "/models/qwen35")
    monkeypatch.setenv("SKILLEV_TOKENIZER_PATH", "/models/qwen35")
    config = _load("configs/serving/qwen35_9b_training_cache_candidate.yaml")
    command = build_command(config)
    assert command[command.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer"
    assert command[command.index("--page-size") + 1] == "64"
    assert "--skillev-fp32-mamba-checkpoints" in command
    assert "--disable-radix-cache" not in command
    with pytest.raises(ValueError):
        build_command({**config, "mamba_radix_cache_strategy": "no_buffer"})
    compatible = build_command(
        {
            **config,
            "mamba_radix_cache_strategy": "no_buffer",
            "disable_overlap_schedule": True,
            "page_size": 1,
            "fp32_mamba_checkpoints": False,
        }
    )
    assert "--disable-overlap-schedule" in compatible


def test_checkpoint_option_is_explicit_and_propagates_to_spawn_workers(monkeypatch):
    from skillev.policy import sglang_mamba_checkpoint
    from skillev.runtime import sglang_server

    calls = []
    # Track the variable even when it was initially absent: the bootstrap writes
    # os.environ directly, and that must not leak into later subprocess tests.
    monkeypatch.setenv("SKILLEV_SGLANG_FP32_MAMBA_CHECKPOINTS", "0")
    monkeypatch.setattr(
        sglang_mamba_checkpoint, "install_fp32_mamba_checkpoints", lambda: calls.append(True)
    )
    ordinary = ["--port", "18467"]
    assert sglang_server.configure_project_options(ordinary) == ordinary
    assert not calls
    assert os.environ["SKILLEV_SGLANG_FP32_MAMBA_CHECKPOINTS"] == "0"
    assert (
        sglang_server.configure_project_options([*ordinary, "--skillev-fp32-mamba-checkpoints"])
        == ordinary
    )
    assert calls == [True]
    assert os.environ["SKILLEV_SGLANG_FP32_MAMBA_CHECKPOINTS"] == "1"
