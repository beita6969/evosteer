"""Replace this process with one role-isolated SGLang server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import UUID

import yaml


def _load(path: str | Path) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "context_length",
        "enable_deterministic_inference",
        "enable_multimodal",
        "enable_metrics",
        "format",
        "host",
        "lora_target_modules",
        "max_lora_rank",
        "model_path_env",
        "port",
        "sampling_backend",
        "served_model_name",
        "supervisor_adapter_name",
        "tensor_parallel_size",
        "tokenizer_path_env",
    }
    version = value.get("format") if isinstance(value, dict) else None
    if version in {
        "skillev-sglang-service@3",
        "skillev-sglang-service@4",
        "skillev-sglang-service@5",
        "skillev-sglang-service@6",
    }:
        expected |= {"max_running_requests", "max_loras_per_batch", "disable_radix_cache"}
    if version in {
        "skillev-sglang-service@4",
        "skillev-sglang-service@5",
        "skillev-sglang-service@6",
    }:
        expected |= {
            "attention_backend",
            "chunked_prefill_size",
            "cuda_graph_max_bs_decode",
            "grammar_backend",
            "linear_attn_backend",
            "max_loaded_loras",
            "max_prefill_tokens",
            "mem_fraction_static",
            "reasoning_parser",
        }
    if version in {"skillev-sglang-service@5", "skillev-sglang-service@6"}:
        expected |= {"mamba_radix_cache_strategy", "page_size", "disable_overlap_schedule"}
    if version == "skillev-sglang-service@6":
        expected.add("fp32_mamba_checkpoints")
    # Older profiles remain readable. Fresh profiles explicitly bind precision
    # and process initialization instead of exposing unresolved `dtype=auto`.
    if isinstance(value, dict):
        expected.update(
            name
            for name in ("enable_request_time_stats_logging", "dtype", "random_seed")
            if name in value
        )
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("SGLang service config has incompatible fields")
    if version not in {
        "skillev-sglang-service@2",
        "skillev-sglang-service@3",
        "skillev-sglang-service@4",
        "skillev-sglang-service@5",
        "skillev-sglang-service@6",
    }:
        raise ValueError("unsupported SGLang service config")
    if value["enable_deterministic_inference"] is not True:
        raise ValueError("SGLang rollout service must honor per-request sampling seeds")
    if value["sampling_backend"] != "pytorch":
        raise ValueError("deterministic SGLang rollout sampling requires the PyTorch backend")
    return value


def build_command(config: dict[str, object]) -> tuple[str, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.startswith("GPU-"):
        UUID(visible[4:])  # Physical UUID, not a mask exposing additional devices.
    elif visible is None or not visible.isdigit():
        raise RuntimeError("SGLang requires exactly one explicit physical CUDA_VISIBLE_DEVICES")
    model_path_env = config["model_path_env"]
    if not isinstance(model_path_env, str) or not model_path_env:
        raise TypeError("model_path_env must be non-empty text")
    model_path = os.environ.get(model_path_env)
    if model_path is None or not model_path.strip():
        raise RuntimeError(f"required model path environment variable is unset: {model_path_env}")
    tokenizer_path_env = config["tokenizer_path_env"]
    if not isinstance(tokenizer_path_env, str) or not tokenizer_path_env:
        raise TypeError("tokenizer_path_env must be non-empty text")
    tokenizer_path = os.environ.get(tokenizer_path_env)
    if tokenizer_path is None or not tokenizer_path.strip():
        raise RuntimeError(
            f"required tokenizer path environment variable is unset: {tokenizer_path_env}"
        )
    command = [
        sys.executable,
        "-m",
        (
            "skillev.runtime.sglang_server"
            if config["format"] != "skillev-sglang-service@2"
            else "sglang.launch_server"
        ),
        "--model-path",
        model_path,
        "--tokenizer-path",
        tokenizer_path,
        "--served-model-name",
        str(config["served_model_name"]),
        "--host",
        str(config["host"]),
        "--port",
        str(config["port"]),
        "--context-length",
        str(config["context_length"]),
        "--tp-size",
        str(config["tensor_parallel_size"]),
        "--sampling-backend",
        str(config["sampling_backend"]),
        "--enable-deterministic-inference",
        "--enable-lora",
        "--trust-remote-code",
        "--max-lora-rank",
        str(config["max_lora_rank"]),
        "--lora-target-modules",
        *[str(module) for module in config["lora_target_modules"]],
    ]
    if "dtype" in config:
        if config["dtype"] != "bfloat16":
            raise ValueError("explicit project service precision must be bfloat16")
        command.extend(["--dtype", "bfloat16"])
    if "random_seed" in config:
        seed = config["random_seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("service initialization seed must be a nonnegative integer")
        command.extend(["--random-seed", str(seed)])
    if config["format"] != "skillev-sglang-service@2":
        if type(config["disable_radix_cache"]) is not bool:
            raise TypeError("prefix cache switch must be boolean")
        if config["disable_radix_cache"]:
            command.append("--disable-radix-cache")
        for name, minimum in (("max_running_requests", 1), ("max_loras_per_batch", 2)):
            value = config[name]
            if type(value) is not int or value < minimum:
                raise ValueError("invalid mixed-serving capacity")
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if config["format"] in {
        "skillev-sglang-service@4",
        "skillev-sglang-service@5",
        "skillev-sglang-service@6",
    }:
        # Qwen3.5's supported full decode graph avoids per-token eager kernel
        # launch overhead. Prefill graphs remain off for LoRA compatibility.
        command.extend(["--cuda-graph-backend-decode", "full", "--disable-prefill-cuda-graph"])
        for name in (
            "cuda_graph_max_bs_decode",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "max_loaded_loras",
        ):
            value = config[name]
            if type(value) is not int or value < 1:
                raise ValueError("serving capacity must be a positive integer")
            command.extend(["--" + name.replace("_", "-"), str(value)])
        fraction = config["mem_fraction_static"]
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError("static memory fraction must lie between zero and one")
        command.extend(["--mem-fraction-static", str(fraction)])
        for name in (
            "attention_backend",
            "linear_attn_backend",
            "grammar_backend",
            "reasoning_parser",
        ):
            value = config[name]
            if not isinstance(value, str) or not value:
                raise ValueError("serving backend must be nonempty text")
            command.extend(["--" + name.replace("_", "-"), value])
    if config["format"] in {"skillev-sglang-service@5", "skillev-sglang-service@6"}:
        strategy, page = config["mamba_radix_cache_strategy"], config["page_size"]
        if strategy not in {"no_buffer", "extra_buffer"}:
            raise ValueError("unsupported explicit Mamba prefix-cache execution strategy")
        if type(page) is not int or page < 1:
            raise ValueError("prefix-cache page size must be a positive integer")
        overlap_disabled = config["disable_overlap_schedule"]
        if type(overlap_disabled) is not bool:
            raise TypeError("overlap scheduling switch must be boolean")
        if not config["disable_radix_cache"] and strategy == "no_buffer":
            if not overlap_disabled or page != 1:
                raise ValueError(
                    "no_buffer cache requires non-overlapped scheduling and page size 1"
                )
        command.extend(["--mamba-radix-cache-strategy", str(strategy), "--page-size", str(page)])
        if overlap_disabled:
            command.append("--disable-overlap-schedule")
    if config["format"] == "skillev-sglang-service@6":
        precise = config["fp32_mamba_checkpoints"]
        if type(precise) is not bool:
            raise TypeError("Mamba checkpoint precision switch must be boolean")
        if precise:
            if (
                config["disable_radix_cache"]
                or config["mamba_radix_cache_strategy"] != "extra_buffer"
                or config["page_size"] % 64
                or config["chunked_prefill_size"] % 64
                or config["linear_attn_backend"] != "triton"
            ):
                raise ValueError("FP32 checkpoints require aligned Triton extra-buffer caching")
            command.append("--skillev-fp32-mamba-checkpoints")
    if config["enable_metrics"] is True:
        command.append("--enable-metrics")
    request_times = config.get("enable_request_time_stats_logging", False)
    if type(request_times) is not bool:
        raise TypeError("request timing logging switch must be boolean")
    if request_times:
        command.append("--enable-request-time-stats-logging")
    if config["enable_multimodal"] is True:
        command.append("--enable-multimodal")
    return tuple(command)


def build_environment(
    base: dict[str, str],
    *,
    executable: str,
    cuda_version: str | None,
    cuda_root: Path = Path("/usr/local"),
) -> dict[str, str]:
    """Expose environment-local build tools and the matching CUDA toolkit."""
    environment = dict(base)
    executable_directory = str(Path(executable).parent)
    path_entries = environment.get("PATH", "").split(os.pathsep)
    environment["PATH"] = os.pathsep.join(
        [executable_directory, *[entry for entry in path_entries if entry != executable_directory]]
    )
    environment.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    if "CUDA_HOME" not in environment and cuda_version is not None:
        candidate = cuda_root / f"cuda-{cuda_version}"
        if (candidate / "bin" / "nvcc").is_file():
            environment["CUDA_HOME"] = str(candidate)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving/qwen35_9b.yaml")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--port", type=int, help="Listening port for this replica")
    arguments = parser.parse_args()
    config = _load(arguments.config)
    if arguments.port is not None:
        config["port"] = arguments.port
    command = build_command(config)
    if arguments.print_command:
        print(" ".join(command))
        return
    try:
        import torch
    except ImportError:
        cuda_version = None
    else:
        cuda_version = torch.version.cuda
    environment = build_environment(
        dict(os.environ), executable=sys.executable, cuda_version=cuda_version
    )
    os.execvpe(  # noqa: S606 - exact argv replaces this trusted launcher process
        command[0], command, environment
    )


if __name__ == "__main__":
    main()
