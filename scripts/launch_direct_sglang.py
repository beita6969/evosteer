#!/usr/bin/env python3
"""Launch an adapter-free SGLang service for the direct-reference suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

_FIELDS = {
    "adapters",
    "configured_repo_id",
    "context_length",
    "enable_deterministic_inference",
    "enable_metrics",
    "format",
    "host",
    "max_running_requests",
    "mem_fraction_static",
    "model_path_env",
    "port",
    "reasoning_parser",
    "sampling_backend",
    "served_model_name",
    "tensor_parallel_size",
    "tokenizer_path_env",
    "trust_remote_code",
}
_SINGLE_GPU = re.compile(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)\Z")


@dataclass(frozen=True, slots=True)
class ModelRuntimeReceipt:
    configured_repo_id: str
    resolved_model_path: str
    resolved_tokenizer_path: str
    served_model_name: str
    model_type: str
    architectures: tuple[str, ...]
    num_hidden_layers: int | None
    max_position_embeddings: int | None
    service_context_length: int
    reasoning_parser: str
    sampling_backend: str
    adapters: str


def build_runtime_receipt(
    config: dict[str, object], environment: dict[str, str]
) -> ModelRuntimeReceipt:
    model_path = Path(environment[str(config["model_path_env"])]).resolve()
    tokenizer_path = Path(environment[str(config["tokenizer_path_env"])]).resolve()
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (tokenizer_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(model_config, dict) or not isinstance(tokenizer_config, dict):
        raise ValueError("model and tokenizer configurations must be JSON objects")
    if "chat_template" not in tokenizer_config:
        raise ValueError("direct tokenizer lacks chat_template")
    architectures = model_config.get("architectures", [])
    if not isinstance(architectures, list) or any(
        not isinstance(item, str) for item in architectures
    ):
        raise ValueError("model architectures must be a list of text values")
    return ModelRuntimeReceipt(
        configured_repo_id=str(config["configured_repo_id"]),
        resolved_model_path=str(model_path),
        resolved_tokenizer_path=str(tokenizer_path),
        served_model_name=str(config["served_model_name"]),
        model_type=str(model_config.get("model_type", "unknown")),
        architectures=tuple(architectures),
        num_hidden_layers=_optional_int(model_config.get("num_hidden_layers")),
        max_position_embeddings=_optional_int(model_config.get("max_position_embeddings")),
        service_context_length=int(config["context_length"]),
        reasoning_parser=str(config["reasoning_parser"]),
        sampling_backend=str(config["sampling_backend"]),
        adapters=str(config["adapters"]),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("model structural fields must be integers or null")
    return value


def load_config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("direct SGLang config has incompatible fields")
    if value["format"] != "skillev-direct-sglang-service@1":
        raise ValueError("unsupported direct SGLang config")
    if value["adapters"] != "forbidden":
        raise ValueError("direct SGLang service must forbid adapters")
    if value["enable_deterministic_inference"] is not True:
        raise ValueError("direct service must honor per-request seeds")
    if value["sampling_backend"] != "pytorch":
        raise ValueError("direct service requires the deterministic PyTorch sampler")
    return value


def build_command(config: dict[str, object], environment: dict[str, str]) -> tuple[str, ...]:
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None or _SINGLE_GPU.fullmatch(visible) is None:
        raise RuntimeError("direct SGLang requires exactly one explicitly visible physical GPU")
    model_env = str(config["model_path_env"])
    tokenizer_env = str(config["tokenizer_path_env"])
    model_path = environment.get(model_env)
    tokenizer_path = environment.get(tokenizer_env)
    if not model_path or not tokenizer_path:
        raise RuntimeError("direct SGLang model/tokenizer path environment is incomplete")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
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
        "--mem-fraction-static",
        str(config["mem_fraction_static"]),
        "--max-running-requests",
        str(config["max_running_requests"]),
        "--reasoning-parser",
        str(config["reasoning_parser"]),
        "--sampling-backend",
        str(config["sampling_backend"]),
        "--enable-deterministic-inference",
    ]
    if config["trust_remote_code"] is True:
        command.append("--trust-remote-code")
    if config["enable_metrics"] is True:
        command.append("--enable-metrics")
    return tuple(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/serving/qwen35_9b_direct_reference.yaml")
    )
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--runtime-receipt", type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    command = build_command(config, dict(os.environ))
    if arguments.runtime_receipt is not None:
        receipt = build_runtime_receipt(config, dict(os.environ))
        arguments.runtime_receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.runtime_receipt.write_text(
            json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if arguments.print_command:
        print(" ".join(command))
        return
    os.execvpe(command[0], command, dict(os.environ))  # noqa: S606


if __name__ == "__main__":
    main()
