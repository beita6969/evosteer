#!/usr/bin/env python3
"""Write one operator-supplied, runtime-owned Qwen service receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from skillev.evaluation.current_iid.protocol13.service_receipts import (
    ModelServiceReceipt,
    write_model_service_receipt,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-instance-id", required=True)
    parser.add_argument("--service-profile-id", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-repo-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chat-template-profile", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--adapter-path", action="append", default=[])
    parser.add_argument("--lora-module", action="append", default=[])
    parser.add_argument("--launch-code-revision", required=True)
    parser.add_argument("--started-at", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    write_model_service_receipt(
        arguments.output,
        ModelServiceReceipt(
            service_instance_id=arguments.service_instance_id,
            service_profile_id=arguments.service_profile_id,
            served_model_name=arguments.served_model_name,
            model_repo_id=arguments.model_repo_id,
            model_revision=arguments.model_revision,
            tokenizer_repo_id=arguments.tokenizer_repo_id,
            tokenizer_revision=arguments.tokenizer_revision,
            chat_template_profile=arguments.chat_template_profile,
            context_length=arguments.context_length,
            dtype=arguments.dtype,
            tensor_parallel_size=arguments.tensor_parallel_size,
            reasoning_parser=arguments.reasoning_parser,
            tool_call_parser=arguments.tool_call_parser,
            adapter_paths=tuple(arguments.adapter_path),
            lora_modules=tuple(arguments.lora_module),
            launch_code_revision=arguments.launch_code_revision,
            started_at=arguments.started_at,
        ),
    )


if __name__ == "__main__":
    main()
