#!/usr/bin/env python3
"""Measure resource-limited rollout concurrency without training side effects."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from skillev.contracts import JsonValue
from skillev.rollout import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
    GenerationPhase,
    PolicySnapshot,
    RolloutGenerationRequest,
)
from skillev.runtime import SGLangGateway, SGLangGatewayConfig
from skillev.training import (
    RolloutBatchWorkflow,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
)


class _HFTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: tuple[int, ...], *, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True, slots=True)
class _TokenizerAdapter:
    tokenizer: _HFTokenizer
    tokenizer_id: str

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "sglang"), default="synthetic")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--model-delay-ms", type=float, default=50.0)
    parser.add_argument("--evaluator-delay-ms", type=float, default=100.0)
    parser.add_argument("--endpoint")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-name")
    parser.add_argument("--adapter-revision", default="bounded-benchmark")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def _binding(concurrency: int) -> RolloutWorkflowBinding:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    return RolloutWorkflowBinding(
        max_resident_trajectories=concurrency + 4,
        max_inflight_model_requests=concurrency,
        max_inflight_environment_calls=max(1, min(concurrency, 8)),
        max_inflight_terminal_evaluations=max(1, min(concurrency, 6)),
        max_inflight_process_graders=max(1, min(concurrency, 3)),
        transport_worker_threads=concurrency,
    )


async def _synthetic(args: argparse.Namespace) -> dict[str, JsonValue]:
    binding = _binding(args.concurrency)
    resources = RolloutWorkflowResources(binding)
    workflow = RolloutBatchWorkflow[int, tuple[int, int]](binding)
    latencies = [0.0] * args.requests

    async def execute(position: int) -> tuple[int, int]:
        started = time.perf_counter()
        async with resources.model_requests.lease():
            await asyncio.sleep(args.model_delay_ms / 1000.0)
        if position % 4 == 0:
            async with resources.terminal_evaluations.lease():
                await asyncio.sleep(args.evaluator_delay_ms / 1000.0)
        latencies[position] = time.perf_counter() - started
        return position, 8

    started = time.perf_counter()
    outputs = await workflow.run(tuple(range(args.requests)), execute)
    wall = time.perf_counter() - started
    return _summary(
        mode="synthetic",
        binding=binding,
        wall_seconds=wall,
        latencies=latencies,
        prompt_tokens=args.requests * 8,
        completion_tokens=sum(item[1] for item in outputs),
        resources=resources,
    )


async def _sglang(args: argparse.Namespace) -> dict[str, JsonValue]:
    required = {
        "endpoint": args.endpoint,
        "base_model": args.base_model,
        "adapter_name": args.adapter_name,
        "tokenizer_path": args.tokenizer_path,
    }
    if any(value is None for value in required.values()):
        raise ValueError("SGLang mode requires endpoint, model, adapter, and tokenizer arguments")
    from transformers import AutoTokenizer

    tokenizer_path = cast(Path, args.tokenizer_path).resolve()
    tokenizer_impl = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = _TokenizerAdapter(cast(_HFTokenizer, tokenizer_impl), tokenizer_path.name)
    gateway = SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base=cast(str, args.endpoint),
            base_model=cast(str, args.base_model),
            supervisor_adapter=cast(str, args.adapter_name),
        )
    )
    gateway.bind_existing_supervisor_adapter(adapter_revision=args.adapter_revision)
    snapshot = PolicySnapshot.create(
        backbone_id="bounded-sglang-benchmark",
        forward_adapter_version=args.adapter_revision,
        tokenizer_id=tokenizer.tokenizer_id,
        backend_id="sglang-native-exact-token",
        initial_trainable_state_hash="0" * 64,
    )
    binding = _binding(args.concurrency)
    resources = RolloutWorkflowResources(binding)
    generator = ExternalSGLangRolloutGenerator(
        config=ExternalSGLangRolloutConfig(
            cast(str, args.endpoint),
            transport_worker_threads=binding.transport_worker_threads,
        ),
        tokenizer=tokenizer,
        gateway=gateway,
        snapshot_provider=lambda: snapshot,
    )
    prompt_ids = tuple(tokenizer.encode("Return the word OK."))
    workflow = RolloutBatchWorkflow[int, tuple[int, int]](binding)
    latencies = [0.0] * args.requests

    async def execute(position: int) -> tuple[int, int]:
        started = time.perf_counter()
        async with resources.model_requests.lease():
            result = await generator.generate(
                RolloutGenerationRequest(
                    phase=GenerationPhase.REASONING,
                    input_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=position,
                    decoding_snapshot_id="bounded-benchmark",
                    expected_policy_snapshot_id=snapshot.snapshot_id,
                )
            )
        latencies[position] = time.perf_counter() - started
        return result.usage.input_tokens, result.usage.output_tokens

    try:
        started = time.perf_counter()
        outputs = await workflow.run(tuple(range(args.requests)), execute)
        wall = time.perf_counter() - started
    finally:
        generator.close()
    return _summary(
        mode="sglang",
        binding=binding,
        wall_seconds=wall,
        latencies=latencies,
        prompt_tokens=sum(item[0] for item in outputs),
        completion_tokens=sum(item[1] for item in outputs),
        resources=resources,
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _summary(
    *,
    mode: str,
    binding: RolloutWorkflowBinding,
    wall_seconds: float,
    latencies: list[float],
    prompt_tokens: int,
    completion_tokens: int,
    resources: RolloutWorkflowResources,
) -> dict[str, JsonValue]:
    request_count = len(latencies)
    return {
        "binding": binding.to_value(),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / wall_seconds,
        "mode": mode,
        "model_resource": resources.model_requests.timing.to_value(),
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": prompt_tokens / wall_seconds,
        "request_count": request_count,
        "requests_per_second": request_count / wall_seconds,
        "trajectory_latency_max_seconds": max(latencies),
        "trajectory_latency_mean_seconds": statistics.fmean(latencies),
        "trajectory_latency_p50_seconds": _percentile(latencies, 0.50),
        "trajectory_latency_p95_seconds": _percentile(latencies, 0.95),
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    args = _arguments()
    if args.requests < 1 or args.max_new_tokens < 1:
        raise ValueError("request and completion counts must be positive")
    summary = asyncio.run(_synthetic(args) if args.mode == "synthetic" else _sglang(args))
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
