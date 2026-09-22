#!/usr/bin/env python3
"""Fixed-artifact execution comparison. No generation, optimizer step or posterior.

Inputs and output belong in private storage. A position subset is a profiling
sample, not a smaller training batch: each contribution still uses the original B.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
from skillev_private.experiments.protocol_v13_training_debug import _read_preparation

from skillev.policy import build_qwen_policy_backbone
from skillev.policy.fla_execution import fla_execution_metrics
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.planning import CollectedTrainingBatch
from skillev.training.scoring_telemetry import scoring_telemetry
from skillev.training.step_math import compute_ttb_gradient_shard


def gpu_state() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("profiling requires exactly one explicitly selected CUDA device")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for the read-only thermal baseline")
    return subprocess.run(  # noqa: S603 - fixed read-only query, no shell
        [
            executable,
            "-i",
            visible,
            "--query-gpu=uuid,temperature.gpu,clocks.sm,utilization.gpu,memory.used,clocks_event_reasons.active",
            "--format=csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positions", help="comma-separated canonical positions; default full batch"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--checkpoint-min-tokens", type=int, nargs="+", default=[1])
    parser.add_argument("--temperature-beta", type=float, required=True)
    args = parser.parse_args()
    if args.sizes[0] != 1 or any(size not in (1, 2, 4) for size in args.sizes):
        parser.error("the unchanged one-edge baseline must run first")
    if args.checkpoint_min_tokens[0] != 1 or any(t < 1 for t in args.checkpoint_min_tokens):
        parser.error("the always-checkpointed baseline must run first")
    if not math.isfinite(args.temperature_beta) or args.temperature_beta <= 0:
        parser.error("temperature beta must be the positive frozen method value")
    variants = tuple(
        (size, threshold) for size in args.sizes for threshold in args.checkpoint_min_tokens
    )
    state_before = gpu_state()
    profile = TrainingPerformanceConfig.load(args.performance)
    config, checkpoint = _read_preparation(args.preparation)
    backbone = build_qwen_policy_backbone(config, performance=profile)
    backbone.load_checkpoint(str(checkpoint.directory))
    backbone.bind_initial_trainable_state(checkpoint.trainable_state)
    batch = CollectedTrainingBatch.from_value(
        json.loads(args.batch.read_text()), tokenizer=backbone.tokenizer
    )
    positions = (
        tuple(range(len(batch.artifacts)))
        if args.positions is None
        else tuple(int(v) for v in args.positions.split(","))
    )
    # Keep the existing fixed-32 CUDA comparison's 1e-3 gradient bound.
    # Check components separately: a large Z gradient must not hide LoRA error.
    # This is NOT a posterior confidence interval or bitwise-Adam guarantee.
    reduced_precision = config.torch_dtype == "bfloat16"
    bounds = {
        "gradient_relative_l2_per_component": 1e-3 if reduced_precision else 1e-5,
        "edge_mean_absolute": 1e-3 if reduced_precision else 1e-6,
    }
    report = {
        "kind": "fixed-artifact-execution-only",
        "global_batch_size": len(batch.artifacts),
        "profiled_trajectories": len(positions),
        "temperature_beta": args.temperature_beta,
        "numeric_bounds": bounds,
        "gpu_before": state_before,
        "variants": [],
        "planned_variants": len(variants),
        "complete": False,
        "all_within_bounds": False,
    }
    # Optional CUDA packages can select a different numerical path. Record the
    # effective environment once; do not infer it from the source lock file.
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report["runtime_packages"] = packages
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    reference = None
    reference_edges = None
    within_bounds = True
    for size, threshold in variants:
        variant = replace(
            profile,
            teacher_forcing=replace(
                profile.teacher_forcing,
                microbatch_size=size,
                checkpoint_min_tokens=threshold,
            ),
        )
        backbone.configure_performance(variant)
        kwargs = {
            "backbone": backbone,
            "parameters": backbone.parameter_groups(),
            "batch": batch,
            "positions": positions,
            "global_batch_size": len(batch.artifacts),
            "temperature_beta": args.temperature_beta,
        }
        # Same first artifact warms each shape/backend; warmup is not a training step.
        execution = {"size": size, "checkpoint_min_tokens": threshold}
        print(json.dumps(execution | {"stage": "warmup-started"}), flush=True)
        compute_ttb_gradient_shard(**(kwargs | {"positions": positions[:1]}))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps(execution | {"stage": "measurement-started"}), flush=True)
        start = time.perf_counter()
        shard = compute_ttb_gradient_shard(**kwargs)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        current = {name: value.detach().float().cpu() for name, value in shard.gradients.items()}
        edge_values = tuple(
            v
            for artifact in shard.artifacts
            for edge in artifact.edges
            for v in (edge.forward_logprob_per_token, edge.backward_logprob_per_token)
        )
        if reference is None:
            reference, reference_edges = current, edge_values
        difference = math.fsum(
            float((current[n] - reference[n]).double().square().sum()) for n in reference
        )
        reference_norm = math.fsum(
            float(value.double().square().sum()) for value in reference.values()
        )
        relative = (
            math.sqrt(difference / reference_norm)
            if reference_norm
            else (0.0 if not difference else math.inf)
        )
        components = {}
        for component in sorted({name.split(".")[0] for name in reference}):
            names = [name for name in reference if name.split(".")[0] == component]
            numerator = math.fsum(
                float((current[n] - reference[n]).double().square().sum()) for n in names
            )
            denominator = math.fsum(float(reference[n].double().square().sum()) for n in names)
            components[component] = (
                math.sqrt(numerator / denominator)
                if denominator
                else (0.0 if not numerator else math.inf)
            )
        edge_error = max(abs(a - b) for a, b in zip(edge_values, reference_edges, strict=True))
        valid = (
            all(v <= bounds["gradient_relative_l2_per_component"] for v in components.values())
            and edge_error <= bounds["edge_mean_absolute"]
        )
        within_bounds &= valid
        report["variants"].append(
            {
                "microbatch_size": size,
                "checkpoint_min_tokens": threshold,
                "execution": variant.to_value(),
                "seconds": duration,
                "gradient_relative_l2": relative,
                "gradient_relative_l2_per_component": components,
                "edge_mean_max_absolute": edge_error,
                "within_bounds": valid,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "scoring": scoring_telemetry(shard.artifacts),
                "fla": fla_execution_metrics(),
                "gpu_after": gpu_state(),
            }
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(execution | {"seconds": duration, "within_bounds": valid}), flush=True)
        del shard, current
    report["complete"] = True
    report["all_within_bounds"] = within_bounds
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not within_bounds:
        raise SystemExit("candidate execution failed the predeclared numeric comparison")


if __name__ == "__main__":
    main()
