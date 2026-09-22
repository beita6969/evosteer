"""Stable Qwen3.5 FLA numerical kernels, independent of cold-rank timing.

The observed H800 cold-start race selected different Triton tile configurations
on otherwise identical ranks. Torch's deterministic switch does not constrain
that choice. Reuse the reference's shape-only choices and make previously unseen
keys select one legal configuration deterministically, never by benchmark time.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from typing import Any

_KERNEL_MODULES = {
    "chunk_local_cumsum_scalar_kernel": "fla.ops.utils.cumsum",
    "layer_norm_gated_fwd_kernel": "fla.modules.fused_norm_gate",
    "layer_norm_gated_bwd_kernel": "fla.modules.fused_norm_gate",
    "l2norm_fwd_kernel": "fla.modules.l2norm",
    "l2norm_bwd_kernel": "fla.modules.l2norm",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "fla.ops.common.chunk_delta_h",
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": "fla.ops.common.chunk_delta_h",
    "chunk_fwd_kernel_o": "fla.ops.common.chunk_o",
    "chunk_bwd_kernel_dv_local": "fla.ops.common.chunk_o",
    "recompute_w_u_fwd_kernel": "fla.ops.gated_delta_rule.wy_fast",
    "prepare_wy_repr_bwd_kernel": "fla.ops.gated_delta_rule.wy_fast",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": "fla.ops.gated_delta_rule.chunk_fwd",
}
_CONFIG_FIELDS = ("kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg")
_INSTALLED: dict[str, Any] = {}


def _configuration_value(config: Any) -> dict[str, Any]:
    return {name: getattr(config, name) for name in _CONFIG_FIELDS}


def _configuration_order(config: Any) -> str:
    return json.dumps(_configuration_value(config), sort_keys=True, separators=(",", ":"))


def stabilize_autotuner(tuner: Any, choices: list[dict[str, Any]]) -> None:
    """Install once before CUDA owners start; do not read or modify disk caches."""
    if getattr(tuner, "_skillev_stable_choices", False):
        if choices != tuner._skillev_requested_choices:
            raise RuntimeError("a different frozen FLA profile requires a fresh process")
        return
    configured = {}
    for choice in choices:
        candidates = [
            config
            for config in tuner.configs
            if _configuration_value(config) == choice["configuration"]
        ]
        if not candidates:
            raise RuntimeError("installed FLA lacks the reference execution configuration")
        configured[tuple(choice["key"])] = candidates[0]

    original_prune = tuner.prune_configs
    tuner._skillev_execution_counts = {"calls": 0, "fallbacks": 0}
    tuner._skillev_original_prune = original_prune

    def deterministic_prune(kwargs: dict[str, Any]) -> list[Any]:
        # Retain Triton's shape/resource eligibility checks for a new key. A
        # singleton leaves no timing-dependent choice, even on a fresh rank.
        candidates = original_prune(kwargs)
        tuner._skillev_execution_counts["fallbacks"] += 1
        return [min(candidates, key=_configuration_order)]

    tuner.cache.clear()
    tuner.cache.update(configured)
    tuner.cache_results = False
    tuner.prune_configs = deterministic_prune
    tuner._skillev_stable_choices = True
    tuner._skillev_requested_choices = choices
    if hasattr(tuner, "run"):
        original_run = tuner.run

        def counted_run(*args: Any, **kwargs: Any) -> Any:
            tuner._skillev_execution_counts["calls"] += 1
            return original_run(*args, **kwargs)

        tuner.run = counted_run


def configure_qwen35_fla_kernels(profile: dict[str, Any] | None = None) -> None:
    """Load the shipped execution profile only in a CUDA Qwen3.5 process."""
    autotuner_type = importlib.import_module("triton.runtime.autotuner").Autotuner
    if profile is None:
        profile = json.loads(files(__package__).joinpath("qwen35_fla_execution.json").read_text())
    assert profile is not None
    for kernel, module_name in _KERNEL_MODULES.items():
        tuner = getattr(importlib.import_module(module_name), kernel)
        # FLA puts Triton heuristics outside some autotuners.
        while not isinstance(tuner, autotuner_type):
            tuner = tuner.fn
        stabilize_autotuner(tuner, profile[kernel])
        _INSTALLED[kernel] = tuner


def fla_execution_metrics() -> dict[str, Any]:
    return {
        name: {
            **tuner._skillev_execution_counts,
            "cached_shapes": len(tuner.cache),
        }
        for name, tuner in _INSTALLED.items()
    }


@contextmanager
def offline_fla_profiling() -> Iterator[dict[str, Any]]:
    """Explicit fixed-artifact profiling, NEVER used by training initialization.

    The caller runs representative forward/backward inputs on a thermally stable
    device, then verifies score/gradient error before freezing the returned table.
    All training choices are restored even when measurement fails.
    """
    if not _INSTALLED:
        raise RuntimeError("configure the installed FLA kernels before offline profiling")
    original = {
        name: (dict(tuner.cache), tuner.prune_configs) for name, tuner in _INSTALLED.items()
    }
    measured: dict[str, Any] = {}
    try:
        for tuner in _INSTALLED.values():
            tuner.cache.clear()
            tuner.prune_configs = tuner._skillev_original_prune
        yield measured
        for name, tuner in _INSTALLED.items():
            # Keep previously frozen shapes that were not exercised this time.
            merged = original[name][0] | tuner.cache
            measured[name] = [
                {"key": list(key), "configuration": _configuration_value(config)}
                for key, config in sorted(merged.items(), key=lambda item: repr(item[0]))
            ]
    finally:
        for name, tuner in _INSTALLED.items():
            cache, prune = original[name]
            tuner.cache.clear()
            tuner.cache.update(cache)
            tuner.prune_configs = prune
