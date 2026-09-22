#!/usr/bin/env python3
"""Validate direct model, frozen IID populations, scorers, and SWE evaluator."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from skillev_private.direct_reference.evaluators import SCORER_REGISTRY
from skillev_private.direct_reference.manifests import load_population_manifest
from skillev_private.direct_reference.populations import load_skillflow_iid_cases
from skillev_private.direct_reference.swe_evaluator import RemoteSWEEvaluatorClient

from skillev.evaluation.direct_baseline.client import (
    DirectGenerationRequest,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark, DirectDecodingProfile
from skillev.evaluation.direct_baseline.protocol import (
    load_direct_reference_protocol,
    validate_runtime_registries,
)

_IID = frozenset(
    {
        DirectBenchmark.HOTPOT_QA,
        DirectBenchmark.TRIVIA_QA,
        DirectBenchmark.AIME_2026,
        DirectBenchmark.MED_QA,
        DirectBenchmark.SWE_BENCH,
    }
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iid-population", type=Path, required=True)
    parser.add_argument("--population-manifest-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    parser.add_argument("--swe-evaluator-base")
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(item.value for item in _IID),
        required=True,
    )
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_direct_reference_protocol(arguments.config)
    validate_runtime_registries(protocol)
    selected = frozenset(DirectBenchmark(value) for value in arguments.benchmarks)
    unknown_scorers = {
        protocol.benchmark(benchmark).scorer_profile
        for benchmark in selected - {DirectBenchmark.SWE_BENCH}
        if protocol.benchmark(benchmark).scorer_profile not in SCORER_REGISTRY
    }
    if unknown_scorers:
        raise ValueError(f"unknown scorers: {sorted(unknown_scorers)}")
    manifests = {
        benchmark: load_population_manifest(
            arguments.population_manifest_dir / f"{benchmark.value}.json"
        )
        for benchmark in selected
    }
    cases = load_skillflow_iid_cases(
        arguments.iid_population,
        protocol=protocol,
        include=selected,
        manifests=manifests,
    )
    if DirectBenchmark.SWE_BENCH in selected:
        if not arguments.swe_evaluator_base:
            raise ValueError("SWE generation is forbidden until an evaluator passes preflight")
        await RemoteSWEEvaluatorClient(arguments.swe_evaluator_base).preflight()
    served_model = arguments.served_model_name or protocol.model.served_model_name
    if served_model != protocol.model.served_model_name:
        raise ValueError("served model differs from executable protocol")
    client = OpenAICompatibleDirectClient(
        endpoint_base=arguments.endpoint_base,
        served_model_name=served_model,
        timeout_seconds=120,
        context_length=arguments.context_length,
        token_counter=QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path),
    )
    routes = await client.model_routes()
    probes = []
    for enable_thinking in (False, True):
        probes.append(
            await client.generate(
                DirectGenerationRequest(
                    request_id=f"direct-runtime-capability-probe-{enable_thinking}",
                    messages=({"role": "user", "content": "Reply with OK."},),
                    profile=DirectDecodingProfile(
                        f"runtime-capability-probe-{enable_thinking}@1",
                        enable_thinking,
                        0.0,
                        1.0,
                        1,
                        0.0,
                        0.0,
                        1.0,
                        8,
                        sampling_mode="greedy",
                    ),
                )
            )
        )
    if any(probe.response_model != served_model for probe in probes):
        raise ValueError("capability probe returned another model route")
    return {
        "format": "skillev-direct-runtime-receipt@1",
        "model_repo": protocol.model.repo_id,
        "served_model": served_model,
        "model_route": protocol.model.route,
        "adapter_policy": protocol.model.adapters,
        "skill_policy": protocol.model.skill_library,
        "rollout_policy": protocol.model.rollout_engine,
        "model_routes": list(routes),
        "response_model": probes[0].response_model,
        "response_id_present": all(probe.response_id is not None for probe in probes),
        "thinking_modes_probed": [False, True],
        "benchmarks": sorted(item.value for item in selected),
        "population_records": len(cases),
        "swe_evaluator_preflight": (
            "passed" if DirectBenchmark.SWE_BENCH in selected else "not-requested"
        ),
    }


def main() -> None:
    arguments = _arguments()
    receipt = asyncio.run(_run(arguments))
    arguments.private_receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.private_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
