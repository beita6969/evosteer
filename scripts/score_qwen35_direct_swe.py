#!/usr/bin/env python3
"""Score an existing complete SWE-bench generation journal with typed verdicts."""

from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import cast

from skillev_private.direct_reference.swe_evaluator import (
    LocalCommandSWEEvaluator,
    RemoteSWEEvaluatorClient,
    SWEDatasetVariant,
    SWEEvaluator,
    SWEPrediction,
    SWEVerdictKind,
)

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol

ONE_SHOT_CONTRACT = "one-shot-unified-diff"
REPOSITORY_AGENT_CONTRACT = "skillflow-repository-agent"
DIRECT_REPOSITORY_AGENT_CONTRACT = "qwen-direct-repository-agent"
DIRECT_RAW_TOOLS_CONTRACT = "qwen-direct-repository-agent-raw-tools"
DIRECT_READONLY_CONTRACT = "qwen-direct-readonly-repository-agent"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    evaluator = parser.add_mutually_exclusive_group(required=True)
    evaluator.add_argument("--official-evaluator-base")
    evaluator.add_argument(
        "--local-evaluator-command-json",
        help="JSON string array containing the local evaluator argv",
    )
    parser.add_argument("--local-evaluator-work-dir", type=Path)
    parser.add_argument("--evaluator-timeout-seconds", type=float, default=86400)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument(
        "--generation-contract",
        choices=(
            ONE_SHOT_CONTRACT,
            REPOSITORY_AGENT_CONTRACT,
            DIRECT_REPOSITORY_AGENT_CONTRACT,
            DIRECT_RAW_TOOLS_CONTRACT,
            DIRECT_READONLY_CONTRACT,
        ),
        default=DIRECT_READONLY_CONTRACT,
    )
    return parser.parse_args()


def _generation_binding(contract: str) -> tuple[str, str, str]:
    if contract == ONE_SHOT_CONTRACT:
        return "direct-patch@2", "qwen35-thinking-code@1", "unified-diff@2"
    if contract == REPOSITORY_AGENT_CONTRACT:
        return (
            "skillflow-code-generation-repository-agent@1",
            "qwen35-repository-agent-supervisor@1",
            "workspace-git-diff@1",
        )
    if contract == DIRECT_REPOSITORY_AGENT_CONTRACT:
        return (
            "qwen-direct-repository-agent@1",
            "qwen35-repository-agent-supervisor@1",
            "workspace-git-diff@1",
        )
    if contract == DIRECT_RAW_TOOLS_CONTRACT:
        return (
            "qwen-direct-repository-agent-raw-tools@1",
            "qwen35-repository-agent-supervisor@1",
            "workspace-git-diff@1",
        )
    if contract == DIRECT_READONLY_CONTRACT:
        return (
            "qwen-direct-readonly-repository-agent@1",
            "qwen35-repository-agent-supervisor@1",
            "unified-diff@2",
        )
    raise ValueError("unknown SWE generation contract")


def _resolved_metric(
    resolved: int, *, reference: Decimal, max_gap_pp_exclusive: Decimal
) -> dict[str, str]:
    if not 0 <= resolved <= 128:
        raise ValueError("resolved count must be within the frozen SWE panel")
    observed = Decimal(resolved) * Decimal(100) / Decimal(128)
    gap = abs(observed - reference)
    return {
        "metric": "resolved",
        "reference_percent": str(reference),
        "diagnostic_observed_percent": str(observed),
        "formal_observed_percent": str(observed),
        "absolute_gap_pp": str(gap),
        "numeric_status": "PASS" if gap < max_gap_pp_exclusive else "FAIL",
    }


def _evaluator(arguments: argparse.Namespace) -> SWEEvaluator:
    if arguments.official_evaluator_base is not None:
        if arguments.local_evaluator_work_dir is not None:
            raise ValueError("local evaluator work directory requires a local command")
        return RemoteSWEEvaluatorClient(
            arguments.official_evaluator_base,
            timeout_seconds=arguments.evaluator_timeout_seconds,
        )
    if arguments.local_evaluator_work_dir is None:
        raise ValueError("local evaluator command requires --local-evaluator-work-dir")
    value = json.loads(arguments.local_evaluator_command_json)
    if type(value) is not list or not value or not all(type(item) is str for item in value):
        raise ValueError("local evaluator command must be a non-empty JSON string array")
    return LocalCommandSWEEvaluator(
        tuple(cast(list[str], value)),
        arguments.local_evaluator_work_dir.resolve(),
        timeout_seconds=arguments.evaluator_timeout_seconds,
    )


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with arguments.predictions.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError("SWE prediction rows must be objects")
            rows.append(cast(dict[str, object], value))
    if len(rows) != 128 or len({row.get("instance_id") for row in rows}) != 128:
        raise ValueError("SWE scoring requires 128 unique frozen instances")
    if any(row.get("generation_infrastructure_error") is not None for row in rows):
        raise ValueError("SWE generation coverage is incomplete")
    predictions = tuple(
        SWEPrediction(
            instance_id=str(row["instance_id"]),
            model_patch=str(row.get("model_patch") or ""),
        )
        for row in rows
    )
    submission_count = sum(bool(prediction.model_patch.strip()) for prediction in predictions)
    protocol = load_direct_reference_protocol(arguments.config)
    spec = protocol.benchmark(DirectBenchmark.SWE_BENCH)
    variant = SWEDatasetVariant(str(spec.dataset_variant))
    evaluator = _evaluator(arguments)
    await evaluator.preflight()
    result = await evaluator.evaluate(
        variant=variant,
        predictions=predictions,
    )
    arguments.private_output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.private_output_dir / "verdicts.jsonl").write_text(
        "".join(
            json.dumps(
                {"instance_id": verdict.instance_id, "status": verdict.kind.value},
                sort_keys=True,
            )
            + "\n"
            for verdict in result.verdicts
        ),
        encoding="utf-8",
    )
    resolved = sum(verdict.resolved for verdict in result.verdicts)
    reference = spec.metrics[0].reference_percent
    prompt_profile, decoding_profile, parser_profile = _generation_binding(
        arguments.generation_contract
    )
    return {
        "swe-bench": {
            "benchmark": "swe-bench",
            "population": spec.population,
            "comparability": spec.comparability.value,
            "dataset_revision": spec.dataset_revision,
            "selection_rule": spec.selection_rule,
            "generation_contract": arguments.generation_contract,
            "prompt_profile": prompt_profile,
            "decoding_profile": decoding_profile,
            "parser_profile": parser_profile,
            "scorer_profile": spec.scorer_profile,
            "seed_aggregation": spec.seed_aggregation.mode.value,
            "comparability_evidence": {
                "population": spec.evidence.population.value,
                "prompt": spec.evidence.prompt.value,
                "decoding": spec.evidence.decoding.value,
                "seed_aggregation": spec.evidence.seed_aggregation.value,
                "scorer": spec.evidence.scorer.value,
                "environment": spec.evidence.environment.value,
            },
            "availability": "complete",
            "dataset_variant": result.dataset_variant.value,
            "evaluator_version": result.evaluator_version,
            "coverage": {
                "planned_count": 128,
                "final_record_count": 128,
                "candidate_response_count": 128,
                "definitive_verdict_count": 128,
                "generation_infrastructure_failures": 0,
                "scorer_infrastructure_failures": 0,
                "environment_infrastructure_failures": 0,
            },
            "submission_rate_percent": str(Decimal(submission_count) * Decimal(100) / Decimal(128)),
            "scorer_reach_rate_percent": "100",
            "verdict_counts": {
                kind.value: sum(verdict.kind is kind for verdict in result.verdicts)
                for kind in SWEVerdictKind
            },
            "metrics": [
                _resolved_metric(
                    resolved,
                    reference=reference,
                    max_gap_pp_exclusive=protocol.parity.max_gap_pp_exclusive,
                )
            ],
        }
    }


def main() -> None:
    print(json.dumps(asyncio.run(_run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
