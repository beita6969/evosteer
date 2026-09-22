"""Strict answer-free Protocol 12 receipt serialization."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from .catalog import CurrentIIDBenchmark
from .conditions import ExecutionContract, ExecutionLane, ReferenceEligibility
from .receipts import MetricObservation, OutcomeCounts, Protocol12RunReceipt, RunProvenance


def execution_to_value(value: ExecutionContract) -> dict[str, object]:
    return {
        "benchmark": value.benchmark.value,
        "condition_id": value.condition_id,
        "lane": value.lane.value,
        "reference_eligibility": value.reference_eligibility.value,
        "actor_model": value.actor_model,
        "actor_route": value.actor_route,
        "actor_service_profile": value.actor_service_profile,
        "context_length": value.context_length,
        "adapter_policy": value.adapter_policy,
        "population_id": value.population_id,
        "dataset_revision": value.dataset_revision,
        "selection_rule": value.selection_rule,
        "expected_count": value.expected_count,
        "prompt_profile": value.prompt_profile,
        "decoding_profile": value.decoding_profile,
        "parser_profile": value.parser_profile,
        "tool_surface": list(value.tool_surface),
        "completion_profile": value.completion_profile,
        "environment_profile": value.environment_profile,
        "scorer_profile": value.scorer_profile,
        "grader_profile": value.grader_profile,
        "metric_profile": value.metric_profile,
        "seed_aggregation": value.seed_aggregation,
    }


def _execution(value: object) -> ExecutionContract:
    if not isinstance(value, dict):
        raise ValueError("receipt execution must be a mapping")
    tools = value.get("tool_surface")
    if not isinstance(tools, list) or any(type(item) is not str for item in tools):
        raise ValueError("receipt tool surface must be text array")
    return ExecutionContract(
        benchmark=CurrentIIDBenchmark(str(value["benchmark"])),
        condition_id=str(value["condition_id"]),
        lane=ExecutionLane(str(value["lane"])),
        reference_eligibility=ReferenceEligibility(str(value["reference_eligibility"])),
        actor_model=str(value["actor_model"]),
        actor_route=str(value["actor_route"]),
        actor_service_profile=str(value["actor_service_profile"]),
        context_length=int(value["context_length"]),
        adapter_policy=str(value["adapter_policy"]),
        population_id=str(value["population_id"]),
        dataset_revision=str(value["dataset_revision"]),
        selection_rule=str(value["selection_rule"]),
        expected_count=int(value["expected_count"]),
        prompt_profile=str(value["prompt_profile"]),
        decoding_profile=str(value["decoding_profile"]),
        parser_profile=str(value["parser_profile"]),
        tool_surface=tuple(tools),
        completion_profile=(
            None if value.get("completion_profile") is None else str(value["completion_profile"])
        ),
        environment_profile=(
            None if value.get("environment_profile") is None else str(value["environment_profile"])
        ),
        scorer_profile=str(value["scorer_profile"]),
        grader_profile=(
            None if value.get("grader_profile") is None else str(value["grader_profile"])
        ),
        metric_profile=str(value["metric_profile"]),
        seed_aggregation=str(value["seed_aggregation"]),
    )


def receipt_to_value(receipt: Protocol12RunReceipt) -> dict[str, object]:
    return {
        "format": "skillev-current-iid-receipt@2",
        "benchmark": receipt.benchmark.value,
        "execution": execution_to_value(receipt.execution),
        "planned_count": receipt.planned_count,
        "outcomes": {
            "scored_success": receipt.outcomes.scored_success,
            "scored_failure": receipt.outcomes.scored_failure,
            "candidate_invalid": receipt.outcomes.candidate_invalid,
            "generation_infrastructure": receipt.outcomes.generation_infrastructure,
            "environment_infrastructure": receipt.outcomes.environment_infrastructure,
            "scorer_infrastructure": receipt.outcomes.scorer_infrastructure,
            "detail": receipt.outcomes.detail,
        },
        "metrics": [
            {
                "metric_id": item.metric_id,
                "numerator": str(item.numerator),
                "denominator": item.denominator,
                "observed_percent": str(item.observed_percent),
                "formula": item.formula,
                "aggregation": item.aggregation,
            }
            for item in receipt.metrics
        ],
        "provenance": {
            name: getattr(receipt.provenance, name)
            for name in (
                "attempt_id",
                "generation_attempt_id",
                "scoring_attempt_id",
                "protocol_version",
                "generation_code_revision",
                "scoring_code_revision",
                "renderer_code_revision",
                "evaluator_version",
                "started_at",
                "completed_at",
            )
        },
        "diagnostics": {key: str(value) for key, value in receipt.diagnostics.items()},
    }


def load_current_iid_receipt(path: Path) -> Protocol12RunReceipt:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-current-iid-receipt@2":
        raise ValueError("legacy/manual receipt cannot enter Protocol 12")
    outcomes = root.get("outcomes")
    provenance = root.get("provenance")
    metrics = root.get("metrics")
    if not isinstance(outcomes, dict) or not isinstance(provenance, dict):
        raise ValueError("receipt lacks outcomes or provenance")
    if not isinstance(metrics, list):
        raise ValueError("receipt metrics must be an array")
    execution = _execution(root["execution"])
    return Protocol12RunReceipt(
        benchmark=CurrentIIDBenchmark(str(root["benchmark"])),
        execution=execution,
        planned_count=int(root["planned_count"]),
        outcomes=OutcomeCounts(
            scored_success=int(outcomes["scored_success"]),
            scored_failure=int(outcomes["scored_failure"]),
            candidate_invalid=int(outcomes["candidate_invalid"]),
            generation_infrastructure=int(outcomes["generation_infrastructure"]),
            environment_infrastructure=int(outcomes["environment_infrastructure"]),
            scorer_infrastructure=int(outcomes["scorer_infrastructure"]),
            detail={str(k): int(v) for k, v in dict(outcomes.get("detail", {})).items()},
        ),
        metrics=tuple(
            MetricObservation(
                metric_id=str(item["metric_id"]),
                numerator=Decimal(str(item["numerator"])),
                denominator=int(item["denominator"]),
                observed_percent=Decimal(str(item["observed_percent"])),
                formula=str(item["formula"]),
                aggregation=str(item["aggregation"]),
            )
            for item in metrics
            if isinstance(item, dict)
        ),
        provenance=RunProvenance(**{key: str(value) for key, value in provenance.items()}),
        diagnostics={str(k): str(v) for k, v in dict(root.get("diagnostics", {})).items()},
    )


def write_receipt_exclusive(path: Path, receipt: Protocol12RunReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt_to_value(receipt), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


__all__ = [
    "execution_to_value",
    "load_current_iid_receipt",
    "receipt_to_value",
    "write_receipt_exclusive",
]
