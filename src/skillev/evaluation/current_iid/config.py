"""YAML loading for current-IID benchmark and condition registries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CurrentIIDBenchmark,
)
from skillev.evaluation.current_iid.conditions import (
    EvaluationCondition,
    EvaluationConditionKind,
    ExecutionContract,
    ExecutionLane,
    ReferenceEligibility,
)
from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11, BenchmarkV11


@dataclass(frozen=True, slots=True)
class MetricContract:
    metric_id: str

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric ID must be non-empty")


@dataclass(frozen=True, slots=True)
class CurrentIIDBenchmarkSpec:
    benchmark: BenchmarkV11
    condition_id: str
    population_id: str
    expected_count: int
    metrics: tuple[MetricContract, ...]
    selection_rule: str
    seed_aggregation: str

    def __post_init__(self) -> None:
        expected = 30 if self.benchmark is BenchmarkV11.AIME_2026 else 128
        if self.expected_count != expected:
            raise ValueError(f"{self.benchmark.value} must use {expected} records")
        if not self.metrics:
            raise ValueError("benchmark requires metrics")
        if any(
            not value.strip()
            for value in (self.condition_id, self.population_id, self.selection_rule)
        ):
            raise ValueError("benchmark identity fields must be non-empty")
        if self.seed_aggregation != "single-seed":
            raise ValueError("the owner requested one seed only")


@dataclass(frozen=True, slots=True)
class CurrentIIDConfig:
    conditions: dict[str, EvaluationCondition]
    benchmarks: tuple[CurrentIIDBenchmarkSpec, ...]

    def __post_init__(self) -> None:
        if tuple(item.benchmark for item in self.benchmarks) != ACTIVE_BENCHMARKS_V11:
            raise ValueError("current-IID config must contain the authoritative ten in order")
        if any(item.condition_id not in self.conditions for item in self.benchmarks):
            raise ValueError("benchmark references an unknown condition")


def _tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("expected a list of strings")
    return tuple(value)


def load_current_iid_config(path: Path) -> CurrentIIDConfig:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-current-iid-config@1":
        raise ValueError("invalid current-IID config")
    condition_rows = root.get("conditions")
    benchmark_rows = root.get("benchmarks")
    if not isinstance(condition_rows, list) or not isinstance(benchmark_rows, list):
        raise ValueError("current-IID config requires condition and benchmark rows")
    conditions: dict[str, EvaluationCondition] = {}
    for raw in condition_rows:
        if not isinstance(raw, dict):
            raise ValueError("condition row must be a mapping")
        condition = EvaluationCondition(
            condition_id=str(raw["id"]),
            kind=EvaluationConditionKind(str(raw["kind"])),
            actor_model=str(raw["actor_model"]),
            actor_route=str(raw["actor_route"]),
            actor_service_profile=str(raw["actor_service_profile"]),
            context_length=int(raw["context_length"]),
            adapter_policy=str(raw["adapter_policy"]),
            prompt_profile=str(raw["prompt_profile"]),
            decoding_profile=str(raw["decoding_profile"]),
            tool_surface=_tuple(raw.get("tool_surface")),
            environment_profile=(
                None if raw.get("environment_profile") is None else str(raw["environment_profile"])
            ),
            scorer_profile=str(raw["scorer_profile"]),
            grader_profile=(
                None if raw.get("grader_profile") is None else str(raw["grader_profile"])
            ),
            auxiliary_models=_tuple(raw.get("auxiliary_models")),
            retrieval_corpus=(
                None if raw.get("retrieval_corpus") is None else str(raw["retrieval_corpus"])
            ),
        )
        if condition.condition_id in conditions:
            raise ValueError("duplicate condition ID")
        conditions[condition.condition_id] = condition
    benchmarks = tuple(
        CurrentIIDBenchmarkSpec(
            benchmark=BenchmarkV11(str(raw["benchmark"])),
            condition_id=str(raw["condition_id"]),
            population_id=str(raw["population_id"]),
            expected_count=int(raw["expected_count"]),
            metrics=tuple(MetricContract(str(item)) for item in raw["metrics"]),
            selection_rule=str(raw["selection_rule"]),
            seed_aggregation=str(raw["seed_aggregation"]),
        )
        for raw in benchmark_rows
        if isinstance(raw, dict)
    )
    return CurrentIIDConfig(conditions=conditions, benchmarks=benchmarks)


_EXECUTION_FIELDS = frozenset(
    {
        "benchmark",
        "condition_id",
        "lane",
        "reference_eligibility",
        "actor_model",
        "actor_route",
        "actor_service_profile",
        "context_length",
        "adapter_policy",
        "population_id",
        "dataset_revision",
        "selection_rule",
        "expected_count",
        "prompt_profile",
        "decoding_profile",
        "parser_profile",
        "tool_surface",
        "completion_profile",
        "environment_profile",
        "scorer_profile",
        "grader_profile",
        "metric_profile",
        "seed_aggregation",
    }
)


def load_execution_contracts(path: Path) -> dict[CurrentIIDBenchmark, ExecutionContract]:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or set(root) != {"format", "executions"}:
        raise ValueError("Protocol 12 condition registry has incompatible fields")
    if root["format"] != "skillev-current-iid-conditions@2":
        raise ValueError("invalid Protocol 12 condition registry")
    values = root["executions"]
    if not isinstance(values, list):
        raise ValueError("Protocol 12 executions must be a list")
    output: dict[CurrentIIDBenchmark, ExecutionContract] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != _EXECUTION_FIELDS:
            raise ValueError(f"execution row {index} has an incompatible field set")
        benchmark = CurrentIIDBenchmark(str(value["benchmark"]))
        contract = ExecutionContract(
            benchmark=benchmark,
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
            tool_surface=_tuple(value["tool_surface"]),
            completion_profile=(
                None if value["completion_profile"] is None else str(value["completion_profile"])
            ),
            environment_profile=(
                None if value["environment_profile"] is None else str(value["environment_profile"])
            ),
            scorer_profile=str(value["scorer_profile"]),
            grader_profile=(
                None if value["grader_profile"] is None else str(value["grader_profile"])
            ),
            metric_profile=str(value["metric_profile"]),
            seed_aggregation=str(value["seed_aggregation"]),
        )
        if benchmark in output:
            raise ValueError("duplicate Protocol 12 benchmark execution")
        output[benchmark] = contract
    if tuple(output) != ACTIVE_CURRENT_IID_BENCHMARKS:
        raise ValueError("Protocol 12 executions must contain exactly the authoritative nine")
    return output


__all__ = [
    "CurrentIIDBenchmarkSpec",
    "CurrentIIDConfig",
    "MetricContract",
    "load_current_iid_config",
    "load_execution_contracts",
]
