"""Strict configuration and receipt loaders for Protocol 13."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import yaml

from skillev.experiments.protocol_v13 import ProtocolV13Spec

from .catalog import ACTIVE_PROTOCOL13_BENCHMARKS, Protocol13Benchmark
from .contracts import (
    ExecutionContractV3,
    ExecutionLane,
    InteractiveExecutionExtension,
    ReferenceEligibility,
)
from .receipts import (
    MetricObservationV3,
    OutcomeCountsV4,
    Protocol13RunReceipt,
    RunProvenanceV3,
)
from .targets import (
    EvidenceLocator,
    MetricProjection,
    ReferenceAggregationV4,
    ReferenceMetricV3,
    ReferencePopulationV4,
    ReferenceTargetV3,
    TargetStatus,
)


def _mapping(
    value: object, *, fields: frozenset[str] | None = None, label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a mapping")
    if fields is not None and set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, label) for item in _list(value, label))


_EXECUTION_FIELDS = frozenset(
    {
        "benchmark",
        "protocol_version",
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
        "panel_manifest_id",
        "evaluator_profile",
        "interactive",
        "code_test_suite",
    }
)


def load_execution_contracts_v3(
    path: Path,
    *,
    protocol: ProtocolV13Spec,
) -> dict[Protocol13Benchmark, ExecutionContractV3]:
    root = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        fields=frozenset({"format", "executions"}),
        label="condition registry",
    )
    if root["format"] != "skillev-current-iid-conditions@4":
        raise ValueError("invalid Protocol 13 condition registry")
    output: dict[Protocol13Benchmark, ExecutionContractV3] = {}
    for index, value in enumerate(_list(root["executions"], "executions")):
        row = _mapping(value, fields=_EXECUTION_FIELDS, label=f"execution {index}")
        benchmark = Protocol13Benchmark(_text(row["benchmark"], "benchmark"))
        contract = ExecutionContractV3(
            protocol_version=_text(row["protocol_version"], "protocol version"),
            benchmark=benchmark,
            condition_id=_text(row["condition_id"], "condition ID"),
            lane=ExecutionLane(_text(row["lane"], "lane")),
            reference_eligibility=ReferenceEligibility(
                _text(row["reference_eligibility"], "reference eligibility")
            ),
            actor_model=_text(row["actor_model"], "actor model"),
            actor_route=_text(row["actor_route"], "actor route"),
            actor_service_profile=_text(row["actor_service_profile"], "actor service profile"),
            context_length=_integer(row["context_length"], "context length"),
            adapter_policy=_text(row["adapter_policy"], "adapter policy"),
            population_id=_text(row["population_id"], "population ID"),
            dataset_revision=_text(row["dataset_revision"], "dataset revision"),
            selection_rule=_text(row["selection_rule"], "selection rule"),
            expected_count=_integer(row["expected_count"], "expected count"),
            prompt_profile=_text(row["prompt_profile"], "prompt profile"),
            decoding_profile=_text(row["decoding_profile"], "decoding profile"),
            parser_profile=_text(row["parser_profile"], "parser profile"),
            tool_surface=_string_tuple(row["tool_surface"], "tool surface"),
            completion_profile=_optional_text(row["completion_profile"], "completion profile"),
            environment_profile=_optional_text(row["environment_profile"], "environment profile"),
            scorer_profile=_text(row["scorer_profile"], "scorer profile"),
            grader_profile=_optional_text(row["grader_profile"], "grader profile"),
            metric_profile=_text(row["metric_profile"], "metric profile"),
            seed_aggregation=_text(row["seed_aggregation"], "seed aggregation"),
            panel_manifest_id=_text(row["panel_manifest_id"], "panel manifest ID"),
            evaluator_profile=_optional_text(row["evaluator_profile"], "evaluator profile"),
            interactive=_load_interactive_extension(row["interactive"]),
            code_test_suite=_optional_text(row["code_test_suite"], "code test suite"),
        )
        if benchmark in output:
            raise ValueError("duplicate Protocol 13 benchmark execution")
        output[benchmark] = contract
    if tuple(output) != protocol.benchmarks or tuple(output) != ACTIVE_PROTOCOL13_BENCHMARKS:
        raise ValueError("Protocol 13 executions must contain the authoritative eight in order")
    populations = {item.population_id: item for item in protocol.populations}
    for contract in output.values():
        population = populations.get(contract.population_id)
        if population is None or population.benchmark is not contract.benchmark:
            raise ValueError("execution refers to an unknown final population")
        if population.role.value != "final-evaluation":
            raise ValueError("execution must refer to a final population")
        if population.expected_count != contract.expected_count:
            raise ValueError("execution and population counts differ")
        if (
            population.dataset_revision != contract.dataset_revision
            or population.selection_rule != contract.selection_rule
        ):
            raise ValueError("execution and population identities differ")
    return output


def _load_interactive_extension(value: object) -> InteractiveExecutionExtension | None:
    if value is None:
        return None
    row = _mapping(
        value,
        fields=frozenset(
            {
                "horizon_policy",
                "max_steps_cap",
                "required_max_steps",
                "history_window_steps",
                "history_maximum_characters",
                "include_reasoning_in_history",
                "invalid_candidate_policy",
                "invalid_environment_action_policy",
                "prompt_asset_id",
                "prompt_asset_source_revision",
            }
        ),
        label="interactive execution extension",
    )
    history_window = row["history_window_steps"]
    return InteractiveExecutionExtension(
        horizon_policy=_text(row["horizon_policy"], "horizon policy"),
        max_steps_cap=_integer(row["max_steps_cap"], "maximum step cap"),
        required_max_steps=(
            None
            if row["required_max_steps"] is None
            else _integer(row["required_max_steps"], "required maximum steps")
        ),
        history_window_steps=(
            None if history_window is None else _integer(history_window, "history window steps")
        ),
        history_maximum_characters=(
            None
            if row["history_maximum_characters"] is None
            else _integer(row["history_maximum_characters"], "history maximum characters")
        ),
        include_reasoning_in_history=_boolean(
            row["include_reasoning_in_history"], "include reasoning in history"
        ),
        invalid_candidate_policy=_text(row["invalid_candidate_policy"], "invalid candidate policy"),
        invalid_environment_action_policy=_text(
            row["invalid_environment_action_policy"],
            "invalid environment action policy",
        ),
        prompt_asset_id=_text(row["prompt_asset_id"], "prompt asset ID"),
        prompt_asset_source_revision=_text(
            row["prompt_asset_source_revision"], "prompt asset source revision"
        ),
    )


_METRIC_TARGET_FIELDS = frozenset(
    {
        "metric_id",
        "unit",
        "denominator_id",
        "projection",
        "formula",
        "aggregation",
        "reference_percent",
        "maximum_gap_pp_exclusive",
        "required_for_formal_gate",
    }
)


def load_targets_v3(
    path: Path,
    *,
    executions: dict[Protocol13Benchmark, ExecutionContractV3],
    protocol: ProtocolV13Spec,
) -> dict[Protocol13Benchmark, ReferenceTargetV3]:
    root = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        fields=frozenset({"format", "targets"}),
        label="target registry",
    )
    if root["format"] != "skillev-current-iid-targets@4":
        raise ValueError("invalid Protocol 13 target registry")
    rows = _mapping(root["targets"], label="targets")
    if list(rows) != [item.value for item in protocol.benchmarks]:
        raise ValueError("target registry must contain the authoritative eight in order")
    output: dict[Protocol13Benchmark, ReferenceTargetV3] = {}
    for benchmark in protocol.benchmarks:
        raw = _mapping(rows[benchmark.value], label=benchmark.value)
        status = TargetStatus(_text(raw.get("status"), "target status"))
        execution = executions[benchmark]
        if status is TargetStatus.UNDEFINED:
            if set(raw) != {"status", "condition_id"}:
                raise ValueError("undefined target cannot carry evidence")
            if raw["condition_id"] != execution.condition_id:
                raise ValueError("undefined target condition differs")
            output[benchmark] = ReferenceTargetV3(
                benchmark,
                status,
                execution,
                None,
                None,
                None,
                (),
                None,
            )
            continue
        if set(raw) != {
            "status",
            "condition_id",
            "reference_condition_id",
            "reference_population",
            "reference_aggregation",
            "evidence",
            "metrics",
        }:
            raise ValueError("defined target fields differ")
        if raw["condition_id"] != execution.condition_id:
            raise ValueError("target condition differs from Protocol 13 execution")
        reference_condition = raw["reference_condition_id"]
        if reference_condition is None:
            reference_execution = None
        elif reference_condition == execution.condition_id:
            reference_execution = execution
        else:
            raise ValueError("reference condition is not registered")
        population_raw = raw["reference_population"]
        reference_population = None
        if population_raw is not None:
            population = _mapping(
                population_raw,
                fields=frozenset(
                    {
                        "population_id",
                        "dataset_revision",
                        "selection_rule",
                        "expected_count",
                        "panel_manifest_id",
                    }
                ),
                label="reference population",
            )
            reference_population = ReferencePopulationV4(
                population_id=_text(population["population_id"], "population ID"),
                dataset_revision=_text(population["dataset_revision"], "dataset revision"),
                selection_rule=_text(population["selection_rule"], "selection rule"),
                expected_count=_integer(population["expected_count"], "expected count"),
                panel_manifest_id=_optional_text(
                    population["panel_manifest_id"], "panel manifest ID"
                ),
            )
        aggregation_raw = raw["reference_aggregation"]
        reference_aggregation = None
        if aggregation_raw is not None:
            aggregation = _mapping(
                aggregation_raw,
                fields=frozenset({"mode", "seeds", "run_count", "reducer", "dispersion"}),
                label="reference aggregation",
            )
            reference_aggregation = ReferenceAggregationV4(
                mode=_text(aggregation["mode"], "aggregation mode"),
                seeds=tuple(
                    _integer(item, "reference seed")
                    for item in _list(aggregation["seeds"], "reference seeds")
                ),
                run_count=_integer(aggregation["run_count"], "run count"),
                reducer=_text(aggregation["reducer"], "reducer"),
                dispersion=_optional_text(aggregation["dispersion"], "dispersion"),
            )
        evidence_raw = _mapping(
            raw["evidence"],
            fields=frozenset(
                {
                    "source_kind",
                    "repository_or_paper",
                    "revision_or_version",
                    "path_or_section",
                    "supported_fields",
                }
            ),
            label="target evidence",
        )
        evidence = EvidenceLocator(
            source_kind=_text(evidence_raw["source_kind"], "source kind"),
            repository_or_paper=_text(evidence_raw["repository_or_paper"], "repository or paper"),
            revision_or_version=_text(evidence_raw["revision_or_version"], "revision or version"),
            path_or_section=_text(evidence_raw["path_or_section"], "path or section"),
            supported_fields=_string_tuple(
                evidence_raw["supported_fields"], "supported evidence fields"
            ),
        )
        metrics: list[ReferenceMetricV3] = []
        for index, value in enumerate(_list(raw["metrics"], "target metrics")):
            metric = _mapping(
                value,
                fields=_METRIC_TARGET_FIELDS,
                label=f"{benchmark.value}.metrics[{index}]",
            )
            metrics.append(
                ReferenceMetricV3(
                    metric_id=_text(metric["metric_id"], "metric ID"),
                    unit=_text(metric["unit"], "metric unit"),
                    denominator_id=_text(metric["denominator_id"], "denominator ID"),
                    projection=MetricProjection(_text(metric["projection"], "projection")),
                    formula=_text(metric["formula"], "formula"),
                    aggregation=_text(metric["aggregation"], "aggregation"),
                    reference_percent=Decimal(str(metric["reference_percent"])),
                    maximum_gap_pp_exclusive=Decimal(str(metric["maximum_gap_pp_exclusive"])),
                    required_for_formal_gate=_boolean(
                        metric["required_for_formal_gate"], "formal gate flag"
                    ),
                )
            )
        output[benchmark] = ReferenceTargetV3(
            benchmark=benchmark,
            status=status,
            observed_execution=execution,
            reference_execution=reference_execution,
            reference_population=reference_population,
            reference_aggregation=reference_aggregation,
            metrics=tuple(metrics),
            evidence=evidence,
        )
    return output


_RECEIPT_FIELDS = frozenset(
    {
        "format",
        "benchmark",
        "condition_id",
        "planned_count",
        "outcomes",
        "metrics",
        "provenance",
        "runtime_execution_attempt_id",
        "runtime_contract_matched",
        "final_panel_used_for_selection",
        "diagnostics",
    }
)


def load_receipts_v3(
    directory: Path,
    *,
    executions: dict[Protocol13Benchmark, ExecutionContractV3],
    protocol: ProtocolV13Spec,
) -> dict[Protocol13Benchmark, Protocol13RunReceipt]:
    output: dict[Protocol13Benchmark, Protocol13RunReceipt] = {}
    for benchmark in protocol.benchmarks:
        path = directory / f"{benchmark.value}.json"
        root = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            fields=_RECEIPT_FIELDS,
            label=f"{benchmark.value} receipt",
        )
        if root["format"] != "skillev-protocol13-receipt@2":
            raise ValueError("invalid Protocol 13 receipt format")
        if root["benchmark"] != benchmark.value:
            raise ValueError("receipt benchmark differs from filename")
        execution = executions[benchmark]
        if root["condition_id"] != execution.condition_id:
            raise ValueError("receipt condition differs from execution")
        outcomes = _mapping(
            root["outcomes"],
            fields=frozenset(
                {
                    "definitive_scored",
                    "candidate_invalid",
                    "generation_infrastructure",
                    "environment_infrastructure",
                    "scorer_infrastructure",
                    "binary_success",
                    "binary_failure",
                    "detail",
                }
            ),
            label="outcomes",
        )
        detail = _mapping(outcomes["detail"], label="outcome detail")
        metrics = tuple(
            _load_metric_observation(value, index=index)
            for index, value in enumerate(_list(root["metrics"], "metrics"))
        )
        provenance = _load_provenance(root["provenance"])
        diagnostics_raw = _mapping(root["diagnostics"], label="diagnostics")
        diagnostics: dict[str, Decimal | int | str] = {}
        for name, value in diagnostics_raw.items():
            if type(value) in {int, str}:
                diagnostics[name] = value  # type: ignore[assignment]
            elif type(value) is float:
                diagnostics[name] = Decimal(str(value))
            else:
                raise ValueError("diagnostic values must be scalar")
        output[benchmark] = Protocol13RunReceipt(
            benchmark=benchmark,
            execution=execution,
            planned_count=_integer(root["planned_count"], "planned count"),
            outcomes=OutcomeCountsV4(
                definitive_scored=_integer(outcomes["definitive_scored"], "definitive scored"),
                candidate_invalid=_integer(outcomes["candidate_invalid"], "candidate invalid"),
                generation_infrastructure=_integer(
                    outcomes["generation_infrastructure"], "generation infrastructure"
                ),
                environment_infrastructure=_integer(
                    outcomes["environment_infrastructure"], "environment infrastructure"
                ),
                scorer_infrastructure=_integer(
                    outcomes["scorer_infrastructure"], "scorer infrastructure"
                ),
                binary_success=(
                    None
                    if outcomes["binary_success"] is None
                    else _integer(outcomes["binary_success"], "binary success")
                ),
                binary_failure=(
                    None
                    if outcomes["binary_failure"] is None
                    else _integer(outcomes["binary_failure"], "binary failure")
                ),
                detail={name: _integer(value, name) for name, value in detail.items()},
            ),
            metrics=metrics,
            provenance=provenance,
            diagnostics=diagnostics,
            runtime_execution_attempt_id=_text(
                root["runtime_execution_attempt_id"], "runtime execution attempt ID"
            ),
            runtime_contract_matched=_boolean(
                root["runtime_contract_matched"], "runtime contract matched"
            ),
            final_panel_used_for_selection=_boolean(
                root["final_panel_used_for_selection"], "final panel used for selection"
            ),
        )
    return output


def _load_metric_observation(value: object, *, index: int) -> MetricObservationV3:
    row = _mapping(
        value,
        fields=frozenset(
            {
                "metric_id",
                "unit",
                "numerator",
                "denominator",
                "denominator_id",
                "projection",
                "observed_percent",
                "formula",
                "aggregation",
            }
        ),
        label=f"metric observation {index}",
    )
    return MetricObservationV3(
        metric_id=_text(row["metric_id"], "metric ID"),
        unit=_text(row["unit"], "unit"),
        numerator=Decimal(str(row["numerator"])),
        denominator=_integer(row["denominator"], "denominator"),
        denominator_id=_text(row["denominator_id"], "denominator ID"),
        projection=MetricProjection(_text(row["projection"], "projection")),
        observed_percent=Decimal(str(row["observed_percent"])),
        formula=_text(row["formula"], "formula"),
        aggregation=_text(row["aggregation"], "aggregation"),
    )


def _load_provenance(value: object) -> RunProvenanceV3:
    fields = frozenset(
        {
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
        }
    )
    row = _mapping(value, fields=fields, label="provenance")
    return RunProvenanceV3(**{name: _text(row[name], name) for name in fields})


__all__ = ["load_execution_contracts_v3", "load_receipts_v3", "load_targets_v3"]
