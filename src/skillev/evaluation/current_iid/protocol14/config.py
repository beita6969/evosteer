"""Strict YAML loading for Protocol 14 conditions and target evidence."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml

from skillev.experiments.protocol_v14 import ProtocolV14Spec

from .catalog import ACTIVE_PROTOCOL14_BENCHMARKS, Protocol14Benchmark
from .contracts import (
    AggregationIdentity,
    DecodingParameters,
    ExecutionContractV4,
    ExecutionLane,
    FormalEligibility,
    InteractiveContract,
    ThinkingMode,
)
from .targets import (
    EvidenceLocator,
    MetricProjection,
    PublishedAnchor,
    ReferenceMetricV4,
    ReferenceTargetV5,
    TargetScope,
    TargetStatus,
)

_EXECUTION_FIELDS = {
    "benchmark",
    "protocol_version",
    "condition_id",
    "lane",
    "formal_eligibility",
    "actor_model",
    "actor_model_revision",
    "actor_route",
    "actor_service_profile",
    "context_length",
    "adapter_policy",
    "population_id",
    "dataset_revision",
    "selection_rule",
    "expected_count",
    "panel_manifest_id",
    "prompt_profile",
    "thinking_mode",
    "decoding_profile",
    "decoding",
    "parser_profile",
    "completion_profile",
    "tool_surface",
    "environment_profile",
    "scorer_profile",
    "grader_profile",
    "metric_profile",
    "aggregation",
    "interactive",
    "code_test_suite",
}


def load_execution_contracts_v4(
    path: Path, *, protocol: ProtocolV14Spec
) -> dict[Protocol14Benchmark, ExecutionContractV4]:
    root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "conditions")
    if set(root) != {"format", "executions"} or root["format"] != (
        "skillev-current-iid-conditions@5"
    ):
        raise ValueError("invalid Protocol 14 conditions registry")
    output: dict[Protocol14Benchmark, ExecutionContractV4] = {}
    populations = {item.benchmark: item for item in protocol.populations}
    for value in _list(root["executions"], "executions"):
        row = _mapping(value, "execution")
        if set(row) != _EXECUTION_FIELDS:
            raise ValueError("Protocol 14 execution fields differ")
        benchmark = Protocol14Benchmark(_text(row["benchmark"], "benchmark"))
        if benchmark in output:
            raise ValueError("Protocol 14 execution benchmark is duplicated")
        decoding = _mapping(row["decoding"], "decoding")
        if set(decoding) != {
            "sampling_mode",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "max_new_tokens",
            "seed",
            "stop",
        }:
            raise ValueError("Protocol 14 decoding fields differ")
        aggregation = _mapping(row["aggregation"], "aggregation")
        if set(aggregation) != {"seeds", "run_count", "reducer", "dispersion"}:
            raise ValueError("Protocol 14 aggregation fields differ")
        interactive_raw = row["interactive"]
        interactive = None
        if interactive_raw is not None:
            interactive_row = _mapping(interactive_raw, "interactive")
            if set(interactive_row) != {
                "horizon_policy",
                "max_steps",
                "history_window_steps",
                "history_maximum_characters",
                "history_observation_characters",
                "include_reasoning_in_history",
                "invalid_candidate_policy",
                "invalid_environment_action_policy",
                "prompt_source_repository",
                "prompt_source_revision",
                "prompt_source_path",
                "prompt_source_symbol",
            }:
                raise ValueError("Protocol 14 interactive fields differ")
            history_window = interactive_row["history_window_steps"]
            interactive = InteractiveContract(
                horizon_policy=_text(interactive_row["horizon_policy"], "horizon policy"),
                max_steps=_integer(interactive_row["max_steps"], "max steps"),
                history_window_steps=(
                    None if history_window is None else _integer(history_window, "history window")
                ),
                history_maximum_characters=_integer(
                    interactive_row["history_maximum_characters"], "history characters"
                ),
                history_observation_characters=(
                    None
                    if interactive_row["history_observation_characters"] is None
                    else _integer(
                        interactive_row["history_observation_characters"],
                        "history observation characters",
                    )
                ),
                include_reasoning_in_history=_boolean(
                    interactive_row["include_reasoning_in_history"], "reasoning history"
                ),
                invalid_candidate_policy=_text(
                    interactive_row["invalid_candidate_policy"], "candidate policy"
                ),
                invalid_environment_action_policy=_text(
                    interactive_row["invalid_environment_action_policy"],
                    "environment action policy",
                ),
                prompt_source_repository=_text(
                    interactive_row["prompt_source_repository"], "prompt repository"
                ),
                prompt_source_revision=_text(
                    interactive_row["prompt_source_revision"], "prompt revision"
                ),
                prompt_source_path=_text(interactive_row["prompt_source_path"], "prompt path"),
                prompt_source_symbol=_text(
                    interactive_row["prompt_source_symbol"], "prompt symbol"
                ),
            )
        stop = tuple(_text(item, "stop") for item in _list(decoding["stop"], "stop"))
        seeds = tuple(
            _integer(item, "aggregation seed") for item in _list(aggregation["seeds"], "seeds")
        )
        contract = ExecutionContractV4(
            benchmark=benchmark,
            condition_id=_text(row["condition_id"], "condition ID"),
            lane=ExecutionLane(_text(row["lane"], "lane")),
            formal_eligibility=FormalEligibility(
                _text(row["formal_eligibility"], "formal eligibility")
            ),
            actor_model=_text(row["actor_model"], "actor model"),
            actor_model_revision=_text(row["actor_model_revision"], "actor revision"),
            actor_route=_text(row["actor_route"], "actor route"),
            actor_service_profile=_text(row["actor_service_profile"], "service profile"),
            context_length=_integer(row["context_length"], "context length"),
            adapter_policy=_text(row["adapter_policy"], "adapter policy"),
            population_id=_text(row["population_id"], "population ID"),
            dataset_revision=_text(row["dataset_revision"], "dataset revision"),
            selection_rule=_text(row["selection_rule"], "selection rule"),
            expected_count=_integer(row["expected_count"], "expected count"),
            panel_manifest_id=_text(row["panel_manifest_id"], "panel manifest ID"),
            prompt_profile=_text(row["prompt_profile"], "prompt profile"),
            thinking_mode=ThinkingMode(_text(row["thinking_mode"], "thinking mode")),
            decoding_profile=_text(row["decoding_profile"], "decoding profile"),
            decoding=DecodingParameters(
                sampling_mode=_text(decoding["sampling_mode"], "sampling mode"),
                temperature=_number(decoding["temperature"], "temperature"),
                top_p=_number(decoding["top_p"], "top p"),
                top_k=_integer(decoding["top_k"], "top k"),
                min_p=_number(decoding["min_p"], "min p"),
                presence_penalty=_number(decoding["presence_penalty"], "presence penalty"),
                repetition_penalty=_number(decoding["repetition_penalty"], "repetition penalty"),
                max_new_tokens=_integer(decoding["max_new_tokens"], "max new tokens"),
                seed=_integer(decoding["seed"], "seed"),
                stop=stop,
            ),
            parser_profile=_text(row["parser_profile"], "parser profile"),
            completion_profile=_optional_text(row["completion_profile"], "completion profile"),
            tool_surface=tuple(
                _text(item, "tool surface") for item in _list(row["tool_surface"], "tools")
            ),
            environment_profile=_optional_text(row["environment_profile"], "environment profile"),
            scorer_profile=_text(row["scorer_profile"], "scorer profile"),
            grader_profile=_optional_text(row["grader_profile"], "grader profile"),
            metric_profile=_text(row["metric_profile"], "metric profile"),
            aggregation=AggregationIdentity(
                seeds=seeds,
                run_count=_integer(aggregation["run_count"], "run count"),
                reducer=_text(aggregation["reducer"], "reducer"),
                dispersion=_optional_text(aggregation["dispersion"], "dispersion"),
            ),
            interactive=interactive,
            code_test_suite=_optional_text(row["code_test_suite"], "code test suite"),
            protocol_version=_text(row["protocol_version"], "protocol version"),
        )
        population = populations[benchmark]
        if (
            contract.population_id,
            contract.dataset_revision,
            contract.selection_rule,
            contract.expected_count,
            contract.panel_manifest_id,
        ) != (
            population.population_id,
            population.dataset_revision,
            population.selection_rule,
            population.expected_count,
            population.panel_manifest_id,
        ):
            raise ValueError("execution population differs from Protocol 14 sources")
        output[benchmark] = contract
    if tuple(output) != ACTIVE_PROTOCOL14_BENCHMARKS:
        raise ValueError("Protocol 14 conditions must contain the ordered exact eight")
    return output


def load_targets_v5(
    path: Path,
    *,
    executions: dict[Protocol14Benchmark, ExecutionContractV4],
) -> dict[Protocol14Benchmark, ReferenceTargetV5]:
    root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "targets")
    if set(root) != {"format", "targets"} or root["format"] != ("skillev-current-iid-targets@5"):
        raise ValueError("invalid Protocol 14 target registry")
    raw_targets = _mapping(root["targets"], "target rows")
    if tuple(raw_targets) != tuple(item.value for item in ACTIVE_PROTOCOL14_BENCHMARKS):
        raise ValueError("Protocol 14 target rows differ from exact-eight catalog")
    output: dict[Protocol14Benchmark, ReferenceTargetV5] = {}
    fields = {
        "status",
        "scope",
        "condition_id",
        "reference_condition_id",
        "reference_receipt_id",
        "frozen_at",
        "evidence",
        "metrics",
        "published_anchors",
    }
    for benchmark in ACTIVE_PROTOCOL14_BENCHMARKS:
        row = _mapping(raw_targets[benchmark.value], "target")
        if set(row) != fields:
            raise ValueError("Protocol 14 target fields differ")
        execution = executions[benchmark]
        if _text(row["condition_id"], "condition ID") != execution.condition_id:
            raise ValueError("target condition differs from execution")
        reference_condition = row["reference_condition_id"]
        reference_execution = None
        if reference_condition is not None:
            if _text(reference_condition, "reference condition") != execution.condition_id:
                raise ValueError("reference condition differs from candidate condition")
            reference_execution = execution
        evidence = _load_evidence(row["evidence"])
        metrics = tuple(
            _load_reference_metric(value, index=index)
            for index, value in enumerate(_list(row["metrics"], "metrics"))
        )
        anchors = tuple(
            _load_published_anchor(value, index=index)
            for index, value in enumerate(_list(row["published_anchors"], "anchors"))
        )
        output[benchmark] = ReferenceTargetV5(
            benchmark=benchmark,
            status=TargetStatus(_text(row["status"], "target status")),
            scope=TargetScope(_text(row["scope"], "target scope")),
            observed_execution=execution,
            reference_execution=reference_execution,
            metrics=metrics,
            evidence=evidence,
            reference_receipt_id=_optional_text(
                row["reference_receipt_id"], "reference receipt ID"
            ),
            frozen_at=_optional_text(row["frozen_at"], "frozen at"),
            published_anchors=anchors,
        )
    return output


def _load_reference_metric(value: object, *, index: int) -> ReferenceMetricV4:
    row = _mapping(value, f"metric {index}")
    if set(row) != {
        "metric_id",
        "unit",
        "denominator_id",
        "projection",
        "formula",
        "aggregation",
        "reference_percent",
        "maximum_gap_pp_exclusive",
        "required_for_formal_gate",
    }:
        raise ValueError("Protocol 14 target metric fields differ")
    return ReferenceMetricV4(
        metric_id=_text(row["metric_id"], "metric ID"),
        unit=_text(row["unit"], "unit"),
        denominator_id=_text(row["denominator_id"], "denominator ID"),
        projection=MetricProjection(_text(row["projection"], "projection")),
        formula=_text(row["formula"], "formula"),
        aggregation=_text(row["aggregation"], "aggregation"),
        reference_percent=Decimal(_text(row["reference_percent"], "reference percent")),
        maximum_gap_pp_exclusive=Decimal(_text(row["maximum_gap_pp_exclusive"], "maximum gap")),
        required_for_formal_gate=_boolean(row["required_for_formal_gate"], "formal metric"),
    )


def _load_published_anchor(value: object, *, index: int) -> PublishedAnchor:
    row = _mapping(value, f"published anchor {index}")
    if set(row) != {
        "label",
        "scope",
        "metric_id",
        "value_percent",
        "usable_as_formal_target",
        "evidence",
    }:
        raise ValueError("published anchor fields differ")
    evidence = _load_evidence(row["evidence"])
    if evidence is None:
        raise ValueError("published anchor evidence is required")
    return PublishedAnchor(
        label=_text(row["label"], "anchor label"),
        scope=TargetScope(_text(row["scope"], "anchor scope")),
        metric_id=_text(row["metric_id"], "anchor metric"),
        value_percent=Decimal(_text(row["value_percent"], "anchor value")),
        evidence=evidence,
        usable_as_formal_target=_boolean(row["usable_as_formal_target"], "anchor eligibility"),
    )


def _load_evidence(value: object) -> EvidenceLocator | None:
    if value is None:
        return None
    row = _mapping(value, "evidence")
    if set(row) != {
        "source_kind",
        "repository_or_paper",
        "revision_or_version",
        "path_or_section",
        "supported_fields",
    }:
        raise ValueError("evidence fields differ")
    return EvidenceLocator(
        source_kind=_text(row["source_kind"], "source kind"),
        repository_or_paper=_text(row["repository_or_paper"], "source"),
        revision_or_version=_text(row["revision_or_version"], "source version"),
        path_or_section=_text(row["path_or_section"], "source section"),
        supported_fields=tuple(
            _text(item, "supported field")
            for item in _list(row["supported_fields"], "supported fields")
        ),
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a mapping")
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


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


__all__ = ["load_execution_contracts_v4", "load_targets_v5"]
