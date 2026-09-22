"""Deterministic, answer-free Markdown rendering for Protocol 13."""

from __future__ import annotations

from decimal import Decimal

from .admission import MetricComparisonMode
from .catalog import ACTIVE_PROTOCOL13_BENCHMARKS


def render_protocol_v13_markdown(aggregate: dict[str, object]) -> str:
    if aggregate.get("format") != "skillev-current-iid-result@4":
        raise ValueError("unsupported Protocol 13 aggregate format")
    expected = [item.value for item in ACTIVE_PROTOCOL13_BENCHMARKS]
    if aggregate.get("catalog") != expected:
        raise ValueError("Protocol 13 catalog differs from the authoritative eight")
    rows = aggregate.get("benchmarks")
    if not isinstance(rows, dict) or list(rows) != expected:
        raise ValueError("Protocol 13 benchmark rows differ from the authoritative eight")
    lines = [
        "# Qwen3.5-9B Protocol 13 Exact-Eight IID Backbone-Only Results",
        "",
        "Seven benchmarks use a frozen 128-record panel; AIME 2026 uses all 30. "
        "This document contains no task, answer, or per-record payload.",
        "",
        "| Benchmark | N | Outcomes (binary/definitive/invalid/infra) | Metrics | "
        "Numeric / Scientific / Formal |",
        "|---|---:|---:|---|---|",
    ]
    for benchmark in expected:
        raw = rows[benchmark]
        if not isinstance(raw, dict):
            raise ValueError("benchmark row must be a mapping")
        coverage = raw.get("coverage")
        metrics = raw.get("metrics")
        admission = raw.get("admission")
        if not isinstance(coverage, dict) or not isinstance(metrics, dict):
            raise ValueError("benchmark row lacks coverage or metrics")
        if not isinstance(admission, dict):
            raise ValueError("benchmark row lacks admission")
        planned = _integer(coverage.get("planned"), "planned count")
        infra = sum(
            _integer(coverage.get(name), name)
            for name in (
                "generation_infrastructure",
                "environment_infrastructure",
                "scorer_infrastructure",
            )
        )
        definitive = _integer(coverage.get("definitive_scored"), "definitive scored")
        invalid = _integer(coverage.get("candidate_invalid"), "candidate invalid")
        total = infra + definitive + invalid
        if total != planned:
            raise ValueError("renderer detected non-conserved outcomes")
        rendered: list[str] = []
        recomputed_pass = True
        admitted_count = 0
        for metric_id, value in metrics.items():
            if not isinstance(metric_id, str) or not isinstance(value, dict):
                raise ValueError("metric must be a named mapping")
            observed = Decimal(_text(value.get("observed_percent"), "observed percent"))
            detail = f"{metric_id}={observed:.2f}%"
            target = admission.get(metric_id)
            if isinstance(target, dict):
                admitted_count += 1
                reference = Decimal(_text(target.get("reference_percent"), "reference percent"))
                recorded_gap = Decimal(_text(target.get("gap_pp"), "gap"))
                threshold = Decimal(
                    _text(
                        target.get("maximum_gap_pp_exclusive"),
                        "maximum gap",
                    )
                )
                comparison_mode = MetricComparisonMode(
                    target.get("comparison_mode", MetricComparisonMode.ABSOLUTE_PARITY)
                )
                if comparison_mode is MetricComparisonMode.MINIMUM_GOAL:
                    gap = max(reference - observed, Decimal(0))
                    passed = observed >= reference
                else:
                    gap = abs(observed - reference)
                    passed = gap < threshold
                if gap != recorded_gap or type(target.get("passed")) is not bool:
                    raise ValueError("renderer admission identity differs")
                if passed is not target["passed"]:
                    raise ValueError("renderer admission recomputation differs")
                if comparison_mode is MetricComparisonMode.MINIMUM_GOAL:
                    detail += f" (minimum goal {reference:.2f}%, shortfall {gap:.2f}pp)"
                else:
                    detail += f" (target {reference:.2f}%, gap {gap:.2f}pp)"
                recomputed_pass &= passed
            rendered.append(detail)
        numeric_status = _text(raw.get("numeric_status"), "numeric status")
        scientific_status = _text(raw.get("scientific_status"), "scientific status")
        formal_status = _text(raw.get("formal_status"), "formal status")
        if numeric_status in {"pass", "fail"}:
            if admitted_count == 0 or (numeric_status == "pass") != recomputed_pass:
                raise ValueError("renderer numeric status recomputation differs")
        if formal_status in {"pass", "fail"} and scientific_status != "matched":
            raise ValueError("unmatched target cannot have a formal verdict")
        binary_success = coverage.get("binary_success")
        binary_failure = coverage.get("binary_failure")
        binary = (
            f"{binary_success}/{binary_failure}"
            if type(binary_success) is int and type(binary_failure) is int
            else "n/a"
        )
        outcome = f"{binary}/{definitive}/{invalid}/{infra}"
        lines.append(
            f"| {benchmark} | {planned} | {outcome} | {'<br>'.join(rendered) or 'N/A'} | "
            f"{numeric_status} / {scientific_status} / {formal_status} |"
        )
    lines.extend(
        [
            "",
            "HealthBench uses the owner-required local Qwen3.5-9B rubric judge and is a "
            "diagnostic rather than an official GPT-4.1-comparable score.",
            "Undefined or execution-mismatched references fail closed. Different benchmark "
            "conditions are not macro-averaged.",
        ]
    )
    return "\n".join(lines) + "\n"


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


__all__ = ["render_protocol_v13_markdown"]
