"""Deterministic answer-free Protocol 14 Markdown publication."""

from __future__ import annotations

from decimal import Decimal

from .catalog import ACTIVE_PROTOCOL14_BENCHMARKS


def render_protocol_v14_markdown(aggregate: dict[str, object]) -> str:
    if aggregate.get("format") != "skillev-current-iid-result@5":
        raise ValueError("unsupported Protocol 14 aggregate")
    expected = [item.value for item in ACTIVE_PROTOCOL14_BENCHMARKS]
    if aggregate.get("catalog") != expected:
        raise ValueError("Protocol 14 catalog differs from the authoritative eight")
    rows = aggregate.get("benchmarks")
    if not isinstance(rows, dict) or list(rows) != expected:
        raise ValueError("Protocol 14 rows differ from the authoritative eight")
    formal_benchmarks = _text_list(aggregate.get("formal_gate_benchmarks"), "formal benchmarks")
    diagnostic_benchmarks = _text_list(
        aggregate.get("diagnostic_benchmarks"), "diagnostic benchmarks"
    )
    if formal_benchmarks + diagnostic_benchmarks != [
        item for item in expected if item in formal_benchmarks
    ] + [item for item in expected if item in diagnostic_benchmarks]:
        raise ValueError("Protocol 14 publication groups are not catalog ordered")
    if set(formal_benchmarks).intersection(diagnostic_benchmarks) or set(
        formal_benchmarks + diagnostic_benchmarks
    ) != set(expected):
        raise ValueError("Protocol 14 publication groups do not partition the catalog")
    lines = [
        "# Qwen3.5-9B Protocol 14 Corrected Exact-Eight IID Backbone-Only Results",
        "",
        "Seven benchmarks use frozen 128-record panels; AIME 2026 uses all 30 "
        "(926 planned records total). Formal parity uses strict absolute gap `<7pp`; "
        "relative error is reported separately. HealthBench is a Qwen3.5-9B "
        "SGLang local-judge diagnostic and is excluded from the formal gate.",
        "",
        "## Formal parity results",
        "",
        "| Benchmark | N | Valid / model-invalid / infra | Metrics | Parity / Floor / Formal |",
        "|---|---:|---:|---|---|",
    ]
    for benchmark in formal_benchmarks:
        row = _mapping(rows[benchmark], "benchmark row")
        admission = _mapping(row.get("admission"), "admission")
        planned, valid, invalid, infra = _coverage(row)
        rendered_metrics = _render_metrics(row, admission)
        lines.append(
            f"| {benchmark} | {planned} | {valid} / {invalid} / {infra} | "
            f"{'<br>'.join(rendered_metrics)} | {row['parity_status']} / "
            f"{row['capability_floor_status']} / {row['formal_status']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic results",
            "",
            "| Benchmark | N | Valid / model-invalid / infra | Metrics | Diagnostic / Formal |",
            "|---|---:|---:|---|---|",
        ]
    )
    for benchmark in diagnostic_benchmarks:
        row = _mapping(rows[benchmark], "benchmark row")
        comparison = _mapping(row.get("diagnostic_comparison"), "diagnostic comparison")
        planned, valid, invalid, infra = _coverage(row)
        rendered_metrics = _render_metrics(row, comparison, diagnostic=True)
        lines.append(
            f"| {benchmark} | {planned} | {valid} / {invalid} / {infra} | "
            f"{'<br>'.join(rendered_metrics)} | {row['diagnostic_status']} / "
            f"{row['formal_status']} |"
        )
    lines.extend(
        [
            "",
            "The HealthBench score is graded directly by the same adapter-free Qwen3.5-9B "
            "SGLang model family, never by OpenAI or GPT. Its matched local reference gap is "
            "diagnostic only and cannot grant formal admission.",
            "Published external aggregates and diagnostic judges are never substituted for "
            "standalone matched panel targets. HumanEval and HumanEval+ remain separate.",
            "This public document contains no task, answer, rubric, test, prompt, response, or "
            "per-record payload.",
        ]
    )
    return "\n".join(lines) + "\n"


def _coverage(row: dict[str, object]) -> tuple[int, int, int, int]:
    coverage = _mapping(row.get("coverage"), "coverage")
    planned = _integer(coverage.get("planned"), "planned")
    valid = _integer(coverage.get("scored_valid"), "valid")
    invalid = _integer(coverage.get("model_output_invalid"), "model invalid")
    infra = sum(
        _integer(coverage.get(name), name)
        for name in (
            "generation_infrastructure",
            "environment_infrastructure",
            "scorer_infrastructure",
        )
    )
    if valid + invalid + infra != planned:
        raise ValueError("publication detected non-conserved outcomes")
    return planned, valid, invalid, infra


def _render_metrics(
    row: dict[str, object],
    comparison: dict[str, object],
    *,
    diagnostic: bool = False,
) -> list[str]:
    metrics = _mapping(row.get("metrics"), "metrics")
    rendered: list[str] = []
    for metric_id, value in metrics.items():
        metric = _mapping(value, "metric")
        observed = Decimal(_text(metric.get("observed_percent"), "observed percent"))
        detail = f"{metric_id}={observed:.2f}%"
        if metric_id in comparison:
            compared = _mapping(comparison[metric_id], "comparison metric")
            reference = Decimal(_text(compared.get("reference_percent"), "reference"))
            gap = Decimal(_text(compared.get("absolute_gap_pp"), "absolute gap"))
            relative_raw = compared.get("relative_error_percent")
            relative = (
                "n/a"
                if relative_raw is None
                else f"{Decimal(_text(relative_raw, 'relative error')):.2f}%"
            )
            if gap != abs(observed - reference):
                raise ValueError("publication comparison gap differs")
            target_label = "diagnostic target" if diagnostic else "target"
            detail += f" ({target_label} {reference:.2f}%, gap {gap:.2f}pp, rel {relative})"
        rendered.append(detail)
    return rendered


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError(f"{label} must be a list of text")
    return value


__all__ = ["render_protocol_v14_markdown"]
