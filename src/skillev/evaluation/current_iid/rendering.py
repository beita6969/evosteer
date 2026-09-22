"""Deterministic Markdown renderer for machine current-IID aggregates."""

from __future__ import annotations


def render_current_iid_markdown(aggregate: dict[str, object]) -> str:
    if aggregate.get("format") != "skillev-current-iid-result@1":
        raise ValueError("unsupported current-IID aggregate format")
    rows = aggregate.get("benchmarks")
    if not isinstance(rows, dict):
        raise ValueError("current-IID aggregate has no benchmark rows")
    lines = [
        "# Qwen3.5-9B Protocol 12 Exact-Nine IID Backbone-Only Results",
        "",
        "This table is generated from machine receipts; it contains no per-item data.",
        "",
        "| Benchmark | Condition | N | Metrics | Coverage | Status |",
        "|---|---|---:|---|---:|---|",
    ]
    for benchmark, raw in rows.items():
        if not isinstance(raw, dict):
            raise ValueError("benchmark row must be a mapping")
        coverage = raw.get("coverage")
        metrics = raw.get("metrics")
        if not isinstance(coverage, dict) or not isinstance(metrics, dict):
            raise ValueError("benchmark row lacks coverage or metrics")
        rendered_metrics: list[str] = []
        for name, metric in metrics.items():
            if not isinstance(metric, dict):
                raise ValueError("metric row must be a mapping")
            observed = metric.get("observed")
            reference = metric.get("reference")
            gap = metric.get("gap_pp")
            if isinstance(observed, bool) or not isinstance(observed, int | float):
                raise ValueError("observed metric must be numeric")
            detail = f"{name}={float(observed):.2f}%"
            if reference is not None and gap is not None:
                detail += f" (target {float(reference):.2f}%, gap {float(gap):.2f}pp)"
            rendered_metrics.append(detail)
        definitive = int(coverage.get("definitive", 0))
        planned = int(coverage.get("planned", 0))
        candidate_failures = int(coverage.get("candidate_failures", 0))
        generation_infra = int(coverage.get("generation_infrastructure_failures", 0))
        scorer_infra = int(coverage.get("scorer_infrastructure_failures", 0))
        environment_infra = int(coverage.get("environment_infrastructure_failures", 0))
        coverage_text = (
            f"{definitive}/{planned}; candidate={candidate_failures}; "
            f"infra(g/s/e)={generation_infra}/{scorer_infra}/{environment_infra}"
        )
        lines.append(
            f"| {benchmark} | {raw.get('condition_id')} | {planned} | "
            f"{'<br>'.join(rendered_metrics) or 'N/A'} | {coverage_text} | "
            f"{raw.get('status')} |"
        )
    lines.extend(
        [
            "",
            "HealthBench local-Qwen grading is a diagnostic and is not an official GPT-4.1 score.",
            "Different conditions are not macro-averaged.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_protocol_v12_markdown(aggregate: dict[str, object]) -> str:
    """Render an admitted v2 aggregate without trusting a pre-rendered status."""

    if aggregate.get("format") != "skillev-current-iid-result@2":
        raise ValueError("unsupported Protocol 12 aggregate format")
    expected = [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "spreadsheetbench",
        "mbpp-plus",
        "humaneval",
    ]
    if aggregate.get("catalog") != expected:
        raise ValueError("Protocol 12 aggregate catalog differs from the authoritative nine")
    rows = aggregate.get("benchmarks")
    if not isinstance(rows, dict) or list(rows) != expected:
        raise ValueError("Protocol 12 benchmark rows differ from the authoritative nine")
    lines = [
        "# Qwen3.5-9B Protocol 12 Nine-IID Backbone-Only Results",
        "",
        "Every item is capped at 128 records; AIME 2026 uses all 30 records. "
        "The table contains no question, answer, or per-item payload.",
        "",
        "| Benchmark | N | Outcome conservation (success/fail/invalid/infra) | Metrics | Status |",
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
        planned = int(coverage["planned"])
        infra = sum(
            int(coverage[name])
            for name in (
                "generation_infrastructure",
                "environment_infrastructure",
                "scorer_infrastructure",
            )
        )
        total = (
            sum(
                int(coverage[name])
                for name in ("scored_success", "scored_failure", "candidate_invalid")
            )
            + infra
        )
        if total != planned:
            raise ValueError("renderer detected non-conserved outcomes")
        rendered: list[str] = []
        recomputed_pass = True
        for metric_id, value in metrics.items():
            if not isinstance(value, dict):
                raise ValueError("metric must be a mapping")
            observed = float(value["observed_percent"])
            detail = f"{metric_id}={observed:.2f}%"
            target = admission.get(metric_id)
            if isinstance(target, dict):
                reference = float(target["reference_percent"])
                gap = abs(observed - reference)
                passed = gap < 7.0
                if abs(gap - float(target["gap_pp"])) > 1e-8 or passed is not target["passed"]:
                    raise ValueError("renderer admission recomputation differs")
                detail += f" (target {reference:.2f}%, gap {gap:.2f}pp)"
                recomputed_pass &= passed
            rendered.append(detail)
        status = str(raw.get("status"))
        if status in {"pass", "fail"} and (status == "pass") != recomputed_pass:
            raise ValueError("renderer status recomputation differs")
        outcome = (
            f"{coverage['scored_success']}/{coverage['scored_failure']}/"
            f"{coverage['candidate_invalid']}/{infra}"
        )
        lines.append(
            f"| {benchmark} | {planned} | {outcome} | {'<br>'.join(rendered) or 'N/A'} | {status} |"
        )
    lines.extend(
        [
            "",
            "HealthBench uses the owner-requested Qwen3.5-9B local judge and is explicitly "
            "diagnostic, not an official GPT-4.1-comparable score.",
            "Undefined or condition-mismatched targets fail closed; no cross-condition macro "
            "is reported.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = ["render_current_iid_markdown", "render_protocol_v12_markdown"]
