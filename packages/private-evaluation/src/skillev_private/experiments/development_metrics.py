"""Canonical scalar-only telemetry for live and finished development collections.

No inference, regrading, training writes, or automatic application judgements.
Uses the same original-artifact/assessment joins as the complete paired report.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from .development_collection import FORMAT, read, save
from .development_comparison import inspect, number

ACTION_COUNTS = (
    "action_count",
    "structure_valid_count",
    "assessed_action_count",
    "admitted_count",
    "executed_count",
    "accepted_submission_count",
    "environment_terminal_count",
    "execution_returned_success_count",
    "skill_invocation_count",
    "skill_body_returned_count",
    "body_visible_input_count",
    "reasoning_output_tokens",
    "action_output_tokens",
    "reasoning_length_cap_hits",
    "action_length_cap_hits",
)
PHASE_MEASUREMENTS = (
    "input_tokens",
    "output_tokens",
    "client_model_queue_seconds",
    "client_transport_queue_seconds",
    "client_response_seconds",
    "server_queue_seconds",
    "server_prefill_seconds",
    "server_decode_seconds",
    "server_cached_tokens",
    "server_prefill_tokens",
    "server_prefill_span_seconds",
    "server_after_prefill_span_seconds",
)


def _sum_known(values: list[Any]) -> float | int | None:
    return sum(values) if values and all(number(value) for value in values) else None


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["native_complete"]]
    count = len(completed)
    metrics: dict[str, Any] = {
        "planned_sources": len(rows),
        "completed_sources": count,
        "pending_sources": sum(row["execution_status"] == "missing" for row in rows),
        "infrastructure_failed_sources": sum(
            row["infrastructure_error"] is not None for row in rows
        ),
        # Partial values explicitly use the completed denominator, not planned N.
        "completed/reward_mean": math.fsum(row["reward"] for row in completed) / count
        if count
        else None,
        "completed/success_count": sum(row["success"] for row in completed) if count else None,
        "completed/success_rate": sum(row["success"] for row in completed) / count
        if count
        else None,
        "completed/skill_invoking_sources": sum(
            row["actions"]["skill_invocation_count"] > 0 for row in completed
        )
        if count
        else None,
    }
    for key in ACTION_COUNTS:
        metrics[f"completed/{key}"] = _sum_known([row["actions"].get(key) for row in completed])
    actions = metrics["completed/action_count"]
    for name, key in (
        ("structural_validity", "structure_valid_count"),
        ("admission_validity", "admitted_count"),
    ):
        value = metrics[f"completed/{key}"]
        metrics[f"completed/{name}"] = value / actions if actions and number(value) else None
    for phase in ("reasoning", "action"):
        totals = [
            row["phase_totals"][phase]
            for row in completed
            if row["phase_totals"] is not None and phase in row["phase_totals"]
        ]
        metrics[f"completed/phase/{phase}/observed_sources"] = len(totals)
        metrics[f"completed/phase/{phase}/requests"] = sum(row["requests"] for row in totals)
        for key in PHASE_MEASUREMENTS:
            values = [row["measurements"].get(key, {}) for row in totals]
            measured = [row for row in values if row.get("measured_requests", 0) > 0]
            prefix = f"completed/phase/{phase}/{key}"
            metrics[f"{prefix}/measured_requests"] = sum(
                row["measured_requests"] for row in measured
            )
            metrics[f"{prefix}/measured_sum"] = _sum_known([row.get("sum") for row in measured])
    return metrics


def snapshot(root: Path, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
    frozen = read(root / "selection-private.json")
    if frozen.get("format") != FORMAT or frozen.get("purpose") != "development-only":
        raise ValueError("development telemetry needs a frozen development selection")
    report = inspect(root, frozen, live_elapsed_seconds=elapsed_seconds)
    rows = report["rows"]
    metrics = _group(rows)
    for domain in sorted({row["source"]["report_domain"] for row in rows}):
        # Domain labels are frozen public benchmark names, never source IDs or text.
        if domain not in {
            "hotpotqa",
            "triviaqa",
            "aime-historical",
            "aime-2026",
            "healthbench",
            "alfworld",
            "mbpp-plus",
            "humaneval",
        }:
            raise ValueError("unknown report domain cannot be exported as a metric name")
        for name, value in _group(
            [row for row in rows if row["source"]["report_domain"] == domain]
        ).items():
            metrics[f"domain/{domain}/{name}"] = value
    count = metrics["completed_sources"]
    elapsed = report["elapsed_seconds"]
    metrics.update(
        complete=report["complete"],
        elapsed_seconds=elapsed,
        completed_sources_per_minute=60 * count / elapsed if elapsed > 0 else None,
        # Work-count extrapolation only. Interactive long tails need phase inspection.
        rough_remaining_seconds=elapsed * (len(rows) - count) / count if count else None,
        training_updates=0,
        posterior_updates=0,
        skill_evolution=0,
        training_evidence_writes=0,
    )
    return {
        "format": "canonical-development-metrics@1",
        "scope": "development-only; not IID, training or autonomous adoption qualification",
        "phase_scope": "completed artifacts only; missing server measurements remain null",
        "input_token_scope": (
            "admitted phase input tokens including cached/repeated tokens; not physical prefill"
        ),
        "eta_scope": "rough completed-work extrapolation, not a long-tail guarantee",
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elapsed-seconds", type=float)
    args = parser.parse_args()
    save(args.output, snapshot(args.root, elapsed_seconds=args.elapsed_seconds))


if __name__ == "__main__":
    main()
