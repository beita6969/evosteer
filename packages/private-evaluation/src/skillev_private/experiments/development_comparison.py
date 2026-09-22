"""Private, all-source paired development report; no collection, regrading or IID admission."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from skillev.contracts.skill_exposure import CATALOG_EXPOSURES
from skillev.contracts.skill_invocation import parse_action_invocation
from skillev.diagnostics.rollout_progress import phase_summary
from skillev.rollout.codec import codec_for_initial_meta
from skillev.runtime.execution import ActionParseStatus
from skillev.training.action_metrics import ActionMetrics
from skillev.training.invocation_evidence import catalog_read_result

from .development_collection import FORMAT, aggregate, read


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def delta(before: Any, after: Any) -> Any:
    """Only shared, actually measured numeric fields have a difference."""
    if isinstance(before, dict) and isinstance(after, dict):
        return {
            key: delta(before.get(key), after.get(key))
            for key in sorted(before.keys() | after.keys())
        }
    return after - before if number(before) and number(after) else None


def differences(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [
            row
            for key in sorted(before.keys() | after.keys())
            for row in differences(before.get(key), after.get(key), f"{path}/{key}")
        ]
    return [{"path": path, "before": before, "after": after}]


def action_evidence(artifact: dict[str, Any], assessments: ActionMetrics) -> dict[str, Any]:
    record = artifact["record"]
    steps = record["steps"]
    meta = record["initial_context"]["meta"]
    codec = codec_for_initial_meta(meta)
    parsed = [codec.parse(step["action_text"]) for step in steps]
    receipts = []
    for step in steps:
        action = parse_action_invocation(step["action_text"], initial_meta=meta)
        if action is not None and action.skill_id is not None:
            receipts.append(
                catalog_read_result(
                    meta,
                    action.skill_id,
                    step["invoked_skill_ids"],
                    step["observation_status"],
                    step["observation_text"],
                )[0]
            )
    inputs = artifact.get("skill_input_evidence")
    manifest = artifact["manifest"]
    result: dict[str, Any] = {
        "action_count": len(steps),
        "structure_valid_count": sum(p.status is ActionParseStatus.VALID for p in parsed),
        **assessments.for_records("collection", [record]),
        "skill_invocation_count": sum(len(step["invoked_skill_ids"]) for step in steps),
        "skill_body_returned_count": sum(r is True for r in receipts)
        if meta.get("skill_exposure") in CATALOG_EXPOSURES and all(r is not None for r in receipts)
        else None,
        "body_visible_input_count": sum(bool(row["visible_skill_body_refs"]) for row in inputs)
        if isinstance(inputs, list)
        and all(isinstance(row.get("visible_skill_body_refs"), list) for row in inputs)
        else None,
    }
    for phase in ("reasoning", "action"):
        counts = (
            [step.get("action_token_count") for step in steps]
            if phase == "action"
            else manifest.get("reasoning_token_counts")
        )
        reasons = manifest.get(f"{phase}_finish_reasons")
        result[f"{phase}_output_tokens"] = (
            sum(counts)
            if isinstance(counts, list)
            and len(counts) == len(steps)
            and all(type(c) is int for c in counts)
            else None
        )
        result[f"{phase}_length_cap_hits"] = (
            sum(r == "length" for r in reasons)
            if isinstance(reasons, list) and len(reasons) == len(steps)
            else None
        )
    # Thinking is actual emitted reasoning, not an inference from success or a budget.
    counts = manifest.get("reasoning_token_counts")
    result["turns_with_reasoning_output"] = (
        sum(c > 0 for c in counts)
        if isinstance(counts, list) and len(counts) == len(steps)
        else None
    )
    return result


def inspect(
    root: Path, frozen: dict[str, Any], *, live_elapsed_seconds: float | None = None
) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = read(summary_path)
    elif live_elapsed_seconds is not None:
        if not number(live_elapsed_seconds) or live_elapsed_seconds < 0:
            raise ValueError("live elapsed time must be measured and nonnegative")
        summary = aggregate(root, frozen, elapsed_seconds=live_elapsed_seconds)
    else:
        raise FileNotFoundError(summary_path)
    if summary.get("format") != FORMAT or summary.get("record_kind") != "development-evaluation":
        raise ValueError("only original development collection outputs are supported")
    controls_path = root / "controls-private.json"
    controls = read(controls_path) if controls_path.exists() else None
    if controls is not None and controls.get("selection") != frozen:
        raise ValueError("saved controls and source selection disagree")
    progress_path = root / "collection/episodes/rollout-progress.json"
    progress = read(progress_path) if progress_path.exists() else None
    trajectories = (
        {} if progress is None else {r["trajectory_id"]: r for r in progress["trajectories"]}
    )
    rows = []
    all_phases = []
    with sqlite3.connect(":memory:") as connection:
        assessments = ActionMetrics(connection)
        events = root / "collection/events.jsonl"
        if events.exists():
            with events.open() as stream:
                for line in stream:
                    # A live writer may not have finished its last JSONL record yet.
                    # Never infer an assessment from that partial line.
                    if live_elapsed_seconds is not None and not line.endswith("\n"):
                        break
                    event = json.loads(line)
                    if event.get("run_id") == "collection" and "assessment" in event.get(
                        "payload", {}
                    ):
                        assessments.observe(event)
        for position, source in enumerate(frozen["source_manifest"]["ordered_sources"]):
            relative = f"collection/episodes/episode-{position:06d}-private.json"
            path = root / relative
            outcome = read(path) if path.exists() else {}
            if outcome and (
                outcome.get("position") != position or outcome.get("task_id") != source["task_id"]
            ):
                raise ValueError("saved outcome differs from the fixed source position")
            artifact = outcome.get("artifact")
            reward: Any = artifact["record"]["reward"] if artifact else None
            native_complete = (
                reward is not None
                and number(reward.get("value"))
                and type(reward.get("success")) is bool
            )
            row: dict[str, Any] = {
                "position": position,
                "source": source,
                "artifact_path": relative if path.exists() else None,
                "execution_status": outcome.get("execution_status", "missing"),
                "infrastructure_error": outcome.get("infrastructure_error"),
                "preparation_chunk_start": outcome.get("preparation_chunk_start"),
                "native_complete": native_complete,
                "reward": reward["value"] if native_complete else None,
                "success": reward["success"] if native_complete else None,
                "native_metrics": None,
                "actions": None,
                "phase_totals": None,
                "sampling_identity": None,
            }
            if artifact:
                row["sampling_identity"] = {
                    key: artifact["manifest"].get(key)
                    for key in ("sampling_coordinate", "policy_snapshot", "library_version")
                }
                if artifact["manifest"]["task_id"] != source["task_id"]:
                    raise ValueError("artifact task differs from its fixed source")
                native = reward["native_payload"].get("public_metrics", {})
                row["native_metrics"] = (
                    {k: v if number(v) else None for k, v in native.items()}
                    if isinstance(native, dict) and native
                    else {reward["native_metric_name"]: reward["value"]}
                )
                row["actions"] = action_evidence(artifact, assessments)
                trace = trajectories.get(artifact["record"]["trajectory_id"])
                if trace is not None:
                    if trace.get("canonical_position") != position:
                        raise ValueError("phase evidence has a different source coordinate")
                    phases = trace.get("phases")
                    if isinstance(phases, list):
                        row["phase_totals"] = phase_summary(phases)
                        all_phases.extend(phases)
            rows.append(row)
    failure_path = root / "failure-private.json"
    failure = read(failure_path) if failure_path.exists() else None
    complete = (
        summary_path.exists()
        and bool(rows)
        and all(
            r["native_complete"]
            and r["execution_status"] == "completed"
            and r["infrastructure_error"] is None
            for r in rows
        )
        and failure is None
        and (root / "collection/collection.json").exists()
        and controls is not None
    )
    measured = aggregate(root, frozen, elapsed_seconds=summary["elapsed_seconds"])
    return {
        "root": str(root),
        "complete": complete,
        "rows": rows,
        "domains": measured["domains"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "consumed_budget": measured["consumed_budget"],
        "phase_totals": phase_summary(all_phases) if progress is not None else None,
        "phase_observed_sources": sum(row["phase_totals"] is not None for row in rows),
        "failure": None
        if failure is None
        else {"error_type": failure.get("error_type"), "evidence_path": str(failure_path)},
        "controls": controls,
    }


def compare(before: Path, after: Path) -> dict[str, Any]:
    frozen = read(before / "selection-private.json")
    if frozen != read(after / "selection-private.json"):
        raise ValueError("paired development requires the exact same saved selection")
    if frozen.get("format") != FORMAT or frozen.get("purpose") != "development-only":
        raise ValueError("this is not a fixed development selection")
    left, right = inspect(before, frozen), inspect(after, frozen)
    complete = left["complete"] and right["complete"]
    pairs = []
    for a, b in zip(left.pop("rows"), right.pop("rows"), strict=True):
        paired = a["native_complete"] and b["native_complete"]
        if paired and a["sampling_identity"] != b["sampling_identity"]:
            raise ValueError(
                "actual source/sample/policy/library coordinates changed between candidates"
            )
        pairs.append(
            {
                "position": a["position"],
                "source": a["source"],
                "before": a,
                "after": b,
                "reward_delta": b["reward"] - a["reward"] if paired else None,
                "success_delta": int(b["success"]) - int(a["success"]) if paired else None,
                "native_delta": delta(a["native_metrics"], b["native_metrics"]) if paired else None,
            }
        )
    declared = differences(left.pop("controls"), right.pop("controls"))
    return {
        "format": "fixed-source-development-comparison@1",
        "purpose": "development-only",
        "comparison_id": frozen["comparison_id"],
        "complete_paired_comparison": complete,
        "planned_source_count": len(pairs),
        "pairs": pairs,
        "before": left,
        "after": right,
        "domain_deltas": delta(left["domains"], right["domains"]) if complete else None,
        "declared_condition_differences": declared,
        "interpretation": (
            "Paired descriptive development outcomes, not IID acceptance or causal efficacy. "
            "Changed formal/prompt/budget controls are condition changes, not "
            "infrastructure-equivalent speedups. Phase sums are request-chain costs, "
            "not disjoint wall time; missing measurements remain unknown. "
            "Body visibility is not proof of following a skill."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.before, args.after)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    if not report["complete_paired_comparison"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
