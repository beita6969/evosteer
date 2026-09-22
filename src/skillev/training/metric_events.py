"""Explicit new-run axes and allowlisted native aggregates; no case data escapes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

EVENT_CONTRACT = "separated-metrics@1"
NATIVE_FIELDS = {
    "hotpotqa": {"em": "answer-exact-match", "f1": "answer-f1"},
    "triviaqa": {"em": "answer-exact-match", "f1": "answer-f1"},
    "aime-2026": {"accuracy": "accuracy"},
    "healthbench": {
        "qwen_local_rubric_score": "qwen-local-rubric-score",
        "luna_medium_rubric_score": "luna-medium-api-rubric-score",
    },
    "mbpp-plus": {
        "base_pass": "base-pass",
        "plus_pass": "plus-pass",
        "base_plus_pass": "base-plus-pass@1",
    },
    "humaneval": {"passed": "passed"},
    "alfworld": {},
}


def scalar(value: Any) -> int | float | None:
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError("public aggregate must be a finite number or null")
    return value  # type: ignore[no-any-return]


def native_fields(event: dict[str, Any]) -> dict[str, Any]:
    """Distinct native verifier metrics are not aliases to add or accuracy claims.

    An explicit null is not replaced by reward, an alias or a subset mean.
    Counts are integers; no per-question field or verifier text is forwarded.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    sources: set[tuple[str, str]] = set()
    memberships: set[tuple[str, str, str]] = set()
    unknown = 0
    for record in event["payload"]["records"]:
        native = record["reward"].get("native_payload", {})
        groups.setdefault(native.get("benchmark_id", "unknown"), []).append(native)
        source = native.get("training_evidence_source")
        if isinstance(source, dict) and all(
            isinstance(source.get(key), str) and source[key]
            for key in ("benchmark_id", "source_question_id")
        ):
            sources.add((source["benchmark_id"], source["source_question_id"]))
            if isinstance(source.get("population_id"), str) and source["population_id"]:
                memberships.add(
                    (source["benchmark_id"], source["population_id"], source["source_question_id"])
                )
        else:
            unknown += 1
    result: dict[str, Any] = {
        # New-run event contract only. Historical exporter rows retain their
        # original population-membership definition for reproducible backfills.
        "train/unique_source_question_count": len(sources),
        "train/source_population_membership_count": len(memberships),
        "train/unknown_source_trajectory_count": unknown,
        "train/source_question_count_definition": "benchmark-plus-canonical-source; aliases-merged",
    }
    for domain, fields in NATIVE_FIELDS.items():
        members = groups.get(domain, [])
        for name, key in fields.items():
            values = []
            for member in members:
                public = member.get("public_metrics", {})
                public = public if isinstance(public, dict) else {}
                value = public.get(key)
                if key == "passed" and key not in public and type(member.get("passed")) is bool:
                    value = int(member["passed"])
                values.append(scalar(value))
            observed = [v for v in values if v is not None]
            prefix = f"train/domain/{domain}/{name}"
            result[prefix] = (
                math.fsum(observed) / len(values)
                if values and len(observed) == len(values)
                else None
            )
            result[prefix + "_observed_count"] = len(observed)
    return result


def committed_native_fields(
    path: Path | None, *, run_id: str, expected_steps: dict[str, int] | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if path is None or not path.exists():
        return result
    with path.open() as source:
        for line in source:
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            if event.get("event_type") != "training_step_committed":
                continue
            if event["run_id"] != run_id:
                raise ValueError("native source event belongs to another run")
            commit = event["event_id"]
            if expected_steps is not None:
                if commit not in expected_steps:
                    continue
                if event["payload"]["optimizer_step"] != expected_steps[commit]:
                    raise ValueError("native source commit optimizer coordinate differs")
            fields = native_fields(event)
            if commit in result and result[commit] != fields:
                raise ValueError("native aggregate differs for the same source commit")
            result[commit] = fields
    return result


def training_point(
    point: dict[str, Any], supplemental: dict[str, Any] | None, native: dict[str, Any] | None
) -> dict[str, Any]:
    result = dict(point)
    if supplemental:
        result.update({k: v for k, v in supplemental.items() if not k.startswith("telemetry/")})
    result.update(native or {})
    result["record_kind"] = "committed_training_step"
    result["train/optimizer_step"] = point["optimizer_step"]
    result["train/sampled_policy_step"] = point["sampled_policy_step"]
    return result


def supplemental_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_kind": "supplemental_telemetry",
        **{k if k.startswith("telemetry/") else "telemetry/" + k: v for k, v in point.items()},
    }
