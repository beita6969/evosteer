"""Frozen LiveMedBench sources and terminal-only, per-criterion Luna grading.

The owner sees narrative/core_request only. Official prompts and arithmetic are
reused without the upstream silent-skip, parse-failure-as-zero or hidden retries.
All case contents, generated answers and criterion ledgers stay private.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from skillev.evaluation.external_judge_policy import (
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_MAX_TOKENS,
    EXTERNAL_JUDGE_MODEL,
    EXTERNAL_JUDGE_PROVIDER,
    EXTERNAL_JUDGE_ROUTING,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.livemedbench import BENCHMARK, PROFILE
from skillev.evaluation.sealed_candidates import FinalCandidate

from .external_judge_api import make_external_judge_client
from .livemedbench_upstream import (
    calculate_case_total_score,
    calculate_max_possible_score,
    create_evaluation_prompt,
)


def validate_rubrics(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise ValueError("LiveMedBench requires a nonempty rubric population")
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("criterion"), str)
            or not item["criterion"].strip()
            or type(item.get("points")) not in (int, float)
            or not math.isfinite(item["points"])
        ):
            raise ValueError("LiveMedBench rubric is malformed; do not skip it")
    return items


def export_livemedbench(
    source: dict[str, Any], directory: Path, *, count: int, seed: int
) -> dict[str, Any]:
    from .ood_export import sample_rows

    paths, snapshot = source["paths"], source["snapshot"]
    if len(paths) != 1 or Path(paths[0]).name != f"LiveMedBench_{snapshot}.json":
        raise ValueError("select exactly one explicitly named LiveMedBench snapshot")
    rows = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(type(row.get("case_id")) is not int for row in rows):
        raise ValueError("LiveMedBench source must be a JSON array with integer case IDs")
    ids = [row["case_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("LiveMedBench source contains duplicate cases")
    selected, population = sample_rows(rows, count, seed)
    projected = []
    for _, row in selected:
        public = {name: row[name] for name in ("narrative", "core_request")}
        task_id = f"{BENCHMARK}:{snapshot}:{row['case_id']}"
        PublicTaskView.from_record(task_id, BENCHMARK, public)
        projected.append(
            {
                "task_id": task_id,
                "public": public,
                "target": {
                    "case_id": row["case_id"],
                    "post_time": row["post_time"],
                    "snapshot": snapshot,
                    "source_metadata": {
                        key: row[key]
                        for key in ("language", "specialty", "department")
                        if key in row
                    },
                    **public,
                    "rubric_items": validate_rubrics(row["rubric_items"]),
                },
            }
        )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "livemedbench-private.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for record in projected:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "ood_sources": {BENCHMARK: str(path)},
        "evaluation_sample_counts": {BENCHMARK: count},
        "ood_provenance": {
            BENCHMARK: {
                "dataset": "JuelieYann/LiveMedBench",
                "snapshot": snapshot,
                "population": population,
                "sampling": "uniform-reservoir-source-order",
                "seed": seed,
                "case_ids": [row["case_id"] for _, row in selected],
                "selected_rubric_count": sum(len(row["rubric_items"]) for _, row in selected),
                "judge_profile": PROFILE,
            }
        },
    }


def parse_verdict(text: str, criterion: str) -> tuple[bool, str]:
    """Only deterministic JSON-fence removal; no semantic answer repair."""
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    value = json.loads(fenced[1] if fenced else text)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("one criterion must receive exactly one verdict")
    item = value[0]
    if item.get("question", "").strip() != criterion.strip():
        raise ValueError("judge did not identify the requested criterion")
    met = item.get("met")
    if isinstance(met, str) and met.lower() in {"true", "false"}:
        met = met.lower() == "true"
    if type(met) is not bool or not isinstance(item.get("reasoning"), str):
        raise ValueError("judge verdict is missing a boolean or its rationale")
    return met, item["reasoning"]


def request_rubric(client: Any, prompt: str, criterion: str) -> dict[str, Any]:
    """One bounded request, including incomplete/error accounting. Never retry."""
    from openai import APIError

    started = time.monotonic()
    result: dict[str, Any] = {"status": "transport-error", "usage": None}
    try:
        response = client.chat.completions.create(
            model=EXTERNAL_JUDGE_MODEL,
            reasoning_effort=EXTERNAL_JUDGE_EFFORT,
            max_completion_tokens=EXTERNAL_JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
            stream=False,
            store=False,
        )
    except (APIError, TimeoutError, ConnectionError) as error:
        result.update(
            error_type=type(error).__name__,
            http_status=getattr(error, "status_code", None),
            request_id=getattr(error, "request_id", None),
            provider=getattr(error, "judge_provider", None),
            endpoint=getattr(error, "judge_endpoint", None),
        )
    else:
        result.update(
            response_id=response.id,
            request_id=getattr(response, "_request_id", None),
            returned_model=response.model,
            provider=getattr(response, "judge_provider", None),
            endpoint=getattr(response, "judge_endpoint", None),
            provider_requested_model=getattr(response, "judge_requested_model", None),
            failover_reason=getattr(response, "judge_failover_reason", None),
            usage=None if response.usage is None else response.usage.model_dump(),
            status="invalid-response",
        )
        if len(response.choices) == 1:
            choice = response.choices[0]
            result.update(raw_output=choice.message.content, finish_reason=choice.finish_reason)
            if choice.finish_reason != "stop" or choice.message.refusal:
                result["status"] = "incomplete-or-refused"
            else:
                try:
                    met, reason = parse_verdict(choice.message.content or "", criterion)
                except (ValueError, TypeError, AttributeError):
                    result["status"] = "parse-error"
                else:
                    result.update(status="resolved", met=met, reasoning=reason)
    result["wall_seconds"] = time.monotonic() - started
    return result


def rubric_result(items: list[dict[str, Any]], verdicts: list[bool]) -> dict[str, Any]:
    validate_rubrics(items)
    if len(items) != len(verdicts) or any(type(met) is not bool for met in verdicts):
        raise ValueError("all criteria must be scored; missing judgements are not zero")
    evaluations = {
        f"rubric_{index}": {
            "criterion": item["criterion"],
            "points": item["points"],
            "score": int(met),
            "weighted_score": item["points"] * int(met),
        }
        for index, (item, met) in enumerate(zip(items, verdicts, strict=True), 1)
    }
    total = calculate_case_total_score(evaluations, items)
    maximum = calculate_max_possible_score(items)
    # The released population includes negative-only cases. Upstream's
    # calculate_model_scores explicitly scores these zero; never drop/resample.
    raw = total / maximum if maximum > 0 else 0.0
    return {
        "evaluations": evaluations,
        "raw_score": raw,
        "positive_denominator": maximum,
        "earned_points": total,
        "triggered_negative_count": sum(
            met and item["points"] < 0 for item, met in zip(items, verdicts, strict=True)
        ),
        "terminal_reward_projection": max(0.0, min(1.0, raw)),
        "positive_criteria_count": sum(item["points"] > 0 for item in items),
        "positive_criteria_met": sum(
            item["points"] > 0 and met for item, met in zip(items, verdicts, strict=True)
        ),
        "positive_earned_points": sum(
            item["points"]
            for item, met in zip(items, verdicts, strict=True)
            if item["points"] > 0 and met
        ),
        "negative_triggered_points": sum(
            item["points"]
            for item, met in zip(items, verdicts, strict=True)
            if item["points"] < 0 and met
        ),
    }


def grade_case(
    candidate: FinalCandidate,
    target: dict[str, Any],
    settings: dict[str, Any],
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Read finalized owner answer and cache every rubric result, true OR false.

    Completed cases/criteria are reused only for the exact same submission.
    A failed request is preserved and stops this condition, not silently retried.
    """
    if (settings.get("profile"), settings.get("judge_model"), settings.get("reasoning_effort")) != (
        PROFILE,
        EXTERNAL_JUDGE_MODEL,
        EXTERNAL_JUDGE_EFFORT,
    ):
        raise ValueError("LiveMedBench requires its declared Luna medium judge condition")
    items = validate_rubrics(target["rubric_items"])
    if type(target["case_id"]) is not int:
        raise ValueError("case identity must be numeric")
    directory = Path(settings["cache_directory"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"case-{target['case_id']}-private.json"
    identity = {
        "scope": [
            candidate.run_id,
            candidate.arm_id,
            candidate.episode_id,
            candidate.owner_call_id,
        ],
        "candidate": candidate.text,
        "target": target,
        "profile": PROFILE,
        "judge_model": EXTERNAL_JUDGE_MODEL,
        "reasoning_effort": EXTERNAL_JUDGE_EFFORT,
        "max_completion_tokens": EXTERNAL_JUDGE_MAX_TOKENS,
        "endpoint": EXTERNAL_JUDGE_API_BASE,
        "provider": EXTERNAL_JUDGE_PROVIDER,
    }
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("saved rubric calls belong to a different submission or condition")
    else:
        state = {"identity": identity, "attempts": [], "complete": False}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False)

    def save() -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    owned_client = client is None
    if owned_client:
        client = make_external_judge_client()
    try:
        for index, item in enumerate(items):
            if index < len(state["attempts"]):
                attempt = state["attempts"][index]
            else:
                prompt = create_evaluation_prompt(
                    item["criterion"],
                    candidate.text,
                    f"{target['narrative']}\n\n{target['core_request']}",
                )
                attempt = request_rubric(client, prompt, item["criterion"])
                state["attempts"].append({"rubric_index": index, "prompt": prompt, **attempt})
                save()
            if attempt["status"] != "resolved":
                raise RuntimeError("LiveMedBench judge failed; see private criterion ledger")
        result = rubric_result(items, [attempt["met"] for attempt in state["attempts"]])
        cost = {
            "model_calls": float(len(state["attempts"])),
            "wall_seconds": sum(a["wall_seconds"] for a in state["attempts"]),
            "unknown_usage_calls": float(sum(a["usage"] is None for a in state["attempts"])),
        }
        for name, field in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            cost[name] = sum((a["usage"] or {}).get(field, 0) for a in state["attempts"])
        state.update(result=result, cost=cost, complete=True)
        save()
        return {
            **result,
            "cost": cost,
            "requested_judge_model": EXTERNAL_JUDGE_MODEL,
            "returned_judge_models": [a.get("returned_model") for a in state["attempts"]],
            "judge_providers": [a.get("provider") for a in state["attempts"]],
            "routing_policy": EXTERNAL_JUDGE_ROUTING,
            "source_metadata": target.get("source_metadata", {}),
            "snapshot": target.get("snapshot"),
        }
    finally:
        if owned_client:
            client.close()
