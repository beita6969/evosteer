"""Explicit training-only format-zero adjudication; native evaluation stays unchanged."""

from __future__ import annotations

import math
from typing import cast

from skillev.contracts import JsonValue

from .external_judge_policy import (
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_MODEL,
    EXTERNAL_JUDGE_PROVIDER,
)

FORMAT_REVIEW_PROFILE = "luna-medium-format-content-review@1"
FORMAT_REVIEW_DOMAINS = ("hotpotqa", "triviaqa", "aime-2026", "healthbench", "mbpp-plus")


def format_review_condition(from_step: int) -> dict[str, JsonValue]:
    if type(from_step) is not int or from_step < 1:
        raise ValueError("format review needs a positive first optimizer step")
    return {
        "profile": FORMAT_REVIEW_PROFILE,
        "effective_from_optimizer_step": from_step,
        "model": EXTERNAL_JUDGE_MODEL,
        "reasoning_effort": EXTERNAL_JUDGE_EFFORT,
        "provider": EXTERNAL_JUDGE_PROVIDER,
        "domains": list(FORMAT_REVIEW_DOMAINS),
        "scope": "training-only-native-zero-format-failure",
        "candidate": "final-action-or-admitted-submission-only-no-reasoning-no-best-of",
        "approved_reward": 1.0,
        "approved_success": True,
        "native_result": "preserved-separately-not-an-official-native-pass",
        "retry": "no-uncertain-post-replay",
    }


def format_review_metrics(records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    """Only scalar aggregates leave the private terminal evidence boundary."""
    rewards = [cast(dict[str, JsonValue], record["reward"]) for record in records]
    originals: list[dict[str, JsonValue]] = []
    reviews: list[dict[str, JsonValue]] = []
    for reward in rewards:
        payload = cast(dict[str, JsonValue], reward["native_payload"])
        review = payload.get("format_content_review")
        if isinstance(review, dict):
            reviews.append(review)
            originals.append(cast(dict[str, JsonValue], review["native_result"]))
        else:
            originals.append(reward)
    values = [float(cast(float, original["value"])) for original in originals]
    metrics: dict[str, JsonValue] = {
        "native_reward_mean": math.fsum(values) / len(records),
        "native_success_fraction": sum(v["success"] is True for v in originals) / len(records),
        "format_review_count": len(reviews),
        "format_review_approved_count": sum(v["approved"] is True for v in reviews),
        "format_review_reward_gain": math.fsum(float(cast(float, v["value"])) for v in rewards)
        - math.fsum(values),
    }
    for field, key in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        counts: list[int] = []
        for review in reviews:
            verdict = review.get("verdict")
            usage = verdict.get("usage") if isinstance(verdict, dict) else None
            value = usage.get(key) if isinstance(usage, dict) else None
            if type(value) is int:
                counts.append(value)
        metrics[f"format_review_api_{field}"] = sum(counts) if len(counts) == len(reviews) else None
    return metrics
