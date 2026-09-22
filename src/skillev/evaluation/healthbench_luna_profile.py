"""Explicit future HealthBench scoring condition; no task-agent configuration."""

from dataclasses import asdict, replace
from typing import cast

from skillev.contracts import JsonValue

from .external_judge_policy import (
    DIRECT_HEALTHBENCH_PROFILE,
    DIRECT_JUDGE_API_BASE,
    DIRECT_JUDGE_MODEL,
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_KEY_ENV,
    EXTERNAL_JUDGE_MAX_TOKENS,
    EXTERNAL_JUDGE_MODEL,
    EXTERNAL_JUDGE_PROVIDER,
    EXTERNAL_JUDGE_ROUTING,
    GATEWAY_HEALTHBENCH_PROFILE,
    HEALTHBENCH_EXTERNAL_PROFILE,
)
from .healthbench_official import HealthBenchExternalJudgeProfile

PROFILE_ID = HEALTHBENCH_EXTERNAL_PROFILE
MODEL = EXTERNAL_JUDGE_MODEL
METRIC = "luna-medium-api-rubric-score"
GATEWAY_VERIFIER = "healthbench-simple-evals-lab-gpt56-luna-medium-flowsteer@2"
FAILOVER_VERIFIER = "healthbench-simple-evals-luna-medium-provider-failover@3"
VERIFIER = FAILOVER_VERIFIER
DIRECT_METRIC = "luna-medium-api-rubric-score"
DIRECT_VERIFIER = "healthbench-simple-evals-gpt56-luna-medium-api@1"
LEGACY_TRAINING_JUDGE = "qwen-local@1"


def luna_profile() -> HealthBenchExternalJudgeProfile:
    """Return the current owner-selected external Judge profile.

    The compatibility name is retained for existing callers; the returned
    identity explicitly permits gateway-first, official fallback for unsent
    requests. Historical gateway-only conditions retain their own identity.
    """

    return HealthBenchExternalJudgeProfile(
        profile_id=PROFILE_ID,
        backend="openai-chat-completions",
        model=MODEL,
        rubric_source_repository="https://github.com/openai/simple-evals",
        rubric_source_revision="652c89d0ca9df547706735883097e9537d40dc47",
        rubric_source_path="healthbench_eval.py",
        endpoint_environment="SKILLEV_JUDGE_BASE_URL",
        api_key_environment=EXTERNAL_JUDGE_KEY_ENV,
        call_mode="per-rubric",
        response_format="json-object",
        max_completion_tokens=EXTERNAL_JUDGE_MAX_TOKENS,
        reasoning_effort=EXTERNAL_JUDGE_EFFORT,
        temperature=None,
        top_p=None,
        request_timeout_seconds=120.0,
        maximum_attempts=1,
    )


def external_healthbench_profile(profile_id: str) -> HealthBenchExternalJudgeProfile:
    current = luna_profile()
    if profile_id == PROFILE_ID:
        return current
    if profile_id == GATEWAY_HEALTHBENCH_PROFILE:
        return replace(current, profile_id=profile_id)
    if profile_id == DIRECT_HEALTHBENCH_PROFILE:
        return replace(
            current,
            profile_id=profile_id,
            model=DIRECT_JUDGE_MODEL,
            endpoint_environment="OPENAI_BASE_URL",
            api_key_environment="OPENAI_API_KEY",
        )
    raise ValueError("unknown future HealthBench judge condition")


def healthbench_condition(profile_id: str) -> dict[str, JsonValue]:
    direct = profile_id == DIRECT_HEALTHBENCH_PROFILE
    # Read historical identities without silently migrating a frozen run.
    profile = external_healthbench_profile(profile_id)
    fallback = profile_id == HEALTHBENCH_EXTERNAL_PROFILE
    return {
        **cast(dict[str, JsonValue], asdict(profile)),
        **({} if direct else {"provider": EXTERNAL_JUDGE_PROVIDER, "upstream_api": "responses"}),
        **(
            {
                "routing_policy": EXTERNAL_JUDGE_ROUTING,
                "fallback_provider": "openai-official",
                "fallback_endpoint": DIRECT_JUDGE_API_BASE,
                "fallback_model": DIRECT_JUDGE_MODEL,
                "fallback_upstream_api": "chat-completions",
            }
            if fallback
            else {}
        ),
        "endpoint": DIRECT_JUDGE_API_BASE if direct else EXTERNAL_JUDGE_API_BASE,
        "metric": DIRECT_METRIC if direct else METRIC,
        "verifier": DIRECT_VERIFIER
        if direct
        else FAILOVER_VERIFIER
        if fallback
        else GATEWAY_VERIFIER,
        "ttb_reward": "clip(raw-rubric-score,0,1)",
        "success": "raw-rubric-score>=0.60-and-no-negative-rubric",
    }
