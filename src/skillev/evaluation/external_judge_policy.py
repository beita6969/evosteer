"""Answer-free identities for the owner's future external-judge condition.

These settings do not change the Qwen task owner or any native/rule scorer.
The old Omni identity remains readable, never relabelled as the new API run.
"""

EXTERNAL_JUDGE_MODEL = "lab-gpt-5.6-luna"
EXTERNAL_JUDGE_EFFORT = "medium"
EXTERNAL_JUDGE_API_BASE = "https://llm-gateway.example.org/v1"
EXTERNAL_JUDGE_PROVIDER = "flowsteer-third-party"
EXTERNAL_JUDGE_KEY_ENV = "SKILLEV_JUDGE_API_KEY"
EXTERNAL_JUDGE_MAX_TOKENS = 8000
EXTERNAL_JUDGE_ROUTING = "flowsteer-first-official-fallback-no-post-replay@1"

OMNI_JUDGE_PROFILE = "omni-official-equivalence-prompt-luna-medium-provider-failover@4"
OMNI_JUDGE_METRIC = "luna-medium-api-equivalence-accuracy"
OMNI_JUDGE_VERIFIER = "omni-math-official-prompt-luna-medium-provider-failover@ood4"
GATEWAY_OMNI_JUDGE_PROFILE = "omni-official-equivalence-prompt-luna-medium-flowsteer@3"
GATEWAY_OMNI_JUDGE_VERIFIER = "omni-math-official-prompt-lab-gpt56-luna-medium-flowsteer@ood3"

DIRECT_JUDGE_MODEL = "gpt-5.6-luna"
DIRECT_JUDGE_API_BASE = "https://api.openai.com/v1"
DIRECT_OMNI_JUDGE_PROFILE = "omni-official-equivalence-prompt-luna-medium-api@2"
DIRECT_OMNI_JUDGE_VERIFIER = "omni-math-official-prompt-gpt56-luna-medium-api@ood2"

LEGACY_OMNI_JUDGE_PROFILE = "omni-official-equivalence-prompt-luna-high@1"
LEGACY_OMNI_JUDGE_METRIC = "luna-high-equivalence-accuracy"
LEGACY_OMNI_JUDGE_VERIFIER = "omni-math-official-prompt-gpt56-luna-high@ood1"

HEALTHBENCH_EXTERNAL_PROFILE = "healthbench-luna-medium-provider-failover-per-rubric@3"
GATEWAY_HEALTHBENCH_PROFILE = "healthbench-luna-medium-flowsteer-per-rubric@2"
DIRECT_HEALTHBENCH_PROFILE = "healthbench-luna-medium-api-per-rubric@1"
