"""LiveMedBench public prompt and explicit non-official-judge metric identity.

The complete-response condition differs from the released no-explanation prompt.
No clinical reference, doctor response, rubric or scorer is imported here.
"""

# ruff: noqa: RUF001 -- patient-facing Chinese punctuation is intentional.

BENCHMARK = "livemedbench"
SNAPSHOT = "v202601"
METRIC = "luna-medium-api-rubric-score"
VERIFIER = "livemedbench-official-rubric-luna-medium-provider-failover@3"
PROFILE = "livemedbench-official-per-criterion-luna-medium-provider-failover@3"
GATEWAY_VERIFIER = "livemedbench-official-rubric-lab-gpt56-luna-medium-flowsteer@2"
DIRECT_VERIFIER = "livemedbench-official-rubric-gpt56-luna-medium-api@1"


def generation_instruction(narrative: str, core_request: str) -> str:
    """Task-independent patient-facing guidance; never constructed from rubrics."""
    if any("\u4e00" <= char <= "\u9fff" for char in f"{narrative}\n\n{core_request}"):
        return (
            "请用中文给患者一个完整、清楚、实用的答复，回应其具体问题。"
            "可以解释医学依据、重要的不确定性和适当的下一步；不要只给结论或药名清单。"
            "根据已提供的信息作答，不虚构病史、检查结果或诊断。"
            "无需展示内部推理过程，但应保留患者理解建议所需的解释。"
        )
    return (
        "Give the patient a complete, clear and useful response addressing their specific "
        "questions. Explain relevant medical reasoning, important uncertainty and appropriate "
        "next steps rather than giving only a conclusion or a list of treatments. Use the "
        "information provided; do not invent history, findings or a confirmed diagnosis. "
        "Do not expose an internal reasoning transcript; do include explanations the patient "
        "needs to understand your advice."
    )
