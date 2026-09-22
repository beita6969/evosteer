"""Answer-free, split-independent applicability coordinates for train and IID.

ALFWorld's subtype is used only when explicitly supplied as public metadata.
Absent subtype stays unspecified, never inferred from a private game path or
successful action trace. Tools are supplied by the actual runtime, not this map.
"""

from dataclasses import dataclass

TASK_FEATURE_MAPPING_VERSION = "benchmark-public-task@1"
_FAMILIES = {
    "hotpotqa": "multi-hop-qa",
    "triviaqa": "factual-qa",
    "aime-2026": "integer-answer",
    "healthbench": "health-dialogue",
    "webshop": "shopping",
    "alfworld": "unspecified",
    "mbpp-plus": "code-generation",
    "humaneval": "code-generation",
    "musique": "multi-hop-qa",
    "nq-open": "factual-qa",
    "omni-math": "mathematical-reasoning",
    "math-hard": "mathematical-reasoning",
    "gpqa-diamond-bioorganic": "scientific-multiple-choice",
    "livemedbench": "health-dialogue",
    "scienceworld": "scientific-interaction",
    "livecodebench": "code-generation",
    "apps-introductory": "code-generation",
}


@dataclass(frozen=True, slots=True)
class PublicTaskFeatures:
    task_family: str
    context_id: str
    mapping_version: str = TASK_FEATURE_MAPPING_VERSION

    def to_value(self) -> dict[str, str]:
        return {
            "task_family": self.task_family,
            "context_id": self.context_id,
            "mapping_version": self.mapping_version,
        }


def public_task_features(
    benchmark: str, *, task_family: str | None = None, context_id: str | None = None
) -> PublicTaskFeatures:
    if benchmark not in _FAMILIES:
        raise ValueError("no public feature mapping for this benchmark")
    family = task_family or _FAMILIES[benchmark]
    if family == "public-task":
        family = _FAMILIES[benchmark]
    if not family.startswith(f"{benchmark}/"):
        family = f"{benchmark}/{family}"
    return PublicTaskFeatures(family, context_id or f"{benchmark}:task")


def configured_public_task_features(
    benchmark: str, settings: dict[str, object]
) -> PublicTaskFeatures:
    # Explicit historical mappings remain possible, but are exported as such.
    # Never accept future horizon, outcome, or evaluator fields as route features.
    if settings.keys() - {"task_family", "context_id", "mapping_version"}:
        raise ValueError("only public task family and context may configure retrieval")
    family, context = settings.get("task_family"), settings.get("context_id")
    if any(
        value is not None and (not isinstance(value, str) or not value.strip())
        for value in (family, context)
    ):
        raise ValueError("public retrieval features must be nonempty text")
    if family is not None and not isinstance(family, str):
        raise TypeError("task family must be text")
    if context is not None and not isinstance(context, str):
        raise TypeError("context must be text")
    if (
        settings.get("mapping_version", TASK_FEATURE_MAPPING_VERSION)
        != TASK_FEATURE_MAPPING_VERSION
    ):
        raise ValueError("unsupported public task feature mapping")
    return public_task_features(benchmark, task_family=family, context_id=context)
