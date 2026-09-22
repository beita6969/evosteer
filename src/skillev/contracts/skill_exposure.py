"""Versioned skill prompts; sampling controls and action budgets stay unchanged."""

CATALOG_EXPOSURE_VERSION = "catalog-then-read@1"
TWO_SKILL_CATALOG_EXPOSURE = "catalog-then-read@2"
PROACTIVE_CATALOG_EXPOSURE = "catalog-then-read@3"
AUTONOMOUS_CATALOG_EXPOSURES = frozenset({CATALOG_EXPOSURE_VERSION, PROACTIVE_CATALOG_EXPOSURE})
CATALOG_EXPOSURES = AUTONOMOUS_CATALOG_EXPOSURES | {TWO_SKILL_CATALOG_EXPOSURE}
SKILL_EXPOSURES = frozenset({"full-inline", *CATALOG_EXPOSURES})


def can_continue_skill_exposure(source: str, target: str) -> bool:
    return (source == "full-inline" and target in CATALOG_EXPOSURES) or (
        source in CATALOG_EXPOSURES and target in CATALOG_EXPOSURES and source != target
    )
