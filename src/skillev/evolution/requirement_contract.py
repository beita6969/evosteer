"""Shared typed requirement edit contract, independent of author/model execution."""

from skillev.contracts import JsonValue
from skillev.runtime import SkillDocument, SkillRequirement


def immutable_requirements(source: SkillDocument) -> tuple[SkillRequirement, ...]:
    return tuple(item for item in source.requirements if item.immutable)


def revision_contract(source: SkillDocument) -> dict[str, JsonValue]:
    return {
        "immutable_source_requirements": [
            item.to_value() for item in immutable_requirements(source)
        ],
        "evolvable_source_strategies": [
            item.to_value() for item in source.requirements if not item.immutable
        ],
        "strategy_revision_rule": (
            "Copy immutable constraints verbatim. You may correct or remove strategies. "
            "For every changed or removed source strategy, a new/changed strategy must "
            "list its ID in replaces (sorted unique) and give a nonempty change_reason "
            "grounded in this request's context/modality evidence. Do not claim private "
            "answer access or new tool permissions. New patches are evolvable-strategy."
        ),
    }


def validate_requirement_revisions(
    source: SkillDocument, requirements: tuple[SkillRequirement, ...]
) -> None:
    """Allow evidenced strategy correction, but never weaken immutable authority."""
    original = {item.requirement_id: item for item in source.requirements}
    produced = {item.requirement_id: item for item in requirements}
    if any(produced.get(item.requirement_id) != item for item in immutable_requirements(source)):
        raise ValueError("draft cannot drop, rewrite or reclassify an immutable constraint")
    changed = {
        key for key, item in original.items() if not item.immutable and produced.get(key) != item
    }
    explained: set[str] = set()
    for item in requirements:
        if original.get(item.requirement_id) == item:
            continue  # Historical revision metadata is not a new edit claim.
        if item.immutable:
            raise ValueError("author cannot promote new strategy into immutable authority")
        if not set(item.replaces) <= changed:
            raise ValueError("strategy revision refers to an unchanged or immutable requirement")
        explained.update(item.replaces)
    if explained != changed:
        raise ValueError("every removed or changed strategy needs a recorded evidence-based reason")
