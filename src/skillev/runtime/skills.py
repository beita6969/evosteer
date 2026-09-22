"""Immutable Skill documents and an exact read-only catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

from skillev.contracts.canonical import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.contracts.identity import validate_identifier, validate_sha256

from .contracts import SkillManifest

SKILL_DOCUMENT_FORMAT = "skillev-skill-document@2"


def _exact_object(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise TypeError(f"{label} must be a normalized JSON object")
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _wire_text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    return value


def _text_tuple(
    values: tuple[str, ...],
    *,
    field: str,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field} cannot be empty")
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError(f"{field} must contain non-empty text")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class SkillApplicability:
    """Explicit task/tool conditions used by the production retriever."""

    task_families: tuple[str, ...]
    contexts: tuple[str, ...]
    required_tools: tuple[str, ...]
    excluded_contexts: tuple[str, ...]

    def __post_init__(self) -> None:
        _text_tuple(self.task_families, field="task_families")
        _text_tuple(self.contexts, field="contexts")
        _text_tuple(self.required_tools, field="required_tools")
        _text_tuple(self.excluded_contexts, field="excluded_contexts")
        if not self.task_families and not self.contexts:
            raise ValueError("Skill applicability requires task_families or contexts")
        for field, values in (
            ("task_families", self.task_families),
            ("contexts", self.contexts),
        ):
            if "*" in values and values != ("*",):
                raise ValueError(f"{field} wildcard must be its only value")
        if set(self.contexts) & set(self.excluded_contexts):
            raise ValueError("included and excluded contexts cannot overlap")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "contexts": list(self.contexts),
            "excluded_contexts": list(self.excluded_contexts),
            "required_tools": list(self.required_tools),
            "task_families": list(self.task_families),
        }

    @classmethod
    def from_value(cls, value: object) -> SkillApplicability:
        normalized = _exact_object(
            value,
            label="Skill applicability",
            expected={
                "contexts",
                "excluded_contexts",
                "required_tools",
                "task_families",
            },
        )
        return cls(
            task_families=_wire_text_tuple(normalized["task_families"], field="task_families"),
            contexts=_wire_text_tuple(normalized["contexts"], field="contexts"),
            required_tools=_wire_text_tuple(normalized["required_tools"], field="required_tools"),
            excluded_contexts=_wire_text_tuple(
                normalized["excluded_contexts"], field="excluded_contexts"
            ),
        )


@dataclass(frozen=True, slots=True)
class SkillRequirement:
    """Interface/security constraints are fixed; explicitly tagged strategies may evolve.

    Missing ``kind`` in historical documents means immutable, never an inferred
    permission to rewrite a constraint. Revision metadata remains in the new
    document and is bound to the original authoring request/proposal journal.
    """

    requirement_id: str
    text: str
    kind: str = "immutable-constraint"
    replaces: tuple[str, ...] = ()
    change_reason: str = ""

    def __post_init__(self) -> None:
        if not self.requirement_id.strip() or not self.text.strip():
            raise ValueError("Skill requirement identity and text cannot be empty")
        validate_identifier(self.requirement_id)
        if self.kind not in ("immutable-constraint", "evolvable-strategy"):
            raise ValueError("unknown requirement kind")
        _text_tuple(self.replaces, field="replaces")
        if self.replaces and (self.kind != "evolvable-strategy" or not self.change_reason.strip()):
            raise ValueError("strategy revisions require an explicit reason")
        if self.change_reason and not self.replaces:
            raise ValueError("revision reason requires replaced strategy IDs")

    @property
    def immutable(self) -> bool:
        return self.kind == "immutable-constraint"

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {"requirement_id": self.requirement_id, "text": self.text}
        # Keep the original representation/identity of unmodified historical
        # constraints. New strategies are always explicitly typed.
        if not self.immutable:
            value["kind"] = self.kind
        if self.replaces:
            value["replaces"] = list(self.replaces)
            value["change_reason"] = self.change_reason
        return value

    @classmethod
    def from_value(cls, value: object) -> SkillRequirement:
        if (
            not isinstance(value, dict)
            or not {"requirement_id", "text"} <= set(value)
            or (set(value) - {"requirement_id", "text", "kind", "replaces", "change_reason"})
        ):
            raise ValueError("Skill requirement has incompatible fields")
        normalized = normalize_json(value)
        assert isinstance(normalized, dict)
        return cls(
            requirement_id=_wire_text(normalized["requirement_id"], field="requirement_id"),
            text=_wire_text(normalized["text"], field="text"),
            kind=_wire_text(normalized.get("kind", "immutable-constraint"), field="kind"),
            replaces=_wire_text_tuple(normalized.get("replaces", []), field="replaces"),
            change_reason=_wire_text(normalized.get("change_reason", ""), field="change_reason"),
        )


def _wire_text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return tuple(_wire_text(item, field=field) for item in value)


@dataclass(frozen=True, slots=True)
class SkillDocument:
    manifest: SkillManifest
    title: str
    summary: str
    instructions: str
    applicability: SkillApplicability
    requirements: tuple[SkillRequirement, ...]
    format: str = SKILL_DOCUMENT_FORMAT

    def __post_init__(self) -> None:
        if self.format != SKILL_DOCUMENT_FORMAT:
            raise ValueError(f"Skill document format must be {SKILL_DOCUMENT_FORMAT!r}")
        if not self.title.strip() or not self.summary.strip() or not self.instructions.strip():
            raise ValueError("Skill title, summary, and instructions cannot be empty")
        if not isinstance(self.applicability, SkillApplicability):
            raise TypeError("Skill applicability must be SkillApplicability")
        if not isinstance(self.requirements, tuple) or not self.requirements:
            raise ValueError("Skill requirements must be a non-empty tuple")
        if any(not isinstance(requirement, SkillRequirement) for requirement in self.requirements):
            raise TypeError("Skill requirements must contain SkillRequirement values")
        requirement_ids = tuple(requirement.requirement_id for requirement in self.requirements)
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("Skill requirement IDs must be unique")
        if self.manifest.content_hash != stable_hash(self.content_value()):
            raise ValueError("Skill document does not match its immutable manifest")

    def content_value(self) -> dict[str, JsonValue]:
        return {
            "applicability": self.applicability.to_value(),
            "instructions": self.instructions,
            "requirements": [requirement.to_value() for requirement in self.requirements],
            "summary": self.summary,
            "title": self.title,
        }

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "applicability": self.applicability.to_value(),
            "format": self.format,
            "instructions": self.instructions,
            "manifest": cast(dict[str, JsonValue], self.manifest.to_value()),
            "requirements": [requirement.to_value() for requirement in self.requirements],
            "summary": self.summary,
            "title": self.title,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillDocument:
        normalized = _exact_object(
            value,
            label="Skill document",
            expected={
                "applicability",
                "format",
                "instructions",
                "manifest",
                "requirements",
                "summary",
                "title",
            },
        )
        if normalized["format"] != SKILL_DOCUMENT_FORMAT:
            raise ValueError("Skill document has an incompatible format")
        raw_requirements = normalized["requirements"]
        if not isinstance(raw_requirements, list):
            raise TypeError("Skill requirements must be a JSON array")
        raw_manifest = normalized["manifest"]
        if not isinstance(raw_manifest, dict):
            raise TypeError("Skill manifest must be a JSON object")
        return cls(
            manifest=SkillManifest.from_value(raw_manifest),
            title=_wire_text(normalized["title"], field="title"),
            summary=_wire_text(normalized["summary"], field="summary"),
            instructions=_wire_text(normalized["instructions"], field="instructions"),
            applicability=SkillApplicability.from_value(normalized["applicability"]),
            requirements=tuple(SkillRequirement.from_value(item) for item in raw_requirements),
        )


def model_visible_skill_content(document: SkillDocument) -> str:
    """Render the complete skill content admitted into rollout ``H_0``."""

    if not isinstance(document, SkillDocument):
        raise TypeError("model-visible skill content requires a SkillDocument")
    return canonical_json(document.content_value())


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    skill_id: str
    version: str
    content_hash: str
    input_schema_id: str
    output_schema_id: str
    license_id: str
    provenance_hash: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.skill_id,
                self.version,
                self.input_schema_id,
                self.output_schema_id,
                self.license_id,
            )
        ):
            raise ValueError("Skill metadata fields cannot be empty")
        validate_identifier(self.skill_id)
        validate_sha256(self.content_hash)
        validate_sha256(self.provenance_hash)

    @classmethod
    def from_document(cls, document: SkillDocument) -> SkillMetadata:
        manifest = document.manifest
        return cls(
            skill_id=manifest.skill_id,
            version=manifest.version,
            content_hash=manifest.content_hash,
            input_schema_id=manifest.input_schema_id,
            output_schema_id=manifest.output_schema_id,
            license_id=manifest.license_id,
            provenance_hash=manifest.provenance_hash,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "content_hash": self.content_hash,
            "input_schema_id": self.input_schema_id,
            "license_id": self.license_id,
            "output_schema_id": self.output_schema_id,
            "provenance_hash": self.provenance_hash,
            "skill_id": self.skill_id,
            "version": self.version,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillMetadata:
        normalized = _exact_object(
            value,
            label="Skill metadata",
            expected={
                "content_hash",
                "input_schema_id",
                "license_id",
                "output_schema_id",
                "provenance_hash",
                "skill_id",
                "version",
            },
        )
        return cls(
            skill_id=_wire_text(normalized["skill_id"], field="skill_id"),
            version=_wire_text(normalized["version"], field="version"),
            content_hash=_wire_text(normalized["content_hash"], field="content_hash"),
            input_schema_id=_wire_text(
                normalized["input_schema_id"],
                field="input_schema_id",
            ),
            output_schema_id=_wire_text(
                normalized["output_schema_id"],
                field="output_schema_id",
            ),
            license_id=_wire_text(normalized["license_id"], field="license_id"),
            provenance_hash=_wire_text(
                normalized["provenance_hash"],
                field="provenance_hash",
            ),
        )


class RetrievalInclusionReason(str, Enum):
    """Auditable reason why a Skill projection was included in ``H_0``."""

    APPLICABILITY_MATCH = "applicability-match"


class SkillCatalog:
    """Read-only exact document catalog; no disclosure or activation state."""

    def __init__(self, documents: tuple[SkillDocument, ...]) -> None:
        by_id: dict[str, SkillDocument] = {}
        for document in documents:
            skill_id = document.manifest.skill_id
            if skill_id in by_id:
                raise ValueError("Skill IDs must be unique within a catalog")
            by_id[skill_id] = document
        self._documents: Mapping[str, SkillDocument] = MappingProxyType(by_id)

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._documents))

    def document(self, skill_id: str) -> SkillDocument:
        return self._documents[skill_id]

    def metadata(self, skill_id: str) -> SkillMetadata:
        return SkillMetadata.from_document(self.document(skill_id))


__all__ = [
    "SKILL_DOCUMENT_FORMAT",
    "RetrievalInclusionReason",
    "SkillApplicability",
    "SkillCatalog",
    "SkillDocument",
    "SkillMetadata",
    "SkillRequirement",
    "model_visible_skill_content",
]
