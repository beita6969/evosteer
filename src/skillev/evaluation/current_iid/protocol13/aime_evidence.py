"""Typed, field-level evidence for AIME generation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AIMEProfileEvidenceStatus(StrEnum):
    AUTHOR_EXACT = "author-exact"
    MODEL_VENDOR_GUIDANCE = "model-vendor-guidance"
    PRE_FINAL_CALIBRATION = "pre-final-calibration"
    UNVERIFIED = "unverified"


_REQUIRED_FORMAL_FIELDS = frozenset(
    {
        "prompt_profile",
        "enable_thinking",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "max_new_tokens",
        "seed_aggregation",
    }
)


@dataclass(frozen=True, slots=True)
class AIMEProfileEvidence:
    status: AIMEProfileEvidenceStatus
    repository_or_paper: str
    revision_or_version: str
    path_or_section: str
    supported_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.repository_or_paper,
            self.revision_or_version,
            self.path_or_section,
        )
        if any(not value.strip() for value in values):
            raise ValueError("AIME profile evidence provenance is incomplete")
        if not self.supported_fields or any(not value.strip() for value in self.supported_fields):
            raise ValueError("AIME profile evidence fields are incomplete")
        if len(self.supported_fields) != len(set(self.supported_fields)):
            raise ValueError("AIME profile evidence fields are duplicated")

    @property
    def formal_reference_eligible(self) -> bool:
        return (
            self.status is AIMEProfileEvidenceStatus.AUTHOR_EXACT
            and _REQUIRED_FORMAL_FIELDS.issubset(set(self.supported_fields))
        )


__all__ = ["AIMEProfileEvidence", "AIMEProfileEvidenceStatus"]
