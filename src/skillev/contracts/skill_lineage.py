"""Immutable, predecessor-linked records for skill-library evolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from .canonical import stable_hash
from .identity import validate_identifier, validate_sha256


class SkillEvolutionAction(StrEnum):
    """The five distinct skill evolution actions defined by the method."""

    RETAIN_COMPRESS = "retain-compress"
    REFINE = "refine"
    SPLIT = "split"
    PRUNE = "prune"
    GENERATE = "generate"


@dataclass(frozen=True, order=True, slots=True)
class SkillVersionRef:
    """A content-addressed version of one skill."""

    skill_id: str
    version: int
    content_hash: str

    def __post_init__(self) -> None:
        validate_identifier(self.skill_id)
        if self.version < 1:
            raise ValueError("Skill versions start at one")
        validate_sha256(self.content_hash)

    def to_value(self) -> dict[str, object]:
        return {
            "content_hash": self.content_hash,
            "skill_id": self.skill_id,
            "version": self.version,
        }

    @classmethod
    def from_value(cls, value: dict[str, object]) -> SkillVersionRef:
        return cls(
            skill_id=str(value["skill_id"]),
            version=int(str(value["version"])),
            content_hash=str(value["content_hash"]),
        )


@dataclass(frozen=True, slots=True)
class SkillEvolutionRecord:
    """One canonical action in an append-only skill lineage.

    ``evidence_hash`` identifies the decision basis without prescribing its
    scientific representation.  Publication state and rollback pointers are
    deliberately outside this method-level action record.
    """

    lineage_id: str
    sequence_number: int
    action: SkillEvolutionAction
    input_versions: tuple[SkillVersionRef, ...]
    output_versions: tuple[SkillVersionRef, ...]
    previous_record_hash: str | None
    evidence_hash: str
    reason: str

    def __post_init__(self) -> None:
        validate_identifier(self.lineage_id)
        if self.sequence_number < 1:
            raise ValueError("Lineage sequence numbers start at one")
        if self.sequence_number == 1:
            if self.previous_record_hash is not None:
                raise ValueError("The first lineage record cannot have a predecessor")
        elif self.previous_record_hash is None:
            raise ValueError("Later lineage records must link to their predecessor")
        if self.previous_record_hash is not None:
            validate_sha256(self.previous_record_hash)
        validate_sha256(self.evidence_hash)
        if not self.reason.strip():
            raise ValueError("A lineage action reason is required")
        self._validate_refs("input", self.input_versions)
        self._validate_refs("output", self.output_versions)
        self._validate_action_shape()

    @staticmethod
    def _validate_refs(label: str, refs: tuple[SkillVersionRef, ...]) -> None:
        if tuple(sorted(refs)) != refs or len(refs) != len(set(refs)):
            raise ValueError(f"{label} skill versions must be unique and sorted")

    def _validate_action_shape(self) -> None:
        inputs = len(self.input_versions)
        outputs = len(self.output_versions)
        expected = {
            SkillEvolutionAction.RETAIN_COMPRESS: inputs == 1 and outputs == 1,
            SkillEvolutionAction.REFINE: inputs == 1 and outputs == 1,
            SkillEvolutionAction.SPLIT: inputs == 1 and outputs >= 2,
            SkillEvolutionAction.PRUNE: inputs == 1 and outputs == 0,
            SkillEvolutionAction.GENERATE: inputs == 0 and outputs >= 1,
        }
        if not expected[self.action]:
            raise ValueError("Input/output skill versions do not match the evolution action")

    def to_value(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "evidence_hash": self.evidence_hash,
            "input_versions": [version.to_value() for version in self.input_versions],
            "lineage_id": self.lineage_id,
            "output_versions": [version.to_value() for version in self.output_versions],
            "previous_record_hash": self.previous_record_hash,
            "reason": self.reason,
            "sequence_number": self.sequence_number,
        }

    @classmethod
    def from_value(cls, value: dict[str, object]) -> SkillEvolutionRecord:
        return cls(
            lineage_id=str(value["lineage_id"]),
            sequence_number=int(str(value["sequence_number"])),
            action=SkillEvolutionAction(str(value["action"])),
            input_versions=cls._refs_from_value(value["input_versions"]),
            output_versions=cls._refs_from_value(value["output_versions"]),
            previous_record_hash=(
                None
                if value["previous_record_hash"] is None
                else str(value["previous_record_hash"])
            ),
            evidence_hash=str(value["evidence_hash"]),
            reason=str(value["reason"]),
        )

    @staticmethod
    def _refs_from_value(value: object) -> tuple[SkillVersionRef, ...]:
        if not isinstance(value, list):
            raise TypeError("Skill version references must be a list")
        refs: list[SkillVersionRef] = []
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("Skill version references must be objects")
            refs.append(SkillVersionRef.from_value(item))
        return tuple(refs)

    @property
    def record_hash(self) -> str:
        """Return the content address of this complete action record."""

        return stable_hash(self.to_value())

    def follows(self, previous: SkillEvolutionRecord) -> bool:
        """Return whether this record is the direct successor of ``previous``."""

        return (
            self.lineage_id == previous.lineage_id
            and self.sequence_number == previous.sequence_number + 1
            and self.previous_record_hash == previous.record_hash
        )


def validate_lineage(records: tuple[SkillEvolutionRecord, ...]) -> None:
    """Validate one complete, ordered predecessor chain."""

    if not records:
        return
    if records[0].sequence_number != 1 or records[0].previous_record_hash is not None:
        raise ValueError("A complete lineage must start with its first record")
    for previous, current in pairwise(records):
        if not current.follows(previous):
            raise ValueError("Lineage records do not form one predecessor chain")
