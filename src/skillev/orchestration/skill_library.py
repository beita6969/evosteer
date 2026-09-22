"""Content-addressed skill-library snapshots and append-only active pointers.

This module deliberately separates two kinds of history:

* :class:`SkillLibrarySnapshot` binds a complete library state to the
  ``SkillEvolutionRecord`` that produced it and to its parent snapshot.
* :class:`SkillLibraryPointerEvent` records publication decisions without
  changing either snapshots or earlier pointer events.

The in-memory store is intentionally a small reference implementation.  Its
objects expose canonical JSON values, so a durable backend can replay the same
validation rules without inheriting storage-specific policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from skillev.contracts import (
    SkillEvolutionRecord,
    SkillVersionRef,
    stable_hash,
    validate_identifier,
    validate_sha256,
)
from skillev.runtime import SkillManifest


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _manifest_from_value(value: object) -> SkillManifest:
    item = _object(value, label="Skill manifest")
    expected = {
        "content_hash",
        "input_schema_id",
        "license_id",
        "output_schema_id",
        "provenance_hash",
        "skill_id",
        "version",
    }
    if set(item) != expected:
        raise ValueError("Skill manifest fields do not match the supported representation")
    return SkillManifest(
        skill_id=str(item["skill_id"]),
        version=str(item["version"]),
        content_hash=str(item["content_hash"]),
        input_schema_id=str(item["input_schema_id"]),
        output_schema_id=str(item["output_schema_id"]),
        license_id=str(item["license_id"]),
        provenance_hash=str(item["provenance_hash"]),
    )


def _manifest_ref(manifest: SkillManifest) -> SkillVersionRef:
    """Project a runtime manifest into the method-level version reference."""

    try:
        version = int(manifest.version)
    except ValueError as error:
        raise ValueError("Skill manifest versions must be positive decimal integers") from error
    if str(version) != manifest.version:
        raise ValueError("Skill manifest versions must use their canonical decimal form")
    return SkillVersionRef(manifest.skill_id, version, manifest.content_hash)


@dataclass(frozen=True, slots=True)
class SkillLibrarySnapshot:
    """An immutable, content-addressed view of a complete skill library."""

    library_id: str
    sequence_number: int
    manifests: tuple[SkillManifest, ...]
    parent_snapshot_hash: str | None
    evolution_record_hash: str

    def __post_init__(self) -> None:
        validate_identifier(self.library_id)
        if type(self.sequence_number) is not int or self.sequence_number < 1:
            raise ValueError("Snapshot sequence numbers start at one")
        validate_sha256(self.evolution_record_hash)
        if self.sequence_number == 1:
            if self.parent_snapshot_hash is not None:
                raise ValueError("The first snapshot cannot have a parent")
        elif self.parent_snapshot_hash is None:
            raise ValueError("Later snapshots must identify their parent")
        if self.parent_snapshot_hash is not None:
            validate_sha256(self.parent_snapshot_hash)

        skill_ids = tuple(manifest.skill_id for manifest in self.manifests)
        if skill_ids != tuple(sorted(skill_ids)) or len(skill_ids) != len(set(skill_ids)):
            raise ValueError("Snapshot manifests must have unique, sorted skill IDs")
        # Projection also validates the identifier, version, and content hash
        # compatibility between runtime and method-level contracts.
        _ = self.version_refs

    @classmethod
    def for_record(
        cls,
        record: SkillEvolutionRecord,
        manifests: Iterable[SkillManifest],
        *,
        parent_snapshot: SkillLibrarySnapshot | None = None,
    ) -> SkillLibrarySnapshot:
        """Build a snapshot whose identity fields are bound to ``record``."""

        return cls(
            library_id=record.lineage_id,
            sequence_number=record.sequence_number,
            manifests=tuple(manifests),
            parent_snapshot_hash=(
                None if parent_snapshot is None else parent_snapshot.snapshot_hash
            ),
            evolution_record_hash=record.record_hash,
        )

    @property
    def version_refs(self) -> tuple[SkillVersionRef, ...]:
        return tuple(_manifest_ref(manifest) for manifest in self.manifests)

    @property
    def snapshot_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, object]:
        return {
            "evolution_record_hash": self.evolution_record_hash,
            "library_id": self.library_id,
            "manifests": [manifest.to_value() for manifest in self.manifests],
            "parent_snapshot_hash": self.parent_snapshot_hash,
            "sequence_number": self.sequence_number,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillLibrarySnapshot:
        item = _object(value, label="Skill-library snapshot")
        manifests_value = item.get("manifests")
        if not isinstance(manifests_value, list):
            raise TypeError("Snapshot manifests must be a list")
        return cls(
            library_id=str(item["library_id"]),
            sequence_number=int(str(item["sequence_number"])),
            manifests=tuple(_manifest_from_value(manifest) for manifest in manifests_value),
            parent_snapshot_hash=(
                None if item["parent_snapshot_hash"] is None else str(item["parent_snapshot_hash"])
            ),
            evolution_record_hash=str(item["evolution_record_hash"]),
        )


class SkillLibraryPointerOperation(StrEnum):
    """The two ways to move the active skill-library pointer."""

    PROMOTE = "promote"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class SkillLibraryPointerEvent:
    """One append-only update to the active snapshot pointer."""

    library_id: str
    sequence_number: int
    operation: SkillLibraryPointerOperation
    target_snapshot_hash: str
    previous_active_snapshot_hash: str | None
    previous_event_hash: str | None
    reason: str

    def __post_init__(self) -> None:
        validate_identifier(self.library_id)
        if type(self.sequence_number) is not int or self.sequence_number < 1:
            raise ValueError("Pointer-event sequence numbers start at one")
        validate_sha256(self.target_snapshot_hash)
        if self.previous_active_snapshot_hash is not None:
            validate_sha256(self.previous_active_snapshot_hash)
        if self.sequence_number == 1:
            if self.previous_event_hash is not None:
                raise ValueError("The first pointer event cannot have a predecessor")
        elif self.previous_event_hash is None:
            raise ValueError("Later pointer events must link to their predecessor")
        if self.previous_event_hash is not None:
            validate_sha256(self.previous_event_hash)
        if self.operation is SkillLibraryPointerOperation.ROLLBACK:
            if self.previous_active_snapshot_hash is None:
                raise ValueError("Rollback requires an existing active snapshot")
        if not self.reason.strip():
            raise ValueError("Pointer updates require a reason")

    @property
    def event_hash(self) -> str:
        return stable_hash(self.to_value())

    def follows(self, previous: SkillLibraryPointerEvent) -> bool:
        return (
            self.library_id == previous.library_id
            and self.sequence_number == previous.sequence_number + 1
            and self.previous_event_hash == previous.event_hash
            and self.previous_active_snapshot_hash == previous.target_snapshot_hash
        )

    def to_value(self) -> dict[str, object]:
        return {
            "library_id": self.library_id,
            "operation": self.operation.value,
            "previous_active_snapshot_hash": self.previous_active_snapshot_hash,
            "previous_event_hash": self.previous_event_hash,
            "reason": self.reason,
            "sequence_number": self.sequence_number,
            "target_snapshot_hash": self.target_snapshot_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillLibraryPointerEvent:
        item = _object(value, label="Skill-library pointer event")
        return cls(
            library_id=str(item["library_id"]),
            sequence_number=int(str(item["sequence_number"])),
            operation=SkillLibraryPointerOperation(str(item["operation"])),
            target_snapshot_hash=str(item["target_snapshot_hash"]),
            previous_active_snapshot_hash=(
                None
                if item["previous_active_snapshot_hash"] is None
                else str(item["previous_active_snapshot_hash"])
            ),
            previous_event_hash=(
                None if item["previous_event_hash"] is None else str(item["previous_event_hash"])
            ),
            reason=str(item["reason"]),
        )


class InMemorySkillLibraryStore:
    """Validated in-memory storage for snapshots, records, and active pointers."""

    def __init__(self, library_id: str) -> None:
        self._library_id = validate_identifier(library_id)
        self._snapshots: dict[str, SkillLibrarySnapshot] = {}
        self._records_by_snapshot: dict[str, SkillEvolutionRecord] = {}
        self._snapshot_by_record: dict[str, str] = {}
        self._pointer_events: list[SkillLibraryPointerEvent] = []

    @property
    def library_id(self) -> str:
        return self._library_id

    @property
    def snapshots(self) -> tuple[SkillLibrarySnapshot, ...]:
        return tuple(self._snapshots.values())

    @property
    def pointer_events(self) -> tuple[SkillLibraryPointerEvent, ...]:
        return tuple(self._pointer_events)

    @property
    def active_snapshot_hash(self) -> str | None:
        if not self._pointer_events:
            return None
        return self._pointer_events[-1].target_snapshot_hash

    @property
    def active_snapshot(self) -> SkillLibrarySnapshot | None:
        active_hash = self.active_snapshot_hash
        return None if active_hash is None else self._snapshots[active_hash]

    def get_snapshot(self, snapshot_hash: str) -> SkillLibrarySnapshot | None:
        return self._snapshots.get(snapshot_hash)

    def record_for(self, snapshot_hash: str) -> SkillEvolutionRecord:
        try:
            return self._records_by_snapshot[snapshot_hash]
        except KeyError as error:
            raise KeyError("Unknown skill-library snapshot") from error

    def add_snapshot(
        self,
        record: SkillEvolutionRecord,
        manifests: Iterable[SkillManifest],
        *,
        parent_snapshot_hash: str | None = None,
    ) -> SkillLibrarySnapshot:
        """Construct, validate, and append the snapshot produced by ``record``."""

        parent = (
            None if parent_snapshot_hash is None else self._required_snapshot(parent_snapshot_hash)
        )
        snapshot = SkillLibrarySnapshot.for_record(
            record,
            manifests,
            parent_snapshot=parent,
        )
        return self.append_snapshot(snapshot, record)

    def append_snapshot(
        self,
        snapshot: SkillLibrarySnapshot,
        record: SkillEvolutionRecord,
    ) -> SkillLibrarySnapshot:
        """Append a snapshot only if both of its predecessor links are valid."""

        if snapshot.library_id != self.library_id or record.lineage_id != self.library_id:
            raise ValueError("Snapshot, record, and store must use one library lineage")
        if snapshot.sequence_number != record.sequence_number:
            raise ValueError("Snapshot and evolution record sequences differ")
        if snapshot.evolution_record_hash != record.record_hash:
            raise ValueError("Snapshot does not bind the supplied evolution record")
        if snapshot.snapshot_hash in self._snapshots:
            raise ValueError("Snapshot is already present")
        if record.record_hash in self._snapshot_by_record:
            raise ValueError("An evolution record can produce only one stored snapshot")

        parent: SkillLibrarySnapshot | None
        parent_record: SkillEvolutionRecord | None
        if snapshot.parent_snapshot_hash is None:
            parent = None
            parent_record = None
            if record.sequence_number != 1 or record.previous_record_hash is not None:
                raise ValueError("A root snapshot requires the first lineage record")
        else:
            try:
                parent = self._required_snapshot(snapshot.parent_snapshot_hash)
            except KeyError as error:
                raise ValueError("Snapshot parent is not present in this store") from error
            parent_record = self._records_by_snapshot[parent.snapshot_hash]
            if not record.follows(parent_record):
                raise ValueError("Evolution record does not follow the parent record")
            if snapshot.sequence_number != parent.sequence_number + 1:
                raise ValueError("Snapshot does not directly follow its parent")

        expected_refs = self._apply_record(
            () if parent is None else parent.version_refs,
            record,
        )
        if snapshot.version_refs != expected_refs:
            raise ValueError("Snapshot contents do not match the evolution record")
        if parent is not None:
            self._validate_inherited_manifests(parent, snapshot, record)

        snapshot_hash = snapshot.snapshot_hash
        self._snapshots[snapshot_hash] = snapshot
        self._records_by_snapshot[snapshot_hash] = record
        self._snapshot_by_record[record.record_hash] = snapshot_hash
        return snapshot

    def promote(self, snapshot_hash: str, *, reason: str) -> SkillLibraryPointerEvent:
        """Move the active pointer forward by appending a promotion event."""

        event = self._new_pointer_event(
            SkillLibraryPointerOperation.PROMOTE,
            snapshot_hash,
            reason,
        )
        return self.append_pointer_event(event)

    def rollback(self, snapshot_hash: str, *, reason: str) -> SkillLibraryPointerEvent:
        """Move the active pointer to an ancestor without changing history."""

        event = self._new_pointer_event(
            SkillLibraryPointerOperation.ROLLBACK,
            snapshot_hash,
            reason,
        )
        return self.append_pointer_event(event)

    def append_pointer_event(
        self,
        event: SkillLibraryPointerEvent,
    ) -> SkillLibraryPointerEvent:
        """Replay or append one pointer event after validating its full link."""

        if event.library_id != self.library_id:
            raise ValueError("Pointer event belongs to a different library")
        self._required_snapshot(event.target_snapshot_hash)

        previous = self._pointer_events[-1] if self._pointer_events else None
        current_active = self.active_snapshot_hash
        if previous is None:
            if event.sequence_number != 1 or event.previous_event_hash is not None:
                raise ValueError("Pointer history must start with its first event")
            if event.previous_active_snapshot_hash is not None:
                raise ValueError("The initial pointer event cannot replace an active snapshot")
        else:
            if not event.follows(previous):
                raise ValueError("Pointer events do not form an append-only chain")
        if event.previous_active_snapshot_hash != current_active:
            raise ValueError("Pointer event is based on a stale active snapshot")

        target_hash = event.target_snapshot_hash
        if event.operation is SkillLibraryPointerOperation.PROMOTE:
            if current_active is not None and not self._is_ancestor(current_active, target_hash):
                raise ValueError("Promotion target must descend from the active snapshot")
        elif current_active is None or not self._is_ancestor(target_hash, current_active):
            raise ValueError("Rollback target must be an ancestor of the active snapshot")

        self._pointer_events.append(event)
        return event

    def to_value(self) -> dict[str, object]:
        """Return a canonical-JSON-compatible replay representation."""

        return {
            "library_id": self.library_id,
            "pointer_events": [event.to_value() for event in self._pointer_events],
            "snapshots": [
                {
                    "record": self._records_by_snapshot[snapshot.snapshot_hash].to_value(),
                    "snapshot": snapshot.to_value(),
                }
                for snapshot in self._snapshots.values()
            ],
        }

    @classmethod
    def from_value(cls, value: object) -> InMemorySkillLibraryStore:
        """Rebuild a store by replaying all links rather than trusting its payload."""

        item = _object(value, label="Skill-library store")
        snapshots_value = item.get("snapshots")
        events_value = item.get("pointer_events")
        if not isinstance(snapshots_value, list) or not isinstance(events_value, list):
            raise TypeError("Store snapshots and pointer events must be lists")

        store = cls(str(item["library_id"]))
        for entry_value in snapshots_value:
            entry = _object(entry_value, label="Stored snapshot entry")
            record_value = _object(entry["record"], label="Skill evolution record")
            store.append_snapshot(
                SkillLibrarySnapshot.from_value(entry["snapshot"]),
                SkillEvolutionRecord.from_value(dict(record_value)),
            )
        for event_value in events_value:
            store.append_pointer_event(SkillLibraryPointerEvent.from_value(event_value))
        return store

    def _new_pointer_event(
        self,
        operation: SkillLibraryPointerOperation,
        snapshot_hash: str,
        reason: str,
    ) -> SkillLibraryPointerEvent:
        previous = self._pointer_events[-1] if self._pointer_events else None
        return SkillLibraryPointerEvent(
            library_id=self.library_id,
            sequence_number=len(self._pointer_events) + 1,
            operation=operation,
            target_snapshot_hash=snapshot_hash,
            previous_active_snapshot_hash=self.active_snapshot_hash,
            previous_event_hash=None if previous is None else previous.event_hash,
            reason=reason,
        )

    def _required_snapshot(self, snapshot_hash: str) -> SkillLibrarySnapshot:
        try:
            return self._snapshots[snapshot_hash]
        except KeyError as error:
            raise KeyError("Unknown skill-library snapshot") from error

    def _is_ancestor(self, ancestor_hash: str, descendant_hash: str) -> bool:
        cursor = self._required_snapshot(descendant_hash)
        while cursor.parent_snapshot_hash is not None:
            if cursor.parent_snapshot_hash == ancestor_hash:
                return True
            cursor = self._required_snapshot(cursor.parent_snapshot_hash)
        return False

    @staticmethod
    def _apply_record(
        current: tuple[SkillVersionRef, ...],
        record: SkillEvolutionRecord,
    ) -> tuple[SkillVersionRef, ...]:
        by_id = {version.skill_id: version for version in current}
        for input_version in record.input_versions:
            if by_id.get(input_version.skill_id) != input_version:
                raise ValueError("Evolution input is not present in the parent snapshot")
            del by_id[input_version.skill_id]
        for output_version in record.output_versions:
            if output_version.skill_id in by_id:
                raise ValueError("Evolution output conflicts with an inherited skill")
            by_id[output_version.skill_id] = output_version
        return tuple(sorted(by_id.values()))

    @staticmethod
    def _validate_inherited_manifests(
        parent: SkillLibrarySnapshot,
        child: SkillLibrarySnapshot,
        record: SkillEvolutionRecord,
    ) -> None:
        changed_ids = {version.skill_id for version in record.input_versions}
        parent_by_id = {manifest.skill_id: manifest for manifest in parent.manifests}
        child_by_id = {manifest.skill_id: manifest for manifest in child.manifests}
        for skill_id, manifest in parent_by_id.items():
            if skill_id not in changed_ids and child_by_id.get(skill_id) != manifest:
                raise ValueError("Unchanged skill manifests must be inherited exactly")
