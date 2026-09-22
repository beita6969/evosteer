from __future__ import annotations

import pytest

from skillev.contracts import (
    SchemaDescriptor,
    SchemaRegistry,
    SchemaSet,
    SchemaVersion,
    stable_hash,
)


def _descriptor(schema_id: str, version: SchemaVersion) -> SchemaDescriptor:
    return SchemaDescriptor(
        schema_id, version, stable_hash({"schema": schema_id, "v": str(version)})
    )


def test_schema_version_round_trip_and_ordering() -> None:
    first = SchemaVersion.parse("1.2")
    later = SchemaVersion.parse("2.0")

    assert str(first) == "1.2"
    assert first < later
    with pytest.raises(ValueError):
        SchemaVersion.parse("0.1")


def test_descriptors_use_the_skillev_namespace() -> None:
    descriptor = _descriptor("skillev.skill-evolution-record", SchemaVersion(1, 0))

    assert SchemaDescriptor.from_value(descriptor.to_value()) == descriptor
    with pytest.raises(ValueError):
        _descriptor("other.lineage-transition", SchemaVersion(1, 0))


def test_schema_set_hash_is_stable_and_round_trips() -> None:
    schemas = (
        _descriptor("skillev.skill-evolution-record", SchemaVersion(1, 0)),
        _descriptor("skillev.skill-version", SchemaVersion(1, 0)),
    )
    first = SchemaSet("core-v1", schemas)
    rebuilt = SchemaSet.from_value(first.to_value())

    assert rebuilt == first
    assert rebuilt.schema_set_hash == first.schema_set_hash


def test_schema_sets_require_unique_sorted_ids() -> None:
    first = _descriptor("skillev.skill-version", SchemaVersion(1, 0))
    earlier = _descriptor("skillev.skill-evolution-record", SchemaVersion(1, 0))

    with pytest.raises(ValueError):
        SchemaSet("unsorted", (first, earlier))
    with pytest.raises(ValueError):
        SchemaSet("duplicate", (first, first))


def test_registry_is_an_exact_index_without_admission_policy() -> None:
    first = _descriptor("skillev.skill-version", SchemaVersion(1, 0))
    second = _descriptor("skillev.skill-version", SchemaVersion(1, 1))
    registry = SchemaRegistry((first, second))

    assert registry.contains(first)
    assert registry.get(first.schema_id, first.version) == first
    assert registry.get(first.schema_id, SchemaVersion(2, 0)) is None
    assert registry.registry_hash == SchemaRegistry((first, second)).registry_hash
