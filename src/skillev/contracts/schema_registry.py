"""Lightweight schema identities and deterministic schema collections.

This module records which schemas exist.  Lifecycle policy and JSON instance
validation belong to their respective callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .canonical import CANONICALIZATION_VERSION, stable_hash
from .identity import validate_sha256

SCHEMA_ID_RE = re.compile(r"^skillev\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, order=True, slots=True)
class SchemaVersion:
    """A schema compatibility version represented as ``major.minor``."""

    major: int
    minor: int

    def __post_init__(self) -> None:
        if self.major < 1 or self.minor < 0:
            raise ValueError("Schema versions require major >= 1 and minor >= 0")

    @classmethod
    def parse(cls, value: str) -> SchemaVersion:
        match = re.fullmatch(r"([1-9][0-9]*)\.([0-9]+)", value)
        if match is None:
            raise ValueError(f"Invalid schema version: {value!r}")
        return cls(int(match.group(1)), int(match.group(2)))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True, order=True, slots=True)
class SchemaDescriptor:
    """Content identity for one versioned SKILLEV schema."""

    schema_id: str
    version: SchemaVersion
    schema_digest: str

    def __post_init__(self) -> None:
        if SCHEMA_ID_RE.fullmatch(self.schema_id) is None:
            raise ValueError(f"Invalid SKILLEV schema ID: {self.schema_id!r}")
        validate_sha256(self.schema_digest)

    def to_value(self) -> dict[str, str]:
        return {
            "schema_digest": self.schema_digest,
            "schema_id": self.schema_id,
            "version": str(self.version),
        }

    @classmethod
    def from_value(cls, value: dict[str, object]) -> SchemaDescriptor:
        return cls(
            schema_id=str(value["schema_id"]),
            version=SchemaVersion.parse(str(value["version"])),
            schema_digest=str(value["schema_digest"]),
        )


@dataclass(frozen=True, slots=True)
class SchemaSet:
    """The unique schema versions bound to one serialized interface."""

    schema_set_id: str
    schemas: tuple[SchemaDescriptor, ...]
    canonicalization_version: str = CANONICALIZATION_VERSION

    def __post_init__(self) -> None:
        if not self.schema_set_id:
            raise ValueError("A schema set identifier is required")
        schema_ids = [schema.schema_id for schema in self.schemas]
        if schema_ids != sorted(schema_ids) or len(schema_ids) != len(set(schema_ids)):
            raise ValueError("Schema set entries must have unique, sorted schema IDs")
        if not self.canonicalization_version:
            raise ValueError("A canonicalization version is required")

    def to_value(self) -> dict[str, object]:
        return {
            "canonicalization_version": self.canonicalization_version,
            "schema_set_id": self.schema_set_id,
            "schemas": [schema.to_value() for schema in self.schemas],
        }

    @classmethod
    def from_value(cls, value: dict[str, object]) -> SchemaSet:
        raw_schemas = value["schemas"]
        if not isinstance(raw_schemas, list):
            raise TypeError("schemas must be a list")
        schemas: list[SchemaDescriptor] = []
        for raw_schema in raw_schemas:
            if not isinstance(raw_schema, dict):
                raise TypeError("schema descriptors must be objects")
            schemas.append(SchemaDescriptor.from_value(raw_schema))
        return cls(
            schema_set_id=str(value["schema_set_id"]),
            schemas=tuple(schemas),
            canonicalization_version=str(value["canonicalization_version"]),
        )

    @property
    def schema_set_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class SchemaRegistry:
    """A deterministic index of known descriptors, without admission policy."""

    schemas: tuple[SchemaDescriptor, ...]

    def __post_init__(self) -> None:
        keys = [(schema.schema_id, schema.version) for schema in self.schemas]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("Registry entries must be unique and sorted by ID and version")

    def to_value(self) -> dict[str, object]:
        return {"schemas": [schema.to_value() for schema in self.schemas]}

    @property
    def registry_hash(self) -> str:
        return stable_hash(self.to_value())

    def get(self, schema_id: str, version: SchemaVersion) -> SchemaDescriptor | None:
        """Return an exact descriptor match, if the registry contains one."""

        return next(
            (
                schema
                for schema in self.schemas
                if schema.schema_id == schema_id and schema.version == version
            ),
            None,
        )

    def contains(self, descriptor: SchemaDescriptor) -> bool:
        """Return whether the exact schema identity is present."""

        return descriptor in self.schemas
