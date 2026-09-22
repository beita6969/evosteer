"""Deterministic JSON serialization and content hashing for SKILLEV.

The representation is deliberately project-local rather than an assertion of
full RFC 8785 compatibility.  Version 1 normalizes strings to NFC, sorts object
keys, emits compact UTF-8 JSON, rejects non-finite numbers, and normalizes
negative zero to ``0.0``.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

CANONICALIZATION_VERSION = "skillev-canonical-json@1"

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class CanonicalizationError(ValueError):
    """Raised when a value cannot safely enter canonical JSON."""


def normalize_json(value: object) -> Any:
    """Return a JSON-only, Unicode-normalized value.

    Mapping keys must be strings.  Two distinct source keys that become equal
    after NFC normalization are rejected instead of silently overwriting data.
    The return type is intentionally the untyped JSON parsing boundary; callers
    validate and narrow their own exact wire schemas before constructing typed
    contracts.
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("Non-finite numbers are not valid canonical JSON")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("Canonical JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise CanonicalizationError("Object keys collide after Unicode normalization")
            normalized[normalized_key] = normalize_json(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [normalize_json(item) for item in value]
    raise CanonicalizationError(f"Unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize ``value`` using the SKILLEV canonical JSON v1 representation."""

    return json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return canonical JSON encoded as UTF-8 bytes."""

    return canonical_json(value).encode("utf-8")


def stable_hash(value: object) -> str:
    """Hash the canonical JSON meaning of ``value``.

    This is a content-addressing primitive, not an authorization mechanism.
    Raw artifact bytes should instead be passed to
    :func:`skillev.contracts.identity.artifact_hash`.
    """

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def parse_canonical_json(payload: str) -> JsonValue:
    """Parse JSON while requiring that the input text is already canonical."""

    parsed = cast(object, json.loads(payload))
    if canonical_json(parsed) != payload:
        raise CanonicalizationError("JSON payload is valid but is not canonical")
    return cast(JsonValue, normalize_json(parsed))
