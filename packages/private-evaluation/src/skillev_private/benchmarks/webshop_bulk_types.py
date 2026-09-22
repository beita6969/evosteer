"""Types and exact source identities for streaming WebShop preparation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue, stable_hash
from skillev.experiments import FIXED_SEED, Benchmark

from .acquisition import (
    BenchmarkAcquisitionLock,
    LockedAcquisitionArtifact,
)

WEBSHOP_BULK_PREPARATION_FORMAT = "skillev-private-webshop-bulk-preparation@1"
WEBSHOP_PRODUCT_STORE_FORMAT = "skillev-webshop-product-store@1"

PRODUCTS_RELATIVE_PATH = "raw/items_shuffle.json"
ATTRIBUTES_RELATIVE_PATH = "raw/items_ins_v2.json"
HUMAN_INSTRUCTIONS_RELATIVE_PATH = "raw/items_human_ins.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an exact string-keyed object")
    return cast(dict[str, object], value)


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value:
        raise TypeError(f"{field} must be text without NUL")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} cannot be empty")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field=field)


def _lower_sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True, slots=True)
class WebShopSourceIdentity:
    """One exact locked WebShop source file."""

    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("WebShop source path must be absolute")
        _positive_int(self.size_bytes, field="WebShop source size")
        _lower_sha256(self.sha256, field="WebShop source SHA-256")

    def verify(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.stat().st_size != self.size_bytes:
            raise ValueError(f"WebShop source size differs from lock: {self.path.name}")
        if _sha256_file(self.path) != self.sha256:
            raise ValueError(f"WebShop source digest differs from lock: {self.path.name}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "name": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class WebShopPreparationProgress:
    """Aggregate-only progress sample safe for logs."""

    phase: str
    raw_products_seen: int
    accepted_products: int
    goals_written: int
    elapsed_seconds: float
    records_per_second: float
    eta_seconds: float | None

    def __post_init__(self) -> None:
        _text(self.phase, field="WebShop preparation phase")
        for name, counter in (
            ("raw_products_seen", self.raw_products_seen),
            ("accepted_products", self.accepted_products),
            ("goals_written", self.goals_written),
        ):
            if type(counter) is not int or counter < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, measurement in (
            ("elapsed_seconds", self.elapsed_seconds),
            ("records_per_second", self.records_per_second),
        ):
            if (
                not isinstance(measurement, float)
                or not math.isfinite(measurement)
                or measurement < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.eta_seconds is not None and (
            not isinstance(self.eta_seconds, float)
            or not math.isfinite(self.eta_seconds)
            or self.eta_seconds < 0.0
        ):
            raise ValueError("eta_seconds must be None or a finite non-negative float")


@dataclass(frozen=True, slots=True)
class WebShopBulkPreparationReceipt:
    """Content identity of one completed streaming preparation."""

    seed: int
    raw_product_limit: int | None
    raw_products_seen: int
    product_count: int
    goal_count: int
    product_store_size_bytes: int
    product_store_sha256: str
    documents_size_bytes: int
    documents_sha256: str
    goals_size_bytes: int
    goals_sha256: str
    source_identities: tuple[dict[str, JsonValue], ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed != FIXED_SEED:
            raise ValueError("WebShop preparation must use the frozen experiment seed")
        _optional_positive_int(self.raw_product_limit, field="raw_product_limit")
        for name, value in (
            ("raw_products_seen", self.raw_products_seen),
            ("product_count", self.product_count),
            ("goal_count", self.goal_count),
            ("product_store_size_bytes", self.product_store_size_bytes),
            ("documents_size_bytes", self.documents_size_bytes),
            ("goals_size_bytes", self.goals_size_bytes),
        ):
            _positive_int(value, field=name)
        if self.raw_product_limit is not None and self.raw_products_seen != self.raw_product_limit:
            raise ValueError("bounded WebShop preparation must consume its exact raw limit")
        if self.product_count > self.raw_products_seen:
            raise ValueError("accepted WebShop products cannot exceed raw products")
        for field, digest in (
            ("product_store_sha256", self.product_store_sha256),
            ("documents_sha256", self.documents_sha256),
            ("goals_sha256", self.goals_sha256),
        ):
            _lower_sha256(digest, field=field)
        if len(self.source_identities) != 3:
            raise ValueError("WebShop preparation must bind exactly three source files")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "documents_sha256": self.documents_sha256,
            "documents_size_bytes": self.documents_size_bytes,
            "format": WEBSHOP_BULK_PREPARATION_FORMAT,
            "goal_count": self.goal_count,
            "goals_sha256": self.goals_sha256,
            "goals_size_bytes": self.goals_size_bytes,
            "product_count": self.product_count,
            "product_store_sha256": self.product_store_sha256,
            "product_store_size_bytes": self.product_store_size_bytes,
            "raw_product_limit": self.raw_product_limit,
            "raw_products_seen": self.raw_products_seen,
            "seed": self.seed,
            "source_identities": list(self.source_identities),
            "store_format": WEBSHOP_PRODUCT_STORE_FORMAT,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_manifest_value(cls, value: object) -> WebShopBulkPreparationReceipt:
        data = _exact_object(value, label="WebShop preparation manifest")
        expected = {
            "content_hash",
            "documents_sha256",
            "documents_size_bytes",
            "format",
            "goal_count",
            "goals_sha256",
            "goals_size_bytes",
            "product_count",
            "product_store_sha256",
            "product_store_size_bytes",
            "raw_product_limit",
            "raw_products_seen",
            "seed",
            "source_identities",
            "store_format",
        }
        if set(data) != expected:
            raise ValueError("WebShop preparation manifest has an incompatible field set")
        if data["format"] != WEBSHOP_BULK_PREPARATION_FORMAT:
            raise ValueError("unsupported WebShop preparation manifest format")
        if data["store_format"] != WEBSHOP_PRODUCT_STORE_FORMAT:
            raise ValueError("unsupported WebShop product-store format")
        sources = data["source_identities"]
        if type(sources) is not list or any(type(item) is not dict for item in sources):
            raise TypeError("WebShop source identities must be an object array")
        receipt = cls(
            seed=cast(int, data["seed"]),
            raw_product_limit=cast(int | None, data["raw_product_limit"]),
            raw_products_seen=cast(int, data["raw_products_seen"]),
            product_count=cast(int, data["product_count"]),
            goal_count=cast(int, data["goal_count"]),
            product_store_size_bytes=cast(int, data["product_store_size_bytes"]),
            product_store_sha256=cast(str, data["product_store_sha256"]),
            documents_size_bytes=cast(int, data["documents_size_bytes"]),
            documents_sha256=cast(str, data["documents_sha256"]),
            goals_size_bytes=cast(int, data["goals_size_bytes"]),
            goals_sha256=cast(str, data["goals_sha256"]),
            source_identities=tuple(cast(list[dict[str, JsonValue]], sources)),
        )
        if data["content_hash"] != receipt.content_hash:
            raise ValueError("WebShop preparation manifest content hash differs")
        return receipt


def _artifact(
    lock: BenchmarkAcquisitionLock,
    relative_path: str,
) -> LockedAcquisitionArtifact:
    webshop = next(
        (entry for entry in lock.benchmarks if entry.benchmark is Benchmark.WEBSHOP),
        None,
    )
    if webshop is None:
        raise ValueError("acquisition lock does not contain WebShop")
    artifact = next(
        (item for item in webshop.source.artifacts if item.relative_path == relative_path),
        None,
    )
    if artifact is None:
        raise ValueError(f"acquisition lock is missing WebShop {relative_path}")
    if artifact.expected_size is None or artifact.expected_sha256 is None:
        raise ValueError("WebShop bulk preparation requires locked source size and digest")
    return artifact


def locked_webshop_sources(
    lock: BenchmarkAcquisitionLock,
    target_root: Path,
) -> tuple[WebShopSourceIdentity, WebShopSourceIdentity, WebShopSourceIdentity]:
    """Resolve the three exact WebShop files under one private target root."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(target_root, Path) or not target_root.is_absolute():
        raise ValueError("target_root must be an absolute Path")
    sources: list[WebShopSourceIdentity] = []
    for relative_path in (
        PRODUCTS_RELATIVE_PATH,
        ATTRIBUTES_RELATIVE_PATH,
        HUMAN_INSTRUCTIONS_RELATIVE_PATH,
    ):
        artifact = _artifact(lock, relative_path)
        sources.append(
            WebShopSourceIdentity(
                path=target_root / Benchmark.WEBSHOP.value / relative_path,
                size_bytes=cast(int, artifact.expected_size),
                sha256=cast(str, artifact.expected_sha256),
            )
        )
    return cast(
        tuple[WebShopSourceIdentity, WebShopSourceIdentity, WebShopSourceIdentity],
        tuple(sources),
    )


__all__ = [
    "ATTRIBUTES_RELATIVE_PATH",
    "HUMAN_INSTRUCTIONS_RELATIVE_PATH",
    "PRODUCTS_RELATIVE_PATH",
    "WEBSHOP_BULK_PREPARATION_FORMAT",
    "WEBSHOP_PRODUCT_STORE_FORMAT",
    "WebShopBulkPreparationReceipt",
    "WebShopPreparationProgress",
    "WebShopSourceIdentity",
    "locked_webshop_sources",
]
