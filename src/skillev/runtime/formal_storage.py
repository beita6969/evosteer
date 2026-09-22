"""Budget-derived storage admission for long Protocol 10 attempts."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

from skillev.contracts import JsonValue, normalize_json

FORMAL_STORAGE_BUDGET_FORMAT: Final = "skillev-formal-storage-budget@1"
FORMAL_STORAGE_BINDING_FORMAT: Final = "skillev-formal-storage-binding@1"


def _positive(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class FormalStorageBudget:
    """Measured peak plus rollback, publication, and 25% safety reserve."""

    estimated_attempt_peak_bytes: int
    rollback_checkpoint_bytes: int
    atomic_publication_bytes: int
    minimum_free_inodes: int
    safety_margin_percent: int = 25
    format: str = FORMAL_STORAGE_BUDGET_FORMAT

    def __post_init__(self) -> None:
        if self.format != FORMAL_STORAGE_BUDGET_FORMAT:
            raise ValueError("unsupported formal storage budget format")
        for field in (
            "estimated_attempt_peak_bytes",
            "rollback_checkpoint_bytes",
            "atomic_publication_bytes",
            "minimum_free_inodes",
        ):
            _positive(getattr(self, field), label=field)
        if self.safety_margin_percent != 25:
            raise ValueError("Protocol 10 storage safety margin must be 25 percent")

    @property
    def minimum_free_bytes(self) -> int:
        subtotal = (
            self.estimated_attempt_peak_bytes
            + self.rollback_checkpoint_bytes
            + self.atomic_publication_bytes
        )
        return (subtotal * (100 + self.safety_margin_percent) + 99) // 100

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "atomic_publication_bytes": self.atomic_publication_bytes,
            "estimated_attempt_peak_bytes": self.estimated_attempt_peak_bytes,
            "format": self.format,
            "minimum_free_bytes": self.minimum_free_bytes,
            "minimum_free_inodes": self.minimum_free_inodes,
            "rollback_checkpoint_bytes": self.rollback_checkpoint_bytes,
            "safety_margin_percent": self.safety_margin_percent,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        fields = {
            "atomic_publication_bytes",
            "estimated_attempt_peak_bytes",
            "format",
            "minimum_free_bytes",
            "minimum_free_inodes",
            "rollback_checkpoint_bytes",
            "safety_margin_percent",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal storage budget has incompatible fields")
        integers = fields - {"format"}
        if any(type(normalized[field]) is not int for field in integers):
            raise TypeError("formal storage budget quantities must be integers")
        format_value = normalized["format"]
        if type(format_value) is not str:
            raise TypeError("formal storage budget format must be text")
        result = cls(
            estimated_attempt_peak_bytes=normalized["estimated_attempt_peak_bytes"],
            rollback_checkpoint_bytes=normalized["rollback_checkpoint_bytes"],
            atomic_publication_bytes=normalized["atomic_publication_bytes"],
            minimum_free_inodes=normalized["minimum_free_inodes"],
            safety_margin_percent=normalized["safety_margin_percent"],
            format=format_value,
        )
        if normalized["minimum_free_bytes"] != result.minimum_free_bytes:
            raise ValueError("formal storage minimum differs from its measured budget")
        return result


@dataclass(frozen=True, slots=True)
class FormalStorageBinding:
    """Private paths and measured capacity floor for one formal attempt."""

    output_root: Path
    temporary_root: Path
    budget: FormalStorageBudget
    format: str = FORMAL_STORAGE_BINDING_FORMAT

    def __post_init__(self) -> None:
        if self.format != FORMAL_STORAGE_BINDING_FORMAT:
            raise ValueError("unsupported formal storage binding format")
        if not self.output_root.is_absolute() or not self.temporary_root.is_absolute():
            raise ValueError("formal storage roots must be absolute")
        if not isinstance(self.budget, FormalStorageBudget):
            raise TypeError("formal storage binding requires a measured budget")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "budget": self.budget.to_value(),
            "format": self.format,
            "output_root": str(self.output_root),
            "temporary_root": str(self.temporary_root),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        fields = {"budget", "format", "output_root", "temporary_root"}
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal storage binding has incompatible fields")
        if any(type(normalized[field]) is not str for field in fields - {"budget"}):
            raise TypeError("formal storage binding paths and format must be text")
        return cls(
            output_root=Path(normalized["output_root"]),
            temporary_root=Path(normalized["temporary_root"]),
            budget=FormalStorageBudget.from_value(normalized["budget"]),
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalStorageObservation:
    available_bytes: int
    available_inodes: int
    device: int


def validate_formal_storage(binding: FormalStorageBinding) -> FormalStorageObservation:
    """Check capacity and prove same-filesystem durable publication semantics."""

    if not isinstance(binding, FormalStorageBinding):
        raise TypeError("formal storage preflight requires a storage binding")
    for root in (binding.output_root, binding.temporary_root):
        if not root.is_dir():
            raise RuntimeError("formal storage root is unavailable")
    output_stat = binding.output_root.stat()
    temporary_stat = binding.temporary_root.stat()
    if output_stat.st_dev != temporary_stat.st_dev:
        raise RuntimeError("formal output and temporary roots must share one filesystem")
    filesystem = os.statvfs(binding.output_root)
    observation = FormalStorageObservation(
        available_bytes=filesystem.f_bavail * filesystem.f_frsize,
        available_inodes=filesystem.f_favail,
        device=output_stat.st_dev,
    )
    if observation.available_bytes < binding.budget.minimum_free_bytes:
        raise RuntimeError("formal output filesystem is below the measured byte reserve")
    if observation.available_inodes < binding.budget.minimum_free_inodes:
        raise RuntimeError("formal output filesystem is below the measured inode reserve")

    descriptor, staging_name = tempfile.mkstemp(
        prefix=".skillev-storage-preflight-",
        dir=binding.temporary_root,
    )
    staging = Path(staging_name)
    published = binding.output_root / f".{staging.name}.published"
    try:
        os.write(descriptor, b"protocol-v10-storage-preflight\n")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(staging, published)
        directory_descriptor = os.open(binding.output_root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        staging.unlink(missing_ok=True)
        published.unlink(missing_ok=True)
    return observation


__all__ = [
    "FORMAL_STORAGE_BINDING_FORMAT",
    "FORMAL_STORAGE_BUDGET_FORMAT",
    "FormalStorageBinding",
    "FormalStorageBudget",
    "FormalStorageObservation",
    "validate_formal_storage",
]
