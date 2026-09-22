"""Typed code-evaluator runtime identities for Protocol 13."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeExecutorContract:
    profile_id: str
    benchmark: str
    python_version: str
    wall_timeout_seconds: str
    cpu_seconds: int
    address_space_bytes: int
    process_count: int
    file_size_bytes: int
    open_file_count: int
    test_profile: str

    def __post_init__(self) -> None:
        texts = (
            self.profile_id,
            self.benchmark,
            self.python_version,
            self.wall_timeout_seconds,
            self.test_profile,
        )
        limits = (
            self.cpu_seconds,
            self.address_space_bytes,
            self.process_count,
            self.file_size_bytes,
            self.open_file_count,
        )
        if any(not value.strip() for value in texts) or any(value <= 0 for value in limits):
            raise ValueError("code executor contract is incomplete")

    def to_mapping(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "benchmark": self.benchmark,
            "python_version": self.python_version,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "cpu_seconds": self.cpu_seconds,
            "address_space_bytes": self.address_space_bytes,
            "process_count": self.process_count,
            "file_size_bytes": self.file_size_bytes,
            "open_file_count": self.open_file_count,
            "test_profile": self.test_profile,
        }


__all__ = ["CodeExecutorContract"]
