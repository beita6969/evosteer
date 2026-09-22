"""Read-only identity for pre-Protocol-12 diagnostic artifacts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LegacyDiagnosticReceipt:
    format: str
    payload: dict[str, object]

    @property
    def permits_formal_admission(self) -> bool:
        return False


__all__ = ["LegacyDiagnosticReceipt"]
