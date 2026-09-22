"""Closed-gate Protocol 11 method application identity."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.experiments.protocol_v11 import ProtocolV11Spec

_METHODS = frozenset(
    {"skillflow-baseline", "bayesian-improve-full", "bayesian-improve-no-calibration"}
)


@dataclass(frozen=True, slots=True)
class ProtocolV11ApplicationInput:
    protocol: ProtocolV11Spec
    method: str
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError("Protocol 11 method is unsupported")
        if self.seed != 0:
            raise ValueError("Protocol 11 uses one seed: 0")
        if self.protocol.executable:
            raise ValueError("Protocol 11 gate must not be bypassed in application input")


__all__ = ["ProtocolV11ApplicationInput"]
