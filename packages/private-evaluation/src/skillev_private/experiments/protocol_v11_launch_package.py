"""Inspectable, deliberately non-executable Protocol 11 launch package."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol_v11_application_input import ProtocolV11ApplicationInput


@dataclass(frozen=True, slots=True)
class ProtocolV11LaunchPackage:
    application: ProtocolV11ApplicationInput
    total_steps: int = 320
    total_episodes: int = 5_120
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.total_steps * self.batch_size != self.total_episodes:
            raise ValueError("Protocol 11 launch shape does not close")

    def require_executable(self) -> None:
        self.application.protocol.require_execution_ready()


__all__ = ["ProtocolV11LaunchPackage"]
