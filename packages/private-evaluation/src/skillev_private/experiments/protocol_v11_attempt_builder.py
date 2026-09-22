"""Protocol 11 attempt admission remains closed until every unblock condition passes."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol_v11_launch_package import ProtocolV11LaunchPackage


@dataclass(frozen=True, slots=True)
class ProtocolV11AttemptBuilder:
    launch_package: ProtocolV11LaunchPackage

    def build(self) -> None:
        self.launch_package.require_executable()


__all__ = ["ProtocolV11AttemptBuilder"]
